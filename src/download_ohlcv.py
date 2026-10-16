"""第一步：下载 3 年日线 OHLCV（未复权 + 公司行动）。

口径说明（对应报告 §3.3 的硬规则）：
  存 as-traded 原始价 + 拆股/分红事件，复权在读取时现算。
  绝不落库「前复权序列」——它对增量 append 会造出假单日跌幅。

用法：
  python src/download_ohlcv.py --universe tight
  python src/download_ohlcv.py --universe oi100 --workers 1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

import db
import vendor_stockanalysis as vendor_sa
import yfutil
from config import (BACKOFF_BASE, BATCH_SIZE, HISTORY_YEARS, RESTATE_MIN_REL_DIFF,
                    history_start, load_universe)
from yfutil import RateLimited

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ohlcv")

# 关掉库内部的静默重试：它会和外层退避叠加，把单次调用拖成几分钟
try:
    yf.config.network.retries = 0
except Exception:  # pragma: no cover
    pass

REQUEST_TIMEOUT = 20  # 秒。Yahoo 慢速限流时会吊住连接，必须有上限

COLS = ["symbol", "date", "open", "high", "low", "close", "volume",
        "dividends", "splits", "fetched_at", "source"]

FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
          "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits"}


def _frames_from_batch(raw: pd.DataFrame, batch: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """把 yf.download 的返回拆成 (symbol, DataFrame)。"""
    out: list[tuple[str, pd.DataFrame]] = []
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        for sym in batch:
            if sym not in lvl0:
                continue
            sub = raw[sym]
            if sub.dropna(how="all").empty:
                continue
            out.append((sym, sub))
    else:  # 单标的
        if not raw.dropna(how="all").empty:
            out.append((batch[0], raw))
    return out


# 物理上不可能的行计数（OHLC 自洽性）。落库前拦掉，而不是让它们污染面板。
_IMPOSSIBLE = {"ohlc": 0}


def _to_rows(sym: str, df: pd.DataFrame, now: str, source: str) -> list[tuple]:
    """把一只标的的 DataFrame 转成待写入的行。

    ⚠️ **物理上不可能的行不落库**（§0：宁可拒收，也不要往库里落不可能的价）。
    实测：本数据源有 470 行满足 `close ∉ [low, high]`，且全部集中在**最后一根 bar**
    （vendor 当日快照未定型/收盘口径不一致）。这类行一旦入库，会污染该日的
    ret1/ATR/触发判定 —— 而它们在物理上根本不可能发生。
    """
    df = df.rename(columns=FIELDS)
    if "close" not in df.columns:
        return []
    df = df.dropna(subset=["close"])
    rows = []
    for ts, r in df.iterrows():
        d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
        o, h, lo, c = _f(r.get("open")), _f(r.get("high")), _f(r.get("low")), _f(r.get("close"))
        if None not in (o, h, lo, c) and (h < max(o, c) - 1e-4 or lo > min(o, c) + 1e-4):
            _IMPOSSIBLE["ohlc"] += 1
            continue                       # 不可能的 OHLC：拒收（走日志，不静默计数）
        rows.append((
            sym, d,
            _f(r.get("open")), _f(r.get("high")), _f(r.get("low")), _f(r.get("close")),
            _f(r.get("volume")), _f(r.get("dividends"), 0.0), _f(r.get("splits"), 0.0),
            now, source,
        ))
    return rows


def _f(v, default=None):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


# ---------------------------------------------------------------- 厂商重述检测与修复
#
# 缺陷本体（2026-10-17 审计 A1）：每日任务是 `--recent 10` 的增量刷新，而写入是
# INSERT OR REPLACE、(symbol,date) 为主键 ⇒ 只有最近 10 天会被改写。厂商在拆股后
# 会把**该日之前的整段历史**重述，于是更早的行永久停在旧基数，在窗口左边界留下
# 一个等于累计拆股比的假单日跌幅。检测阈值见 config.RESTATE_MIN_REL_DIFF 的注释。

def _stored_closes(con, symbol: str, dates: list[str]) -> dict[str, float]:
    """写入**之前**读取这些日期上已落盘的 close。"""
    if not dates:
        return {}
    q = ",".join("?" * len(dates))
    rows = con.execute(
        f"SELECT date, close FROM ohlcv WHERE symbol=? AND date IN ({q})",
        [symbol, *dates]).fetchall()
    return {d: c for d, c in rows if c is not None}


def restated_dates(stored: dict[str, float], fresh: dict[str, float],
                   min_rel_diff: float = RESTATE_MIN_REL_DIFF
                   ) -> list[tuple[str, float, float]]:
    """重叠日期上「新取的 close」与「已落盘的 close」不一致 ⇒ 厂商重述过这段历史。

    返回 [(date, 旧值, 新值)]（按日期升序）。纯函数 —— 判定逻辑与 IO 分开，便于测试。

    只比**重叠**日期：不在旧集合里的日期是无从谈起"变化"的新行；旧值为空或非正的行
    也不参与（没有可比基准）。
    """
    out: list[tuple[str, float, float]] = []
    for d, new in fresh.items():
        old = stored.get(d)
        if old is None or new is None or old <= 0 or new <= 0:
            continue
        if abs(new - old) / old > min_rel_diff:
            out.append((d, float(old), float(new)))
    return sorted(out)


def repair_symbol(con, sym: str, start: date, end: date, now: str, run_id: str) -> int:
    """**整段重下并重写**一只标的 —— 厂商重述后的唯一正确反应。

    为什么不能局部修补：重述影响的是当前增量窗口**之外**的更早历史（拆股日之前的
    全部行）。只见窗口的行就只能把断层从窗口左边界挪到别处，问题依旧存在。
    整段重下之后，该标的的整条序列来自**同一个 vintage** —— 这才是要钉住的不变量。

    代价是这只标的一次全量请求；触发条件是"厂商真的改了历史"，所以它的频率等于
    拆股/修错价的频率，不会成为常态开销。
    """
    raw, _bad = fetch_with_split([sym], start, end)
    frames = _frames_from_batch(raw, [sym]) if raw is not None else []
    n = 0
    for _s, df in frames:
        n += db.upsert_many(con, "ohlcv", COLS, _to_rows(_s, df, now, SOURCE))
    log.warning("厂商重述 ⇒ 整段重下重写 %s（%d 行）", sym, n)
    db.log(con, run_id, now, "ohlcv", sym, "repaired", f"整段重写 {n} 行")
    return n


def write_symbol(con, sym: str, df: pd.DataFrame, now: str, run_id: str
                 ) -> tuple[int, list[tuple[str, float, float]]]:
    """写入一只标的（先检测重述、再写）。返回 (写入行数, 重述日期列表)。"""
    rows = _to_rows(sym, df, now, SOURCE)
    if not rows:
        return 0, []
    fresh = {r[1]: r[5] for r in rows}          # COLS: ..., date(1), ..., close(5), ...
    drift = restated_dates(_stored_closes(con, sym, list(fresh)), fresh)
    n = db.upsert_many(con, "ohlcv", COLS, rows)
    db.log(con, run_id, now, "ohlcv", sym, "ok", f"{n} rows")
    if drift:
        d0, old, new = drift[0]
        db.log(con, run_id, now, "ohlcv", sym, "restated",
               f"{len(drift)} 个重叠日期的 close 变了,例 {d0} {old:.4f}->{new:.4f}")
    return n, drift


# 取数源。两源的**价格口径相同**（都已拆股复权、未分红复权），差别在事件来源：
#   yahoo          —— Close + 显式的 dividends/splits 两列（本库原来的唯一路径）
#   stockanalysis  —— Close + 由 (a,c) **反解**的分红；拆股事件不可观测（见该模块 docstring）
# 限流语义两源共用（vendor 抛的也是 yfutil.RateLimited），所以上游的
# "限流不拆批 / 中止本轮"决策路径对两个源完全一致。
SOURCE_YAHOO = "yahoo"
SOURCE_SA = "stockanalysis"
SOURCE = SOURCE_YAHOO          # 由 main() 按 --source 覆盖


def _source_probe() -> bool:
    """当前源的探针。两源的探针语义一致：只回答"服务现在通不通"。"""
    return vendor_sa.probe() if SOURCE == SOURCE_SA else yfutil.probe()


def _fetch_batch(batch: list[str], start: date, end: date) -> pd.DataFrame:
    """一批标的的原始下载。

    Yahoo 侧必须经 `yfutil.download`（= 全局节流 + 指数退避 + 429 分类）—— 见 yfutil.download
    的 docstring：曾经这里的路径绕过了整个限流守卫，把限流伪装成「这些标的没有数据」。
    备用源侧同样带节流与退避（vendor_stockanalysis._get），并抛同一种 RateLimited。
    """
    if SOURCE == SOURCE_SA:
        return vendor_sa.fetch_batch(batch, start, end)[0]
    return yfutil.download(
        batch, start=start.isoformat(), end=end.isoformat(),
        auto_adjust=False, actions=True, progress=False,
        group_by="ticker", threads=False, timeout=REQUEST_TIMEOUT,
    )


def fetch_with_split(batch: list[str], start: date, end: date,
                     depth: int = 0) -> tuple[pd.DataFrame | None, list[str]]:
    """下载一批；失败就二分拆批，把坏标的隔离出来。

    返回 (DataFrame 或 None, 判定为坏的标的列表)。

    ⚠️ 限流**不拆批**（`RateLimited` 直接上抛）：拆批会把请求数翻倍，在限流下正是
       最不该做的事；而且它会把「服务不可用」写成「这些标的没有数据」——
       两类事实混进同一张 fetch_log，运维上再也分不开。
    """
    try:
        return _fetch_batch(batch, start, end), []
    except RateLimited:
        raise
    except Exception as exc:
        if len(batch) == 1 or depth >= 4:
            log.warning("放弃 %s: %s", batch, str(exc)[:120])
            return None, list(batch)
        mid = len(batch) // 2
        log.info("批次失败（%s），拆成 %d + %d 重试",
                 str(exc)[:60], mid, len(batch) - mid)
        a, bad_a = fetch_with_split(batch[:mid], start, end, depth + 1)
        b, bad_b = fetch_with_split(batch[mid:], start, end, depth + 1)
        if a is None:
            return b, bad_a + bad_b
        if b is None:
            return a, bad_a + bad_b
        return pd.concat([a, b], axis=1), bad_a + bad_b


# 整批无数据时的重试次数。只在**探针确认 Yahoo 不通**时才重试（见 _fetch_resilient）。
THROTTLE_RETRIES = 3


def _fetch_resilient(batch: list[str], start: date, end: date
                     ) -> tuple[pd.DataFrame | None, list[str]]:
    """取一批，并对「整批无数据」这个有歧义的信号做一次显式判定。

    整批无数据有两种完全不同的成因，必须分开（§0 Fail-Fast / §1 时间确定性）：

      * **被限流（假死）**：Yahoo 活着、但对我们 429，`yf.download` 会返回空表而不抛错。
        正确反应是退避重试，重试耗尽则**中止本轮**并让 run_daily 报错 ——
        绝不写成一个「这批标的就是没数据」的既成事实。
      * **这批标的确实没有数据**：退市、改名、坏 ticker。这是正常事实，记 `empty` 后继续。

    判据只有一条：探针（`yfutil.probe`）通不通。不引入任何新参数。
    """
    for attempt in range(THROTTLE_RETRIES + 1):
        raw, bad = fetch_with_split(batch, start, end)
        frames = _frames_from_batch(raw, batch) if raw is not None else []
        if frames or bad:
            return raw, bad
        if not _source_probe():
            if attempt >= THROTTLE_RETRIES:
                raise RateLimited(
                    f"整批 {len(batch)} 只无数据且探针连续 {attempt + 1} 次失败 —— "
                    f"判定为被限流，中止本轮（已写入的数据保留，重跑自动续）")
            delay = BACKOFF_BASE ** (attempt + 1)
            log.warning("整批无数据且探针不通（疑似限流），%.0fs 后重试本批 %d/%d",
                        delay, attempt + 1, THROTTLE_RETRIES)
            time.sleep(delay)
            continue
        log.info("整批无数据但探针正常 ⇒ 这批确实没有数据（退市/坏 ticker %d 只）",
                 len(batch))
        return raw, bad
    raise RateLimited("不可达")  # pragma: no cover —— 循环内必然 return 或 raise


def download(universe_name: str, limit: int | None, skip_covered: bool,
             recent_days: int | None = None,
             as_of: date | None = None) -> dict:
    """recent_days 为 None 时做 3 年全量回补；给定值则只拉最近 N 天做增量刷新。

    返回逐只状态计数 {ok, empty, error, rows} —— `main()` 用它决定 exit code。
    被限流时抛 `yfutil.RateLimited`（已写入的数据保留，重跑自动续）。
    """
    uni = load_universe(universe_name)
    symbols = list(uni.symbols[:limit] if limit else uni.symbols)
    # as_of 用来**钉住**数据 vintage:默认取今天,给了值就按给定日期算窗口。
    # 不钉住的话同一份代码在两个日期会下到两份不同的数据,结果不可复现(§0)。
    today = as_of or date.today()
    if recent_days:
        start = today - timedelta(days=recent_days)
    else:
        start = history_start(today)
    # 整段重下用的起点：与本次的增量窗口无关。重述修复必须覆盖 3 年全历史，
    # 否则"单一时序 vintage"这条不变量不成立（见 repair_symbol 的 docstring）。
    full_start = history_start(today)
    end = today
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = f"ohlcv-{now}"

    db.init()
    total = 0
    # 逐只状态计数：返回给 main() 决定 exit code，并让「这轮到底成不成」在日志里可读。
    stats = {"ok": 0, "empty": 0, "error": 0, "restated": 0, "repaired": 0, "rows": 0}
    with db.session() as con:
        # 增量模式必须刷所有标的；全量模式才跳过已覆盖的
        if skip_covered and recent_days is None:
            have = db.covered_symbols(con, "ohlcv", (end.replace(day=1)).isoformat())
            before = len(symbols)
            symbols = [s for s in symbols if s not in have]
            log.info("跳过已覆盖 %d 只，剩 %d 只", before - len(symbols), len(symbols))

        log.info("下载 %s: %d 只, %s → %s", universe_name, len(symbols), start, end)
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            raw, bad = _fetch_resilient(batch, start, end)

            frames = _frames_from_batch(raw, batch) if raw is not None else []
            got = {s for s, _ in frames}
            to_repair: list[str] = []
            for sym, df in frames:
                n, drift = write_symbol(con, sym, df, now, run_id)
                total += n
                stats["ok"] += 1
                if drift:
                    stats["restated"] += 1
                    to_repair.append(sym)
                con.commit()   # 逐标的提交：别让写锁一直握着，阻塞其他脚本
            # 重述修复必须在**本批写完**之后做：它要拿整段历史覆盖刚才写的增量行。
            # 放在这里而不是延后到全量结束后，是为了让「同一只标的的历史来自单一
            # vintage」这条不变量在每一批结束时都成立（失败也只影响这一批）。
            for sym in to_repair:
                total += repair_symbol(con, sym, start=full_start, end=end,
                                       now=now, run_id=run_id)
                stats["repaired"] += 1
                con.commit()
            for s in batch:
                if s in got:
                    continue
                status = "error" if s in bad else "empty"
                stats[status] += 1
                db.log(con, run_id, now, "ohlcv", s, status,
                       "download failed" if s in bad else "no rows returned")

            con.commit()
            db.refresh_coverage(con, "ohlcv", "ohlcv", "date", now, batch)
            log.info("[%d/%d] 本批 %d/%d 只有数据，累计 %d 行（失败 %d / 无数据 %d）",
                     min(i + BATCH_SIZE, len(symbols)), len(symbols),
                     len(got), len(batch), total, stats["error"], stats["empty"])
            time.sleep(0.4)

    stats["rows"] = total
    stats["impossible_ohlc"] = _IMPOSSIBLE["ohlc"]
    if _IMPOSSIBLE["ohlc"]:
        log.warning("拒收 %d 行不可能的 OHLC（close 落在 [low, high] 之外）—— "
                    "vendor 当日快照未定型是已知成因，下一轮增量刷新会重取",
                    _IMPOSSIBLE["ohlc"])
    log.info("完成，写入 %d 行（成功 %d 只 / 失败 %d 只 / 无数据 %d 只 / "
             "重述 %d 只→已整段重写 %d 只 / 拒收 %d 行）",
             total, stats["ok"], stats["error"], stats["empty"],
             stats["restated"], stats["repaired"], stats["impossible_ohlc"])
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="tight", choices=["all", "oi100", "tight"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true", help="不跳过已覆盖的标的")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None,
                    metavar="YYYY-MM-DD",
                    help="钉住数据窗口的右端(默认为今天);用于复现一份既有 vintage")
    ap.add_argument("--source", default=SOURCE_YAHOO,
                    choices=[SOURCE_YAHOO, SOURCE_SA],
                    help="取数源（两源价格口径相同，事件来源不同；见 download_ohlcv 头注释）")
    ap.add_argument("--recent", type=int, default=None, metavar="DAYS",
                    help="增量模式：只拉最近 N 天（每日任务用，约 1 分钟）")
    a = ap.parse_args()
    global SOURCE
    SOURCE = a.source
    log.info("取数源：%s", SOURCE)
    try:
        stats = download(a.universe, a.limit, not a.no_skip, recent_days=a.recent,
                         as_of=a.as_of)
    except RateLimited as exc:
        log.error("被限流中止：%s（已写入的数据保留，重跑会自动续）", exc)
        return 2
    # exit code 的语义（run_daily.sh 只看 exit code，所以这里必须说真话）：
    #   0 = 没有失败；3 = 有标的**抛错**（真问题，需要人看）
    #   `empty`（退市/坏 ticker）不算失败：它每天都一样，把它当失败会训练出
    #   「反正每天都红」的告警疲劳，真问题反而被淹没。系统性陈旧由
    #   validate.py 的新鲜度检查负责（那才是「数据其实没更新」的正确告警点）。
    if stats["error"]:
        log.error("本轮有 %d 只标的下载报错（成功 %d / 无数据 %d）",
                  stats["error"], stats["ok"], stats["empty"])
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
