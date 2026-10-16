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
import yfutil
from config import (BACKOFF_BASE, BATCH_SIZE, HISTORY_YEARS, history_start,
                    load_universe)
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
        "dividends", "splits", "fetched_at"]

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


def _to_rows(sym: str, df: pd.DataFrame, now: str) -> list[tuple]:
    df = df.rename(columns=FIELDS)
    if "close" not in df.columns:
        return []
    df = df.dropna(subset=["close"])
    rows = []
    for ts, r in df.iterrows():
        d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
        rows.append((
            sym, d,
            _f(r.get("open")), _f(r.get("high")), _f(r.get("low")), _f(r.get("close")),
            _f(r.get("volume")), _f(r.get("dividends"), 0.0), _f(r.get("splits"), 0.0),
            now,
        ))
    return rows


def _f(v, default=None):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _fetch_batch(batch: list[str], start: date, end: date) -> pd.DataFrame:
    """一批标的的原始下载。

    必须经 `yfutil.download`（= 全局节流 + 指数退避 + 429 分类）—— 见 yfutil.download
    的 docstring：曾经这里的路径绕过了整个限流守卫，把限流伪装成「这些标的没有数据」。
    """
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
        if not yfutil.probe():
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
    end = today
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = f"ohlcv-{now}"

    db.init()
    total = 0
    # 逐只状态计数：返回给 main() 决定 exit code，并让「这轮到底成不成」在日志里可读。
    stats = {"ok": 0, "empty": 0, "error": 0, "rows": 0}
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
            for sym, df in frames:
                rows = _to_rows(sym, df, now)
                total += db.upsert_many(con, "ohlcv", COLS, rows)
                stats["ok"] += 1
                db.log(con, run_id, now, "ohlcv", sym, "ok", f"{len(rows)} rows")
                con.commit()   # 逐标的提交：别让写锁一直握着，阻塞其他脚本
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
    log.info("完成，写入 %d 行（成功 %d 只 / 失败 %d 只 / 无数据 %d 只）",
             total, stats["ok"], stats["error"], stats["empty"])
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="tight", choices=["all", "oi100", "tight"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true", help="不跳过已覆盖的标的")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None,
                    metavar="YYYY-MM-DD",
                    help="钉住数据窗口的右端(默认为今天);用于复现一份既有 vintage")
    ap.add_argument("--recent", type=int, default=None, metavar="DAYS",
                    help="增量模式：只拉最近 N 天（每日任务用，约 1 分钟）")
    a = ap.parse_args()
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
