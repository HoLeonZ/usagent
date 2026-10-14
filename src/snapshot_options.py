"""第二步：每日期权链快照 —— 唯一「不落盘就永久丢失」的数据。

为什么它是第一优先级：Yahoo 的期权链只有当前快照、没有任何历史版本
（option_chain 的 date 参数是「到期日」而不是「截至某日」）。今天不落盘，
今天的链就永久消失，事后无法回补。

落盘口径：
  - 只保留现货 ±10% 的行权价、DTE 7-120、最接近 ATM 的 N 个到期
    （全链全到期会落成几十 GB，而单腿只需要这些）
  - 快照日期 = 数据所反映的美东交易日，不是本地日期
  - 主键含 snapshot_date，所以同一天重复跑是幂等的

用法：
  python src/snapshot_options.py --universe tight
  python src/snapshot_options.py --universe oi100 --dry-run
  python src/snapshot_options.py --symbols NVDA,AAPL
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

import db
import yfutil
from config import (CHAIN_DTE_MAX, CHAIN_DTE_MIN, CHAIN_MAX_EXPIRIES,
                    CHAIN_MAX_STRIKES, CHAIN_STRIKE_BAND, load_universe)
from yfutil import RateLimited

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chain")
ET = ZoneInfo("America/New_York")

COLS = ["snapshot_date", "symbol", "expiry", "dte", "strike", "right",
        "bid", "ask", "last", "volume", "open_interest",
        "implied_volatility", "underlying_price", "fetched_at"]


def resolve_snapshot_date(now_utc: datetime, con) -> date:
    """快照日期 = 数据反映的美东交易日。

    优先用价格表里最新的交易日（权威）；否则按日历推（美东 17:00 后算当日，
    否则算前一工作日）。
    """
    row = con.execute("SELECT MAX(date) FROM ohlcv").fetchone()
    if row and row[0]:
        return date.fromisoformat(row[0])
    et = now_utc.astimezone(ET)
    d = et.date() if et.hour >= 17 else et.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def is_standard_monthly(d: date) -> bool:
    """标准月度 = 每月第三个周五。报告 §4.1 实测：选错到期会让价差差一个数量级。"""
    return d.weekday() == 4 and 15 <= d.day <= 21


def pick_expiries(symbol: str, snapshot: date) -> list[tuple[str, int]]:
    t = yf.Ticker(symbol)
    exps = yfutil.expirations(t)
    if not exps:
        return []
    scored = []
    for e in exps:
        d = date.fromisoformat(e)
        dte = (d - snapshot).days
        if not (CHAIN_DTE_MIN <= dte <= CHAIN_DTE_MAX):
            continue
        # 标准月度优先，其次按距离 45 天排
        scored.append((0 if is_standard_monthly(d) else 1, abs(dte - 45), e, dte))
    scored.sort()
    return [(e, dte) for _, _, e, dte in scored[:CHAIN_MAX_EXPIRIES]]


def rows_for(symbol: str, snapshot: str, now: str) -> tuple[list[tuple], str]:
    snap_d = date.fromisoformat(snapshot)
    picked = pick_expiries(symbol, snap_d)
    if not picked:
        return [], "no_expiry_in_window"

    rows: list[tuple] = []
    for expiry, dte in picked:
        t = yf.Ticker(symbol)
        ch = yfutil.chain(t, expiry)
        spot = None
        try:
            spot = float(ch.underlying["regularMarketPrice"])
        except Exception:
            spot = None
        if not spot or spot != spot:
            continue
        lo, hi = spot * (1 - CHAIN_STRIKE_BAND), spot * (1 + CHAIN_STRIKE_BAND)
        # 先取两个方向的行权价并集，再挑「最接近现货的 N 档」——
        # 用并集是为了让 call/put 落在同一组行权价上（后续做偏斜要用）。
        strikes = set()
        for frame in (ch.calls, ch.puts):
            if frame is None or len(frame) == 0:
                continue
            col = frame["strike"]
            strikes |= set(col[(col >= lo) & (col <= hi)].tolist())
        if not strikes:
            continue
        keep = sorted(strikes, key=lambda k: abs(k - spot))[:CHAIN_MAX_STRIKES]
        keep_set = set(keep)

        for right, frame in (("C", ch.calls), ("P", ch.puts)):
            if frame is None or len(frame) == 0:
                continue
            f = frame[frame["strike"].isin(keep_set)]
            for _, r in f.iterrows():
                rows.append((
                    snapshot, symbol, expiry, dte, float(r["strike"]), right,
                    _f(r.get("bid")), _f(r.get("ask")), _f(r.get("lastPrice")),
                    _f(r.get("volume")), _f(r.get("openInterest")),
                    _f(r.get("impliedVolatility")), spot, now,
                ))
    return rows, "ok" if rows else "empty_after_filter"



def _f(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def preflight() -> tuple[bool, str]:
    """飞行前检查：Yahoo 在美东深夜会把期权报价整体清零。

    现象（2026-10-14 实测）：美东 23:48 时 AAPL/NVDA 的平值档 bid/ask 正常，
    到 00:51 变成全 0，且一直持续到次日开盘前。此时整批链都无意义
    （20,532 行里 19,921 行 bid=0），反解不出任何 IV。

    宁可拒绝写入，也不要往库里落 2 万行垃圾 —— 那会污染 IV 序列且很难察觉。
    """
    t = yf.Ticker("AAPL")
    exps = yfutil.expirations(t)
    if not exps:
        return False, "AAPL 无到期日（网络或接口异常？）"
    ch = yfutil.chain(t, exps[0])
    try:
        spot = float(ch.underlying["regularMarketPrice"])
        row = ch.calls.iloc[(ch.calls.strike - spot).abs().argsort()].iloc[0]
        bid, ask = float(row["bid"]), float(row["ask"])
    except Exception as exc:
        return False, f"探针解析失败：{exc}"
    if bid <= 0 and ask <= 0:
        return False, (f"AAPL 平值档报价为 0（bid={bid} ask={ask}）—— "
                       "Yahoo 正处于隔夜清零窗口。请在美东收盘后至 23:45 之间重跑"
                       "（北京 05:15–11:45 夏令时 / 06:15–12:45 冬令时）")
    return True, f"AAPL 平值档 bid={bid} ask={ask}（报价有效）"


def snapshot(universe_name: str, symbols: list[str] | None, dry: bool,
             force: bool = False) -> int:
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat(timespec="seconds")
    run_id = f"chain-{now}"
    db.init()

    ok, msg = preflight()
    log.info("飞行前检查：%s", msg)
    if not ok and not force:
        log.error("中止：%s", msg)
        return -1
    if not ok:
        log.warning("--force：忽略飞行前检查，继续（写入的数据可能无意义）")

    if symbols:
        syms = symbols
    else:
        syms = list(load_universe(universe_name).symbols)

    total = 0
    with db.session() as con:
        snapshot_date = resolve_snapshot_date(now_dt, con)
        log.info("快照日期 = %s（美东交易日）| 标的 %d 只 | dry_run=%s",
                 snapshot_date, len(syms), dry)
        existing = {r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM option_chain WHERE snapshot_date=?",
            (snapshot_date.isoformat(),))}
        if existing:
            log.info("该日期已有 %d 只，将覆盖重写", len(existing))

        for i, sym in enumerate(syms, 1):
            try:
                rows, status = rows_for(sym, snapshot_date.isoformat(), now)
            except RateLimited as exc:
                log.error("限流中止于 %s：%s（已写入保留，重跑续）", sym, exc)
                raise
            except Exception as exc:
                db.log(con, run_id, now, "chain", sym, "error", str(exc)[:200])
                continue

            if not dry:
                total += db.upsert_many(con, "option_chain", COLS, rows)
                db.log(con, run_id, now, "chain", sym, status, f"{len(rows)} rows")
                if i % 25 == 0:
                    con.commit()
            if i % 25 == 0 or i == len(syms):
                log.info("[%d/%d] %s 累计 %d 行", i, len(syms), sym, total)

        if not dry:
            con.commit()
            db.refresh_coverage(con, "chain", "option_chain", "snapshot_date", now,
                               [s for s in syms])

    log.info("完成，写入 %d 行", total)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="tight", choices=["all", "oi100", "tight"])
    ap.add_argument("--symbols", default=None, help="逗号分隔，覆盖 universe")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="跳过飞行前检查（报价清零时仍写入，数据可能无意义）")
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    try:
        rc = snapshot(a.universe, syms, a.dry_run, force=a.force)
        return 3 if rc == -1 else 0
    except RateLimited as exc:
        log.error("被限流中止：%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
