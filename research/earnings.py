"""财报日历 —— 事件维度信号源(hiagent preset_search §6 指的那条路)。

为什么需要:§6 的原话是「若要取得可证实的 alpha,应**更换信号源**(引入基本面/事件
维度),而不是继续在价格派生的 A~E 条件上做参数或排序层面的改动」。

数据源:Nasdaq 官方免费 JSON,**无鉴权**。
    https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD
需要带浏览器 User-Agent,否则被拒。

为什么这件事重要(机制,不是直觉):
    财报是**不可预测的二值事件**,且以跳空形式释放。本策略用 1 ATR 的固定止损,
    跳空会直接穿透它 —— 止损价形同虚设。持有窗口 18 天里夹着财报,等于把一个
    无法定价的事件风险塞进一笔"技术面回调"里。这不是可以用价格派生条件修的东西,
    必须引入事件数据。

缓存:按日期增量写 `results/earnings_cache.json`,可断点续跑(已取过的日期不再请求)。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

API = "https://api.nasdaq.com/api/calendar/earnings?date={d}"
UA = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json",
}
CACHE = Path(__file__).resolve().parent / "results" / "earnings_cache.json"
THROTTLE_S = 0.30           # 对官方免费端点保持克制
MAX_RETRY = 3


class EarningsError(RuntimeError):
    """取数失败 —— fail-fast,不静默返回空日历。"""


def _read_cache(path: Path) -> dict[str, list[str]]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("缓存损坏,重建: %s", path)
    return {}


def _write_cache(path: Path, cache: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, sort_keys=True))
    tmp.replace(path)                      # 原子替换,避免半截文件


def _fetch_one(day: str) -> list[str]:
    last: Exception | None = None
    for attempt in range(MAX_RETRY):
        try:
            req = urllib.request.Request(API.format(d=day), headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = json.loads(r.read().decode())
            rows = (payload.get("data") or {}).get("rows") or []
            return sorted({str(x.get("symbol", "")).upper().strip()
                           for x in rows if x.get("symbol")})
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise EarningsError(f"{day} 取数失败: {last}")


def fetch_calendar(start: str, end: str, *, path: Path = CACHE,
                   throttle: float = THROTTLE_S,
                   workers: int = 1) -> dict[str, list[str]]:
    """抓 [start, end] 内每个**工作日**的财报日历,增量写入缓存。

    周末不会有财报,跳过以省请求。非交易日(假期)返回空列表也无害。

    workers > 1 时用线程池并发。实测本机到 Nasdaq 单请求延迟约 4.5s,
    串行抓 700 个工作日要 ~1 小时;6 个 worker 降到 ~8 分钟。
    对公开无鉴权端点保持克制:并发压在 6 以内,并逐请求留间隔。
    """
    days = [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(start, end)]
    cache = _read_cache(path)
    todo = [d for d in days if d not in cache]
    logger.info("日历: 共 %d 个工作日,已缓存 %d,待取 %d (workers=%d)",
                len(days), len(days) - len(todo), len(todo), workers)

    def _work(d: str) -> tuple[str, list[str]]:
        time.sleep(throttle)
        return d, _fetch_one(d)

    done = 0
    if workers <= 1:
        for d in todo:
            cache[d] = _fetch_one(d)
            done += 1
            if done % 25 == 0 or done == len(todo):
                _write_cache(path, cache)
                logger.info("[%d/%d] %s -> %d 只", done, len(todo), d, len(cache[d]))
            time.sleep(throttle)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_work, d) for d in todo]
            try:
                for fut in as_completed(futures):
                    d, syms = fut.result()
                    cache[d] = syms
                    done += 1
                    if done % 50 == 0 or done == len(todo):
                        _write_cache(path, cache)
                        logger.info("[%d/%d] %s -> %d 只", done, len(todo), d, len(syms))
            finally:
                for fut in futures:
                    fut.cancel()
                _write_cache(path, cache)      # 中断也落盘,保住已完成的

    _write_cache(path, cache)
    return cache


def load_calendar(path: Path = CACHE) -> dict[str, list[str]]:
    if not path.exists():
        raise EarningsError(
            f"没有财报缓存 {path} —— 先跑:\n"
            f"  python -m research.earnings --start 2024-04-01 --end 2026-10-14"
        )
    return _read_cache(path)


def earnings_dates_by_symbol(cache: dict[str, list[str]]) -> dict[str, list[pd.Timestamp]]:
    """倒排成 {symbol: [财报日]} —— 信号层要按标的查,不是按日查。"""
    out: dict[str, list[pd.Timestamp]] = {}
    for day, syms in cache.items():
        ts = pd.Timestamp(day)
        for s in syms:
            out.setdefault(s, []).append(ts)
    for s in out:
        out[s].sort()
    return out


def has_earnings_between(by_symbol: dict[str, list[pd.Timestamp]], symbol: str,
                         lo: pd.Timestamp, hi: pd.Timestamp) -> bool:
    """[lo, hi] 闭区间内该标的是否有财报 —— 二分查找。"""
    ds = by_symbol.get(symbol)
    if not ds:
        return False
    import bisect
    i = bisect.bisect_left(ds, lo)
    return i < len(ds) and ds[i] <= hi


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="抓 Nasdaq 财报日历到本地缓存")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--throttle", type=float, default=THROTTLE_S)
    ap.add_argument("--workers", type=int, default=6,
                    help="并发数;对公开端点保持克制,建议 <= 6")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cal = fetch_calendar(a.start, a.end, throttle=a.throttle, workers=a.workers)
    n_events = sum(len(v) for v in cal.values())
    print(f"完成: {len(cal)} 个日历日, {n_events} 条财报记录, 覆盖 {len(earnings_dates_by_symbol(cal))} 只标的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
