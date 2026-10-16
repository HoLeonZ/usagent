"""备用行情源：stockanalysis.com 的公开历史接口（**无 key**）。

为什么需要这一层（2026-10-17）：本机被 Yahoo 边缘封锁（`yf.download` 与
`Ticker.history` 都 429，`query{1,2}.finance.yahoo.com/v8/finance/chart` 直接 403），
而这是实测唯一一条**无 key**且深度够用的通道：

    实测 `?range=5Y` → 1255 根日线（2021-10-18 .. 2026-10-16），一只一个请求。
    ⚠️ `range=MAX` / `Max` / `max` 会**静默退回 1 年**（252 根）—— 必须写 `5Y`。

口径（逐条说清，否则会静默污染面板）:

  * 接口的 `c` = **已拆股复权、未分红复权**的收盘价 —— 与 Yahoo 的 Close **同一约定**。
    实测 NVDA 2024-06-10 10:1 拆股前后：120.888 → 121.79（量级未变，不是 1208.88）。
    所以它可以直接当本库的 `close` 落盘（本库一律存"拆股后价位"这一口径，
    `series.source_is_split_adjusted` 就是为这个约定写的）。
  * 接口**不提供**公司行动事件。分红用 `(a, c)` 反解：

        D(t) = c(t−1) × (1 − F(t−1)/F(t))，  F = a/c

    这正是 `series._back_factor` 的逆运算（那边是 `1 − D/前收`），所以反解出来的
    分红可以直接喂回本库现成的复权数学，不需要第二套口径。
    实测对照已知值：AAPL $0.26/$0.27、MSFT $0.91、PTEN $0.10、NVDA $0.004→$0.01→$0.25
    全部对上。
  * **拆股事件无法反解**（`c` 已经是拆股后价位，因子在拆股日不跳）⇒ 一律写 `splits=0`。
    这不影响价格口径（本库本来就不对拆股再缩放，见 series._back_factor 的注释），
    但 `validate.py` 的"公司行动已捕获"这一条只能靠 dividends 满足。
  * **噪声地板**：`a` 只有约 8 位有效数字，反解的除法噪声可达 1e-6 相对量级。
    不过门槛就会把舍入噪声当成分红（实测 TSLA 无分红却"反解出" 60 次 $0.0003）。
    门槛取 `DIV_FLOOR_REL`（见下），实测既清掉噪声（TSLA → 0 次），
    又保住低股息率名字（NVDA 拆股后 $0.004 档仍在）。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime

import pandas as pd

import yfutil  # 复用同一条限流语义（yfutil.RateLimited）与 is_rate_limited 分类

logger = logging.getLogger("vendor_stockanalysis")

BASE = "https://stockanalysis.com/api/symbol/s/{sym}/history"
RANGE = "5Y"              # ⚠️ 必须精确写 5Y；MAX/Max/max 会静默退回 1 年
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")

MIN_INTERVAL = 0.35       # 秒。与 Yahoo 通道同量级；本接口实测容错，但仍主动限速
TIMEOUT = 25
MAX_RETRY = 3
BACKOFF_BASE = 3.0

# 分红反解的相对噪声地板：|D| / c 必须 **大于** 它才算一次分红。
# 1e-5 = 万分之一：真实分红最低档（NVDA 拆股后 $0.004 / $120 ≈ 3.3e-5）仍在门槛之上，
# 而 (a,c) 的舍入噪声（实测 ~1e-6）落在门槛之下。
DIV_FLOOR_REL = 1e-5

# 分红不能离谱：超过前收的 30% 更可能是数据错误而不是分红（真实分红极少超过 10%）。
DIV_MAX_REL = 0.30

PROBE_SYMBOL = "AAPL"
PROBE_TIMEOUT = 20


class VendorError(RuntimeError):
    """本源的取数失败（非限流）。"""


class _Throttle:
    def __init__(self, min_interval: float) -> None:
        self._min, self._lock, self._last = min_interval, threading.Lock(), 0.0

    def wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self._min:
                time.sleep(self._min - gap)
            self._last = time.monotonic()


THROTTLE = _Throttle(MIN_INTERVAL)


def _get(url: str, timeout: int) -> dict:
    """一次 GET + 解析。带节流、退避重试；重试耗尽 ⇒ `yfutil.RateLimited`。

    抛 `yfutil.RateLimited` 而不是自定义类型，是为了让 `download_ohlcv` 的
    "限流不拆批 / 中止本轮"那条决策路径对两个源**完全一致** —— 限流的语义只能有一份。
    """
    last: Exception | None = None
    for attempt in range(MAX_RETRY + 1):
        THROTTLE.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                last = exc
                if attempt >= MAX_RETRY:
                    break
                delay = BACKOFF_BASE ** (attempt + 1)
                logger.warning("HTTP %s，%.0fs 后重试 (%d/%d)", exc.code, delay,
                               attempt + 1, MAX_RETRY)
                time.sleep(delay)
                continue
            if exc.code == 404:
                return {"data": None, "status": 404}      # 无此标的：是事实，不是故障
            raise VendorError(f"HTTP {exc.code}: {url}") from exc
        except Exception as exc:                          # noqa: BLE001
            last = exc
            if attempt >= MAX_RETRY:
                break
            time.sleep(BACKOFF_BASE ** (attempt + 1))
    raise yfutil.RateLimited(f"stockanalysis 重试 {MAX_RETRY} 次仍失败：{last}")


def probe() -> bool:
    """Yahoo 那侧 probe 的对应物：本源现在通不通（一次最小请求）。"""
    try:
        j = _get(BASE.format(sym=PROBE_SYMBOL.lower()) + "?range=1Y", PROBE_TIMEOUT)
    except Exception:  # noqa: BLE001 —— 探针只回答"通/不通"
        return False
    d = j.get("data") if isinstance(j, dict) else None
    return bool(d)


def _rows(payload: dict) -> list[dict]:
    d = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(d, dict):
        d = d.get("data")
    if not isinstance(d, list):
        return []
    return d


def derive_dividends(rows: list[dict], *, floor_rel: float = DIV_FLOOR_REL,
                     max_rel: float = DIV_MAX_REL) -> dict[str, float]:
    """从 (a, c) 反解分红：`{除息日: 每股现金}`。**只有现金**，不含拆股（见模块 docstring）。

    rows 必须按日期升序。返回的键是**除息日当天**（与 Yahoo 的 dividends 列对齐：
    Yahoo 也把 D 记在除息日那一行）。
    """
    out: dict[str, float] = {}
    for i in range(1, len(rows)):
        p, q = rows[i - 1], rows[i]
        try:
            cp, cq = float(p["c"]), float(q["c"])
            fp, fq = float(p["a"]) / cp, float(q["a"]) / cq
        except (KeyError, TypeError, ZeroDivisionError, ValueError):
            continue
        if not (cp > 0 and cq > 0 and fq > 0):
            continue
        d = cp * (1.0 - fp / fq)
        if d <= floor_rel * cp:
            continue
        if d > max_rel * cp:
            # 超过前收 30% 的"分红"几乎一定是数据问题：宁可报出来让人看，
            # 也不要静默写进账本（§0 Fail-Fast 的"不静默"精神）。
            logger.warning("%s 反解出离谱分红 %.4f（前收 %.4f，%.1f%%）—— 已忽略",
                           q.get("t"), d, cp, 100 * d / cp)
            continue
        out[str(q["t"])[:10]] = round(d, 6)
    return out


def fetch_history(symbol: str, start: date | None = None, end: date | None = None
                  ) -> pd.DataFrame:
    """取一只标的的日线，返回 **yfinance 同构** 的 DataFrame（index=date）。

    列名刻意与 `yf.download` 对齐（Open/High/Low/Close/Volume/Dividends/Stock Splits），
    这样 `download_ohlcv` 的整条写入链路（含厂商重述检测）不需要第二个分支 ——
    口径变换只发生在本模块这一处。
    """
    j = _get(BASE.format(sym=symbol.lower()) + f"?range={RANGE}", TIMEOUT)
    rows = _rows(j)
    if not rows:
        return pd.DataFrame()
    rows = sorted(rows, key=lambda r: str(r["t"]))
    divs = derive_dividends(rows)
    idx, recs = [], []
    for r in rows:
        t = pd.Timestamp(str(r["t"])[:10])
        if start is not None and t.date() < start:
            continue
        if end is not None and t.date() > end:
            continue
        idx.append(t)
        recs.append({
            "Open": float(r["o"]), "High": float(r["h"]), "Low": float(r["l"]),
            "Close": float(r["c"]), "Volume": float(r["v"]),
            "Dividends": float(divs.get(str(r["t"])[:10], 0.0)),
            # 见模块 docstring：c 已是拆股后价位 ⇒ 拆股事件在本源不可观测。
            "Stock Splits": 0.0,
        })
    if not idx:
        return pd.DataFrame()
    df = pd.DataFrame(recs, index=pd.DatetimeIndex(idx))
    df.index.name = "Date"
    return df


def fetch_batch(batch: list[str], start: date, end: date
                ) -> tuple[pd.DataFrame | None, list[str]]:
    """与 `download_ohlcv.fetch_with_split` 同构的一批取数（逐只请求）。

    返回 (MultiIndex 列、按 symbol 分层的 DataFrame 或 None, 判定为坏的标的列表)。
    单只失败不拖垮整批：坏标的进 bad 列表（与 Yahoo 路径的语义一致）。
    """
    frames: dict[str, pd.DataFrame] = {}
    bad: list[str] = []
    for sym in batch:
        try:
            df = fetch_history(sym, start=start, end=end)
        except yfutil.RateLimited:
            raise                    # 限流的语义只有一份：交给上游决定退避/中止
        except Exception as exc:     # noqa: BLE001
            logger.warning("%s 取数失败：%s", sym, str(exc)[:120])
            bad.append(sym)
            continue
        if df.empty:
            bad.append(sym)
            continue
        frames[sym] = df
    if not frames:
        return None, bad
    raw = pd.concat(frames, axis=1)   # 逐只都是唯一列名 ⇒ 天然是按 symbol 分层的 MultiIndex
    return raw, bad


def selfcheck(symbol: str = PROBE_SYMBOL) -> tuple[bool, str]:
    """自检：用**反解出的分红**重建总收益因子，与厂商自己的 `a` 对照。

    这是本模块唯一能证明"反解没说谎"的手段：如果推导出的分红是错的，
    重建出来的复权序列就不会与 `a` 吻合。用于 validate，不进热路径。
    """
    j = _get(BASE.format(sym=symbol.lower()) + f"?range={RANGE}", TIMEOUT)
    rows = sorted(_rows(j), key=lambda r: str(r["t"]))
    if len(rows) < 30:
        return False, f"{symbol} 行数不足（{len(rows)}），无法自检"
    divs = derive_dividends(rows)
    if not divs:
        return False, f"{symbol} 未反解出任何分红 —— 无法自检（选一只有分红的名字）"
    # 重建：F_rec(t) = ∏_{除息日 > t} (1 − D/前收)，与 a/c 逐点比
    factor_by_date = {str(r["t"])[:10]: float(r["a"]) / float(r["c"]) for r in rows}
    dates = [str(r["t"])[:10] for r in rows]
    prev_close = {dates[i]: float(rows[i - 1]["c"]) for i in range(1, len(rows))}
    f = 1.0
    rec = {}
    for d in reversed(dates):
        rec[d] = f
        if d in divs and d in prev_close and prev_close[d] > 0:
            f *= (1.0 - divs[d] / prev_close[d])
    worst = 0.0
    for d in dates:
        want, got = factor_by_date[d], rec[d]
        if want > 0:
            worst = max(worst, abs(got - want) / want)
    ok = worst < 0.02          # 2% 是"够用"的判据：分红反解得准，重建因子就该基本重合
    return ok, (f"{symbol}: 反解 {len(divs)} 次分红，重建复权因子与厂商 a 的最大相对偏差 "
                f"{worst:.4%}" + ("" if ok else "（>2% ⇒ 反解不可信，不要用这份分红）"))
