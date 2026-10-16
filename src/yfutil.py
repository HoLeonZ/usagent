"""Yahoo 访问层：全局节流 + 指数退避重试。

为什么需要这个：本轮调研里 Yahoo 的配额被我打爆过两次（6 线程跑 997 只全 429）。
Yahoo 不公布额度、响应里也没有 Retry-After 头，所以只能被动限速 + 退避。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, TypeVar

from config import BACKOFF_BASE, MAX_RETRY, MIN_INTERVAL

log = logging.getLogger("yfutil")
T = TypeVar("T")

# 探针标的与超时。探针只回答「Yahoo 现在通不通」，不参与任何研究结论，
# 所以固定用一只必然存在的大盘股即可（换标的没有意义，且会引入可调参数）。
PROBE_SYMBOL = "AAPL"
PROBE_TIMEOUT = 20  # 秒

try:  # yfinance 各版本的异常位置不完全一致
    from yfinance.exceptions import YFRateLimitError
except Exception:  # pragma: no cover
    class YFRateLimitError(Exception):
        pass


class RateLimited(Exception):
    """重试耗尽后仍被限流。"""


class _Throttle:
    """进程内全局节流：所有线程共享同一个最小间隔。"""

    def __init__(self, min_interval: float) -> None:
        self._min = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            gap = now - self._last
            if gap < self._min:
                time.sleep(self._min - gap)
            self._last = time.monotonic()


THROTTLE = _Throttle(MIN_INTERVAL)


def call(fn: Callable[..., T], *args: Any, what: str = "", **kwargs: Any) -> T:
    """执行一次 Yahoo 调用，带节流与退避。

    抛出 RateLimited 表示重试耗尽；其余异常原样抛出。
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRY + 1):
        THROTTLE.wait()
        try:
            return fn(*args, **kwargs)
        except YFRateLimitError as exc:
            last_exc = exc
            if attempt >= MAX_RETRY:
                break
            delay = BACKOFF_BASE ** (attempt + 1)
            log.warning("429 限流 %s，%.0fs 后重试 (%d/%d)", what, delay, attempt + 1, MAX_RETRY)
            time.sleep(delay)
        except Exception as exc:  # 非限流错误不重试，交给调用方
            raise
    raise RateLimited(f"{what}: 重试 {MAX_RETRY} 次仍被限流") from last_exc


def history(ticker_obj: Any, **kwargs: Any) -> Any:
    from config import OHLCV_ACTIONS, OHLCV_AUTO_ADJUST
    kw = dict(auto_adjust=OHLCV_AUTO_ADJUST, actions=OHLCV_ACTIONS)
    kw.update(kwargs)
    return call(ticker_obj.history, **kw, what=f"history({getattr(ticker_obj,'ticker','?')})")


def download(symbols: list[str], **kwargs: Any) -> Any:
    """**批量下载的唯一受控入口**（`yf.download` 必须经此调用）。

    为什么必须有这一层（2026-10-17 审计 §4/§0）：

      `download_ohlcv` 曾经只 `from yfutil import RateLimited` —— 导入了异常类型，
      却**没有把调用接进 `call()`**。于是本模块的全局节流与指数退避（config 的
      MIN_INTERVAL / MAX_RETRY / BACKOFF_BASE，动机写在模块 docstring：「配额被打爆过
      两次」）对**唯一每天运行的生产路径完全无效**：限流来了既不退避，还被
      `fetch_with_split` 当成「这批标的坏了」，二分拆批 —— 在限流下把请求数翻倍，
      最后把整批标成 `empty` 写进 fetch_log，而 `main()` 仍 `return 0`，
      `run_daily.sh` 只看 exit code ⇒「数据已更新」与「数据一个月没动」不可区分。

      本函数把 download 接回 `call()`，于是：节流真实生效、429 触发退避与重试、
      重试耗尽抛 `RateLimited`（而不是伪装成「这些标的没有数据」）。
    """
    import yfinance as yf
    return call(yf.download, symbols, what=f"download({len(symbols)} 只)", **kwargs)


def is_rate_limited(exc: BaseException) -> bool:
    """这个异常是否表示「被 Yahoo 限流」。

    调用方用它决定「退避重试」还是「当成数据问题」——两者绝不能混。
    """
    return isinstance(exc, (YFRateLimitError, RateLimited))


def probe() -> bool:
    """探针：Yahoo 现在还能不能取到数据。

    用途是在「整批无数据」这个**有歧义**的信号面前做一次判定：整批无数据既可能是
    被限流（假死），也可能是这批标的确实都没有数据（退市、坏 ticker）。区别在于
    探针能不能拿到数据。

    探针标的选择口径：用池子里**必然存在**的高流动性大盘股，且只取最近 5 个自然日
    （一次最小请求）。探针失败一律返回 False（把「不确定」当作「不可用」，
    宁可多退避一次，也不要静默把限流写成一堆 empty）。
    """
    import yfinance as yf
    try:
        THROTTLE.wait()
        df = yf.download([PROBE_SYMBOL], period="5d", auto_adjust=False, actions=True,
                         progress=False, group_by="ticker", threads=False,
                         timeout=PROBE_TIMEOUT)
    except Exception:  # noqa: BLE001 —— 探针只回答「通/不通」
        return False
    return df is not None and not df.empty


def expirations(ticker_obj: Any) -> tuple[str, ...]:
    return call(lambda: ticker_obj.options, what=f"options({getattr(ticker_obj,'ticker','?')})")


def chain(ticker_obj: Any, expiry: str) -> Any:
    return call(ticker_obj.option_chain, expiry,
                what=f"chain({getattr(ticker_obj,'ticker','?')},{expiry})")
