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


def expirations(ticker_obj: Any) -> tuple[str, ...]:
    return call(lambda: ticker_obj.options, what=f"options({getattr(ticker_obj,'ticker','?')})")


def chain(ticker_obj: Any, expiry: str) -> Any:
    return call(ticker_obj.option_chain, expiry,
                what=f"chain({getattr(ticker_obj,'ticker','?')},{expiry})")
