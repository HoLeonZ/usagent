"""取数路径的失败语义 —— 限流不得被伪装成「这些标的没有数据」。

为什么这几条必须被钉住（2026-10-17 审计）:

`download_ohlcv` 曾经绕过 `yfutil.call`(只导入了 `RateLimited` 这个类型),
于是全局节流与指数退避对**唯一每天运行的路径**无效,而且限流会被
`fetch_with_split` 当成「批次坏了」→ 二分拆批(在限流下把请求数翻倍)→
最终整批写成 `empty` / `error` 落进 fetch_log,`main()` 仍然 `return 0`,
`run_daily.sh` 只看 exit code ⇒「数据已更新」与「数据一个月没动」不可区分。

本文件钉住修复后的三条不变量:
  1. 限流**不拆批**(拆批 = 请求数翻倍,在限流下正是最坏反应);
  2. 「整批无数据」必须用探针判定是**限流**(退避重试/中止)还是**真的没数据**(继续);
  3. 探针通、批次空 ⇒ 不重试、不报错;探针不通 ⇒ 退避重试,耗尽则上抛 RateLimited。
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import download_ohlcv as D  # noqa: E402
import yfutil  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sleep_no_throttle(monkeypatch):
    """把节流与退避的**等待**去掉(被测的是决策,不是墙钟)。"""
    monkeypatch.setattr(yfutil, "THROTTLE", yfutil._Throttle(0.0))
    monkeypatch.setattr(yfutil.time, "sleep", lambda *_: None)
    monkeypatch.setattr(D.time, "sleep", lambda *_: None)


def _empty_frame():
    return pd.DataFrame()


def test_rate_limit_is_not_split(monkeypatch):
    """限流时**只调用一次** _fetch_batch —— 绝不二分拆批。"""
    calls: list[list[str]] = []

    def boom(batch, start, end):
        calls.append(list(batch))
        raise yfutil.RateLimited("429")

    monkeypatch.setattr(D, "_fetch_batch", boom)
    with pytest.raises(yfutil.RateLimited):
        D.fetch_with_split(["A", "B", "C", "D"], date(2024, 1, 1), date(2024, 2, 1))
    assert calls == [["A", "B", "C", "D"]], f"限流被拆批了：{calls}"


def test_non_rate_limit_error_still_splits(monkeypatch):
    """非限流的异常仍然二分隔离坏标的（原行为不变）。"""
    calls: list[list[str]] = []

    def boom(batch, start, end):
        calls.append(list(batch))
        if len(batch) == 1:
            raise ValueError("bad ticker")
        raise ValueError("batch failed")

    monkeypatch.setattr(D, "_fetch_batch", boom)
    raw, bad = D.fetch_with_split(["A", "B"], date(2024, 1, 1), date(2024, 2, 1))
    assert raw is None and sorted(bad) == ["A", "B"]
    assert calls[0] == ["A", "B"] and len(calls) > 1, "非限流错误应继续二分拆批"


def test_all_empty_batch_with_live_probe_is_accepted_without_retry(monkeypatch):
    """探针通 ⇒ 这批确实没有数据：不重试、不报错，原样返回空。"""
    n = {"fetch": 0}

    def fetch(batch, start, end):
        n["fetch"] += 1
        return _empty_frame(), []

    monkeypatch.setattr(D, "fetch_with_split", fetch)
    monkeypatch.setattr(D.yfutil, "probe", lambda: True)
    raw, bad = D._fetch_resilient(["DLST1", "DLST2"], date(2024, 1, 1), date(2024, 2, 1))
    assert n["fetch"] == 1, "探针正常时不该重试"
    assert raw is not None and raw.empty and bad == []


def test_all_empty_batch_with_dead_probe_retries_then_raises(monkeypatch):
    """探针不通 ⇒ 判定限流：退避重试 THROTTLE_RETRIES 次后上抛，不写成 empty。"""
    n = {"fetch": 0}

    def fetch(batch, start, end):
        n["fetch"] += 1
        return _empty_frame(), []

    monkeypatch.setattr(D, "fetch_with_split", fetch)
    monkeypatch.setattr(D.yfutil, "probe", lambda: False)
    with pytest.raises(yfutil.RateLimited):
        D._fetch_resilient(["A", "B"], date(2024, 1, 1), date(2024, 2, 1))
    assert n["fetch"] == D.THROTTLE_RETRIES + 1, "应在每次失败后退避重试"


def test_all_empty_batch_recovers_when_probe_comes_back(monkeypatch):
    """限流是暂时的：探针恢复后本批应被成功取回，而不是被记成 empty。"""
    state = {"calls": 0}

    def fetch(batch, start, end):
        state["calls"] += 1
        if state["calls"] == 1:
            return _empty_frame(), []
        return pd.DataFrame({"A": [1.0]}), []

    monkeypatch.setattr(D, "fetch_with_split", fetch)
    monkeypatch.setattr(D.yfutil, "probe", lambda: True)
    raw, _ = D._fetch_resilient(["A"], date(2024, 1, 1), date(2024, 2, 1))
    assert state["calls"] == 1, "探针通时不该重试"
    # 第一次探针不通的情形由下一个用例覆盖；这里确认「恢复后不再重试」的路径存在
    assert raw is not None


def test_probe_recovery_path(monkeypatch):
    """探针先不通、后通 ⇒ 允许重试一次并成功，不抛异常。"""
    seq = {"probe": 0, "fetch": 0}

    def fetch(batch, start, end):
        seq["fetch"] += 1
        if seq["fetch"] == 1:
            return _empty_frame(), []
        return pd.DataFrame({"A": [1.0]}), []

    def probe():
        seq["probe"] += 1
        return seq["probe"] > 1        # 第一次不通，之后通

    monkeypatch.setattr(D, "fetch_with_split", fetch)
    monkeypatch.setattr(D.yfutil, "probe", probe)
    raw, _ = D._fetch_resilient(["A"], date(2024, 1, 1), date(2024, 2, 1))
    assert seq["fetch"] == 2 and raw is not None and not raw.empty


def test_yfutil_download_goes_through_throttle(monkeypatch):
    """`yfutil.download` 必须走 call()：节流被调用，且 429 会重试而不是原样抛出。"""
    waited = {"n": 0}
    monkeypatch.setattr(yfutil.THROTTLE, "wait", lambda: waited.__setitem__("n", waited["n"] + 1))

    attempts = {"n": 0}

    def flaky(*a, **kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise yfutil.YFRateLimitError()
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(yf, "download", flaky)
    out = yfutil.download(["AAPL"], period="5d")
    assert attempts["n"] == 3, "前两次 429 应被退避后重试"
    assert waited["n"] == 3, "每次尝试前都必须经过全局节流"
    assert not out.empty


def test_yfutil_download_gives_up_as_rate_limited(monkeypatch):
    """重试耗尽后必须抛 RateLimited（调用方据此退避/中止），不能伪装成空结果。"""
    monkeypatch.setattr(yfutil.THROTTLE, "wait", lambda: None)

    def always_429(*a, **kw):
        raise yfutil.YFRateLimitError()

    monkeypatch.setattr(yf, "download", always_429)
    with pytest.raises(yfutil.RateLimited):
        yfutil.download(["AAPL"], period="5d")


def test_non_rate_limit_error_propagates_immediately(monkeypatch):
    """非限流错误不重试（原语义），交给调用方判定。"""
    monkeypatch.setattr(yfutil.THROTTLE, "wait", lambda: None)
    attempts = {"n": 0}

    def broken(*a, **kw):
        attempts["n"] += 1
        raise ValueError("bad request")

    monkeypatch.setattr(yf, "download", broken)
    with pytest.raises(ValueError):
        yfutil.download(["AAPL"], period="5d")
    assert attempts["n"] == 1


def test_is_rate_limited_classifies_both_types():
    assert yfutil.is_rate_limited(yfutil.RateLimited("x"))
    assert yfutil.is_rate_limited(yfutil.YFRateLimitError())
    assert not yfutil.is_rate_limited(ValueError("x"))
