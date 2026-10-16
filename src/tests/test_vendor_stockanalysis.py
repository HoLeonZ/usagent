"""备用行情源：(a,c) 反解分红必须与已知值对得上，且噪声不会被当成分红。

背景（2026-10-17）：Yahoo 被本机封了（429 + 403），换成 stockanalysis.com 这条
无 key 通道。它的 `c` 与 Yahoo 的 Close **同口径**（已拆股复权、未分红复权），
但**不提供公司行动事件** —— 分红只能从 (a, c) 反解：

    D(t) = c(t−1) × (1 − F(t−1)/F(t))，F = a/c

这是 `series._back_factor` 的逆运算，所以反解结果能直接喂回本库现成的复权数学。
本文件钉住三件事：
  1. 反解公式在**解析可验**的合成数据上精确成立；
  2. 相对噪声地板真的挡住舍入噪声（这是实测踩到的：TSLA 无分红却"反解出" 60 次）；
  3. 无 key 通道的失败语义与 Yahoo 侧一致（限流抛同一种 `RateLimited`，
     坏标的进 bad 列表而不是拖垮整批）。
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import vendor_stockanalysis as V  # noqa: E402
import yfutil  # noqa: E402


def _row(t: str, c: float, a: float | None = None, v: float = 1e6) -> dict:
    a = c if a is None else a
    return {"t": t, "o": c, "h": c, "l": c, "c": c, "a": a, "v": v, "ch": 0.0}


def _series(dates: list[str], closes: list[float], divs: dict[str, float]) -> list[dict]:
    """按本库的复权约定生成 (c, a)：a(t) = c(t) × ∏_{除息日 > t}(1 − D/前收)。"""
    rows = []
    for i, (d, c) in enumerate(zip(dates, closes)):
        f = 1.0
        for j in range(i + 1, len(dates)):
            dj = dates[j]
            if dj in divs and closes[j - 1] > 0:
                f *= (1.0 - divs[dj] / closes[j - 1])
        rows.append(_row(d, c, round(c * f, 8)))
    return rows


DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
CLOSES = [100.0, 102.0, 101.0, 103.0, 104.0]


# ---------------------------------------------------------------- 反解公式

def test_derive_recovers_a_known_dividend_exactly():
    rows = _series(DATES, CLOSES, {"2024-01-05": 1.0})
    got = V.derive_dividends(rows)
    assert list(got) == ["2024-01-05"]
    assert got["2024-01-05"] == pytest.approx(1.0, abs=1e-6)


def test_derive_handles_multiple_dividends():
    rows = _series(DATES, CLOSES, {"2024-01-04": 0.5, "2024-01-08": 0.75})
    got = V.derive_dividends(rows)
    assert got["2024-01-04"] == pytest.approx(0.5, abs=1e-6)
    assert got["2024-01-08"] == pytest.approx(0.75, abs=1e-6)


def test_no_dividend_no_event():
    assert V.derive_dividends(_series(DATES, CLOSES, {})) == {}


def test_rounding_noise_is_not_a_dividend():
    """`a` 只有约 8 位有效数字 ⇒ 反解会有 1e-6 量级的除法噪声。

    实测：没有这道地板时，**无分红的 TSLA 会"反解出" 60 次 $0.0003**。
    """
    rows = _series(DATES, CLOSES, {})
    rows = [dict(r, a=round(r["a"] * (1 + 1e-7), 8)) for r in rows]   # 注入 1e-7 噪声
    assert V.derive_dividends(rows) == {}


def test_small_but_real_dividends_survive_the_floor():
    """地板不能把真实的低档分红也切掉（NVDA 拆股后有 $0.004 档）。"""
    closes = [120.0] * 5
    rows = _series(DATES, closes, {"2024-01-04": 0.004})
    got = V.derive_dividends(rows)
    assert got.get("2024-01-04") == pytest.approx(0.004, abs=1e-6)


def test_absurd_dividend_is_dropped_not_written(caplog):
    """超过前收 30% 的"分红"几乎一定是数据问题 —— 忽略并出声，不静默写进账本。"""
    rows = _series(DATES, CLOSES, {"2024-01-05": 40.0})
    with caplog.at_level("WARNING"):
        got = V.derive_dividends(rows)
    assert got == {}
    assert any("离谱分红" in r.message for r in caplog.records)


def test_malformed_rows_are_skipped_not_crashed():
    rows = [_row("2024-01-02", 100.0), {"t": "2024-01-03"},
            _row("2024-01-04", 0.0), _row("2024-01-05", 101.0)]
    assert V.derive_dividends(rows) == {}


# ---------------------------------------------------------------- 响应形状

@pytest.mark.parametrize("payload", [
    {"status": 200, "data": [{"t": "2024-01-02", "c": 1.0}]},
    {"status": 200, "data": {"data": [{"t": "2024-01-02", "c": 1.0}]}},
])
def test_rows_accepts_both_response_shapes(payload):
    """实测两种形状都出现过（`range=3Y` 返回 list，默认返回 {"data": list}）。"""
    assert len(V._rows(payload)) == 1
    assert V._rows({"status": 404, "data": None}) == []


def test_fetch_history_shapes_like_yfinance(monkeypatch):
    """必须与 yf.download 同构 —— 否则 download_ohlcv 的整条写入链路要长出第二个分支。"""
    monkeypatch.setattr(V, "_get", lambda url, timeout: {"data": _series(DATES, CLOSES, {})})
    df = V.fetch_history("aaa")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume",
                                "Dividends", "Stock Splits"]
    assert isinstance(df.index, pd.DatetimeIndex) and len(df) == 5
    assert (df["Stock Splits"] == 0.0).all(), "本源观测不到拆股事件（c 已是拆股后价位）"


def test_fetch_history_filters_the_window(monkeypatch):
    monkeypatch.setattr(V, "_get", lambda url, timeout: {"data": _series(DATES, CLOSES, {})})
    df = V.fetch_history("aaa", start=date(2024, 1, 4), end=date(2024, 1, 8))
    assert [d.strftime("%Y-%m-%d") for d in df.index] == DATES[2:]


def test_fetch_history_empty_on_no_data(monkeypatch):
    monkeypatch.setattr(V, "_get", lambda url, timeout: {"status": 404, "data": None})
    assert V.fetch_history("nope").empty


# ---------------------------------------------------------------- 批次语义（与 Yahoo 侧一致）

def test_batch_builds_symbol_level_multiindex(monkeypatch):
    def fake(sym, start=None, end=None):
        return pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0],
                             "Volume": [1.0], "Dividends": [0.0], "Stock Splits": [0.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]))

    monkeypatch.setattr(V, "fetch_history", fake)
    raw, bad = V.fetch_batch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 2, 1))
    assert bad == [] and set(raw.columns.get_level_values(0)) == {"AAA", "BBB"}


def test_batch_isolates_bad_symbols(monkeypatch):
    def fake(sym, start=None, end=None):
        if sym == "BAD":
            raise ValueError("boom")
        return pd.DataFrame({"Open": [1.0], "Close": [1.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]))

    monkeypatch.setattr(V, "fetch_history", fake)
    raw, bad = V.batch = V.fetch_batch(["AAA", "BAD"], date(2024, 1, 1), date(2024, 2, 1))
    assert bad == ["BAD"] and set(raw.columns.get_level_values(0)) == {"AAA"}


def test_batch_reraises_rate_limit(monkeypatch):
    """限流**不**是"这个标的坏了" —— 必须上抛，交上游决定退避/中止。"""
    def fake(sym, start=None, end=None):
        raise yfutil.RateLimited("429")

    monkeypatch.setattr(V, "fetch_history", fake)
    with pytest.raises(yfutil.RateLimited):
        V.fetch_batch(["AAA"], date(2024, 1, 1), date(2024, 2, 1))


def test_get_marks_429_as_rate_limited(monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def boom(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", None, None)

    monkeypatch.setattr(V.urllib.request, "urlopen", boom)
    monkeypatch.setattr(V.THROTTLE, "wait", lambda: None)
    monkeypatch.setattr(V.time, "sleep", lambda *_: None)
    with pytest.raises(yfutil.RateLimited):
        V._get("http://x", 5)
    assert calls["n"] == V.MAX_RETRY + 1
    assert yfutil.is_rate_limited(yfutil.RateLimited("x"))     # 与 Yahoo 侧共用同一种语义


def test_get_404_is_a_fact_not_a_failure(monkeypatch):
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(V.urllib.request, "urlopen", boom)
    monkeypatch.setattr(V.THROTTLE, "wait", lambda: None)
    assert V._get("http://x", 5)["status"] == 404


# ---------------------------------------------------------------- 自检

def _long_series(n: int = 60, div_date_offset: int = 30) -> list[dict]:
    """60 个**互不相同**的连续交易日，中间一次分红（自检需要单调的唯一日期轴）。"""
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-01-02", periods=n)]
    closes = [100.0 + i * 0.5 for i in range(n)]
    divs = {dates[div_date_offset]: 1.25}
    return _series(dates, closes, divs)


def test_selfcheck_accepts_an_exact_reconstruction(monkeypatch):
    monkeypatch.setattr(V, "_get", lambda url, timeout: {"data": _long_series()})
    ok, msg = V.selfcheck("aaa")
    assert ok, msg
    assert "偏差" in msg


def test_selfcheck_rejects_a_broken_derivation(monkeypatch):
    """若反解被破坏（这里把 a 换成另一套口径），自检必须拒绝 —— 它是唯一能证明
    "反解没说谎"的手段。"""
    # 因子**连续漂移**（而不是阶跃）：任何"分红"都解释不了它 ——
    # 反解要么找不到事件、要么找出来的与厂商 a 对不上。这是本源口径坏掉时的形态。
    rows = [dict(r, a=r["c"] * (0.90 + 0.0005 * i))
            for i, r in enumerate(_long_series(n=60, div_date_offset=30))]
    monkeypatch.setattr(V, "_get", lambda url, timeout: {"data": rows})
    ok, msg = V.selfcheck("aaa")
    assert not ok, msg
    assert ("未反解出任何分红" in msg) or ("偏差" in msg and "不要用这份分红" in msg)
