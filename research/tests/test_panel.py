"""面板层口径 —— 双价格体系(CLAUDE.md §3(b))与成交额代理量(§3(a)/§3(b))。

为什么 amount 必须用 as-traded 价:panel 用 `volume × close` 合成成交额代理量,而 close
是**分红回补**后的价(因子 = ∏_{事件晚于 t})。E 条件把这个量拿去比**绝对美元常数**
[3e7, 3e8],于是「T 日能不能进池」变成了 T 之后除息日的函数。凭证:2026-10-15 审计实测
全库 390,168 个就绪格中有 740 格的 E 判定被未来分红翻转(VISN 的因子低到 0.278,即
成交额被压低到真值的 28%)。把 amount 换成 as-traded 后,12m preset 的信号数 19→21、
CAGR +44.54%→+47.19%。

比例型指标(close>ma60、mom120、pullback)对同一个因子免疫 —— 分子分母同乘一个常数。
所以问题**只**出在「绝对美元阈值」这一处比对上,这也是为什么修的是 amount 而不是指标。
"""
from __future__ import annotations

import pandas as pd
import pytest

from research.panel import (
    MAX_LOOKBACK_BARS, RAW_COLS, PanelError, data_vintage, first_signal_date,
    to_panel_frame,
)

DATES = pd.bdate_range("2024-01-01", periods=3)


def _series(close: list[float], dividends: list[float] | None = None,
            dates: pd.DatetimeIndex = DATES) -> pd.DataFrame:
    """模仿 src/series.daily 的输出:index=date,含 OHLCV + dividends/splits。"""
    n = len(close)
    df = pd.DataFrame({
        "date": dates,
        "open": close, "high": [c * 1.01 for c in close],
        "low": [c * 0.99 for c in close], "close": close,
        "volume": 1000.0,
        "dividends": dividends if dividends is not None else [0.0] * n,
        "splits": 0.0,
    })
    return df.set_index("date")


def test_amount_uses_as_traded_close():
    out = to_panel_frame("AAA", _series([50.0, 50.0, 50.0]), _series([100.0, 100.0, 100.0]))
    assert list(out["close"]) == [50.0] * 3, "指标口径必须仍是复权价"
    assert list(out["raw_close"]) == [100.0] * 3, "执行口径必须是 as-traded 价"
    assert list(out["amount"]) == [1000.0 * 100.0] * 3, "amount 必须用 as-traded 价算"


def test_frame_carries_both_price_series():
    out = to_panel_frame("AAA", _series([50.0] * 3), _series([100.0] * 3))
    assert set(RAW_COLS) <= set(out.columns)
    assert set(out["symbol"]) == {"AAA"}
    assert set(out["dividends"]) == {0.0}


def test_frame_fails_fast_when_the_two_series_do_not_align():
    """两条序列日期对不上 ⇒ fail-fast,不得静默留 NaN(§0 Fail-Fast)。"""
    short = pd.bdate_range("2024-01-01", periods=2)
    with pytest.raises(PanelError, match="as-traded"):
        to_panel_frame("AAA", _series([50.0] * 3), _series([100.0] * 2, dates=short))


# ---------------------------------------------------------------- 预热边界与 vintage

def test_first_signal_date_is_the_first_bar_past_the_warmup():
    """预热边界取「全表第 MAX_LOOKBACK_BARS 个交易日」= 第 121 根 K 线。"""
    dates = pd.bdate_range("2020-01-01", periods=200)
    panel = pd.DataFrame({"symbol": "AAA", "date": dates})
    assert first_signal_date(panel) == dates[MAX_LOOKBACK_BARS]


def test_first_signal_date_rejects_a_panel_shorter_than_the_warmup():
    dates = pd.bdate_range("2020-01-01", periods=MAX_LOOKBACK_BARS)
    with pytest.raises(PanelError, match="预热"):
        first_signal_date(pd.DataFrame({"symbol": "AAA", "date": dates}))


def test_data_vintage_is_a_stable_fingerprint():
    panel = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
    })
    assert data_vintage(panel) == "rows=3;symbols=2;from=2024-01-01;to=2024-01-02"
