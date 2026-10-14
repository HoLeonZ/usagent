"""财报窗口过滤(H1)—— 纯函数测试,不联网。"""
from __future__ import annotations

import pandas as pd
import pytest

from research.earnings import earnings_dates_by_symbol, has_earnings_between
from research.signals import ENTRY_COLS, filter_earnings_window


def _sig(day: str, symbol: str) -> pd.DataFrame:
    row = {c: float("nan") for c in ENTRY_COLS}
    row.update({"date": pd.Timestamp(day), "symbol": symbol, "score": 1.0, "atr_pct": 0.05})
    return pd.DataFrame([row])[ENTRY_COLS]


def _cal(entries: dict[str, list[str]]) -> dict[str, list[pd.Timestamp]]:
    """{date: [symbols]} -> 倒排 {symbol: [dates]}"""
    raw = {d: syms for d, syms in entries.items()}
    return earnings_dates_by_symbol(raw)


def test_has_earnings_between_is_inclusive():
    m = earnings_dates_by_symbol({"2024-05-10": ["AAA"]})
    lo, hi = pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-10")
    assert has_earnings_between(m, "AAA", lo, hi)          # 恰好落在上界
    assert not has_earnings_between(m, "AAA", lo, pd.Timestamp("2024-05-09"))
    assert not has_earnings_between(m, "AAA",
                                    pd.Timestamp("2024-05-11"), pd.Timestamp("2024-06-01"))
    assert not has_earnings_between(m, "BBB", lo, hi)       # 不相关标的


def test_filter_drops_signal_whose_hold_window_contains_earnings():
    """信号日 +1 .. +forward_days 内夹财报 ⇒ 剔除。"""
    m = _cal({"2024-05-20": ["AAA"]})
    sig = pd.concat([_sig("2024-05-15", "AAA"), _sig("2024-01-02", "AAA")],
                    ignore_index=True)
    out = filter_earnings_window(sig, m, forward_days=26)
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2024-01-02")
    # 2024-01-02 + 26 日 = 01-28,早于 05-20,所以留下


def test_filter_keeps_earnings_that_fall_before_entry():
    """财报在信号日**之前**不属于持有窗口,不该剔除。"""
    m = _cal({"2024-05-01": ["AAA"]})
    out = filter_earnings_window(_sig("2024-05-15", "AAA"), m, forward_days=26)
    assert len(out) == 1


def test_filter_forward_days_boundary():
    """forward_days 用**日历日**,窗口是 [signal+1, signal+forward_days]。

    信号 2024-05-15,财报 2024-05-20:
      forward_days=3  → 窗口 05-16..05-18,不含财报  → 保留(1)
      forward_days=6  → 窗口 05-16..05-21,含财报    → 剔除(0)
    """
    m = _cal({"2024-05-20": ["AAA"]})
    sig = _sig("2024-05-15", "AAA")
    assert len(filter_earnings_window(sig, m, forward_days=3)) == 1
    assert len(filter_earnings_window(sig, m, forward_days=6)) == 0


def test_filter_forward_days_uses_calendar_not_trading_days():
    """max_hold=18 个交易日 ≈ 26 个自然日。若误把交易日数当日历日用,
    窗口会短 ~28%,尾部一周的财报会被漏掉。"""
    m = _cal({"2024-06-10": ["AAA"]})
    sig = _sig("2024-05-15", "AAA")            # 05-15 + 26 天 = 06-10
    assert len(filter_earnings_window(sig, m, forward_days=26)) == 0   # 刚好命中,剔除
    assert len(filter_earnings_window(sig, m, forward_days=18)) == 1   # 用 18 就漏了


def test_filter_is_identity_without_earnings_data():
    sig = pd.concat([_sig("2024-05-15", "AAA"), _sig("2024-06-15", "BBB")],
                    ignore_index=True)
    out = filter_earnings_window(sig, {}, forward_days=26)
    assert len(out) == 2


def test_filter_handles_empty_input():
    out = filter_earnings_window(pd.DataFrame(columns=ENTRY_COLS), {}, forward_days=26)
    assert out.empty
