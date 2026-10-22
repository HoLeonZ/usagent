"""财报窗口过滤(H1)—— 纯函数测试,不联网。

⚠️ 2026-10-17 审计 A5 改过的语义：`has_earnings_between` / `filter_earnings_window`
的 `coverage` 现在是**必填**的。此前"符号不在日历里"与"那段日期根本没取过"都返回
**False**（= 窗口内没有财报），于是**过滤器在样本尾部静默失效** —— 你看到的是一个
"已剔除财报窗"的信号表，而它其实只是没数据。**未知 ≠ 没有。**
"""
from __future__ import annotations

import pandas as pd
import pytest

from research.earnings import (EarningsError, calendar_coverage,
                               earnings_dates_by_symbol, has_earnings_between)
from research.signals import ENTRY_COLS, filter_earnings_window

# 测试窗口都落在这段覆盖区间内；coverage 现在是必填参数。
COV = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))


def _sig(day: str, symbol: str) -> pd.DataFrame:
    row = {c: float("nan") for c in ENTRY_COLS}
    row.update({"date": pd.Timestamp(day), "symbol": symbol, "score": 1.0, "atr_pct": 0.05})
    return pd.DataFrame([row])[ENTRY_COLS]


def _cal(entries: dict[str, list[str]]) -> dict[str, list[pd.Timestamp]]:
    """{date: [symbols]} -> 倒排 {symbol: [dates]}"""
    return earnings_dates_by_symbol({d: syms for d, syms in entries.items()})


# ---------------------------------------------------------------- 覆盖区间

def test_calendar_coverage_is_the_min_max_of_fetched_dates():
    assert calendar_coverage({"2024-03-01": ["AAA"], "2024-01-05": ["BBB"]}) == (
        pd.Timestamp("2024-01-05"), pd.Timestamp("2024-03-01"))


def test_empty_calendar_has_no_coverage():
    assert calendar_coverage({}) is None


# ---------------------------------------------------------------- 判定本身

def test_has_earnings_between_is_inclusive():
    m = _cal({"2024-05-10": ["AAA"]})
    lo = pd.Timestamp("2024-05-01")
    hi = pd.Timestamp("2024-05-10")
    assert has_earnings_between(m, "AAA", lo, hi, coverage=COV)          # 恰好落在上界
    assert not has_earnings_between(m, "AAA", lo, pd.Timestamp("2024-05-09"),
                                    coverage=COV)
    assert not has_earnings_between(m, "BBB", lo, hi, coverage=COV)      # 不相关标的


def test_filter_drops_signal_whose_hold_window_contains_earnings():
    m = _cal({"2024-06-01": ["AAA"]})
    out = filter_earnings_window(_sig("2024-05-15", "AAA"), m, forward_days=26,
                                 coverage=COV)
    assert out.empty


def test_filter_keeps_earnings_that_fall_before_entry():
    m = _cal({"2024-05-01": ["AAA"]})
    out = filter_earnings_window(_sig("2024-05-15", "AAA"), m, forward_days=26,
                                 coverage=COV)
    assert len(out) == 1


def test_filter_forward_days_boundary():
    m = _cal({"2024-05-20": ["AAA"]})
    sig = _sig("2024-05-15", "AAA")
    assert len(filter_earnings_window(sig, m, forward_days=3, coverage=COV)) == 1
    assert len(filter_earnings_window(sig, m, forward_days=6, coverage=COV)) == 0


def test_filter_forward_days_uses_calendar_not_trading_days():
    """forward_days 是**日历日**：18 个交易日 ≈ 26 个自然日，用 18 会漏掉尾部一周。"""
    m = _cal({"2024-06-05": ["AAA"]})            # 入场 05-15 之后的第 21 个自然日
    sig = _sig("2024-05-15", "AAA")
    assert len(filter_earnings_window(sig, m, forward_days=26, coverage=COV)) == 0
    assert len(filter_earnings_window(sig, m, forward_days=18, coverage=COV)) == 1


def test_filter_handles_empty_input():
    empty = pd.DataFrame(columns=ENTRY_COLS)
    out = filter_earnings_window(empty, {}, forward_days=26, coverage=COV)
    assert out.empty


# ---------------------------------------------------------------- 未知 ≠ 没有（A5 的核心）

def test_missing_coverage_raises_instead_of_saying_no_earnings():
    """**这就是 A5**：不给覆盖区间宁可不回答，也不许把"不知道"答成"没有财报"。"""
    m = _cal({"2024-06-01": ["AAA"]})
    with pytest.raises(EarningsError, match="日历为空"):
        has_earnings_between(m, "AAA", pd.Timestamp("2024-05-01"),
                             pd.Timestamp("2024-06-10"), coverage=None)


def test_window_outside_coverage_raises_with_an_actionable_message():
    """窗口超出覆盖 ⇒ 抛错，并说清"未知不等于没有"以及两条正路。"""
    m = _cal({"2024-06-01": ["AAA"]})
    with pytest.raises(EarningsError) as e:
        has_earnings_between(m, "AAA", pd.Timestamp("2024-05-01"),
                             pd.Timestamp("2025-03-01"), coverage=COV)
    msg = str(e.value)
    assert "未知不等于没有" in msg and "静默失效" in msg
    assert "补齐" in msg and "不用财报过滤" in msg


def test_filter_on_a_window_beyond_coverage_fails_loudly():
    """样本尾部超出日历覆盖 ⇒ **整条过滤抛错**，而不是悄悄返回未过滤的信号表。"""
    m = _cal({"2024-06-01": ["AAA"]})
    sig = pd.concat([_sig("2024-05-15", "AAA"), _sig("2024-12-20", "AAA")],
                    ignore_index=True)
    with pytest.raises(EarningsError):
        filter_earnings_window(sig, m, forward_days=26, coverage=COV)   # 第二笔越界


def test_in_coverage_symbol_absent_is_a_definite_false():
    """覆盖之内、符号不在日历里 ⇒ 那段时间它确实没有财报 —— 这是**确定性**答案，
    不是"未知"。把这条与越界区分开，正是 A5 的要点。"""
    m = _cal({"2024-06-01": ["BBB"]})
    assert has_earnings_between(m, "AAA", pd.Timestamp("2024-05-01"),
                                pd.Timestamp("2024-06-10"), coverage=COV) is False
