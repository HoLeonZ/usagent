"""事件过滤器的**唯一应用点**与它的键 —— 都不需要真库。

守三件事：
  1. 默认全关 ⇒ 不读日历、行为逐位不变（这是既有 4 条 preset 与 7 个哈希的前提）；
  2. preset 声明的键**真的被读**（"声明了却没人读"是本仓库反复吃亏的那类缺陷 ——
     `avoid_earnings` 此前只能由 CLI 开关打开，preset 里根本写不了）；
  3. 两条入口（CLI 与 golden 载体）**走同一个入口**，否则行为锁锁住的又是另一条曲线。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.preset_contract import DEFAULTS, ENGINE_KEYS, resolve  # noqa: E402
from research import signals  # noqa: E402

BASE = {
    "intraday_tiebreak": "sl_first", "position_fraction": 1.0, "max_positions": 1,
    "max_volume_participation": 0.05, "cost_bps_roundtrip": 5.0, "atr_slip_scale": 0.0,
    "sl_pct": 0.02, "tp_pct": 0.20, "max_hold": 18, "initial_cash": 100000.0,
}


def _sig(days, symbol="AAA"):
    return pd.DataFrame({"date": pd.to_datetime(days), "symbol": symbol})


def test_event_keys_are_declared_and_default_to_off():
    for k in ("avoid_earnings", "avoid_earnings_forward_calendar_days",
              "require_earnings_within_trading_days"):
        assert k in ENGINE_KEYS, f"{k} 不在引擎键里 ⇒ 它会是报告里看不见的旋钮"
    assert DEFAULTS["avoid_earnings"] is False
    assert DEFAULTS["avoid_earnings_forward_calendar_days"] == 26
    assert DEFAULTS["require_earnings_within_trading_days"] is None


def test_default_is_off_and_does_not_even_read_the_calendar(monkeypatch):
    """默认全关时必须**零成本**：连日历都不该去读（否则每条 preset 都隐含依赖外部文件）。"""
    from research import earnings as mod_earnings

    def _boom(*a, **k):                                        # pragma: no cover
        raise AssertionError("默认关闭时不该去读财报日历")

    monkeypatch.setattr(mod_earnings, "load_calendar", _boom)
    sig = _sig(["2024-05-06", "2024-05-07"])
    out = signals.apply_event_filters(sig, dict(BASE))
    pd.testing.assert_frame_equal(out, sig)


def test_declared_key_is_actually_read(monkeypatch):
    """preset 声明 `avoid_earnings: True` ⇒ 引擎必须真的读它（这是本次补上的那处缺口）。"""
    axis = pd.bdate_range("2024-01-01", periods=60)
    monkeypatch.setattr("research.earnings.load_calendar",
                        lambda *a, **k: {str(d.date()): [] for d in axis} | {
                            str(axis[30].date()): ["AAA"]})
    p = {**BASE, "avoid_earnings": True}
    sig = _sig([axis[10], axis[29]])            # 第二笔的前瞻窗罩住轴上的第 30 天
    out = signals.apply_event_filters(sig, p, trading_days=axis)
    assert len(out) == 1 and out["date"].iloc[0] == axis[10]


def test_recent_filter_keeps_only_signals_just_after_a_report(monkeypatch):
    axis = pd.bdate_range("2024-01-01", periods=60)
    monkeypatch.setattr("research.earnings.load_calendar",
                        lambda *a, **k: {str(d.date()): [] for d in axis} | {
                            str(axis[40].date()): ["AAA"]})
    p = {**BASE, "require_earnings_within_trading_days": 3}
    sig = _sig([axis[41], axis[42], axis[44], axis[50]])       # gap 1 / 2 / 4 / 10
    out = signals.apply_event_filters(sig, p, trading_days=axis)
    assert list(out["date"]) == [axis[41], axis[42]], "只该留下距上一份财报 ≤3 个交易日的"


def test_missing_trading_axis_fails_fast(monkeypatch):
    axis = pd.bdate_range("2024-01-01", periods=20)
    monkeypatch.setattr("research.earnings.load_calendar",
                        lambda *a, **k: {str(d.date()): [] for d in axis})
    with pytest.raises(ValueError, match="交易日轴"):
        signals.apply_event_filters(_sig([axis[5]]),
                                    {**BASE, "require_earnings_within_trading_days": 3})


def test_signal_day_outside_calendar_coverage_fails_fast(monkeypatch):
    """覆盖之外必须抛错 —— 让过滤器在样本尾部静默失效是最危险的失效方式（审计 A5）。"""
    axis = pd.bdate_range("2024-01-01", periods=30)
    monkeypatch.setattr("research.earnings.load_calendar",
                        lambda *a, **k: {str(d.date()): [] for d in axis[:20]})
    with pytest.raises(Exception, match="未知不等于没有"):
        signals.apply_event_filters(_sig([axis[25]]),
                                    {**BASE, "require_earnings_within_trading_days": 3},
                                    trading_days=axis)


def test_cli_flag_forces_the_filter_on(monkeypatch):
    axis = pd.bdate_range("2024-01-01", periods=60)
    monkeypatch.setattr("research.earnings.load_calendar",
                        lambda *a, **k: {str(d.date()): [] for d in axis} | {
                            str(axis[30].date()): ["AAA"]})
    sig = _sig([axis[29]])
    assert len(signals.apply_event_filters(sig, dict(BASE), trading_days=axis)) == 1
    forced = signals.apply_event_filters(sig, dict(BASE), trading_days=axis,
                                         force_avoid_earnings=True)
    assert forced.empty


# ---------------------------------------------------------------- 一条路（护栏）

@pytest.mark.parametrize("path,func", [
    ("research/run_baseline.py", "def _signals"),
    ("research/golden_baseline.py", "def run_preset"),
])
def test_both_entries_use_the_single_application_point(path, func):
    """CLI 与 golden 载体必须走同一个入口 —— 两处实现 = 两条曲线（§6 的老毛病）。"""
    src = (ROOT / path).read_text(encoding="utf-8")
    body = src[src.index(func):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    assert "apply_event_filters" in body, f"{path} 的 {func} 没有走唯一入口"
    assert "filter_earnings_window" not in body, f"{path} 自己又过滤了一遍"


def test_resolve_exposes_the_event_keys_with_defaults():
    spec = resolve(dict(BASE))
    assert spec["avoid_earnings"] is False
    assert spec["require_earnings_within_trading_days"] is None
