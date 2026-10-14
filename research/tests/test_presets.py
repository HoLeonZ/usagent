"""preset 自检 —— 参数自洽 + 纪律字段在位。"""
from __future__ import annotations

import pytest

from research import presets as PS
from research.broker import simulate
from research.signals import V21_SIGNAL_PARAMS


def test_preset_loads_and_is_deep_copied():
    a = PS.get_preset()
    b = PS.get_preset()
    a["signal"]["min_mom120"] = 999.0
    assert b["signal"]["min_mom120"] != 999.0, "get_preset 必须返回深拷贝"


def test_signal_params_match_v21_exactly():
    """基线的意义就是「一个都没改」,这条测试守着它。"""
    assert PS.get_preset()["signal"] == V21_SIGNAL_PARAMS


def test_preset_declares_statistical_discipline_fields():
    p = PS.get_preset()
    assert p["n_comparisons"] >= 1, "CLAUDE.md §5 要求显式记账 n_trials"
    assert p["intraday_tiebreak"] == "sl_first", "CLAUDE.md §4"
    assert p["max_volume_participation"] == 0.10, "CLAUDE.md §4"


def test_preset_carries_negative_evidence():
    """§5 要求负面证据与参数同时记录 —— 不许只报好的一面。"""
    p = PS.get_preset()
    assert p["validated"] is False
    assert p["verdict"].strip(), "verdict 不能为空"
    assert "alpha" in p["verdict"] or "alpha" in p["verdict"].lower()


def test_atr_geometry_preserves_v21_payoff_ratio():
    p = PS.get_preset()
    ratio = p["atr_tp_mult"] / p["atr_sl_mult"]
    assert ratio == pytest.approx(PS.V21_PAYOFF_GEOMETRY, rel=1e-12)


def test_risk_budget_is_priori_not_tuned():
    """risk_per_trade 必须是先验常数,不是从回测里挑的。"""
    assert PS.DEFAULT_RISK_PER_TRADE == 0.01
    assert PS.get_preset()["risk_per_trade"] == PS.DEFAULT_RISK_PER_TRADE


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="未知 preset"):
        PS.get_preset("nope")


def test_preset_is_runnable_by_the_broker(synth_panel):
    """preset 的字段必须能被 broker 吃下去(端到端接线检查)。"""
    import pandas as pd
    from research.indicators import compute_indicators
    from research.signals import ENTRY_COLS

    ind = compute_indicators(synth_panel)
    row = {c: float("nan") for c in ENTRY_COLS}
    dates = sorted(synth_panel["date"].unique())
    row.update({"date": pd.Timestamp(dates[10]), "symbol": "AAA",
                "score": 1.0, "atr_pct": 0.05})
    sig = pd.DataFrame([row])[ENTRY_COLS]

    p = PS.get_preset()
    res = simulate(p, sig, ind, start=dates[0], end=dates[-1],
                   cost_bps_roundtrip=p["cost_bps_roundtrip"])
    assert "trades" in res and "equity" in res and "metrics" in res


def test_h1_is_recorded_as_tested_and_rejected():
    """H1 已检验且被机制检验证伪 —— 不许被当成已验证的改进。"""
    p = PS.get_preset()
    assert p["earnings_filter_validated"] is False
    assert p["avoid_earnings_forward_calendar_days"] == 26
    assert p["n_comparisons"] >= 16, "H1/H2 各一次试验,必须计入 n_trials"


def test_h2_is_recorded_as_tested_and_rejected():
    """H2(移动止损)已检验且不成立 —— 默认必须关闭。"""
    p = PS.get_preset()
    assert p["trailing_sl"] is False
    assert p["trailing_sl_validated"] is False
    assert p["validated"] is False
    assert "五组假设" in p["verdict"] or "假设" in p["verdict"]
