"""评估口径测试。重点是 DSR 的单位一致性 —— hiagent 在这里踩过坑。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.metrics import (beta_attribution, cagr, deflated_sharpe_ratio,
                              monthly_windows, r_multiples, tail_stats)


# ---------------------------------------------------------------- DSR 单位

def test_dsr_matches_hiagent_documented_correct_value():
    """年化 Sharpe=0.84 / k=19 / T=1071 ⇒ DSR ≈ 0.44。

    hiagent preset_search §1 记录:他们曾拿**年化** Sharpe 去比**逐期**阈值,
    把这一组判成 DSR=0.000,而正确值约 0.44 —— 把「有边际优势」误读成「毫无优势」。
    这条测试把那个修复钉住。
    """
    r = deflated_sharpe_ratio(observed_sharpe=0.84, n_trials=19, n_returns=1071)
    assert r["deflated_sharpe"] == pytest.approx(0.44, abs=0.02), (
        f"DSR={r['deflated_sharpe']:.4f} 偏离 ~0.44 —— 单位一致性又坏了?"
    )
    # 明确排除两个错误方向的塌陷
    assert r["deflated_sharpe"] > 0.20
    assert r["deflated_sharpe"] < 0.70


def test_dsr_does_not_collapse_moderate_sharpe():
    """单位 bug 的特征是把中等 Sharpe 折减到 ~0;这里必须显著非零。"""
    for sr in (0.5, 0.84, 1.2):
        d = deflated_sharpe_ratio(sr, n_trials=10, n_returns=500)["deflated_sharpe"]
        assert d > 0.05, f"Sharpe={sr} 被折减到 {d:.4f}"


def test_dsr_decreases_with_more_trials():
    """试得越多,同一 Sharpe 的证据越弱。"""
    vals = [deflated_sharpe_ratio(1.0, n_trials=k, n_returns=500)["deflated_sharpe"]
            for k in (1, 5, 20, 100, 500)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), vals


def test_dsr_single_trial_degenerates_to_psr():
    r = deflated_sharpe_ratio(1.0, n_trials=1, n_returns=500)
    assert r["expected_max_sharpe"] == 0.0
    assert 0.0 < r["deflated_sharpe"] <= 1.0


def test_dsr_rejects_bad_n_trials():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(1.0, n_trials=0)


# ---------------------------------------------------------------- E[R]

def _trades(pnl: list[float], risk: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "net_pnl": pnl, "risk_amount": risk,
        "entry_date": pd.date_range("2024-01-01", periods=len(pnl), freq="D"),
        "exit_date": pd.date_range("2024-01-02", periods=len(pnl), freq="D"),
    })


def test_r_multiples_uses_planned_risk_not_atr():
    t = _trades([100.0, -50.0], [100.0, 100.0])
    r = r_multiples(t)
    assert list(np.round(r, 6)) == [1.0, -0.5]


def test_r_multiples_rejects_zero_risk():
    with pytest.raises(ValueError, match="risk_amount"):
        r_multiples(_trades([10.0], [0.0]))


def test_tail_stats_basic_shape():
    t = _trades([100.0, -50.0, 200.0, -25.0], [100.0] * 4)
    s = tail_stats(t)
    assert s["n"] == 4
    assert s["expectancy_r"] == pytest.approx((1.0 - 0.5 + 2.0 - 0.25) / 4)
    assert s["win_pct"] == pytest.approx(0.5)
    assert s["payoff"] == pytest.approx(150.0 / 37.5)
    assert s["profit_factor"] == pytest.approx(300.0 / 75.0)
    assert s["p_r_ge_5"] == 0.0


def test_tail_stats_top_decile_share_detects_tail_dependence():
    """前 10% 交易贡献了净盈亏的绝大部分 ⇒ top_decile_share 应显著 > 1。

    注意分母是**净盈亏之和**,而 top decile 里会混进 9 笔输单,所以这个量不会
    无限大:99 笔 -2 加 1 笔 +250 ⇒ 净 52,top10 = 250-18 = 232,share = 4.46。
    (hiagent 实测该族是 2.97,与他们 79% 止损 / 盈亏比 5.42 的分布同向。)
    """
    pnl = [-2.0] * 99 + [250.0]
    t = _trades(pnl, [1.0] * 100)
    s = tail_stats(t)
    assert s["top_decile_share"] > 3.0, s["top_decile_share"]
    assert s["expectancy_r"] > 0
    assert s["win_pct"] == pytest.approx(0.01)


def test_tail_stats_empty():
    s = tail_stats(pd.DataFrame(columns=["net_pnl", "risk_amount", "exit_date"]))
    assert s["n"] == 0 and math.isnan(s["expectancy_r"])


# ---------------------------------------------------------------- 窗口与 CAGR

def test_monthly_windows_count_and_bounds():
    w = monthly_windows("2024-04", "2026-10", window_months=2, step_months=1)
    assert w[0] == ("2024-04-01", "2024-06-01")
    assert w[-1] == ("2026-08-01", "2026-10-01")
    assert len(w) == 29
    # 全部左闭右开且不超界
    for s, e in w:
        assert s < e and e <= "2026-10-01"


def test_monthly_windows_non_overlapping_mode():
    w = monthly_windows("2024-04", "2026-10", window_months=2, step_months=2)
    assert w[0] == ("2024-04-01", "2024-06-01")
    assert w[1] == ("2024-06-01", "2024-08-01")
    assert len(w) == 15


def test_cagr_is_calendar_annualized():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2025-01-01"]))
    eq = pd.Series([100.0, 110.0])       # 366 天 +10%
    got = cagr(eq, dates)
    want = 1.10 ** (365.25 / 366) - 1
    assert got == pytest.approx(want, rel=1e-9)


# ---------------------------------------------------------------- beta 归因

def test_beta_attribution_recovers_known_beta():
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0005, 0.01, 400))
    strat = 2.0 * bench                     # 纯 2 倍 beta,零 alpha
    expo = pd.Series(1.0, index=bench.index)
    out = beta_attribution(strat, bench, expo)
    assert out["beta"] == pytest.approx(2.0, abs=1e-9)
    assert out["alpha_annual"] == pytest.approx(0.0, abs=1e-9)
    assert out["corr"] == pytest.approx(1.0, abs=1e-9)


def test_beta_attribution_separates_timing_from_exposure():
    """只在市场上涨的那些天持仓 ⇒ 择时有效,但 bench_on_exposed_days 会揭露它。"""
    bench = pd.Series([0.01, -0.01] * 200)
    strat = pd.Series([0.01 if b > 0 else 0.0 for b in bench])
    expo = (bench > 0).astype(float)
    out = beta_attribution(strat, bench, expo)
    assert out["pct_days_exposed"] == pytest.approx(0.5)
    # 只在上涨日持有基准,本身就能拿到全部涨幅 —— 这正是要揭穿的"伪 alpha"
    assert out["bench_on_exposed_days"] > 5.0


def test_beta_attribution_handles_short_series():
    out = beta_attribution(pd.Series([0.01]), pd.Series([0.01]), pd.Series([1.0]))
    assert math.isnan(out["beta"])
    assert out["n_days"] == 1
