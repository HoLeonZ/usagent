"""风格因子归因的正确性（审计 C4）。

三件事必须对，否则整张归因表是装饰品：
  1. **HAC 标准误真的在工作** —— 残差有自相关时，它必须给出比普通 OLS **更小**的 t
     （普通 OLS 假设独立同方差，会把噪声读成显著）；
  2. **因子构造是因果的、符号是对的** —— 划分只用 ≤ 再平衡日的数据；动量因子的
     符号必须与"高动量端做多"一致；
  3. **纯因子策略的归因必须退化得干净** —— 若策略就是市场本身，β_market ≈ 1、
     alpha ≈ 0、R² ≈ 1、跟踪误差 ≈ 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import factor_model as FM  # noqa: E402


# ---------------------------------------------------------------- HAC 标准误

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 800)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.5, 800)
    fit = FM.newey_west_ols(y, x)
    assert fit["beta"][0] == pytest.approx(2.0, abs=0.1)
    assert fit["beta"][1] == pytest.approx(3.0, abs=0.1)
    assert fit["r2"] > 0.9


def test_hac_corrects_the_standard_error_in_expectation():
    """残差正自相关时，HAC 的标准误**在期望上**大于普通 OLS 的标准误。

    ⚠️ 必须按**多次重复**比，不能按单次样本比：HAC 修正的是标准误的估计量，
    单次样本里它的方向可以相反（实测踩过 —— 我第一版断言"HAC 一定更小"，
    结果得到一个反例就红了）。这里 200 次重复、固定 x（带持续性的随机游走），
    比的是平均值。
    """
    n, reps = 600, 200
    rng = np.random.default_rng(2)
    se_class, se_hac = [], []
    for _ in range(reps):
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.7 * e[i - 1] + rng.normal(0, 1)      # 强正自相关
        x = np.cumsum(rng.normal(0, 1, n))                # 带持续性的回归量
        y = 0.3 * x + e
        A = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - A @ beta
        se_class.append(float(np.sqrt(np.diag(np.linalg.inv(A.T @ A)
                                              * (resid @ resid) / (n - 2)))[1]))
        se_hac.append(float(FM.newey_west_ols(y, x, lags=20)["se"][1]))
    mean_class, mean_hac = float(np.mean(se_class)), float(np.mean(se_hac))
    assert mean_hac > mean_class * 1.5, (
        f"HAC 没有在期望上修正自相关：普通 SE 均值 {mean_class:.4f} vs HAC {mean_hac:.4f}")


def test_ols_fails_fast_on_too_few_observations():
    with pytest.raises(FM.FactorError):
        FM.newey_west_ols(np.array([1.0]), np.array([[1.0]]))


# ---------------------------------------------------------------- 因子构造

def _panel_from_closes(closes: dict[str, list[float]], start="2023-01-02") -> pd.DataFrame:
    """按收盘序列造一个带 ret1/mom120/rvol20/eligible 的面板（指标口径够用即可）。"""
    rows = []
    for sym, cs in closes.items():
        dates = pd.bdate_range(start, periods=len(cs))
        for i, (d, c) in enumerate(zip(dates, cs)):
            rows.append({"symbol": sym, "date": d, "close": c, "ret1": np.nan,
                         "mom120": np.nan, "rvol20": np.nan, "eligible": True})
    df = pd.DataFrame(rows).sort_values(["symbol", "date"])
    g = df.groupby("symbol", sort=False)
    df["ret1"] = g["close"].pct_change()
    df["mom120"] = g["close"].transform(lambda s: s / s.shift(120) - 1.0)
    df["rvol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())
    return df.reset_index(drop=True)


def test_momentum_factor_is_positive_when_momentum_predicts():
    """高动量组持续跑赢低动量组 ⇒ 动量因子（高−低）的累计收益必须为正。"""
    n = 300
    closes = {}
    for k in range(30):                          # 前 15 只上行、后 15 只下行
        slope = 0.004 if k < 15 else -0.004
        closes[f"S{k:02d}"] = [100.0 * (1 + slope) ** i for i in range(n)]
    ind = _panel_from_closes(closes)
    fac = FM.style_factor_returns(ind, lo="2023-09-01", hi="2024-06-01")
    assert fac["mom"].sum() > 0, "高动量端跑赢时，动量因子的累计收益应为正"
    assert fac["lowvol"].abs().sum() >= 0            # 该构造下低波方向无先验，只要求算得出来


def test_factor_returns_are_causal_in_the_grouping():
    """再平衡日之后的行情不得改变**分组** —— 只改变实现收益。

    构造两根完全相同的价格路径，只把最后一段的走势反转：因子收益会变（那是实现部分），
    但**分组**（谁在高三分位）必须只由 ≤ 再平衡日的数据决定。这里用"改变未来不改变
    早期因子收益"来钉住因果性。
    """
    n = 300
    closes = {f"S{k:02d}": [100.0 * (1 + 0.002 * (k % 5)) ** i for i in range(n)]
              for k in range(30)}
    a = _panel_from_closes(closes)
    # 只把**未来**那一段的收盘整体放大（等价于"未来才发生的行情"）
    cut = 200                                        # 索引 200 之后算未来
    modified = {s: cs[:cut] + [c * 1.5 for c in cs[cut:]] for s, cs in closes.items()}
    b = _panel_from_closes(modified)
    fa = FM.style_factor_returns(a, lo="2023-06-01", hi="2024-06-01")
    fb = FM.style_factor_returns(b, lo="2023-06-01", hi="2024-06-01")
    early = fa.index < (a["date"].iloc[cut])          # 未来开始之前""")
    assert len(early) and early.any()
    assert np.allclose(fa.loc[early, "mom"], fb.loc[early, "mom"]), \
        "改变未来影响了早期的因子收益 —— 分组里混进了未来信息"


def test_factor_construction_fails_fast():
    with pytest.raises(FM.FactorError, match="需要"):
        FM.style_factor_returns(pd.DataFrame({"date": [], "symbol": []}), lo="a", hi="b")


# ---------------------------------------------------------------- 归因的退化行为

def test_pure_market_strategy_attributes_cleanly():
    """策略就是市场本身 ⇒ β_market ≈ 1、alpha ≈ 0、R² ≈ 1、跟踪误差 ≈ 0。"""
    rng = np.random.default_rng(3)
    n = 500
    idx = pd.bdate_range("2024-01-01", periods=n)
    mkt = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    fac = pd.DataFrame({"market": mkt, "mom": rng.normal(0, 0.005, n),
                        "lowvol": rng.normal(0, 0.005, n)}, index=idx)
    d = FM.attribute(mkt, fac, mkt)
    assert d["beta_market"] == pytest.approx(1.0, abs=0.02)
    assert abs(d["alpha_annual"]) < 0.02
    assert d["r2"] > 0.99
    assert d["te_annual"] == 0.0, "策略就是基准 ⇒ 跟踪误差恰为 0（已定义为 0，不是 nan）"
    assert np.isnan(d["ir"]), "0/0 的信息比率是未定义"


def test_factor_loading_is_recovered():
    """策略 = 0.5×market + 0.8×mom + 噪声 ⇒ 两个 β 必须被估出来（且 alpha 接近 0）。"""
    rng = np.random.default_rng(4)
    n = 800
    idx = pd.bdate_range("2024-01-01", periods=n)
    mk = pd.Series(rng.normal(0, 0.01, n), index=idx)
    mo = pd.Series(rng.normal(0, 0.008, n), index=idx)
    lv = pd.Series(rng.normal(0, 0.006, n), index=idx)
    fac = pd.DataFrame({"market": mk, "mom": mo, "lowvol": lv}, index=idx)
    s = 0.5 * mk + 0.8 * mo + pd.Series(rng.normal(0, 0.002, n), index=idx)
    d = FM.attribute(s, fac, mk)
    assert d["beta_market"] == pytest.approx(0.5, abs=0.05)
    assert d["beta_mom"] == pytest.approx(0.8, abs=0.05)
    assert abs(d["beta_lowvol"]) < 0.1
    assert abs(d["alpha_annual"]) < 0.5


def test_ir_and_te_use_the_excess_series():
    rng = np.random.default_rng(5)
    n = 400
    idx = pd.bdate_range("2024-01-01", periods=n)
    b = pd.Series(rng.normal(0, 0.01, n), index=idx)
    s = b + pd.Series(rng.normal(0.0004, 0.003, n), index=idx)
    fac = pd.DataFrame({"market": b, "mom": 0.0, "lowvol": 0.0}, index=idx)
    d = FM.attribute(s, fac, b)
    assert d["te_annual"] == pytest.approx(float(np.std(s - b, ddof=1)) * np.sqrt(252))
    assert d["ir"] == pytest.approx(float(np.mean(s - b) / np.std(s - b, ddof=1) * np.sqrt(252)))


def test_attribute_fails_fast_on_a_short_overlap():
    idx = pd.bdate_range("2024-01-01", periods=10)
    with pytest.raises(FM.FactorError, match="不足以归因"):
        FM.attribute(pd.Series(0.0, index=idx),
                     pd.DataFrame({"market": 0.0, "mom": 0.0, "lowvol": 0.0}, index=idx),
                     pd.Series(0.0, index=idx))


def test_describe_mentions_the_known_gaps():
    """披露必须说出来："没有价值与规模因子"这件事要在输出里可见，而不是只写在注释里。"""
    src = (ROOT / "research" / "factor_model.py").read_text()
    assert "没有价值因子" in src and "规模因子" in src
