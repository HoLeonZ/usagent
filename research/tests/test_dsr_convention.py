"""DSR 的口径只能有一个（审计 B3）—— 从收益序列派生，而不是各入口各说各话。

缺陷本体：`deflated_sharpe_ratio` 需要四个量（T、偏度、峰度、一阶自相关），
而四个调用点各传各的：`run_baseline` 传交易日数、`family_search`/`wfv_search` 传
**成交笔数**、`factor_probe` 传**月数**，且**没有任何一处**传过偏度/峰度
（非正态修正恒为默认值 ⇒ 惰性）；`factor_probe` 还同时把月频 Sharpe 除以 √252。
同一个指标在两个入口得到两种口径的数字，而这个数字是用来做判据的。

本文件钉住：
  1. 派生量（年化 Sharpe / 偏度 / 峰度 / 自相关 / T）都从**同一条序列**算出来，且算对；
  2. **单位一致性**：同一段行情写成日频（252）或月频（12）应当给出相近的 Sharpe 与 DSR；
  3. `sharpe_significance_pvalue` 的同一个单位 bug 不得回归；
  4. 逐窗拼接只在**窗口不重叠**时发生（重叠 ⇒ 拒绝，不报那个数）；
  5. **源码级护栏**：除 metrics.py 外没有第二处直接调用 `deflated_sharpe_ratio`。
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import metrics as M  # noqa: E402


def _ar1(n=800, phi=0.3, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------- 派生量算对

def test_annualized_sharpe_matches_the_definition():
    r = np.array([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
    want = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert M.annualized_sharpe(r) == pytest.approx(want)


def test_moments_are_computed_from_the_series():
    rng = np.random.default_rng(3)
    sym = rng.normal(0, 0.01, 4000)
    out = M.deflated_sharpe_from_returns(sym, 10)
    assert abs(out["skewness"]) < 0.2, "对称序列的偏度应接近 0"
    assert abs(out["kurtosis"] - 3.0) < 0.3, "正态峰度应为 3（非超额）"
    assert out["n_returns"] == 4000

    right = np.exp(rng.normal(0, 0.5, 4000))          # 重右尾
    out2 = M.deflated_sharpe_from_returns(right - right.mean() + 0.001, 10)
    assert out2["skewness"] > 1.0
    assert out2["kurtosis"] > 4.0


def test_autocorrelation_is_measured():
    out = M.deflated_sharpe_from_returns(_ar1(phi=0.3), 10)
    assert out["autocorr_lag1"] > 0.2, "AR(1) 正自相关必须被估出来（DSR 的自相关修正依赖它）"


def test_moments_are_fed_to_the_deflated_formula():
    """把同样的四个量手工喂给底层函数，结果必须一致（否则说明"派生"没接上）。"""
    r = _ar1(seed=5)
    a = M.deflated_sharpe_from_returns(r, 25)
    z = (r - r.mean()) / r.std(ddof=1)
    b = M.deflated_sharpe_ratio(M.annualized_sharpe(r), 25,
                                skewness=float(np.mean(z ** 3)),
                                kurtosis=float(np.mean(z ** 4)),
                                autocorr_lag1=float(np.corrcoef(r[:-1], r[1:])[0, 1]),
                                n_returns=len(r))
    assert a["deflated_sharpe"] == pytest.approx(b["deflated_sharpe"], rel=1e-12)


# ---------------------------------------------------------------- 单位一致性

def test_same_market_expressed_daily_or_monthly_agrees():
    daily = _ar1(n=756, phi=0.05, seed=11)
    monthly = np.add.reduceat(daily, np.arange(0, len(daily), 21))
    d = M.deflated_sharpe_from_returns(daily, 50, periods_per_year=252)
    m = M.deflated_sharpe_from_returns(monthly, 50, periods_per_year=12)
    assert d["observed_sharpe"] == pytest.approx(m["observed_sharpe"], rel=0.35)
    # ⚠️ **不**断言 DSR 相等：DSR 里 se ∝ 1/√(T−1) 且矩由序列估出来，
    # 所以它天然依赖"有多少个观测"，不是表示不变的量（同一段行情按日聚合 T=756、
    # 按月聚合 T=36，证据量本来就不同）。这正是它必须显式声明 `periods_per_year`
    # 而不能靠猜的原因。
    assert d["n_returns"] == 756 and m["n_returns"] == 36
    assert math.isfinite(d["deflated_sharpe"]) and math.isfinite(m["deflated_sharpe"])


def test_sharpe_significance_uses_per_period_units():
    """年化 Sharpe 必须先换算成逐期再比逐期标准误。
    旧实现直接相除，会把 p 值系统性压低（= 凭空放大显著性）。"""
    ann, T = 2.0, 500
    want = 1.0 - M._norm_cdf((ann / math.sqrt(252)) / (1.0 / math.sqrt(T - 1)))
    assert M.sharpe_significance_pvalue(ann, T) == pytest.approx(want)
    wrong = 1.0 - M._norm_cdf(ann / (1.0 / math.sqrt(T - 1)))
    assert M.sharpe_significance_pvalue(ann, T) > wrong * 10, "单位 bug 不得回归"


# ---------------------------------------------------------------- 退化输入

@pytest.mark.parametrize("bad", [[], [0.01], [0.01, 0.02]])
def test_too_few_observations_is_reported_not_faked(bad):
    out = M.deflated_sharpe_from_returns(bad, 5)
    assert math.isnan(out["deflated_sharpe"]) and "样本不足" in out["reason"]


def test_zero_variance_is_reported():
    out = M.deflated_sharpe_from_returns([0.01] * 50, 5)
    assert math.isnan(out["deflated_sharpe"]) and "方差为 0" in out["reason"]


# ---------------------------------------------------------------- 逐窗拼接

def test_pooling_concatenates_non_overlapping_windows_in_date_order():
    per = [(("2025-07-09", "2025-10-09"), [3.0, 4.0]),
           (("2024-10-09", "2025-01-09"), [1.0, 2.0])]
    assert M.pool_windows_rets(per) == [1.0, 2.0, 3.0, 4.0]


def test_pooling_allows_touching_windows():
    per = [(("2025-01-09", "2025-04-09"), [2.0]),
           (("2024-10-09", "2025-01-09"), [1.0])]
    assert M.pool_windows_rets(per) == [1.0, 2.0]


def test_pooling_refuses_overlapping_windows():
    """重叠会把同一段行情算两次 ⇒ 拼出来的序列比真实观测长、自相关被抬高。
    此时**不报** DSR，而不是拿一个拼不出来的数去报。"""
    per = [(("2024-10-09", "2025-02-09"), [1.0, 2.0]),
           (("2025-01-09", "2025-04-09"), [3.0, 4.0])]
    assert M.pool_windows_rets(per) is None


def test_pooling_empty():
    assert M.pool_windows_rets([]) == []


# ---------------------------------------------------------------- 源码级护栏

def test_only_metrics_may_call_the_raw_dsr_function():
    """除 metrics.py 外不得有第二处直接调用 `deflated_sharpe_ratio` ——
    否则"口径只有一个"会在下一次改动里悄悄失效（这正是 B3 的成因）。"""
    offenders = []
    for path in sorted((ROOT / "research").glob("*.py")):
        if path.name == "metrics.py":
            continue
        src = path.read_text()
        for m in re.finditer(r"\bdeflated_sharpe_ratio\s*\(", src):
            line = src[:m.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}")
    assert not offenders, (
        f"这些地方绕过了 deflated_sharpe_from_returns：{offenders} —— "
        f"请改用它（T/矩/自相关必须从同一条收益序列派生）")
