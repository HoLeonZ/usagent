"""买卖价差估计：方向必须对、必须因果、偏差必须被**测量**过而不是假设。

背景（2026-10-17 审计 E1）：成本模型此前只有规费（5bp 双边）+ 冲击滑点，
**没有买卖价差**——而价差是尺寸无关的那一半成本。对一个允许 $1 级别票的池子，
把它当 0 是数量级错误。

本文件钉住的四件事：
  1. **方向**：注入已知价差 ⇒ 必须估出正的、量级相近的值；且
     **系统性低估**（实测 100bp→72bp、300bp→247bp）—— 低估是可接受的，
     高估不是（那会凭想象压低收益）；
  2. **因果**：只用 ≤t 的两根 bar；截断序列不得改变过去的值；
  3. **稳健**：单日尖峰不得主导滚动中位；不足窗口 ⇒ NaN（不可判定，不建模）；
  4. **退化边界**：非正价、缺失值、负估计（按 0）都不许产生假的成本。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core import spread as S  # noqa: E402


def synth(spread: float, n: int = 200):
    """已知价差的合成序列：确定的中间价来回走，H/L 各偏 ±半价差。"""
    mid = 100.0 + 0.5 * np.sin(np.arange(n) / 3.0)
    return mid * (1 + spread / 2), mid * (1 - spread / 2)


# ---------------------------------------------------------------- 方向与量级

@pytest.mark.parametrize("injected,lo,hi", [
    (0.01, 40.0, 160.0),        # 100bp → 实测约 72bp
    (0.03, 200.0, 340.0),       # 300bp → 实测约 247bp
])
def test_recovers_a_positive_estimate_of_the_right_order(injected, lo, hi):
    h, l = synth(injected)
    f = S.corwin_schultz(h, l)
    f = f[np.isfinite(f)]
    assert f.size > 0
    assert lo <= float(np.median(f)) <= hi


def test_is_monotone_in_the_spread():
    med = []
    for sp in (0.002, 0.01, 0.03, 0.06):
        h, l = synth(sp)
        v = S.corwin_schultz(h, l)
        med.append(float(np.median(v[np.isfinite(v)])))
    assert med == sorted(med), f"价差越大估计应越大，得到 {med}"


def test_underestimates_rather_than_overestimates():
    """**偏差方向**被钉住：低估可以接受（不留想象空间），高估不行。"""
    for sp in (0.01, 0.03, 0.06):
        h, l = synth(sp)
        v = S.corwin_schultz(h, l)
        med = float(np.median(v[np.isfinite(v)]))
        assert med <= sp * 1e4 * 1.05, f"注入 {sp*1e4:.0f}bp 却估出 {med:.0f}bp（高估）"


def test_never_negative_and_capped():
    rng = np.random.default_rng(7)
    mid = 100 * np.cumprod(1 + rng.normal(0, 0.05, 500))     # 剧烈波动
    h = mid * (1 + rng.uniform(0, 0.2, 500))
    l = mid * (1 - rng.uniform(0, 0.2, 500))
    v = S.corwin_schultz(h, l)
    f = v[np.isfinite(v)]
    assert (f >= 0).all(), "负估计必须按 0 处理"
    assert (f <= S.MAX_SPREAD_BPS).all(), "必须截断在硬上界"


# ---------------------------------------------------------------- 因果性

def test_truncating_history_does_not_change_past_values():
    h, l = synth(0.02, n=300)
    full = S.corwin_schultz(h, l)
    for cut in (150, 220, 299):
        part = S.corwin_schultz(h[:cut], l[:cut])
        assert np.allclose(np.nan_to_num(full[:cut]), np.nan_to_num(part), rtol=0, atol=0), \
            f"截断到 {cut} 后过去的值变了 —— 估计量看了未来"


def test_rolling_median_is_causal_and_spike_robust():
    x = np.full(60, 10.0)
    x[30] = 400.0                     # 单日尖峰
    sm = S.rolling_median(x, lookback=20)
    # 尖峰进入窗口后，中位只被抬高一点点（20 个里只有 1 个是尖峰）
    assert sm[31] < 40.0, f"单日尖峰主导了平滑值：{sm[31]}"
    # 追加未来数据不得改变过去
    longer = S.rolling_median(np.append(x, [999.0] * 10), lookback=20)
    assert np.allclose(sm, longer[:len(sm)], equal_nan=True)


def test_rolling_median_head_is_undefined():
    x = np.full(10, 5.0)
    sm = S.rolling_median(x, lookback=20)
    assert np.isnan(sm).all(), "不足一个完整窗口 ⇒ 不可判定（不建模），不是猜一个值"


def test_rolling_median_respects_min_periods():
    x = np.array([np.nan] * 15 + [7.0] * 5)
    assert np.isnan(S.rolling_median(x, lookback=5, min_periods=5)[18])
    assert S.rolling_median(x, lookback=5, min_periods=3)[18] == pytest.approx(7.0)


# ---------------------------------------------------------------- 退化边界

def test_bad_bars_are_undefined_not_zero():
    h = np.array([101.0, 0.0, 103.0, 104.0])
    l = np.array([99.0, 98.0, 97.0, 96.0])
    v = S.corwin_schultz(h, l)
    assert np.isnan(v[1]) and np.isnan(v[2]), "非正价不得当成 0 价差"
    assert not np.isnan(v[3])


def test_high_below_low_is_undefined():
    h = np.array([100.0, 99.0])
    l = np.array([101.0, 98.0])
    assert np.isnan(S.corwin_schultz(h, l)[1])


def test_too_short_series():
    assert np.isnan(S.corwin_schultz(np.array([1.0]), np.array([1.0]))[0])
    assert S.corwin_schultz(np.array([]), np.array([])).size == 0


# ---------------------------------------------------------------- 费率换算

@pytest.mark.parametrize("bps,rate", [(100.0, 0.005), (0.0, 0.0), (float("nan"), 0.0),
                                      (-5.0, 0.0)])
def test_half_spread_rate(bps, rate):
    assert S.half_spread_rate(bps) == pytest.approx(rate)


def test_describe_mentions_the_known_bias():
    assert "Corwin" in S.describe()
