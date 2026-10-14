"""扩展指标测试:因果性 + 取值域 + 横截面正确性。不联网、不读真库。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.indicators import (EXT_COLS, add_cross_sectional,
                                 add_extended_indicators, compute_all,
                                 compute_indicators)
from research.tests.conftest import make_synth_panel


@pytest.fixture
def ext():
    return compute_all(make_synth_panel(("AAA", "BBB", "CCC"), bars=420))


# ---------------------------------------------------------------- 因果性

def test_extended_indicators_are_causal():
    """截断到 T 重算,T 处每个扩展指标必须逐位相同。

    这是本组指标唯一真正危险的地方:ADX / KDJ / OBV 都有多种平滑变体,
    用错一个(如 rolling(center=True) 或对整列做 z-score)就会偷看未来,
    而结果看起来完全正常。
    """
    full = make_synth_panel(("AAA",), bars=420)
    dates = sorted(full["date"].unique())
    probe = pd.Timestamp(dates[350])

    a = compute_all(full)
    b = compute_all(full[full["date"] <= probe])

    ra = a[a["date"] == probe]
    rb = b[b["date"] == probe]
    assert len(ra) == 1 and len(rb) == 1
    for c in EXT_COLS:
        va, vb = ra.iloc[0][c], rb.iloc[0][c]
        if isinstance(va, float) and np.isnan(va):
            assert isinstance(vb, float) and np.isnan(vb), f"{c}: NaN 不一致"
            continue
        assert float(va) == pytest.approx(float(vb), abs=0.0), \
            f"{c} 在 {probe.date()} 被未来数据改变了: {va!r} -> {vb!r}"


def test_extended_indicators_group_by_symbol():
    """把 BBB 在 T 之后的数据 ×3,AAA 的扩展指标必须一字不变。"""
    panel = make_synth_panel(("AAA", "BBB"), bars=420)
    dates = sorted(panel["date"].unique())
    cut = pd.Timestamp(dates[300])

    base = compute_all(panel)
    mut = panel.copy()
    m = (mut["symbol"] == "BBB") & (mut["date"] > cut)
    for c in ("open", "high", "low", "close"):
        mut.loc[m, c] = mut.loc[m, c] * 3.0
    mut.loc[m, "amount"] = mut.loc[m, "volume"] * mut.loc[m, "close"]
    after = compute_all(mut)

    probe = pd.Timestamp(dates[300])
    for sym in ("AAA", "BBB"):
        ra = base[(base.symbol == sym) & (base.date == probe)].iloc[0]
        rb = after[(after.symbol == sym) & (after.date == probe)].iloc[0]
        for c in EXT_COLS:
            va, vb = ra[c], rb[c]
            if isinstance(va, float) and np.isnan(va):
                continue
            assert float(va) == pytest.approx(float(vb), abs=0.0), f"{sym} {c} 跨股票泄漏"


# ---------------------------------------------------------------- 取值域

def test_rsi_bounds_and_warmup(ext):
    for c in ("rsi14", "rsi7"):
        s = ext[c].dropna()
        assert s.between(0, 100).all(), f"{c} 越界"
        assert len(s) > 0
    a = ext[ext.symbol == "AAA"].reset_index(drop=True)
    assert a["rsi14"].iloc[:13].isna().all()
    assert not np.isnan(a["rsi14"].iloc[14])


def test_rsi_is_high_when_price_monotonic_up():
    """构造只涨不跌的序列 → RSI 应贴近 100。"""
    panel = make_synth_panel(("AAA",), bars=120)
    panel = panel.copy()
    panel["close"] = 100.0 + np.arange(len(panel)) * 0.5
    panel["high"] = panel["close"] + 0.1
    panel["low"] = panel["close"] - 0.1
    panel["open"] = panel["close"]
    panel["amount"] = panel["volume"] * panel["close"]
    a = add_extended_indicators(compute_indicators(panel))
    assert a["rsi14"].iloc[-1] > 99.0


def test_bollinger_relationships(ext):
    s = ext.dropna(subset=["bb_up"]).reset_index(drop=True)
    assert (s["bb_up"] >= s["bb_mid"]).all()
    assert (s["bb_mid"] >= s["bb_lo"]).all()
    # %B:收盘在带中 → 约 0.5
    mid_pctb = s.loc[(s["close"] - s["bb_mid"]).abs() < 1e-9, "bb_pctb"]
    assert len(mid_pctb) == 0 or np.allclose(mid_pctb, 0.5, atol=1e-6)
    assert (s["bb_width"] >= 0).all()


def test_stochastic_bounds(ext):
    s = ext["stoch_k"].dropna()
    assert s.between(-1e-9, 100 + 1e-9).all()
    d = ext["stoch_d"].dropna()
    assert d.between(-1e-9, 100 + 1e-9).all()


def test_adx_bounds_and_warmup(ext):
    s = ext["adx14"].dropna()
    assert s.between(-1e-9, 100 + 1e-9).all(), "ADX 应在 0-100"
    for c in ("plus_di", "minus_di"):
        assert ext[c].dropna().between(-1e-9, 100 + 1e-9).all()
    # ADX 需要 tr 与 dx 两次 Wilder 平滑 → 预热应长于单次平滑
    a = ext[ext.symbol == "AAA"].reset_index(drop=True)
    assert a["adx14"].iloc[:27].isna().all()


def test_52week_position(ext):
    s = ext.dropna(subset=["dist_high252"]).reset_index(drop=True)
    assert (s["dist_high252"] <= 1e-9).all(), "相对 252 日最高应为非正"
    assert (s["dist_low252"] >= -1e-9).all(), "相对 252 日最低应为非负"
    assert (s["dist_low252"] >= s["dist_high252"]).all()


def test_obv_direction():
    """单调上涨 → OBV 单调递增;单调下跌 → 单调递减。"""
    for slope, want in ((0.5, 1), (-0.5, -1)):
        panel = make_synth_panel(("AAA",), bars=80)
        panel = panel.copy()
        panel["close"] = 100.0 + np.arange(len(panel)) * slope
        panel["high"] = panel["close"] + 0.1
        panel["low"] = panel["close"] - 0.1
        panel["open"] = panel["close"]
        panel["amount"] = panel["volume"] * panel["close"]
        a = add_extended_indicators(compute_indicators(panel))
        d = a["obv"].diff().dropna()
        assert (np.sign(d) == want).all() or (d == 0).all()


def test_vol_z_matches_formula(ext):
    """直接验公式:vol_z == (volume − MA20) / SD20。

    不做分布假设 —— 合成数据的成交量是线性增长的,20 日均线滞后会让 z 分
    系统性偏正,那是数据性质不是实现错误。验公式才是对的测法。
    """
    a = ext[ext.symbol == "AAA"].reset_index(drop=True)
    v = a["volume"]
    m = v.rolling(20, min_periods=20).mean()
    s = v.rolling(20, min_periods=20).std(ddof=1)
    want = (v - m) / s
    ok = want.notna()
    assert ok.sum() > 0, "样本不足,测试无效"
    assert np.allclose(a.loc[ok, "vol_z"], want[ok], equal_nan=True)


def test_obv_slope_and_rvol_are_finite_where_defined(ext):
    for c in ("obv_slope", "rvol20", "mom60"):
        s = ext[c].dropna()
        assert len(s) > 0, f"{c} 全为 NaN"
        assert np.isfinite(s).all(), f"{c} 含 inf"


# ---------------------------------------------------------------- 横截面

def test_cross_sectional_rank_is_per_date_and_bounded():
    panel = make_synth_panel(("AAA", "BBB", "CCC", "DDD"), bars=420)
    ind = add_cross_sectional(add_extended_indicators(compute_indicators(panel)))
    for c in ("xs_mom120", "xs_mom60", "xs_rvol20"):
        s = ind[c].dropna()
        assert s.between(0, 1).all(), f"{c} 应在 [0,1]"
    # 每个日期的排名必须在当天完成:某天的最大值应为 1(样本齐全时)
    d = ind.dropna(subset=["xs_mom120"]).groupby("date")["xs_mom120"].max()
    assert np.allclose(d.values, 1.0)


def test_cross_sectional_removes_the_market_drift():
    """横截面的作用:把"市场整体涨了多少"剥掉,只留"相对同池的强弱"。

    ⚠️ 不能用"xs 与原始 mom120 的相关性"来测 —— 在**单个日期内**,排名是原始
    动量的单调变换,两者必然高度相关。真正的区别在**跨时间**:原始动量的水平
    随市场漂移(牛市里全池都很高),而横截面排名的分布恒定居中。
    """
    panel = make_synth_panel(tuple(f"S{i:02d}" for i in range(21)), bars=420)
    ind = add_cross_sectional(add_extended_indicators(compute_indicators(panel)))
    s = ind.dropna(subset=["xs_mom120", "mom120"])
    by_month = s.groupby(s["date"].dt.to_period("M"))

    raw_med = by_month["mom120"].median()
    xs_med = by_month["xs_mom120"].median()

    # 判据是**对比**,不是绝对阈值:原始动量的月度中位数漂移应远大于横截面排名。
    # (绝对阈值依赖合成数据的趋势强度,换个数据就失效;比值不依赖。)
    assert raw_med.std() > 5.0 * xs_med.std(), (
        f"横截面没有剥掉市场漂移:raw std={raw_med.std():.4f} vs xs std={xs_med.std():.4f}")
    assert xs_med.std() < 0.02, f"横截面中位数不该漂移 ({xs_med.std():.4f})"
    # 注意 pct 排名的中位数不是 0.5:rank(pct=True) = i/n,故 n 只时为 (n+1)/(2n)。
    # n=21 → 0.5238。容差按此设,不要把定义性偏置当成 bug。
    assert abs(float(xs_med.mean()) - (21 + 1) / (2 * 21)) < 0.02


def test_cross_sectional_uses_only_same_day_data():
    """每个日期的排名必须只由**当天**的横截面决定,不跨日期。"""
    panel = make_synth_panel(("AAA", "BBB", "CCC", "DDD"), bars=400)
    ind = add_cross_sectional(add_extended_indicators(compute_indicators(panel)))
    ok = ind.dropna(subset=["xs_mom120"])
    # 用当天的 mom120 独立重算一遍排名,必须逐位一致
    want = ok.groupby("date")["mom120"].rank(pct=True)
    assert np.allclose(ok["xs_mom120"].to_numpy(), want.to_numpy())
