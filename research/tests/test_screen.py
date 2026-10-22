"""筛选器测试。不联网、不读真库。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import screen
from research.indicators import compute_indicators
from research.tests.conftest import make_synth_panel


@pytest.fixture
def ind():
    return compute_indicators(make_synth_panel(("AAA", "BBB"), bars=300))


# ---------------------------------------------------------------- 未来函数防线

def test_masks_use_only_past_data():
    """选股掩码在 T 处只依赖 ≤T —— 截断重算必须逐位相同。

    这是整个工具的**根本前提**:清单必须能用当天收盘的数据算出来。
    若掩码里混进 shift(-x) 或中心化 rolling,这条会红。
    """
    full = make_synth_panel(("AAA",), bars=300)
    dates = sorted(full["date"].unique())
    cut = pd.Timestamp(dates[250])

    m_full = screen._masks(compute_indicators(full))
    m_trunc = screen._masks(compute_indicators(full[full["date"] <= cut]))

    for tier in screen.TIER_ORDER:
        row_full = m_full[tier][compute_indicators(full)["date"] == cut].iloc[0]
        row_trunc = m_trunc[tier][
            compute_indicators(full[full["date"] <= cut])["date"] == cut].iloc[0]
        assert bool(row_full) == bool(row_trunc), f"{tier} 在 {cut.date()} 用到了未来数据"


def test_forward_returns_is_nan_at_the_tail():
    """远期收益必须有尾巴 NaN —— 否则就是在用不存在的未来价格。"""
    panel = make_synth_panel(("AAA",), bars=60)
    ind = compute_indicators(panel)
    h = 18
    fwd = screen.forward_returns(ind, h)
    assert fwd.iloc[-1] != fwd.iloc[-1]          # 末行为 NaN
    assert fwd.iloc[-h] != fwd.iloc[-h]          # 第 -h 行也拿不到 T+h 收盘
    assert fwd.iloc[-h - 1] == fwd.iloc[-h - 1]  # 再往前一行就有值
    # fwd[i] = close[i+h]/open[i+1] - 1,需要 i+1 ≤ n-1 且 i+h ≤ n-1
    # → i ≤ n-h-1,即 i ∈ [0, n-h-1],共 n-h 个有效值
    assert fwd.notna().sum() == 60 - h


def test_forward_returns_matches_manual_computation():
    panel = make_synth_panel(("AAA",), bars=40)
    ind = compute_indicators(panel)
    fwd = screen.forward_returns(ind, 5)
    # 手算第 0 行:入场 = 第 1 行 open,出场 = 第 5 行 close
    want = ind["close"].iloc[5] / ind["open"].iloc[1] - 1
    assert fwd.iloc[0] == pytest.approx(want)


def test_forward_returns_does_not_cross_symbols():
    """分组必须按 symbol —— 否则会用下一只票的价格当本票的"未来"。"""
    panel = make_synth_panel(("AAA", "BBB"), bars=40)
    ind = compute_indicators(panel)
    fwd = screen.forward_returns(ind, 5)
    a = ind[ind["symbol"] == "AAA"].reset_index(drop=True)
    fa = fwd[ind["symbol"] == "AAA"].reset_index(drop=True)
    assert fa.iloc[0] == pytest.approx(a["close"].iloc[5] / a["open"].iloc[1] - 1)
    assert fa.iloc[-1] != fa.iloc[-1]


# ---------------------------------------------------------------- 分层

def test_tiers_are_nested_in_the_expected_order():
    """层级必须逐级收窄:trend ⊇ trend_pullback ⊇ ... ⊇ v21_full。

    不满足就说明某一层加的条件不是"再 ∧ 一个",而是把集合放大了 ——
    那样"边际贡献"的解读就完全不成立。
    """
    panel = make_synth_panel(("AAA", "BBB", "CCC"), bars=400)
    m = screen._masks(compute_indicators(panel))
    order = ["base", "trend", "trend_pullback", "trend_pullback_macd", "v21_full"]
    for a, b in zip(order, order[1:]):
        assert (m[a].sum()) >= (m[b].sum()), f"{a} 应 ⊇ {b}"
        assert not (m[b] & ~m[a]).any(), f"{b} 有 {a} 之外的行"


def test_base_tier_includes_everything_ready():
    panel = make_synth_panel(("AAA",), bars=400)
    ind = compute_indicators(panel)
    m = screen._masks(ind)
    # base 要求指标就绪;预热期必须是 False
    assert not m["base"].iloc[:120].any()
    assert m["base"].iloc[130:].all()


# ---------------------------------------------------------------- measure / candidates

def test_measure_returns_one_row_per_tier(ind):
    m = screen.measure(ind, 18, lo="2020-01-01", hi="2021-06-01")
    assert list(m["tier"]) == screen.TIER_ORDER
    assert "t_adj" in m.columns and "vs_base" in m.columns


def test_base_row_is_self_referential(ind):
    """base 行相对自己是 0,且不该被拿去判读优劣。"""
    m = screen.measure(ind, 18, lo="2020-01-01", hi="2021-06-01")
    b = m[m["tier"] == "base"].iloc[0]
    assert float(b["vs_base"]) == 0.0


def test_overlap_correction_shrinks_t(ind):
    """校正后的 t 必须严格小于原 t —— 这是重叠采样校正存在的意义。"""
    m = screen.measure(ind, 18, lo="2020-01-01", hi="2021-06-01")
    for _, r in m.iterrows():
        if np.isfinite(r["t_raw"]) and np.isfinite(r["t_adj"]):
            assert abs(r["t_adj"]) < abs(r["t_raw"]) or r["t_raw"] == 0


def test_candidates_rejects_non_trading_day(ind):
    with pytest.raises(SystemExit, match="不是库里的交易日"):
        screen.candidates(ind, "trend", "1999-01-01", top=5)


def test_candidates_columns_and_ordering(ind):
    last = str(ind["date"].max().date())
    c = screen.candidates(ind, "base", last, top=5)
    assert list(c.columns) == ["symbol", "close", "mom120", "pullback", "down_streak",
                               "ret1", "atr_pct", "amount60", "above_ma60_ratio", "score"]
    assert c["score"].is_monotonic_decreasing


# ------------------------------------------------- 多重比较校正(CLAUDE.md §5(c))
# 铁律原文:evaluation scripts MUST output Deflated Sharpe Ratio (DSR) or apply
# Bonferroni corrections when reporting backtest results from multi-parameter sweeps.
#
# 2026-10-15 审计实测:screen.py 是仓库里唯一把统计判词**直接印给人看**的评估脚本,
# 而它逐个层级下判词、阈值却硬编码 |t|=2(等价 α=0.05 单次),全文没有 DSR /
# Bonferroni / n_trials。已实测触发过 t=−2.52 的判词,而 4 层级 × 3 个推荐 horizon
# 的族门槛是 2.865。

def test_bonferroni_threshold_grows_with_family_size():
    assert screen.bonferroni_t_threshold(1) == pytest.approx(1.96, abs=0.01)
    assert screen.bonferroni_t_threshold(4) == pytest.approx(2.498, abs=0.01)
    assert screen.bonferroni_t_threshold(12) == pytest.approx(2.865, abs=0.02)
    with pytest.raises(ValueError, match="family_size"):
        screen.bonferroni_t_threshold(0)


def test_verdict_uses_the_corrected_threshold():
    r = pd.Series({"t_adj": 2.2, "ann": 0.10})
    assert "优于" in screen._verdict(r, 1.96), "族规模 1 时 2.2 应判优于"
    assert "无差别" in screen._verdict(r, 2.498), "族规模 4 时 2.2 达不到门槛,应判无差别"
    assert "劣于" in screen._verdict(pd.Series({"t_adj": -2.9, "ann": -0.1}), 2.498)


def test_measure_records_the_family_size_and_threshold(ind):
    m = screen.measure(ind, 18, lo=str(ind["date"].min().date()),
                       hi=str(ind["date"].max().date()), family_size=4)
    assert m.attrs["family_size"] == 4
    assert m.attrs["t_crit"] == pytest.approx(2.498, abs=0.01)


def test_measure_defaults_the_family_to_the_number_of_tested_tiers(ind):
    """族规模按**被检验**的层级算 —— base 是全池基准本身,不是一次比较。"""
    m = screen.measure(ind, 18, lo=str(ind["date"].min().date()),
                       hi=str(ind["date"].max().date()))
    assert m.attrs["family_size"] == len(screen.COMPARED_TIERS) == 4
    assert m.attrs["t_crit"] == pytest.approx(2.498, abs=0.01)


# ---------------------------------------------------------------- 与引擎同源（F4）

def test_v21_full_tier_is_exactly_the_engines_own_selection(ind):
    """`v21_full` 必须**就是**引擎的 v21 —— 手抄条件迟早与引擎分叉（审计 F4）。

    此前这一层多一个 `ma20 > ma60`（引擎的 A 组没有这一条），于是用户在候选清单里
    跟着的那个"历史基准率"量的是**另一个策略**。现在它由 `signals.select_entries`
    直接给出，所以两者的 (date, symbol) 命中集合必须逐个相同。
    """
    from research import signals
    from research.screen import _masks
    df = ind
    m = _masks(df)["v21_full"]
    d0 = str(df["date"].min().date())
    d1 = str((df["date"].max() + pd.Timedelta(days=1)).date())
    eng = signals.select_entries(df, start_date=d0, end_date=d1)
    got = set(map(tuple, df.loc[m, ["date", "symbol"]].to_numpy()))
    want = set(map(tuple, eng[["date", "symbol"]].to_numpy()))
    assert got == want, (
        f"screen 的 v21_full 与引擎的 v21 不一致：多 {len(got - want)} 条、少 "
        f"{len(want - got)} 条（例：{(got - want) or (want - got)}）")


def test_v21_full_is_not_hand_copied_conditions():
    """源码级护栏：这一层不得再出现手抄的条件组合。"""
    src = (Path(__file__).resolve().parents[2] / "research" / "screen.py").read_text()
    assert '"v21_full": trend & pullback' not in src, \
        "v21_full 又被手抄成条件组合了 —— 它会与引擎分叉"
    assert "signals.select_entries" in src
