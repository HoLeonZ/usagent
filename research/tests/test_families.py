"""策略族层的测试 —— 重点是**两条路径的对称性**。

family_search 用它自己的 families.select 构造信号表,而 search_12m 走
signals.select_entries。「新族 vs uptrend-pullback」的结论只有在两条路径构造出的信号表
**逐位相同**时才成立 —— 否则比的是两台不同的机器。参考族 upb_ref 就是干这个用的。
"""
from __future__ import annotations

import pandas as pd
import pytest

from research import families as F, signals
from research.indicators import compute_all
from research.tests.conftest import make_synth_panel


@pytest.fixture
def ind():
    return F.add_family_indicators(compute_all(make_synth_panel(("AAA", "BBB"), bars=360)))


def _window(df: pd.DataFrame) -> tuple[str, str]:
    days = sorted(df["date"].unique())
    return str(days[125])[:10], str(days[-1])[:10]


# ⚠️ 对称性检查**不在合成面板上做**:A~F 的条件很严,合成序列在该窗口出 0 个信号,
#    两条路径都是 0 个,断言 0==0 是空断言(实测踩过)。见 test_golden_trades 的真实数据版。


def test_reference_family_is_excluded_from_the_comparison():
    """参考族不得出现在默认的横向比较里(它等于基准本身)。"""
    assert "upb_ref" in F.MASK
    assert "upb_ref" not in F.NEW_FAMILIES
    assert set(F.NEW_FAMILIES) == set(F.MASK) - F.REFERENCE_FAMILIES


def test_family_grids_are_pre_registered_and_non_empty():
    for fid in F.NEW_FAMILIES:
        assert F.grid_size(fid) >= 4, f"{fid} 的网格太小,不足以说明什么"
        assert all(isinstance(v, list) and v for v in F.GRIDS[fid].values())


def test_families_do_not_require_fundamentals(ind):
    """用户明确排除基本面:族只能读价格/量/分红这些列。

    (本测试也顺带把每个族的网格取第一格跑通一遍 —— 网格写错会在这里炸。)
    """
    from research import regime

    reg = regime.compute_regime(ind)
    banned = ("eps", "revenue", "net_income", "book", "market_cap", "pe_")
    for fid in F.NEW_FAMILIES:
        p = {k: v[0] for k, v in F.GRIDS[fid].items()}
        out = F.select(ind, fid, p, *_window(ind), reg=reg)
        assert list(out.columns) == list(signals.ENTRY_COLS)
    for col in ind.columns:
        assert not any(b in col.lower() for b in banned), f"面板里出现了 {col}"


def test_regime_dependent_families_insist_on_the_regime_frame(ind):
    """需要状态门的族在没给 reg 时必须报错,不能静默用一列 NaN 跑出一堆空信号。"""
    from research import regime

    assert F.NEEDS_REGIME, "至少 state_flip / volshock 需要 regime 帧"
    for fid in sorted(F.NEEDS_REGIME):
        with pytest.raises(ValueError, match="regime"):
            F.select(ind, fid, {k: v[0] for k, v in F.GRIDS[fid].items()}, *_window(ind))


def test_unknown_family_raises(ind):
    with pytest.raises(KeyError, match="未知策略族"):
        F.select(ind, "nope", {}, *_window(ind))


# ---------------------------------------------------------------- 价格下限的价轴（A4）

def test_price_floor_uses_the_as_traded_axis(ind):
    """`MIN_PRICE` 是**绝对美元数** ⇒ 必须比 as-traded 价（`raw_close`）。

    审计 A4：此前比的是 `close` —— 那是**分红回补价**，其回补因子在 T 日包含了 T 之后
    的所有分红，所以"T 日价格 ≥ $1"会变成未来分红的函数（与 panel.py 已修掉的 `amount`
    是同一类未来函数）。实测（真数据）：1,301 个标的×日期（0.066%）被**错误排除**，
    涉及 177 只标的；反向（错误纳入）0 行 —— 复权把历史价压低了，所以此前更严。
    """
    d = ind.copy()
    day = d["date"].iloc[200]
    d.loc[d.index[200], ["close", "raw_close"]] = [0.50, 1.20]   # 复权后 <$1，实际 ≥$1
    d.loc[d.index[300], ["close", "raw_close"]] = [1.50, 0.80]   # 复权后 ≥$1，实际 <$1
    floor = F.above_price_floor(d)
    assert bool(floor.iloc[200]) is True, "实际价 ≥$1 就不该被价格下限挡掉"
    assert bool(floor.iloc[300]) is False, "实际价 <$1 就该被挡掉（不许看复权价）"
    assert day is not None


def test_price_floor_fails_fast_without_the_raw_axis():
    """缺 `raw_close` 必须抛错 —— 静默退化回复权轴正是这条缺陷的成因。"""
    import pandas as pd
    bad = pd.DataFrame({"close": [1.5], "date": [pd.Timestamp("2024-01-02")]})
    with pytest.raises(ValueError, match="raw_close"):
        F.above_price_floor(bad)


def test_no_family_compares_the_floor_against_the_adjusted_axis():
    """源码级护栏：不许再出现 `ind["close"] >= MIN_PRICE` 这种写法。"""
    from pathlib import Path
    src = (Path(F.__file__)).read_text(encoding="utf-8")
    assert 'ind["close"] >= MIN_PRICE' not in src, \
        "价格下限又比回复权价了 —— 那是未来函数（审计 A4）"
    assert src.count("above_price_floor(ind)") >= 5, "五处调用点都应当走同一个 helper"
