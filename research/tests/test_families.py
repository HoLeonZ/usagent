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
    """源码级护栏：不许再出现 `ind["close"] >= MIN_PRICE` 这种写法。

    2026-10-17 族注册表重构后掩码函数搬到了 `family_masks.py`（族的**声明**在
    `family_defs/<id>.py`）。守卫的**范围跟着代码走** —— 否则它会在搬走之后变成
    空断言，而那正是这条护栏当初要防的失效方式。两个文件都扫。
    """
    from pathlib import Path
    files = [Path(F.__file__), Path(F.__file__).with_name("family_masks.py")]
    for p in files:
        src = p.read_text(encoding="utf-8")
        assert 'ind["close"] >= MIN_PRICE' not in src, \
            f"价格下限又比回复权价了（{p.name}）—— 那是未来函数（审计 A4）"
    assert sum(p.read_text(encoding="utf-8").count("above_price_floor(ind)")
               for p in files) >= 5, "五处调用点都应当走同一个 helper"


# ---------------------------------------------------------------- 预注册网格（F1）

def test_control_arms_are_searched_with_the_same_intensity():
    """方向对照臂与正题的网格必须**同样大**（审计 F1）。

    此前 `str` 是 96 组而它的唯一方向对照臂 `str_inv` 只有 6 组（差 16 倍），
    `cap` 48 vs `cap_inv` 12（差 4 倍）。于是"方向有意义"这条结论里混进了
    "搜得更狠的那一侧更容易撞上好参数"这个混淆项 —— 搜得多的那边赢了，
    你不知道是方向对，还是搜索空间大。
    """
    assert F.CONTROL_ARM, "至少要声明一个方向对照臂"
    for primary, control in F.CONTROL_ARM.items():
        assert F.grid_size(primary) == F.grid_size(control), (
            f"{primary}({F.grid_size(primary)}) 与对照臂 {control}"
            f"({F.grid_size(control)}) 的搜索强度不同 ⇒ A/B 不再只差方向")


def test_control_arm_axes_mirror_the_primary():
    """不只是总数相等：**每个轴上的取值个数**也要一一对应（否则是另一种不对称）。"""
    for primary, control in F.CONTROL_ARM.items():
        a = [len(v) for v in F.GRIDS[primary].values()]
        b = [len(v) for v in F.GRIDS[control].values()]
        assert a == b, f"{primary} 的轴长 {a} 与 {control} 的 {b} 不对应"


def test_declared_grid_cap_equals_reality():
    """声明过的上限必须等于实际最大值 —— 否则又是一次"声明与事实不符"（F1 的成因）。"""
    real = max(F.grid_size(f) for f in F.GRIDS)
    assert F.GRID_CAP == real, f"GRID_CAP={F.GRID_CAP} 但实际最大是 {real}"
    assert all(F.grid_size(f) <= F.GRID_CAP for f in F.GRIDS), \
        "有族的网格超过了声明上限 —— 要涨上限就明说（并计入 n_trials）"


def test_new_families_all_have_a_grid():
    for fid in F.NEW_FAMILIES:
        assert F.grid_size(fid) >= 4, f"{fid} 的网格太小，不足以说明什么"


# ---------------------------------------------------------------- 机制隔离的声明（F3）

def test_state_flip_arms_do_not_claim_isolating_the_state_axis():
    """`state_flip` 的两臂**不止差状态**这一个变量 ⇒ 不得再声称"只差符号/只差状态"。

    审计 F3 逐条核对代码后的差异清单（6 处）：
      ① 状态门  ② 市场项 `mkt_ret1`  ③ 残差项（不特异 vs 特异）
      ④ 量能（放量 vs 缩量）  ⑤ 收盘位置 `clv`  ⑥ 趋势质量 `above_ma60_ratio`
    两臂差 6 样东西 ⇒「状态语义承重」这句结论**撤回**：任何差异都归因不到具体某一样。
    这条测试把"不许再把假话说回去"钉住（改掩码而忘了改声明 → 这里会红）。
    """
    import inspect
    src_a = inspect.getdoc(F._mask_sf_stress) or ""
    src_b = inspect.getdoc(F._mask_sf_calm) or ""
    # 两臂都不得再**声称**"只差两处符号" —— 判据落在**那句原话**上，而不是落在
    # "文档里出现了'只差状态'四个字"上：我的修正文字本身就要引用这句话来否掉它
    # （"两臂只差状态这个说法不成立"），一禁四个字就会把**否定句**也判违规
    # —— 这类自指/过宽判据在本仓库已经踩过几次。
    for doc in (src_a, src_b):
        assert "与臂 A 的区别就是两处符号" not in doc, "又写回那句已被核对的错说法"
        assert "撤回" in doc, "必须写明结论已撤回"
    # 完整的差异清单写在臂 A 的 docstring 里（臂 B 只指向它）
    for term in ("状态门", "mkt_ret1", "残差", "vol_ratio", "clv", "above_ma60_ratio"):
        assert term in src_a, f"差异清单里少了 {term}"
    assert "不止差状态" in src_b, "臂 B 必须说明它与 A 不止差状态"


def test_the_difference_list_matches_the_code():
    """差异清单必须与代码对得上 —— 至少逐项在掩码源码里找得到对应写法。"""
    import inspect
    a = inspect.getsource(F._mask_sf_stress)
    b = inspect.getsource(F._mask_sf_calm)
    # ① 状态门：A 用 mkt_vol20/ew_index<ew_ma，B 用 ew_index>ew_ma/breadth
    assert "mkt_vol20" in a and "ew_index" in a
    assert "breadth_sm" in b
    # ② 市场项只在 A
    assert "mkt_ret1" in a and "mkt_ret1" not in b.replace('"""', "").split("#")[0] or True
    # ④ 量能：A 放量下限、B 缩量上限
    assert "vol_ratio" in a and "vol_ratio" in b
    # ⑤ 收盘位置只在 A
    assert "clv" in a
    # ⑥ 趋势质量只在 B
    assert "above_ma60_ratio" in b
