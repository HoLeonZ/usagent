"""曝光匹配口径（`research/overlay.py`）的测试 —— 一把尺，且这把尺的两个端点可判。

为什么值得单独测：这条口径是"新 preset 是否更好"的**唯一裁判**。
它错了，或者它与族路径各算各的，那么"更好"这句话就没有意义 ——
而本仓库已经吃过一次"两条入口读两个面板、把分歧固化成基线"的亏。
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from research import overlay as O


def _eq(equity, exposure, dates):
    return pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity,
                         "exposure": exposure}).set_index("date")


def _bench(dates, rets):
    return pd.Series(rets, index=pd.to_datetime(dates))


DAYS = pd.bdate_range("2024-01-01", periods=60)


def test_full_exposure_equals_the_strategy_itself():
    """敞口恒为 1、池子零收益 ⇒ overlay 就是策略本身（口径的右端点）。

    注意不能拿"池子非零"来测：`w.shift(1)` 在**第一天**恒为 0（前一日没有敞口），
    所以第一天一定吃池子收益 —— 那是口径有意的语义，不是瑕疵。
    """
    r = np.resize([0.01, -0.005, 0.002], 60)
    eq = _eq(np.cumprod(1 + np.r_[0.0, r[1:]]), np.ones(60), DAYS)
    out = O.overlay_from_equity(eq, _bench(DAYS, [0.0] * 60))
    own = float(eq["equity"].pct_change().fillna(0.0).add(1).prod()
                ** (365.25 / (DAYS[-1] - DAYS[0]).days) - 1)
    assert out["cagr"] == pytest.approx(own, rel=1e-12)
    # 平均敞口按 w.shift(1) 算 ⇒ 60 行里的第 1 行恒为 0，均值是 59/60 而不是 1
    assert out["exposure"] == pytest.approx(59 / 60)


def test_zero_exposure_equals_the_pool():
    """敞口恒为 0 ⇒ overlay 就是池指数（口径的左端点）。

    ⚠️ 从**第二天**起：第一行的 `pct_change` 是"窗口开始之前一天 → 首日"的收益，
    组合赚不到它，所以被置零（2026-10-17 台账 H1）。不置零的话，零敞口策略会显示
    +0.44pp 的"超额"—— 实测过。
    """
    r = np.resize([0.002, -0.001], 60)
    eq = _eq(np.ones(60), np.zeros(60), DAYS)
    out = O.overlay_from_equity(eq, _bench(DAYS, r))
    pool = float(pd.Series(r[1:], index=DAYS[1:]).add(1).prod()
                 ** (365.25 / (DAYS[-1] - DAYS[0]).days) - 1)
    assert out["cagr"] == pytest.approx(pool, rel=1e-9)


def test_zero_exposure_gives_exactly_zero_excess():
    """**不变量**：零敞口 ⇒ 超额恰好 0 —— 这是 H1 那处偏置的护栏。

    它与上面那条测的不是一件事：上面测"overlay 等于池子"（同一条序列内的identity），
    这条测"overlay 与**基准那另一把尺**相等"。偏置正是出在两个函数的**锚**不同上，所以
    只有这条能在缺陷发生时变红（实测：修之前这里会得到 +0.44pp）。
    """
    r = np.resize([0.002, -0.001], 60)
    reg = pd.DataFrame({"date": DAYS, "ew_index": np.cumprod(1 + np.r_[0.0, r[1:]]) * 100.0})
    eq = _eq(np.ones(60), np.zeros(60), DAYS)
    lo, hi = str(DAYS[0].date()), str((DAYS[-1] + pd.Timedelta(days=1)).date())
    out = O.excess_vs_pool(eq, reg, lo, hi)
    assert abs(out["excess"]) < 1e-9, \
        f"零敞口却报出超额 {out['excess']:+.4%} —— 两把尺的锚又不一致了"


def test_exposure_is_lagged_by_one_day():
    """**因果**：用的是前一交易日的敞口。当日敞口用上去等于同日前视。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    # 第 2 天敞口变 1；若用当日敞口，第 2 天就吃到策略收益（错）；用滞后的则第 3 天才吃到
    eq = _eq([1.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 1.0, 1.0, 0.0], dates)
    bench = _bench(dates, [0.0] * 5)
    ov = O.overlay_from_equity(eq, bench)
    assert ov["n_days"] == 5 and ov["exposure"] == pytest.approx(0.6)   # (0+0+1+1+1)/5


def test_missing_exposure_column_fails_fast():
    eq = pd.DataFrame({"date": DAYS, "equity": np.ones(60)}).set_index("date")
    with pytest.raises(O.OverlayError, match="exposure"):
        O.overlay_from_equity(eq, _bench(DAYS, [0.0] * 60))


def test_too_short_sample_fails_fast():
    d = pd.bdate_range("2024-01-01", periods=2)
    eq = _eq([1.0, 1.0], [0.0, 0.0], d)
    with pytest.raises(O.OverlayError, match="样本太短"):
        O.overlay_from_equity(eq, _bench(d, [0.0, 0.0]))


def test_pool_daily_returns_uses_the_single_index_definition():
    reg = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=3),
                        "ew_index": [100.0, 110.0, 99.0]})
    r = O.pool_daily_returns(reg)
    assert list(r.iloc[1:]) == pytest.approx([0.10, -0.10])


# ---------------------------------------------------------------- 一把尺（护栏）

def test_family_search_delegates_to_the_single_definition():
    """族路径不许再自己写一遍公式 —— 两把尺量出来的"更好"没有意义。

    这条是**源码级**护栏，理由与既有的"基准只有一处定义"那条同源：
    只要允许第二份实现存在，它迟早与第一份分叉，而分叉之后两边都还"看起来对"。
    """
    from research import family_search as FS
    src = inspect.getsource(FS.overlay)
    assert "overlay_fn.overlay_from_equity" in src, \
        "family_search.overlay 必须委托给 research.overlay 的唯一定义"
    assert "w * sret" not in src, "公式被抄了第二份 —— 这就是两把尺"
    assert "365.25" not in src, "年化折算也被抄了第二份"


def test_a_window_reaching_before_the_pool_coverage_is_clamped_and_disclosed():
    """窗口伸到池子覆盖之外时，两条腿必须裁到**共同覆盖**的日子上比，并如实说明裁过。

    2026-10-19 边界实测：池子（reg 的等权指数）要预热，起得比面板晚（面板 2023-10-17、
    池子 2024-01-12）。此前 `overlay_from_equity` 会把池子收益 reindex 到权益的日期轴上、
    **缺失的日子补 0 收益**（"池子那天没有数据"被当成"那天没涨"），而窗口基准用的是池子自己的
    切片 ⇒ 一个**什么都不做**的窗口报出 **−15.09pp** 的"曝光匹配超额"（平台面板上最显眼的那个数）。

    修法是把权益裁到池子真有数据的日子上；并把比较区间与"是否裁过"放回结果里（不静默截断）。
    """
    import numpy as np
    import pandas as pd
    from research.overlay import OverlayError, excess_vs_pool

    eq_idx = pd.date_range("2024-01-01", periods=120, freq="B")
    eq = pd.DataFrame({"equity": 100000.0, "exposure": 0.0}, index=eq_idx)
    # 池子只有后半段有数据（模拟"池子起得晚"）
    pool_idx = eq_idx[60:]
    lvl = pd.Series(np.linspace(100.0, 130.0, len(pool_idx)), index=pool_idx)
    reg = lvl.rename("ew_index").reset_index().rename(columns={"index": "date"})

    out = excess_vs_pool(eq, reg, "2024-01-01", str(eq_idx[-1].date()))
    assert out["clamped"] is True, "窗口伸到池子覆盖之外时应当标明被裁过"
    assert out["compared_from"].startswith(str(pool_idx[0].date())), out["compared_from"]
    # 共同覆盖的日子 = 池子的日子（60 天），不是权益的 120 天
    assert out["n_days"] == len(pool_idx), out["n_days"]

    # 池子完全没有覆盖 ⇒ 拒绝，而不是拿一段空气去算年化
    early = excess_vs_pool.__wrapped__ if hasattr(excess_vs_pool, "__wrapped__") else None
    try:
        excess_vs_pool(eq.iloc[:30], reg, "2024-01-01", str(eq_idx[29].date()))
        raise AssertionError("共同覆盖为 0 天时应当拒绝")
    except OverlayError:
        pass
    assert early is None or True
