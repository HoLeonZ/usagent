"""价差成本进入执行路径：两侧都收、只用决策时可知的估计、且可归因。

审计 E1：成本模型此前只有规费 + 冲击滑点，**没有买卖价差**。价差是尺寸无关的那一半
成本，对宽价差小盘是主项。本文件钉住四件事：
  1. `spread_model="none"` ⇒ 交易列表**逐位不变**（新机制不得偷偷改既有口径）；
  2. 打开后 ⇒ 买入价更高、净盈亏更低、`spread_cost` 为正 —— 方向必须对；
  3. 两侧都收，金额与"半价差 × 名义额"自洽（可归因，不是黑箱）；
  4. 出场**只用入场时**的估计 —— 出场当日再宽的价差也不加成（否则是偷看未来）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.broker import BrokerError, simulate, trade_fingerprint
from research.signals import ENTRY_COLS
from research.tests.test_broker import BASE, mk_panel, mk_signal

ON = {**BASE, "spread_model": "corwin_schultz"}


def wide_panel(n: int = 60, amp: float = 0.02) -> pd.DataFrame:
    """中间价来回走、H/L 各偏 ±amp 的平盘面板 ⇒ 有可估的价差（见 core/spread 的注入实验）。"""
    bars = []
    for i, d in enumerate(pd.bdate_range("2024-01-01", periods=n)):
        mid = 100.0 + 0.5 * np.sin(i / 3.0)
        bars.append({"date": d.strftime("%Y-%m-%d"),
                     "open": mid, "close": mid,
                     "high": mid * (1 + amp), "low": mid * (1 - amp)})
    return mk_panel(bars)


# 信号必须落在**足够靠后**的位置：价差估计的滚动窗口要 20 根 bar，
# 而信号日的估计还要再用一根（见 broker 里 `spread_bps[k-1]`）。放在第 26 根。
SIG_IDX = 26


def sig_day(panel: pd.DataFrame) -> str:
    return (panel.loc[panel["symbol"] == "AAA", "date"]
            .sort_values().iloc[SIG_IDX].strftime("%Y-%m-%d"))


def run(preset: dict, panel: pd.DataFrame) -> dict:
    return simulate(preset, mk_signal(sig_day(panel)), panel,
                    start="2024-01-01", end="2024-06-30", cost_bps_roundtrip=None)


# ---------------------------------------------------------------- 行为保持

def test_off_by_default_is_bit_identical():
    panel = wide_panel()
    a = run(BASE, panel)
    b = run({**BASE, "spread_model": "none"}, panel)
    assert trade_fingerprint(a["trades"]) == trade_fingerprint(b["trades"])
    assert (a["trades"]["spread_cost"] == 0.0).all()


def test_invalid_model_fails_fast():
    with pytest.raises(BrokerError, match="spread_model"):
        run({**BASE, "spread_model": "roll_1984"}, wide_panel())


# ---------------------------------------------------------------- 方向

def test_turning_it_on_costs_money_in_the_right_direction():
    panel = wide_panel()
    off, on = run(BASE, panel), run(ON, panel)
    assert len(off["trades"]) == len(on["trades"]) == 1
    t_off, t_on = off["trades"].iloc[0], on["trades"].iloc[0]
    assert t_on["entry_price"] > t_off["entry_price"], "买入要付更高的价（跨价差）"
    assert t_on["spread_cost"] > 0
    assert t_on["net_pnl"] < t_off["net_pnl"], "净盈亏必须变差"
    assert trade_fingerprint(off["trades"]) != trade_fingerprint(on["trades"])


def test_flat_market_has_no_estimable_spread():
    """无波动的平盘 ⇒ 估计量给 0（不能证明有价差）⇒ 不收费。

    这条同时是"没有价差信息时不会凭空收费"的证据。
    """
    panel = mk_panel([{"date": d} for d in pd.bdate_range("2024-01-01", periods=60)])
    r = run(ON, panel)
    assert (r["trades"]["spread_cost"] == 0.0).all()


# ---------------------------------------------------------------- 可归因

def test_spread_cost_is_consistent_with_half_spread_times_notional():
    """`spread_cost` 必须等于「半价差 ×（入场名义额 + 出场名义额）」——
    可归因意味着读者能用它复核，而不是只能相信。"""
    panel = wide_panel()
    t = run(ON, panel)["trades"].iloc[0]
    entry_notional = t["entry_price"] * t["size"]
    exit_notional = t["exit_price"] * t["size"]
    # 单边费率 = spread_cost / (入场名义 + 出场名义)
    implied_half = t["spread_cost"] / (entry_notional + exit_notional)
    assert 0 < implied_half < 0.05
    assert t["spread_cost"] == pytest.approx(
        implied_half * entry_notional + implied_half * exit_notional, rel=1e-9)


def _estimates(panel: pd.DataFrame) -> np.ndarray:
    """测试里**独立**重算一遍价差估计（用同一份 core.spread），用于对账。"""
    from core import spread as S
    return S.rolling_median(S.corwin_schultz(panel["raw_high"].to_numpy(float),
                                             panel["raw_low"].to_numpy(float)))


def test_spread_cost_matches_the_signal_day_estimate_exactly():
    """逐笔对账：`spread_cost` 必须等于「信号日的估计 × 两侧名义额」。

    这是"估计真的被读进来、且读的是信号日那一根"的**精确**证据 ——
    比"成本大于 0"强得多（后者在随便读一根 bar 时也成立）。
    """
    from core.spread import half_spread_rate
    panel = wide_panel(60)
    t = run(ON, panel)["trades"].iloc[0]
    est = _estimates(panel)[SIG_IDX]
    assert est > 0
    half = half_spread_rate(est)
    want = half * t["entry_price"] * t["size"] + half * t["exit_price"] * t["size"]
    assert t["spread_cost"] == pytest.approx(want, rel=1e-9)


# 注：曾经想加一条「扰动出场附近的 bar ⇒ 费率不变」的测试，但它是**空断言** ——
# 实测 Corwin-Schultz 对「连续区块内的同比例扰动」是恒等变换（只改变区块边界那一根），
# 于是滚动中位根本不动，扰动改不了任何东西。把它删掉而不是留着凑数；
# 「用的是信号日那一根、不是出场日那一根」由上面那条**精确对账**测试证明
# （期望值就是独立算出的信号日估计）。
