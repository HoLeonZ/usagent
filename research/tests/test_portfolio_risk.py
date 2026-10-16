"""组合层风险预算（审计 D1/D2/D3）—— 机构风控的核心，此前**完全不存在**。

修之前，引擎的组合层只有"数仓位个数"（`max_positions`）：没有总风险（heat）上限、
没有单名权重上限、没有回撤停机；而唯一的尺寸规则还会被现金闸门**静默改写**
（实测 12m preset 的第二仓从声明 5% 权益掉到 ~0.6%，报告里看不出来）。

本文件钉住四件事：
  1. 单名权重上限真的约束名义额；
  2. heat（Σ 计划风险）上限真的约束同日第二仓，且**观测值**从不超过上限；
  3. 回撤停机真的挡住后续入场，且"停机天数"可见；
  4. **声明风险 vs 实际风险**逐笔可查（`sizing_cap` / `risk_frac`），
     并且不设上限时行为**逐位不变**（新机制不得偷偷改掉既有口径）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from research.broker import BrokerError, trade_fingerprint
from research.signals import ENTRY_COLS
from research.tests.test_broker import BASE, flat_days, mk_panel, mk_signal

FLAT = 100.0


def two_symbol_days(n: int, start: str = "2024-01-01") -> list[dict]:
    """同一天两只票的平盘 bar（用来构造"同日多个信号"）。"""
    out = []
    for d in pd.bdate_range(start, periods=n):
        for sym in ("AAA", "BBB"):
            out.append({"date": d.strftime("%Y-%m-%d"), "symbol": sym})
    return out


def two_signals(day: str) -> pd.DataFrame:
    a = mk_signal(day, "AAA", score=2.0)
    b = mk_signal(day, "BBB", score=1.0)
    return pd.concat([a, b], ignore_index=True)[ENTRY_COLS]


# ---------------------------------------------------------------- 单名权重上限

def test_max_weight_caps_single_name_notional():
    panel = mk_panel(flat_days(25))
    r = simulate_ok({**BASE, "max_weight": 0.25}, mk_signal("2024-01-01"), panel)
    tr = r["trades"]
    assert len(tr) == 1
    eq0 = BASE["initial_cash"]
    assert tr["entry_price"].iloc[0] * tr["size"].iloc[0] <= 0.25 * eq0 * 1.001
    assert tr["sizing_cap"].iloc[0] == "max_weight"
    assert r["metrics"]["peak_name_weight"] <= 0.26


def test_without_max_weight_the_same_name_takes_the_whole_book():
    """反证：不设上限时 all-in 会把几乎全部权益压在一只票上（这就是要修的东西）。"""
    panel = mk_panel(flat_days(25))
    r = simulate_ok(BASE, mk_signal("2024-01-01"), panel)
    assert r["metrics"]["peak_name_weight"] > 0.95


# ---------------------------------------------------------------- heat 上限

WIDE = {**BASE, "sl_pct": 0.20, "tp_pct": 0.60, "risk_per_trade": 0.05,
        "max_positions": 3, "position_fraction": 1.0}


def test_max_heat_caps_the_second_same_day_position():
    panel = mk_panel(two_symbol_days(25))
    r = simulate_ok({**WIDE, "max_heat": 0.06}, two_signals("2024-01-01"), panel)
    tr = r["trades"]
    assert len(tr) == 2, "两个信号都该成交，但第二个被 heat 额度压缩"
    assert set(tr["sizing_cap"]) == {"policy", "max_heat"}
    # 计划风险之和不得超过上限（允许 1 股的向下取整误差）
    total_risk = (tr["sl_width"] * tr["entry_price"] * tr["size"]).sum()
    assert total_risk <= 0.06 * BASE["initial_cash"] + 1e-6
    assert r["metrics"]["peak_heat"] <= 0.061


def test_without_max_heat_two_positions_double_the_heat():
    panel = mk_panel(two_symbol_days(25))
    r = simulate_ok(WIDE, two_signals("2024-01-01"), panel)
    assert r["metrics"]["peak_heat"] > 0.09, "两仓各 5% ⇒ 应看到 ~10% 的 heat"


def test_heat_room_shrinks_within_the_same_day():
    """heat 额度必须**当日递减**：第二个信号看到的剩余额度已经扣掉第一仓。"""
    panel = mk_panel(two_symbol_days(25))
    r = simulate_ok({**WIDE, "max_heat": 0.06}, two_signals("2024-01-01"), panel)
    by_sym = r["trades"].set_index("symbol")
    risk = lambda x: x["sl_width"] * x["entry_price"] * x["size"]  # noqa: E731
    # 同日按 score 降序抢额度：AAA(score 2.0) 拿到完整政策仓位，
    # BBB(score 1.0) 看到的是扣掉 AAA 之后的剩余额度 ⇒ 明显更小。
    assert by_sym.loc["AAA", "sizing_cap"] == "policy"
    assert by_sym.loc["BBB", "sizing_cap"] == "max_heat"
    assert risk(by_sym.loc["BBB"]) < risk(by_sym.loc["AAA"])


# ---------------------------------------------------------------- 回撤停机

def _crash_panel(n: int = 40) -> list[dict]:
    """第 2 天开盘入场(=第 1 天收盘后出信号)，第 3 天暴跌触发止损。"""
    bars = flat_days(n)
    for i, b in enumerate(bars):
        if i >= 3:
            b.update({"open": 88.0, "high": 90.0, "low": 85.0, "close": 86.0})
    return bars


def test_drawdown_halt_blocks_later_entries():
    panel = mk_panel(_crash_panel(40))
    sig = pd.concat([mk_signal("2024-01-01"), mk_signal("2024-01-08")],
                    ignore_index=True)[ENTRY_COLS]
    r = simulate_ok({**BASE, "sl_pct": 0.05, "tp_pct": 0.30,
                     "max_drawdown_halt": 0.04}, sig, panel)
    assert len(r["trades"]) == 1, "停机之后不得再开新仓"
    assert r["trades"]["exit_reason"].iloc[0].startswith("sl")
    assert r["metrics"]["pct_days_halted"] > 0.5


def test_without_halt_the_second_signal_still_trades():
    panel = mk_panel(_crash_panel(40))
    sig = pd.concat([mk_signal("2024-01-01"), mk_signal("2024-01-08")],
                    ignore_index=True)[ENTRY_COLS]
    r = simulate_ok(BASE | {"sl_pct": 0.05, "tp_pct": 0.30}, sig, panel)
    assert len(r["trades"]) >= 2, "不设停机时后续信号照常入场（反证上一条是停机生效）"


# ---------------------------------------------------------------- 声明风险 vs 实际风险（D3）

def test_cash_gate_is_recorded_not_silent():
    """风险预算买不起 ⇒ 现金闸门截断，且**记下来**。

    这就是审计 D3 指出的现象：声明 risk_per_trade=5%，实际只冒了 ~2% 的险，
    而此前这条改写无日志、无字段、报告里看不出来。
    """
    panel = mk_panel(flat_days(25))
    r = simulate_ok({**BASE, "risk_per_trade": 0.05, "sl_pct": 0.02},  # 窄止损 ⇒ 预算买不起
                    mk_signal("2024-01-01"), panel)
    tr = r["trades"]
    assert tr["sizing_cap"].iloc[0] == "cash"
    assert 0 < tr["risk_frac"].iloc[0] < 0.05, "实际风险小于声明风险 —— 这正是要可见的那件事"
    assert r["metrics"]["capped_share"] == 1.0
    assert r["metrics"]["max_risk_frac"] < 0.05


def test_uncapped_sizing_is_labelled_policy():
    panel = mk_panel(flat_days(25))
    r = simulate_ok({**BASE, "risk_per_trade": 0.01, "sl_pct": 0.02, "atr_sl_mult": None},
                    mk_signal("2024-01-01"), panel)
    tr = r["trades"]
    assert tr["sizing_cap"].iloc[0] == "policy"
    assert tr["risk_frac"].iloc[0] == pytest.approx(0.01, rel=0.02)
    assert r["metrics"]["capped_share"] == 0.0


# ---------------------------------------------------------------- 行为保持与校验

def test_new_keys_absent_or_none_are_identical_to_baseline():
    """不设上限 ⇒ 交易列表逐位不变（新机制不得偷偷改掉既有口径）。"""
    panel = mk_panel(flat_days(25))
    sig = mk_signal("2024-01-01")
    a = simulate_ok(BASE, sig, panel)
    b = simulate_ok({**BASE, "max_weight": None, "max_heat": None,
                     "max_drawdown_halt": None}, sig, panel)
    c = simulate_ok({**BASE, "max_weight": 1.0, "max_heat": 1.0,
                     "max_drawdown_halt": 0.99}, sig, panel)
    fa, fb, fc = (trade_fingerprint(r["trades"]) for r in (a, b, c))
    assert fa == fb, "显式写 None 与不写必须等价"
    assert fa == fc, "上限宽松到不咬合时，交易列表必须逐位不变"


@pytest.mark.parametrize("bad", [
    {"max_weight": 0.0}, {"max_weight": 1.5}, {"max_heat": -0.1}, {"max_heat": 1.01},
    {"max_drawdown_halt": 0.0}, {"max_drawdown_halt": 1.0},
])
def test_invalid_limits_fail_fast(bad):
    panel = mk_panel(flat_days(25))
    with pytest.raises(BrokerError):
        simulate_ok({**BASE, **bad}, mk_signal("2024-01-01"), panel)


def simulate_ok(preset: dict, signals: pd.DataFrame, panel: pd.DataFrame) -> dict:
    from research.broker import simulate
    return simulate(preset, signals, panel, start="2024-01-01", end="2024-06-30",
                    cost_bps_roundtrip=None)
