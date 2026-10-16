"""止损穿价深度（审计 E2）—— 把"按触发价成交"这条假定的代价**量出来**。

缺陷本体：盘中触发止损时成交价恒为 `sl_price`（再叠 ATR 滑点），与当日 bar 实际
往下穿过多少**无关**。方向一律乐观，而且**不等**：各变体的止损占比不同 ⇒ 偏差不等，
足以扭曲以 E[R] 为判据的变体比较。

⚠️ 修法**不是**把成交价改成当日最低价 —— 那同样是错的：止损市价单在**首次触及**
那一刻成交，最低点常常出现在触发之后（审计的对抗验证员推翻了第一版提议）。
所以这里做两件事，都不引入未标定的假设：
  1. **把穿价深度记下来**（`sl_penetration`），并在指标里给中位/p90 ⇒ 偏差有多大
     是**读数**，不是信念；
  2. 给一个**显式声明**的敏感性系数 `sl_overshoot`（默认 0 = 既有行为），
     θ=1 只作为"按最低价成交"的悲观端点，不作为推荐值。

实测（真数据、12m preset、24 个月窗）：穿价中位 **101.6bp**、p90 **175.6bp**；
θ 从 0 到 1 让总收益从 −9.10% 走到 −12.68% ⇒ **这条假定值 1~3.5pp**，不是四舍五入。
"""
from __future__ import annotations

import numpy as np
import pytest

from research.broker import BrokerError, simulate, trade_fingerprint
from research.signals import ENTRY_COLS
from research.tests.test_broker import BASE, mk_panel, mk_signal

# 止损宽 5%：入场 ~100 ⇒ 止损 95；把某一天的 low 压到 90 ⇒ 穿价 ≈ 5.3%
DEEP = {**BASE, "sl_pct": 0.05, "tp_pct": 0.60, "max_hold": 60,
        "atr_slip_scale": 0.0, "cost_bps_roundtrip": 0.0}

# 第 1 天信号 ⇒ 第 2 天开盘入场；第 4 天砸穿止损
CRASH_DAY = 4


def crash_bars(n: int = 30, low_at_crash: float = 90.0) -> list[dict]:
    bars = []
    for i, d in enumerate(__import__("pandas").bdate_range("2024-01-01", periods=n)):
        b = {"date": d.strftime("%Y-%m-%d")}
        if i == CRASH_DAY:
            b.update({"open": 100.0, "high": 100.5, "low": low_at_crash, "close": 92.0})
        bars.append(b)
    return bars


def run(preset: dict, low: float = 90.0) -> dict:
    return simulate(preset, mk_signal("2024-01-01"), mk_panel(crash_bars(30, low)),
                    start="2024-01-01", end="2024-03-01", cost_bps_roundtrip=None)


# ---------------------------------------------------------------- 记录与度量

def test_penetration_is_recorded_on_an_intrabar_stop():
    tr = run(DEEP)["trades"]
    t = tr.iloc[0]
    assert t["exit_reason"] == "sl"
    sl_price = t["entry_price"] * (1 - 0.05)
    want = (sl_price - 90.0) / sl_price
    assert t["sl_penetration"] == pytest.approx(want, rel=1e-9)
    assert t["sl_penetration"] > 0.05, "这条用例必须真的有穿价，否则断言是空的"


def test_penetration_is_nan_when_it_does_not_apply():
    """止盈/跳空/到期都谈不上"盘中止损穿价"—— 不许拿 0 冒充。"""
    tp = {**BASE, "sl_pct": 0.90, "tp_pct": 0.01, "max_hold": 5,
          "atr_slip_scale": 0.0, "cost_bps_roundtrip": 0.0}
    tr = run(tp)["trades"]
    assert (tr["exit_reason"] != "sl").all()
    assert tr["sl_penetration"].isna().all()


def test_metrics_report_the_penetration_distribution():
    m = run(DEEP)["metrics"]
    assert m["n_sl"] == 1
    assert m["median_sl_penetration_bps"] > 300, "5% 止损、low=90 ⇒ 穿价应在数百 bp 量级"
    assert m["median_sl_penetration_bps"] == pytest.approx(m["p90_sl_penetration_bps"])


# ---------------------------------------------------------------- 系数语义

def test_theta_zero_preserves_behaviour_bit_for_bit():
    a = run(DEEP)
    b = run({**DEEP, "sl_overshoot": 0.0})
    assert trade_fingerprint(a["trades"]) == trade_fingerprint(b["trades"])


def test_theta_moves_the_fill_toward_the_bar_low_monotonically():
    prices, pnls = [], []
    for th in (0.0, 0.25, 0.5, 1.0):
        t = run({**DEEP, "sl_overshoot": th})["trades"].iloc[0]
        prices.append(t["exit_price"])
        pnls.append(t["net_pnl"])
    assert prices == sorted(prices, reverse=True), f"θ 越大成交价应越低：{prices}"
    # 净盈亏随 θ 变大而变**更差**（更负）⇒ 是降序，不是升序
    assert pnls == sorted(pnls, reverse=True), f"θ 越大净盈亏应越差：{pnls}"
    # θ=1 的触发价就是当日最低价（文档里写明这是**悲观端点**，不是推荐值）
    hi = run({**DEEP, "sl_overshoot": 1.0})["trades"].iloc[0]
    assert hi["exit_trigger"] == pytest.approx(90.0, rel=1e-9)


def test_the_knob_is_not_inert():
    """θ 真的会改变结论的量级 —— 这条保证上面的"记录"不是自我安慰。"""
    a = run(DEEP)["trades"].iloc[0]["net_pnl"]
    b = run({**DEEP, "sl_overshoot": 0.5})["trades"].iloc[0]["net_pnl"]
    risk = run(DEEP)["trades"].iloc[0]["risk_amount"]
    assert abs(b - a) / risk > 0.25, f"θ=0.5 只改变了 {abs(b-a)/risk:.2f}R —— 与实测不符"


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_invalid_overshoot_fails_fast(bad):
    with pytest.raises(BrokerError, match="sl_overshoot"):
        run({**DEEP, "sl_overshoot": bad})


def test_shallow_penetration_is_small():
    """穿价浅（low 只低一点点）⇒ 读数就小。这条排除"记录的是另一个量"的可能。"""
    # 止损在 95：low=94.5 ⇒ 只穿 0.53%（浅）；low 必须**低于**止损才叫穿价
    t = run(DEEP, low=94.5)["trades"].iloc[0]
    assert t["exit_reason"] == "sl"
    assert 0 < t["sl_penetration"] < 0.06
