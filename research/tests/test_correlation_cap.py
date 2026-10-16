"""组合内两两相关性上限（审计 D6）。

缺陷本体：组合层**没有任何相关性/行业约束** —— `max_positions` 数的是**名字**，
不是**赌注**。两只 60 日收益相关 0.95 的名字同时持有，经济上就是把单名风险加倍，
而引擎看不见这件事。

修法：入场时与**已在场**的每个标的比尾部相关（窗口右端取**信号日**），超限就不建。
为什么用数据驱动的相关性而不是行业分类：行业数据我们**没有**，而相关性可以从
已有价格里因果地算出来；而且真正决定风险的是相关性，不是 GICS 标签。
对齐必须按**日期**而不是下标（各标的日期覆盖不同，按下标配会把不同日期配成一对）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.broker import BrokerError, simulate, trade_fingerprint
from research.signals import ENTRY_COLS
from research.tests.test_broker import BASE, mk_panel, mk_signal

# ⚠️ position_fraction 必须 < 1，否则第一笔吃光现金、第二笔被**现金闸门**挡掉
#    （那不是相关性闸门在起作用）—— 实测踩过：加了相关性上限却看不到任何拦截，
#    因为第二笔压根没走到那一步。
CAP = {**BASE, "max_positions": 3, "position_fraction": 0.40,
       "sl_pct": 0.90, "tp_pct": 9.0, "max_hold": 500,
       "atr_slip_scale": 0.0, "cost_bps_roundtrip": 0.0}


def _two_symbol_bars(n: int = 200, mode: str = "same"):
    """两只标的的日线：same = 完全同向（相关 ≈ +1）；opposite = 反向（≈ −1）。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    bars = []
    for i, d in enumerate(dates):
        up = (1.0 if i % 2 else -1.0)
        bars.append({"date": d.strftime("%Y-%m-%d"), "symbol": "AAA",
                     "open": 100.0 + up, "high": 100.0 + up + 0.5,
                     "low": 100.0 + up - 0.5, "close": 100.0 + up})
        sgn = up if mode == "same" else -up
        bars.append({"date": d.strftime("%Y-%m-%d"), "symbol": "BBB",
                     "open": 50.0 + sgn, "high": 50.0 + sgn + 0.25,
                     "low": 50.0 + sgn - 0.25, "close": 50.0 + sgn})
    return bars


def _signals(day: str = "2024-04-01"):    # 必须在第 60 根之后：相关性要 60 个观测
    a = mk_signal(day, "AAA", score=2.0)
    b = mk_signal(day, "BBB", score=1.0)
    return pd.concat([a, b], ignore_index=True)[ENTRY_COLS]


def _run(preset: dict, mode: str = "same"):
    return simulate(preset, _signals(), mk_panel(_two_symbol_bars(mode=mode)),
                    start="2024-01-01", end="2025-02-01", cost_bps_roundtrip=None)


# ---------------------------------------------------------------- 闸门生效

def test_highly_correlated_pair_is_blocked():
    r = _run({**CAP, "max_pair_corr": 0.80})
    tr = r["trades"]
    assert len(tr) == 1, "同向的两只名字不该同时持有"
    assert tr["symbol"].iloc[0] == "AAA", "先建的（score 高）应当保留"
    assert r["metrics"]["n_blocked_by_corr"] == 1


def test_uncorrelated_pair_is_not_blocked():
    r = _run({**CAP, "max_pair_corr": 0.80}, mode="opposite")
    assert len(r["trades"]) == 2, "反向（相关 ≈ −1）不是同向下注，不该被挡"
    assert r["metrics"]["n_blocked_by_corr"] == 0


def test_off_by_default_is_bit_identical():
    a, b = _run(CAP), _run({**CAP, "max_pair_corr": None})
    assert trade_fingerprint(a["trades"]) == trade_fingerprint(b["trades"])
    assert a["metrics"]["n_blocked_by_corr"] == 0


def test_bad_threshold_fails_fast():
    with pytest.raises(BrokerError, match="max_pair_corr"):
        _run({**CAP, "max_pair_corr": 1.5})


# ---------------------------------------------------------------- 因果与退化

def test_correlation_uses_only_data_up_to_the_signal_day():
    """信号日**之后**的行情不得影响判定 —— 窗口右端取信号日。

    构造：把信号日之后 BBB 的走势整体反转（让它从同向变反向）。
    若实现偷看了未来，这条闸门就会放开；只看 ≤ 信号日则结论不变。
    """
    base = mk_panel(_two_symbol_bars(mode="same"))
    flipped = mk_panel(_two_symbol_bars(mode="same"))
    cut = pd.Timestamp("2024-04-01")
    m = (flipped["symbol"] == "BBB") & (flipped["date"] > cut)
    # 镜像翻转要**交换 high/low**：直接对 high/low 各自取镜像会把区间倒过来
    # （low > high），而引擎会（正确地）拒绝在不自洽的 bar 上成交 —— 实测踩过。
    for hc, lc in (("high", "low"), ("raw_high", "raw_low")):
        hi_old = flipped.loc[m, hc].to_numpy().copy()
        lo_old = flipped.loc[m, lc].to_numpy().copy()
        flipped.loc[m, hc] = 100.0 - (lo_old - 48.0)      # 旧的最低点变成新的最高点
        flipped.loc[m, lc] = 100.0 - (hi_old - 48.0)
    for c in ("open", "close", "raw_open", "raw_close"):
        flipped.loc[m, c] = 100.0 - (flipped.loc[m, c] - 48.0)
    r1 = simulate({**CAP, "max_pair_corr": 0.80}, _signals(), base,
                  start="2024-01-01", end="2025-02-01", cost_bps_roundtrip=None)
    r2 = simulate({**CAP, "max_pair_corr": 0.80}, _signals(), flipped,
                  start="2024-01-01", end="2025-02-01", cost_bps_roundtrip=None)
    assert r1["metrics"]["n_blocked_by_corr"] == r2["metrics"]["n_blocked_by_corr"] == 1


def test_insufficient_overlap_does_not_block():
    """共同观测不足 ⇒ 相关不可判定 ⇒ **不挡**（并留给别的约束去管）。

    这里让 BBB 只有很少的 bar 与 AAA 重叠。
    """
    dates = pd.bdate_range("2024-01-01", periods=200)
    bars = [{"date": d.strftime("%Y-%m-%d"), "symbol": "AAA"} for d in dates]
    bars += [{"date": d.strftime("%Y-%m-%d"), "symbol": "BBB"} for d in dates[-10:]]
    r = simulate({**CAP, "max_pair_corr": 0.80}, _signals(),
                 mk_panel(bars), start="2024-01-01", end="2025-02-01",
                 cost_bps_roundtrip=None)
    assert r["metrics"]["n_blocked_by_corr"] == 0, \
        "不可判定就不判定 —— 不许拿不足的样本算一个相关系数出来"
