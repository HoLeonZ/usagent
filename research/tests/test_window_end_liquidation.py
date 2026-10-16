"""窗口末尾清算（审计 C5）—— 头条收益与 E[R] 必须落在**同一个样本**上。

缺陷本体：`simulate` 在窗口末尾不做任何清算，于是
  * 头条收益 / Sharpe / maxDD 由权益曲线算出（**含未实现浮盈**），
  * 而 E[R] / 胜率 / 盈亏比 / PF 只覆盖 trades 里的**已平仓成交**。
同一份报告里的两组数来自两个不同的样本；窗口末尾刚好浮盈时会系统性抬高头条
（risk=1% 口径下一次 +5R 的浮盈约等于 +5% 权益，足以主导一个三个月窗）。
`EXIT_REASONS` 里那个 `eod` 此前**没有任何产生点** —— 只存在于枚举里。

钉住的不变量：
  1. `liquidate_at_end=True` ⇒ 末尾未平仓按最后一根收盘价成交（reason=eod），
     且 **末值权益 == 初始现金 + Σ 已实现净盈亏**（两组数同源的恒等式）；
  2. 该日为停牌/无 bar 的仓位**不得造假成交**（准则 §1）⇒ 留作未平仓，
     并把"样本未对齐"显式报出来；
  3. `liquidate_at_end=False` 时上述恒等式**不成立** —— 这条反证说明修的是真问题。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.broker import simulate
from research.signals import ENTRY_COLS
from research.tests.test_broker import BASE, mk_panel, mk_signal

# 永不触发的止损/止盈 + 超过窗口的持有期 ⇒ 仓位必然活到窗口末尾
OPEN_FOREVER = {**BASE, "sl_pct": 0.90, "tp_pct": 9.00, "max_hold": 500,
                "atr_slip_scale": 0.0, "cost_bps_roundtrip": 5.0}


def flat(n: int = 30) -> pd.DataFrame:
    return mk_panel([{"date": d.strftime("%Y-%m-%d")}
                     for d in pd.bdate_range("2024-01-01", periods=n)])


def run(preset: dict, panel: pd.DataFrame, day: str = "2024-01-01") -> dict:
    return simulate(preset, mk_signal(day), panel, start="2024-01-01",
                    end="2024-02-15", cost_bps_roundtrip=None)


# ---------------------------------------------------------------- 清算发生

def test_open_position_is_liquidated_at_the_last_close():
    panel = flat(30)
    r = run(OPEN_FOREVER, panel)
    tr = r["trades"]
    assert len(tr) == 1
    assert tr["exit_reason"].iloc[0] == "eod"
    assert tr["exit_date"].iloc[0] == panel["date"].max()
    assert r["metrics"]["samples_aligned"] is True
    assert r["metrics"]["n_unliquidated"] == 0


def test_eod_exit_charges_costs():
    """清算按真实成交处理：平盘价 + 非零成本 ⇒ 净盈亏为负（不是"免费平掉"）。"""
    r = run(OPEN_FOREVER, flat(30))
    assert r["trades"]["net_pnl"].iloc[0] < 0
    assert r["trades"]["fees"].iloc[0] > 0


# ---------------------------------------------------------------- 同源恒等式

def test_final_equity_equals_initial_plus_realized_pnl():
    """**核心不变量**：末尾权益 == 初始现金 + Σ 已实现净盈亏。

    只要还有未平仓，这个恒等式就不成立 —— 它正是"两组数是否同源"的判据。
    """
    r = run(OPEN_FOREVER, flat(30))
    final = float(r["equity"]["equity"].iloc[-1])
    realized = float(r["trades"]["net_pnl"].sum())
    assert final == pytest.approx(BASE["initial_cash"] + realized, rel=1e-9)


def test_without_liquidation_the_two_samples_diverge():
    """反证：关掉清算 ⇒ 恒等式不成立、末尾仍有仓位。"""
    r = run({**OPEN_FOREVER, "liquidate_at_end": False}, flat(30))
    assert len(r["trades"]) == 0, "未平仓 ⇒ 交易口径里什么都没有"
    assert r["equity"]["n_pos"].iloc[-1] == 1
    assert r["metrics"]["samples_aligned"] is False
    final = float(r["equity"]["equity"].iloc[-1])
    assert final != pytest.approx(BASE["initial_cash"], rel=1e-6), \
        "权益曲线含未实现部分，而交易口径为空 —— 这就是要修的分叉"


# ---------------------------------------------------------------- 停牌语义

def test_suspended_at_window_end_is_not_fabricated_into_a_fill():
    """末尾当日无 bar ⇒ 不造成交（准则 §1），留作未平仓并**显式报告**未对齐。"""
    # 需要**两只**标的：BBB 一直有 bar（撑住全局交易日轴），AAA 在末尾前 5 天断掉。
    # 只留一只标的的话，轴的最后一天就是它自己的最后一天，"末尾无 bar"根本构造不出来。
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=30)]
    bars = [{"date": d, "symbol": "BBB"} for d in dates]
    bars += [{"date": d, "symbol": "AAA"} for d in dates[:-5]]   # AAA 末尾 5 天无 bar
    panel = mk_panel(bars)
    r = simulate(OPEN_FOREVER, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-15", cost_bps_roundtrip=None)
    assert len(r["trades"]) == 0, "停牌不得产生成交"
    assert r["metrics"]["n_unliquidated"] == 1
    assert r["metrics"]["samples_aligned"] is False, "未对齐必须被报出来，不能沉默"
    # 权益仍按最近一次已知收盘 mark（不停工、不造 bar）
    assert r["equity"]["equity"].iloc[-1] > 0
    # 入场之后一直持有到窗口末尾（第 0 天还没建仓，所以只从第 1 天起断言）
    assert r["equity"]["n_pos"].iloc[1:].eq(1).all()
    assert r["equity"]["n_pos"].iloc[-1] == 1


def test_no_open_positions_means_aligned_by_construction():
    """仓位在窗口内已被止损/止盈清掉 ⇒ 本来就同源，清算机制不得多插一笔成交。"""
    r = run({**BASE, "sl_pct": 0.02, "tp_pct": 0.02}, flat(30))
    assert "eod" not in set(r["trades"]["exit_reason"])
    assert r["metrics"]["samples_aligned"] is True


# ---------------------------------------------------------------- 与 E[R] 的一致性

def test_e_r_sees_the_end_of_window_trade():
    """清算出来的那笔必须进 R 分布 —— 否则"同源"只做了一半。"""
    from research import metrics as M
    r = run(OPEN_FOREVER, flat(30))
    st = M.tail_stats(r["trades"])
    assert st["n"] == 1
    assert np.isfinite(st["expectancy_r"])


# ---------------------------------------------------------------- 陈旧重估读数（A3）

def test_stale_position_days_are_counted_mid_holding():
    """持仓期内出现无 bar 的日子（停牌）⇒ 那些日子权益是按**陈价**重估的，必须可读。

    审计 A3：此前这件事没有任何读数，于是「这只票到底停了多久」只能靠人肉看数据。
    """
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=30)]
    keep = [d for i, d in enumerate(dates) if not (5 <= i <= 9)]      # AAA 中间停 5 天
    bars = [{"date": d, "symbol": "AAA"} for d in keep]
    bars += [{"date": d, "symbol": "BBB"} for d in dates]
    r = simulate(OPEN_FOREVER, mk_signal("2024-01-01"), mk_panel(bars),
                 start="2024-01-01", end="2024-02-15", cost_bps_roundtrip=None)
    m = r["metrics"]
    assert m["stale_position_days"] == 5, "停牌 5 个交易日应被记成 5 个持仓-日"
    assert 0 < m["pct_position_days_stale"] < 0.3
    assert (r["equity"]["equity"] > 0).all(), "陈价重估不得产生非正权益"


def test_no_stale_days_when_every_bar_is_present():
    r = run(OPEN_FOREVER, flat(30))
    assert r["metrics"]["stale_position_days"] == 0
    assert r["metrics"]["pct_position_days_stale"] == 0.0


# ---------------------------------------------------------------- 双报 Sharpe 口径（B7）

def _oscillating_bars(n: int = 30) -> list[dict]:
    """价格上下摆动的**自洽** bar。

    ⚠️ 只给 close 而让 high/low 取默认值会造出 `close ∉ [low, high]` 的不自洽 bar，
    引擎会（正确地）拒绝在上面成交 —— 实测就是这么被 BrokerError 拦住的，
    也正是数据层那道 OHLC 闸门存在的理由。
    """
    out = []
    for i, d in enumerate(pd.bdate_range("2024-01-01", periods=n)):
        c = 100.0 + (1.0 if i % 2 else -1.0)
        out.append({"date": d.strftime("%Y-%m-%d"), "open": c,
                    "high": c + 0.5, "low": c - 0.5, "close": c})
    return out


def _half_invested_run() -> dict:
    """前半段在场（价格在动）、后半段空仓（收益恒为 0）的一次回测。"""
    bars = _oscillating_bars()
    return simulate({**BASE, "sl_pct": 0.90, "tp_pct": 9.0, "max_hold": 12},
                    mk_signal("2024-01-01"), mk_panel(bars),
                    start="2024-01-01", end="2024-03-01", cost_bps_roundtrip=None)


def test_flat_days_change_the_sharpe_convention():
    """低敞口策略上「含空仓日」与「只在场日」会差很多 —— 只报一个会被读成另一个。

    审计 B7：DSR 用的那条收益序列把空仓日（收益恰为 0）算了进去。对"作为一个账户"
    这是对的，但它与"信号在它真在场时有多好"是两个问题，必须并排可见。
    """
    m = _half_invested_run()["metrics"]
    assert 0 < m["n_days_invested"] < m["days"], "这条用例必须真的有空仓日"
    assert np.isfinite(m["sharpe"]) and np.isfinite(m["sharpe_invested"])
    assert abs(m["sharpe"] - m["sharpe_invested"]) > 1e-6, \
        f"有空仓日时两个口径必须给出不同的数：{m['sharpe']} vs {m['sharpe_invested']}"


def test_fully_invested_conventions_nearly_coincide():
    """基本上一直满仓 ⇒ 两个口径应当几乎相等（反证上一条不是"随手加了个字段"）。

    留一点容差：入场日与窗口首日这两个边界上必然有"有权益但零敞口"的日子。
    """
    m = simulate(OPEN_FOREVER, mk_signal("2024-01-01"), mk_panel(_oscillating_bars()),
                 start="2024-01-01", end="2024-03-01", cost_bps_roundtrip=None)["metrics"]
    assert m["n_days_invested"] >= m["days"] - 2
    # 用**绝对差**而不是相对差：策略的日均收益接近 0 时，相对差会被小分母放大，
    # 而两个口径实际只差一个"边界空仓日"。
    assert abs(m["sharpe"] - m["sharpe_invested"]) < 1.0, \
        f"基本满仓时两个口径不该差这么多：{m['sharpe']} vs {m['sharpe_invested']}"


def test_dsr_is_fed_the_all_days_series():
    """DSR 必须吃**含空仓日**那条序列（策略作为账户的收益流），并在报告里说明。"""
    src = (Path(__file__).resolve().parents[2] / "research" / "run_baseline.py").read_text()
    assert "deflated_sharpe_from_returns(\n        sret" in src, \
        "DSR 的输入变了 —— 必须显式用 sret（含空仓日）并同步改注释"
    assert "含空仓日" in src and "只在场日" in src, "报告必须并排说明两个口径"
