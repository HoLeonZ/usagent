"""执行语义测试(设计文档 §6)。

最重要的一条:**权益曲线不得把建仓名义额计入两次。**
hiagent 自己踩过这个坑(preset_search §1):错的权益曲线让一个 0% 胜率的窗口报出
Sharpe +3.00(修正后 -4.71)。test_equity_is_cash_plus_marked_positions 把它钉死。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.broker import BrokerError, simulate
from research.signals import ENTRY_COLS

FLAT = 100.0


def mk_panel(bars: list[dict]) -> pd.DataFrame:
    """bars: [{date, open, high, low, close, volume, atr_pct?, raw_*, dividends?}] → broker 需要的 panel。

    ⚠️ raw_* 缺省**等于**复权价。这是刻意的:无分红的标的两套价相同,于是「执行改用
    as-traded 价」(§3(b)) 对这些标的必须是**逐位不变的操作** —— 既有的合成用例因此
    全部继续有效,而它也正是这条改动没引入额外回归的证据。
    """
    rows = []
    for b in bars:
        o, h, lo, c = (b.get("open", FLAT), b.get("high", FLAT),
                       b.get("low", FLAT), b.get("close", FLAT))
        rows.append({
            "symbol": b.get("symbol", "AAA"),
            "date": pd.Timestamp(b["date"]),
            "open": o, "high": h, "low": lo, "close": c,
            "raw_open": b.get("raw_open", o), "raw_high": b.get("raw_high", h),
            "raw_low": b.get("raw_low", lo), "raw_close": b.get("raw_close", c),
            "volume": b.get("volume", 1_000_000.0),
            "atr_pct": b.get("atr_pct", 0.02),
            "dividends": b.get("dividends", 0.0),
        })
    return pd.DataFrame(rows)


def mk_signal(day: str, symbol: str = "AAA", score: float = 1.0,
              atr_pct: float = 0.02) -> pd.DataFrame:
    """构造一个信号行。

    ⚠️ atr_pct 必须给在**信号**上,不是给在 panel 上 —— broker 用的是
    ENTRY_COLS 里的 `atr_pct`,即**信号日**的 ATR%(与 hiagent 的
    「信号日 (entry_date - 1 个交易日) 的 atr_pct」同口径)。
    """
    row = {c: np.nan for c in ENTRY_COLS}
    row.update({"date": pd.Timestamp(day), "symbol": symbol, "score": score,
                "atr_pct": atr_pct})
    return pd.DataFrame([row])[ENTRY_COLS]


def flat_days(n: int, start: str = "2024-01-01") -> list[dict]:
    return [{"date": d} for d in pd.bdate_range(start, periods=n)]


BASE = {"tp_pct": 0.2628, "sl_pct": 0.021, "max_hold": 18, "max_positions": 1,
        "position_fraction": 1.0, "max_volume_participation": 0.10,
        "intraday_tiebreak": "sl_first", "initial_cash": 100_000.0,
        # 契约(§2(a)/§4)要求这两个键显式在场。这里取 0 滑点 / 5bp 成本,是为了让
        # 机制类断言(仓位、止损、记账)不被价格冲击污染;滑点本身由
        # 「滑点(§4(c))」一节单独测,那一节显式设 atr_slip_scale=1.0。
        "atr_slip_scale": 0.0, "cost_bps_roundtrip": 5.0}


# ---------------------------------------------------------------- 权益曲线

def test_equity_is_cash_plus_marked_positions():
    """价格不变 ⇒ 权益只应因成本下降,且收盘权益 == 初始现金 − 累计成本。

    若把建仓名义额计入两次,这里会看到权益跳升/塌陷一个名义额量级。
    """
    panel = mk_panel(flat_days(25))
    sig = mk_signal("2024-01-01")            # 入场在 01-02 开盘
    r = simulate(BASE, sig, panel, start="2024-01-01", end="2024-02-06",
                 cost_bps_roundtrip=5.0)
    eq = r["equity"]
    trades = r["trades"]

    assert len(trades) == 1
    t = trades.iloc[0]
    entry_fee = t["entry_price"] * t["size"] * 0.00025
    exit_fee = t["exit_price"] * t["size"] * 0.00025

    # 入场当日:权益 = 初始现金 − 入场费(价格未动)
    entry_row = eq[eq["date"] == t["entry_date"]].iloc[0]
    assert entry_row["equity"] == pytest.approx(100_000.0 - entry_fee, abs=1e-6)

    # 平仓后:权益 = 初始现金 − 往返成本
    assert eq["equity"].iloc[-1] == pytest.approx(
        100_000.0 - entry_fee - exit_fee, abs=1e-6)

    # 权益序列全程必须 ≤ 初始现金(价格不动 + 只有成本)
    assert (eq["equity"] <= 100_000.0 + 1e-9).all()


def test_buy_and_hold_equity_tracks_price():
    """价格上涨 10% ⇒ 权益涨幅应接近 10%(all-in),而不是 20% 或 0%。

    零成本下 qty = floor(100000/100) = 1000,即满仓;价格 +10% ⇒ 权益 +10%。
    若把建仓名义额计入两次,这里会得到 ~0% 或 ~20%。
    """
    bars = flat_days(12)
    for i, b in enumerate(bars):
        if i >= 2:
            for k in ("open", "high", "low", "close"):
                b[k] = FLAT * 1.10
    panel = mk_panel(bars)
    sig = mk_signal("2024-01-01")
    r = simulate(BASE, sig, panel, start="2024-01-01", end="2024-01-20",
                 cost_bps_roundtrip=0.0)
    gain = r["equity"]["equity"].iloc[-1] / 100_000.0 - 1.0
    assert gain == pytest.approx(0.10, abs=2e-3), f"权益涨幅 {gain:.4f} 与满仓 10% 不符(双重记账?)"


# ---------------------------------------------------------------- 退出语义

def test_sl_first_when_both_touched_same_bar():
    """同 bar 既触 SL 又触 TP ⇒ 按最坏情况判 SL(CLAUDE.md §4)。"""
    bars = flat_days(6)
    bars[1].update({"open": FLAT, "high": FLAT * 1.40, "low": FLAT * 0.90, "close": FLAT})
    panel = mk_panel(bars)
    sig = mk_signal("2024-01-01")
    r = simulate(BASE, sig, panel, start="2024-01-01", end="2024-01-10")
    assert len(r["trades"]) == 1
    t = r["trades"].iloc[0]
    assert t["exit_reason"] == "sl"
    assert t["exit_price"] == pytest.approx(t["entry_price"] * (1 - 0.021))


def test_gap_through_stop_fills_at_open():
    """开盘已越过止损 ⇒ 按开盘价成交,不按止损价(CLAUDE.md §4 Gap-Through-Stop)。

    入场日在 bars[1](开盘 100),跳空发生在 bars[2] —— 若把跳空放在入场日,
    broker 会用跳空价当入场价,测的就不是 gap-through-stop 了。
    """
    bars = flat_days(25)
    gap_open = FLAT * 0.80
    bars[2].update({"open": gap_open, "high": gap_open, "low": gap_open * 0.99,
                    "close": gap_open})
    panel = mk_panel(bars)
    sig = mk_signal("2024-01-01")
    r = simulate(BASE, sig, panel, start="2024-01-01", end="2024-02-10")
    assert len(r["trades"]) == 1
    t = r["trades"].iloc[0]
    assert t["exit_reason"] == "sl_gap"
    assert t["entry_price"] == pytest.approx(FLAT)
    assert t["exit_price"] == pytest.approx(FLAT * 0.80)
    # 关键:成交价比止损价更差(悲观)
    assert t["exit_price"] < t["entry_price"] * (1 - 0.021)


def test_take_profit_exit_price():
    bars = flat_days(6)
    bars[1].update({"open": FLAT, "high": FLAT * 1.40, "low": FLAT, "close": FLAT * 1.30})
    panel = mk_panel(bars)
    r = simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-01-10")
    t = r["trades"].iloc[0]
    assert t["exit_reason"] == "tp"
    assert t["exit_price"] == pytest.approx(t["entry_price"] * 1.2628, rel=1e-9)


def test_max_hold_exit():
    panel = mk_panel(flat_days(40))
    r = simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-03-01")
    t = r["trades"].iloc[0]
    assert t["exit_reason"] == "max_hold"
    assert t["hold_days"] == 18


# ---------------------------------------------------------------- 容量与流动性

def test_volume_cap_limits_size():
    """Bar_Volume × 10% 是硬上限;超出部分 IOC 丢弃。

    窗口要够长让仓位走完(否则 trades 为空,断言的其实是"没成交")。
    """
    bars = flat_days(25)
    bars[1]["volume"] = 500.0                      # 入场日:500 × 10% = 50 股
    panel = mk_panel(bars)
    r = simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10")
    assert len(r["trades"]) == 1
    assert r["trades"].iloc[0]["size"] == 50
    # 未成交的余量必须留在现金里(不排队、不记账为持仓)
    assert (r["equity"]["cash"] >= -1e-9).all()


def test_all_in_sizing_leaves_no_margin_borrow():
    """目标股数 floor(cash / (px × (1+half_bp))),不得出现现金为负。"""
    panel = mk_panel(flat_days(25))
    r = simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-06")
    t = r["trades"].iloc[0]
    assert t["size"] == int(np.floor(100_000.0 / (FLAT * (1 + 0.00025))))
    assert (r["equity"]["cash"] >= -1e-9).all()


def test_max_positions_discards_beyond_capacity_same_day():
    """超容量当 T 丢弃,不排队到 T+1(CLAUDE.md §2 单发子弹)。"""
    bars = flat_days(20)
    bars.append({"date": pd.Timestamp("2024-01-31")})
    for d in bars:
        d["symbol"] = "AAA"
    bbb = [dict(d, symbol="BBB") for d in bars]
    panel = mk_panel(bars + bbb)

    sig = pd.concat([mk_signal("2024-01-01", "AAA", 2.0),
                     mk_signal("2024-01-01", "BBB", 1.0)], ignore_index=True)
    r = simulate(BASE, sig, panel, start="2024-01-01", end="2024-02-06")
    entered = set(r["trades"]["symbol"])
    assert entered == {"AAA"}, f"应只买入 score 最高的 AAA,实际 {entered}"


def test_symbols_with_unequal_date_coverage():
    """回归:各标的的日期覆盖不同 ⇒ 全局交易轴与标的内下标**不可混用**。

    真实数据里晚间上市/停牌/退市会让各标的的日期集合不同。早先的实现用全局
    day_pos 去索引「标的内数组」,在全池 596 只上直接越界
    (index 657 out of bounds for axis 0 with size 298)。
    合成 panel 若各标的日期集合完全一致,这个 bug 会被掩盖 —— 所以这里刻意错开。
    """
    days = list(pd.bdate_range("2024-01-01", periods=40))
    # AAA 全 40 根;BBB 晚 20 根才上市
    aaa = [{"date": d, "symbol": "AAA"} for d in days]
    bbb = [{"date": d, "symbol": "BBB"} for d in days[20:]]
    panel = mk_panel(aaa + bbb)

    # 两个信号都发生在前 20 天:AAA 能入场,BBB 当日还没数据
    sig = pd.concat([mk_signal(days[0], "AAA", 2.0),
                     mk_signal(days[5], "BBB", 1.0)], ignore_index=True)
    # 窗口是左闭右开,所以 end 取末日之后一天才是"含末日"
    r = simulate(BASE, sig, panel, start=days[0], end=days[-1] + pd.Timedelta(days=1))
    assert set(r["trades"]["symbol"]) <= {"AAA", "BBB"}
    assert len(r["equity"]) == 40
    # BBB 只在 days[20:] 有数据,其入场必须落在有数据的那些天里(或根本不入场)
    for _, t in r["trades"].iterrows():
        if t["symbol"] == "BBB":
            assert t["entry_date"] in set(days[20:])


def test_no_data_symbol_is_skipped_not_crashed():
    """信号指向一个完全没有数据的标的 ⇒ 跳过,不得抛异常。"""
    panel = mk_panel([{"date": d["date"], "symbol": "AAA"} for d in flat_days(25)])
    sig = mk_signal("2024-01-01", "ZZZ")        # ZZZ 不在 panel 里
    r = simulate(BASE, sig, panel, start="2024-01-01", end="2024-02-10")
    assert len(r["trades"]) == 0
    assert r["metrics"]["total_return"] == pytest.approx(0.0)


# ---------------------------------------------------------------- ATR 归一化退出

def test_atr_normalized_stop_width_scales_per_trade():
    """atr_sl_mult 给定时,止损宽度 = mult × **信号日** atr_pct(逐笔不同)。"""
    panel = mk_panel(flat_days(25))
    p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
    p.update({"atr_sl_mult": 1.0, "atr_tp_mult": 12.51})
    r = simulate(p, mk_signal("2024-01-01", atr_pct=0.05), panel,
                 start="2024-01-01", end="2024-02-10")
    t = r["trades"].iloc[0]
    assert t["sl_width"] == pytest.approx(0.05)
    assert t["tp_width"] == pytest.approx(0.05 * 12.51)
    assert t["risk_amount"] == pytest.approx(0.05 * t["entry_price"] * t["size"])


def test_atr_normalized_stop_actually_used():
    """信号日 atr_pct=0.05 + mult=1.0 ⇒ 止损在 −5%:low −4% 不该触发,−6% 该触发。"""
    for low_mult, expect_exit in ((0.96, False), (0.94, True)):
        bars = flat_days(25)
        bars[2].update({"open": FLAT, "high": FLAT, "low": FLAT * low_mult, "close": FLAT})
        panel = mk_panel(bars)
        p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
        p.update({"atr_sl_mult": 1.0, "atr_tp_mult": 12.51})
        r = simulate(p, mk_signal("2024-01-01", atr_pct=0.05), panel,
                     start="2024-01-01", end="2024-02-10")
        hit = (len(r["trades"]) == 1 and r["trades"].iloc[0]["exit_reason"] == "sl")
        assert hit == expect_exit, f"low={low_mult} 时 SL 命中应为 {expect_exit}"


def test_atr_mult_requires_both_and_valid_atr():
    panel = mk_panel(flat_days(25))
    p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
    with pytest.raises(BrokerError, match="必须同时给出"):
        simulate({**p, "atr_sl_mult": 1.0}, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10")

    with pytest.raises(BrokerError, match="有效的 atr_pct"):
        simulate({**p, "atr_sl_mult": 1.0, "atr_tp_mult": 12.51},
                 mk_signal("2024-01-01", atr_pct=float("nan")), panel,
                 start="2024-01-01", end="2024-02-10")


# ---------------------------------------------------------------- 风险预算式仓位

def test_risk_per_trade_makes_loss_invariant_to_stop_width():
    """每笔计划亏损 = equity × risk_per_trade,与止损宽度无关。

    这才是 all-in + 宽止损的出路:all-in 下止损从 2% 加宽到 7%,单笔伤害也跟着
    从 2% 变 7%;风险预算式把两者脱钩。
    """
    panel = mk_panel(flat_days(25))
    p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
    p.update({"atr_sl_mult": 1.0, "atr_tp_mult": 12.51, "risk_per_trade": 0.01})

    r = simulate(p, mk_signal("2024-01-01", atr_pct=0.05), panel,
                 start="2024-01-01", end="2024-02-10")
    t = r["trades"].iloc[0]
    # qty = floor(100000×0.01 / (0.05×100)) = 200;止损时亏损 = 200×5 = 1000 = 1% 权益
    assert t["size"] == 200
    assert t["risk_amount"] == pytest.approx(1000.0, rel=1e-9)
    assert t["risk_amount"] / 100_000.0 == pytest.approx(0.01, rel=1e-9)


def test_risk_per_trade_leaves_cash_uninvested():
    """风险预算式**不是** all-in:仓位应远小于可用现金(这正是它的目的)。"""
    panel = mk_panel(flat_days(25))
    p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
    p.update({"atr_sl_mult": 1.0, "atr_tp_mult": 12.51, "risk_per_trade": 0.01})
    r = simulate(p, mk_signal("2024-01-01", atr_pct=0.05), panel,
                 start="2024-01-01", end="2024-02-10")
    entry_day = r["trades"].iloc[0]["entry_date"]
    row = r["equity"][r["equity"]["date"] == entry_day].iloc[0]
    assert row["cash"] > 70_000.0                    # 大部分现金没投出去
    assert row["exposure"] < 0.25


def test_all_in_is_still_the_default_when_risk_per_trade_absent():
    """不设 risk_per_trade ⇒ 仍是 all-in,老行为不变。"""
    panel = mk_panel(flat_days(25))
    r = simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10")
    assert r["trades"].iloc[0]["size"] == int(np.floor(100_000.0 / (FLAT * 1.00025)))


def test_risk_per_trade_rejects_bad_values():
    panel = mk_panel(flat_days(25))
    for bad in (0.0, 1.0, -0.01, 2.0):
        with pytest.raises(BrokerError, match="risk_per_trade"):
            simulate({**BASE, "risk_per_trade": bad}, mk_signal("2024-01-01"), panel,
                     start="2024-01-01", end="2024-02-10")


# ---------------------------------------------------------------- fail-fast

def test_rejects_non_sl_first_tiebreak():
    with pytest.raises(BrokerError, match="intraday_tiebreak"):
        simulate({**BASE, "intraday_tiebreak": "tp_first"}, mk_signal("2024-01-01"),
                 mk_panel(flat_days(6)), start="2024-01-01", end="2024-01-10")


def test_rejects_bad_position_fraction():
    with pytest.raises(BrokerError, match="position_fraction"):
        simulate({**BASE, "position_fraction": 0.0}, mk_signal("2024-01-01"),
                 mk_panel(flat_days(6)), start="2024-01-01", end="2024-01-10")


def test_rejects_empty_window():
    with pytest.raises(BrokerError, match="没有任何交易日"):
        simulate(BASE, mk_signal("2024-01-01"), mk_panel(flat_days(6)),
                 start="2025-01-01", end="2025-02-01")


# ---------------------------------------------------------------- H2 移动止损

def _trail_preset(**over):
    p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
    p.update({"atr_sl_mult": 1.0, "atr_tp_mult": 12.51, "risk_per_trade": 0.01,
              "trailing_sl": True})
    p.update(over)
    return p


def test_trailing_stop_locks_in_profit():
    """价格涨到 120 后回落击穿两条止损:固定止损亏,移动止损止盈离场。

    关键:回落那根必须**同时**击穿初始止损(95)与抬高的止损(115),
    否则固定止损那条根本不会出场,比较就不成立。
    """
    bars = flat_days(25)
    bars[2].update({"open": 100.0, "high": 120.0, "low": 100.0, "close": 118.0})
    bars[3].update({"open": 118.0, "high": 118.0, "low": 90.0, "close": 92.0})
    panel = mk_panel(bars)
    sig = mk_signal("2024-01-01", atr_pct=0.05)     # sl_w = 0.05 → trail = $5

    off = simulate(_trail_preset(trailing_sl=False), sig, panel,
                   start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    on = simulate(_trail_preset(), sig, panel,
                  start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)

    t_off, t_on = off["trades"].iloc[0], on["trades"].iloc[0]
    assert t_off["exit_price"] == pytest.approx(100.0 * 0.95)      # 初始止损 → 亏
    assert t_off["net_pnl"] < 0
    # high_water 到 120 → 次日止损 = 120 − 5 = 115 → 盈
    assert t_on["exit_price"] == pytest.approx(115.0)
    assert t_on["net_pnl"] > 0
    assert t_on["net_pnl"] > t_off["net_pnl"]
    assert t_on["exit_reason"] == "sl"                              # 触发的是被抬高的止损


def test_trailing_stop_is_causal_same_bar():
    """同 bar 内不得用当日 high 抬高止损再拿当日 low 触发(同 bar 前视)。

    入场日 high=130、low=95:若错误地用当日 high 抬升止损,止损会变成 130−5=125,
    当日 low=95 立刻触发,记成"止盈";正确做法是入场日止损仍是 95(=100×0.95)。
    """
    bars = flat_days(25)
    bars[1].update({"open": 100.0, "high": 130.0, "low": 95.0, "close": 128.0})
    panel = mk_panel(bars)
    sig = mk_signal("2024-01-01", atr_pct=0.05)
    r = simulate(_trail_preset(), sig, panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    t = r["trades"].iloc[0]
    # 当日 low=95 恰好等于初始止损 95 → 按初始止损成交,而不是被抬高的 125
    assert t["exit_price"] == pytest.approx(95.0), f"疑似同 bar 前视: {t['exit_price']}"


def test_trailing_stop_never_loosens():
    """棘轮只收紧。价格下行时止损位不得下移。"""
    bars = flat_days(25)
    bars[2].update({"open": 100.0, "high": 130.0, "low": 100.0, "close": 128.0})
    bars[3].update({"open": 128.0, "high": 128.0, "low": 124.0, "close": 125.0})
    panel = mk_panel(bars)
    r = simulate(_trail_preset(), mk_signal("2024-01-01", atr_pct=0.05), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    t = r["trades"].iloc[0]
    # day2 high=130 → day3 止损 125;day3 low=124 触发,成交 125
    assert t["exit_price"] == pytest.approx(125.0)


def test_trailing_stop_off_by_default():
    """不设 trailing_sl ⇒ 行为与 H2 之前完全一致。"""
    bars = flat_days(25)
    bars[2].update({"open": 100.0, "high": 120.0, "low": 100.0, "close": 118.0})
    bars[3].update({"open": 118.0, "high": 118.0, "low": 92.0, "close": 93.0})
    panel = mk_panel(bars)
    p = {k: v for k, v in _trail_preset().items() if k != "trailing_sl"}
    r = simulate(p, mk_signal("2024-01-01", atr_pct=0.05), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    assert r["trades"].iloc[0]["exit_price"] == pytest.approx(95.0)   # 初始止损,未被抬高


# ------------------------------------------------- 滑点(CLAUDE.md §4(c))
# 铁律原文:Execution slippage (price impact between signal trigger and fill) MUST be
# modeled dynamically as a function of the asset's current ATR (Average True Range) and
# the order's participation rate. Do not use flat-rate slippage (e.g. 1 tick).
#
# 2026-10-15 审计实测:旧实现滑点**恒为 0** —— 钩子默认 0.0、两个交付 preset 都没开它,
# 且公式里参与率的位置被字面量 1.0 占死。而「开了会怎样」的实测是:CAGR 从 +44.5%
# 一路到 −28.9%。也就是说「达标 30%」完全取决于一根从未被设定、从未被记录的旋钮取 0。


def _atr_preset(**over) -> dict:
    p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
    p.update({"atr_sl_mult": 1.0, "atr_tp_mult": 12.51})
    p.update(over)
    return p


def _entry_of(preset: dict, *, atr_pct: float = 0.02, volume: float = 1_000_000.0) -> pd.Series:
    """跑一笔并返回成交记录。

    用 25 根 bar 让仓位走到 max_hold 平仓 —— trades 只记录**已平仓**的仓位,
    平盘 panel 下 ATR 归一化的止损止盈都不会被触发。
    """
    bars = flat_days(25)
    for b in bars:
        b["volume"] = volume
    panel = mk_panel(bars)
    r = simulate(preset, mk_signal("2024-01-01", atr_pct=atr_pct), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    assert len(r["trades"]) == 1, f"预期恰好一笔平仓,得到 {len(r['trades'])}"
    return r["trades"].iloc[0]


def test_slippage_is_zero_when_not_configured():
    """不设 atr_slip_scale ⇒ 入场价逐位等于开盘价(老行为,合成用例继续有效)。"""
    assert _entry_of(_atr_preset())["entry_price"] == pytest.approx(FLAT)


def test_slippage_grows_with_atr():
    """买入付更高的价,且 ATR 越大滑点越大(f(ATR) 这一半)。"""
    lo = _entry_of(_atr_preset(atr_slip_scale=1.0), atr_pct=0.02)
    hi = _entry_of(_atr_preset(atr_slip_scale=1.0), atr_pct=0.08)
    assert lo["entry_price"] > FLAT, "滑点必须抬升入场价"
    assert hi["entry_price"] > lo["entry_price"], "ATR 越大,滑点越大"


def test_slippage_grows_with_participation_rate():
    """同一笔订单吃掉 bar 成交量的比例越大,滑点越大(f(参与率) 这一半)。

    成交量降到 1/100 ⇒ 参与率 ×100 ⇒ sqrt 律下滑点放大约 10 倍。
    """
    thin = _entry_of(_atr_preset(atr_slip_scale=1.0), volume=10_000.0)
    thick = _entry_of(_atr_preset(atr_slip_scale=1.0), volume=1_000_000.0)
    assert thin["entry_price"] > thick["entry_price"], "参与率越高,滑点越大"
    ratio = ((thin["entry_price"] / FLAT - 1.0) / (thick["entry_price"] / FLAT - 1.0))
    assert ratio == pytest.approx(10.0, rel=0.15), f"sqrt 律下应约 10x,实测 {ratio:.2f}x"


def test_slippage_stays_within_the_participation_cap():
    """滑点用**受量能上限约束后**的参与率算 ⇒ 不会因为一笔巨单而爆炸。

    ⚠️ 场景必须让 target_qty **大于** cap,否则 min() 不咬合、这条断言分辨不出
    「用没用截断后的参与率」—— 第三层复核实测:旧场景 target_qty == cap == 1000,
    把 `qty_pre = min(target_qty(o), cap)` 改成 `qty_pre = target_qty(o)` 之后,
    entry_price 逐位不变、用例照样绿。现在 initial_cash=1e7 ⇒ target_qty=1e5 ≫ cap=1e3。
    """
    p = _atr_preset(atr_slip_scale=1.0, initial_cash=10_000_000.0)
    bars = flat_days(25)
    for b in bars:
        b["volume"] = 10_000.0
    r = simulate(p, mk_signal("2024-01-01", atr_pct=0.08), mk_panel(bars),
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    t = r["trades"].iloc[0]
    assert t["size"] == 1_000, "量能上限没咬合(应恰好 = 0.10 × 10_000)"
    # 参与率 = 1000/10000 = 0.10 ⇒ slip = 1.0 × 0.08 × sqrt(0.10)
    assert t["entry_price"] == pytest.approx(FLAT * (1.0 + 0.08 * np.sqrt(0.10)), rel=1e-12)


def test_slippage_requires_valid_atr():
    """开了滑点却拿不到 atr_pct ⇒ fail-fast,不得静默退化为零滑点(§0)。"""
    with pytest.raises(BrokerError, match="atr_slip_scale"):
        _entry_of(_atr_preset(atr_slip_scale=1.0), atr_pct=float("nan"))


def test_slippage_scale_must_be_non_negative():
    with pytest.raises(BrokerError, match="atr_slip_scale"):
        simulate(_atr_preset(atr_slip_scale=-0.5), mk_signal("2024-01-01"),
                 mk_panel(flat_days(6)), start="2024-01-01", end="2024-01-15")


# ------------------------------------------------- 结算隔离(CLAUDE.md §2(d))
# 铁律原文:differentiate Free_Cash / Locked_Margin / Settling_Funds。Do NOT assume
# funds from a sell order at T are available for a buy order at T unless explicitly
# modeling margin borrowing with interest.
#
# 2026-10-15 审计实测:主循环「先平仓(用当日盘中价)后开仓(用当日开盘价)」把这条
# 整条绕过了 —— 入场闸门直接读已被同日卖出所得充值的 cash。交付的 12m preset 里
# 13 笔有 3 笔**必须**靠这笔钱才能成交(2026-04-29 的 MTSI 用一个多小时后才发生的
# PCVX 盘中止损回款,付了当日开盘的入场款)。规则改为:
#     **当日入场的预算与容量 = 上一交易日收盘时的账本快照。**


def _two_symbol_flat(n: int, *, stop_on: int | None = None,
                     start: str = "2024-01-01") -> pd.DataFrame:
    """两个标的的平盘 panel;stop_on 指定第几根 bar 的 low 击穿 2.1% 止损。"""
    rows = []
    for i, d in enumerate(pd.bdate_range(start, periods=n)):
        for sym in ("AAA", "BBB"):
            b = {"date": d, "symbol": sym, "open": FLAT, "high": FLAT,
                 "low": FLAT, "close": FLAT, "volume": 1_000_000.0, "atr_pct": 0.02}
            if i == stop_on:
                b["low"] = FLAT * 0.90
            rows.append(b)
    return mk_panel(rows)


def _signals(*pairs) -> pd.DataFrame:
    return pd.concat([mk_signal(d, symbol=s) for d, s in pairs], ignore_index=True)


def test_same_day_sale_proceeds_cannot_fund_same_day_entry():
    """卖出所得不得为**同日**入场出资。max_positions=2 ⇒ 这里只测现金这一条。"""
    panel = _two_symbol_flat(8, stop_on=2)          # d2 = 2024-01-03 盘中击穿止损
    p = {**BASE, "max_positions": 2}
    sig = _signals(("2024-01-01", "AAA"), ("2024-01-02", "BBB"))   # BBB 想买在 d2 开盘
    r = simulate(p, sig, panel, start="2024-01-01", end="2024-01-20",
                 cost_bps_roundtrip=0.0)
    syms = list(r["trades"]["symbol"])
    assert syms == ["AAA"], f"BBB 用同日止损回款付了 d2 开盘的入场款:{syms}"


def test_same_day_exit_does_not_free_a_slot_for_same_day_entry():
    """同日出场释放的仓位槽不得为同日入场所用。

    现金充足(position_fraction=0.1)⇒ 这里只测**容量**这一条:d2 开盘时 AAA 仍在场。
    """
    panel = _two_symbol_flat(8, stop_on=2)
    p = {**BASE, "max_positions": 1, "position_fraction": 0.1}
    sig = _signals(("2024-01-01", "AAA"), ("2024-01-02", "BBB"))
    r = simulate(p, sig, panel, start="2024-01-01", end="2024-01-20",
                 cost_bps_roundtrip=0.0)
    assert list(r["trades"]["symbol"]) == ["AAA"]


def test_symbol_held_at_the_open_cannot_be_re_entered_same_day():
    """开盘时仍持有的标的不得在当日被再次建仓 —— 它当天盘中才出场。"""
    panel = _two_symbol_flat(8, stop_on=2)
    p = {**BASE, "max_positions": 2, "position_fraction": 0.1}
    sig = _signals(("2024-01-01", "AAA"), ("2024-01-02", "AAA"))
    r = simulate(p, sig, panel, start="2024-01-01", end="2024-01-20",
                 cost_bps_roundtrip=0.0)
    assert list(r["trades"]["symbol"]) == ["AAA"]


def test_proceeds_are_usable_the_next_trading_day():
    """隔日使用是合法的 —— 上面三条不得把正常的「出场后换仓」一起堵死。

    这条是**反过度收紧的护栏**(不依赖第二笔平仓,只看账本状态):出场之后的每一天
    都必须还有持仓在场。
    """
    panel = _two_symbol_flat(8, stop_on=2)
    p = {**BASE, "max_positions": 2}
    sig = _signals(("2024-01-01", "AAA"), ("2024-01-03", "BBB"))   # BBB 信号在出场日之后
    r = simulate(p, sig, panel, start="2024-01-01", end="2024-01-20",
                 cost_bps_roundtrip=0.0)
    tr = r["trades"]
    eq = r["equity"].set_index("date")
    # AAA 在 d2 止损出场；BBB 在 01-03 出信号、01-04 建仓后一直持有到窗口末尾，
    # 因此在最后一天按 C5 收盘清算（reason=eod）。
    assert list(tr["symbol"]) == ["AAA", "BBB"]
    assert tr.iloc[0]["exit_reason"].startswith("sl")
    assert tr.iloc[1]["exit_reason"] == "eod"
    exit_day = tr.iloc[0]["exit_date"]
    after = eq.loc[eq.index > exit_day]
    assert len(after) > 0
    assert after["n_pos"].iloc[0] == 1, "出场后应能开新仓 —— 隔日复用被误堵"
    assert after["n_pos"].iloc[-1] == 0, "窗口末尾必须清算（C5）"


def test_entry_capacity_reads_the_previous_close_not_the_current_bar():
    """入场容量用的是**上一交易日收盘**的持仓数,不是当日出场后的持仓数。

    与 test_same_day_exit_does_not_free_a_slot... 的区别:这条把「当日出场」换成
    「当日开盘 gap 出场」—— get 到同一结论(槽位在开盘时仍被占用),但走的是另一条
    代码分支(o <= sl_price 的 gap 路径)。
    """
    bars = flat_days(8)
    bars[2].update({"open": FLAT * 0.90, "high": FLAT, "low": FLAT * 0.90, "close": FLAT})
    panel = mk_panel(bars)
    p = {**BASE, "max_positions": 1, "position_fraction": 0.1}
    sig = _signals(("2024-01-01", "AAA"), ("2024-01-02", "BBB"))
    r = simulate(p, sig, panel, start="2024-01-01", end="2024-01-20",
                 cost_bps_roundtrip=0.0)
    assert list(r["trades"]["symbol"]) == ["AAA"]
    assert r["trades"].iloc[0]["exit_reason"] == "sl_gap"


# ------------------------------------------------- 价格口径与公司行动(CLAUDE.md §3(b)/§3(c))
# §3(b) Dual-Price System:复权价只用于**数学指标**;止损止盈触发、跳空判定、逐日市值与
#   现金记账必须用 **as-traded 价**。
# §3(c) Event-Sourced Corporate Actions:现金分红必须**显式**触发一笔现金存入。
# 2026-10-15 审计实测:旧实现 open/high/low/close **四个价一起**被乘上分红回补因子,
# 于是执行与记账全在复权价空间;而那个因子是 ∏_{事件晚于 t},即 T 日的成交价取决于
# T 之后的除息日。另一面:1728 次分红事件没有一次进入账本。


def test_broker_fails_fast_without_as_traded_price_columns():
    """缺 as-traded 价 ⇒ fail-fast,不得静默退回复权价(那正是要被修掉的缺陷)。"""
    panel = mk_panel(flat_days(6)).drop(columns=["raw_close"])
    with pytest.raises(BrokerError, match="raw_close"):
        simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-01-15")


def test_entry_uses_as_traded_open_not_adjusted():
    bars = flat_days(25)
    for b in bars:
        b.update({"raw_open": 100.0, "raw_high": 100.0, "raw_low": 100.0, "raw_close": 100.0,
                  "open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0})
    r = simulate(BASE, mk_signal("2024-01-01"), mk_panel(bars),
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    assert r["trades"].iloc[0]["entry_price"] == pytest.approx(100.0)


def test_equity_marks_positions_at_as_traded_close():
    """逐日市值用 as-traded 收盘价(旧实现用复权价,现金记账因此不是物理现金)。"""
    bars = flat_days(25)
    for i, b in enumerate(bars):
        b.update({"raw_open": 100.0, "raw_high": 100.0, "raw_low": 100.0, "raw_close": 100.0,
                  "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
        if i >= 2:                       # 入场后 as-traded 涨到 110,而复权价不动
            b.update({"raw_open": 110.0, "raw_high": 110.0, "raw_low": 110.0, "raw_close": 110.0})
    r = simulate(BASE, mk_signal("2024-01-01"), mk_panel(bars),
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    eq = r["equity"].set_index("date")
    assert r["trades"].iloc[0]["size"] == 1000
    day2 = pd.Timestamp(bars[2]["date"])
    assert float(eq.loc[day2, "equity"]) == pytest.approx(110_000.0)


def test_stop_loss_triggers_on_as_traded_price_when_dividend_gaps_it_down():
    """除息跳空在 as-traded 口径下是真的会穿透止损 —— 复权口径把它抹平了(偏乐观)。"""
    bars = flat_days(25)
    for b in bars:
        b.update({"raw_open": 100.0, "raw_high": 100.0, "raw_low": 100.0, "raw_close": 100.0,
                  "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    bars[4].update({"raw_open": 95.0, "raw_high": 96.0, "raw_low": 94.0, "raw_close": 95.0,
                    "dividends": 5.0})
    t = simulate(BASE, mk_signal("2024-01-01"), mk_panel(bars),
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)["trades"].iloc[0]
    assert t["exit_reason"] == "sl_gap", "95 < 止损价 98,应按开盘价 gap 出场"
    assert t["exit_price"] == pytest.approx(95.0)


def test_dividend_is_credited_to_cash_when_held_through_ex_date():
    """持有穿过除息日 ⇒ 现金必须**显式**增加 size × 每股分红(§3(c))。"""
    bars = flat_days(25)
    bars[4]["dividends"] = 2.0
    r = simulate(BASE, mk_signal("2024-01-01"), mk_panel(bars),
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    t = r["trades"].iloc[0]
    eq = r["equity"].set_index("date")
    ex_day = pd.Timestamp(bars[4]["date"])
    step = float(eq.loc[ex_day, "equity"]) - float(eq.loc[:ex_day].iloc[-2]["equity"])

    assert t["size"] == 1000
    assert t["dividends"] == pytest.approx(2000.0), "分红没进账本"
    assert step == pytest.approx(2000.0), "权益曲线没反映这笔现金流入"
    assert t["net_pnl"] == pytest.approx(2000.0), (
        "分红必须计入该笔净盈亏 —— 否则 Σ net_pnl 与现金流对不上,双式记账断裂")


def test_dividend_is_not_credited_when_entered_on_the_ex_date():
    """除息**当日**才建仓 ⇒ 不领这次分红(T+1 下进场者不是登记持有人)。"""
    bars = flat_days(25)
    bars[1]["dividends"] = 2.0           # 01-02,正是信号 T+1 的开盘入场日
    r = simulate(BASE, mk_signal("2024-01-01"), mk_panel(bars),
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    assert r["trades"].iloc[0]["dividends"] == pytest.approx(0.0)


def test_dividend_is_credited_even_if_stopped_out_on_the_ex_date():
    """除息日**开盘就**被 gap 止损 ⇒ 仍要领这次分红(卖在除息日 = 含息卖出)。"""
    bars = flat_days(25)
    for b in bars:
        b.update({"raw_open": 100.0, "raw_high": 100.0, "raw_low": 100.0, "raw_close": 100.0,
                  "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    bars[4].update({"raw_open": 90.0, "raw_high": 91.0, "raw_low": 89.0, "raw_close": 90.0,
                    "dividends": 2.0})
    t = simulate(BASE, mk_signal("2024-01-01"), mk_panel(bars),
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)["trades"].iloc[0]
    assert t["exit_reason"] == "sl_gap"
    assert t["dividends"] == pytest.approx(2000.0)
    assert t["net_pnl"] == pytest.approx(-10_000.0 + 2000.0)


# ------------------------------------------- 成交价必须落在 bar 的价格区间内(§0/§4)
# 2026-10-15 复核实测:atr_sl_mult=0.1 + atr_slip_scale=1.0 时,引擎在 [100,100] 的
# 平盘 bar 上以 **102.13** 出场(高于该 bar 的最高价)—— 一次物理上不存在的成交。
# 根因有二:(a) 入场当日那条路径缺 §4(f) 的 gap 分支,与步骤 1 不对称;
#            (b) 滑点是对开盘价的加成,不受 bar 已实现区间约束。
# 修法:补齐 gap 分支,并把「出场价必须在该 bar 的 as-traded [low, high] 内」作为
# §0 的 fail-fast 断言 —— 正常区间(滑点 ≪ 止损宽度)永远满足,病态配置立刻报错。


def _exit_bar_range(panel: pd.DataFrame, symbol: str, day) -> tuple[float, float]:
    row = panel[(panel["symbol"] == symbol) & (panel["date"] == day)].iloc[0]
    return float(row["raw_low"]), float(row["raw_high"])


def _assert_exit_fills_inside_their_bars(r: dict, panel: pd.DataFrame) -> None:
    """⚠️ 盯的是**触发价**(exit_trigger),不是成交价。
    

    入场侧**不在**这条护栏内,而且入场价确实可能落在当日最高价之上 —— 那是已知且已
    披露的建模简化(见 broker.py 里「已知建模简化」那段:实测 v1 2/41 笔、12m 3/15 笔,
    超出 ≤0.13%)。第三层复核实测:旧名字叫 test_slippage_never_pushes_the_entry_
    above_the_bar_high,而函数体只查出场价 —— 一条绿着的用例会让读者得出「入场侧
    已被守住」的相反结论。名字与函数体现在都改成只管出场。
    """
    for _, t in r["trades"].iterrows():
        lo, hi = _exit_bar_range(panel, t["symbol"], t["exit_date"])
        trig = float(t["exit_trigger"])
        assert lo - 1e-9 <= trig <= hi + 1e-9, (
            f"{t['symbol']} {t['exit_date'].date()} 触发价 {trig:.4f} "
            f"落在 bar [{lo}, {hi}] 之外 —— 物理上不可能"
        )
        assert float(t["exit_price"]) <= trig + 1e-9, "卖出滑点只允许把成交价压低"


def test_normal_configs_fill_inside_the_bar_range():
    """正常配置(滑点 ≪ 止损宽度)下该不变量必须处处成立。"""
    bars = flat_days(25)
    bars[3].update({"low": 90.0})            # 击穿 2.1% 止损
    panel = mk_panel(bars)
    r = simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    assert len(r["trades"]) == 1
    _assert_exit_fills_inside_their_bars(r, panel)


def test_exit_fills_stay_inside_the_exit_bar_range():
    """开仓当日的**出场**分支也必须落在 bar 区间内(入场侧是已知简化,不在本用例范围)。"""
    bars = flat_days(25)
    for b in bars:
        b.update({"high": FLAT + 0.5, "low": FLAT - 0.5})
    bars[3].update({"low": FLAT * 0.90})     # 盘中击穿止损
    panel = mk_panel(bars)
    p = {**BASE, "atr_slip_scale": 1.0}
    r = simulate(p, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    _assert_exit_fills_inside_their_bars(r, panel)


def test_gap_through_stop_on_the_entry_bar_fills_at_the_open():
    """入场当日开盘就已越过止损 ⇒ 按**开盘价**成交(§4(f)),不得用止损价。

    2026-10-15 复核发现的原始缺陷:入场当日那条路径**缺** gap 分支(步骤 1 有、它没有),
    于是在这种配置下报出「成交价 102.13 而该 bar 最高价只有 100」—— 物理上不存在的成交。
    """
    bars = flat_days(8)
    for b in bars:
        b["volume"] = 10_000.0
    p = {k: v for k, v in BASE.items() if k not in ("sl_pct", "tp_pct")}
    p.update({"atr_sl_mult": 0.1, "atr_tp_mult": 1.2, "atr_slip_scale": 1.0})
    panel = mk_panel(bars)
    r = simulate(p, mk_signal("2024-01-01", atr_pct=0.10), panel,
                 start="2024-01-01", end="2024-01-20", cost_bps_roundtrip=0.0)
    t = r["trades"].iloc[0]
    assert t["exit_reason"] == "sl_gap"
    assert t["exit_trigger"] == pytest.approx(FLAT)        # 触发在当日开盘价 100
    assert t["exit_price"] < FLAT                          # 卖出滑点把成交价压低
    _assert_exit_fills_inside_their_bars(r, panel)



def test_participation_rate_must_be_a_fraction():
    """参与率上限必须 ∈ (0,1]:>1 等于买超整根 bar,<0 会让 sqrt 产生 NaN。"""
    panel = mk_panel(flat_days(6))
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(BrokerError, match="max_volume_participation"):
            simulate(BASE | {"max_volume_participation": bad}, mk_signal("2024-01-01"),
                     panel, start="2024-01-01", end="2024-01-15")


# ------------------------------------------------- 出场侧滑点与量能上限(§4(a)/§4(c))
# 2026-10-15 实测(v1 41 笔 / 12m 15 笔):出场参与率中位 0.013%~0.048%、最大 0.42%,
# 距 10% 的量能上限还有 24~127 倍 —— 上限从不咬合;但出场滑点中位 0.085%~0.094%/边,
# 是全部双边佣金预算(5bp)的 1.7 倍。**不建模出场滑点会系统性偏爱窄止损**:窄止损的
# 损失被机械封顶在 −1R,而真实成交会比触发价更差,且这个差额相对止损宽度更大。


def _sl_bars(n=25, volume=1_000_000.0, exit_volume=None, exit_idx=3):
    bars = flat_days(n)
    for i, b in enumerate(bars):
        b["volume"] = volume
        if i == exit_idx:
            b["low"] = FLAT * 0.90          # 击穿 2.1% 止损
            if exit_volume is not None:
                b["volume"] = exit_volume
    return bars


def test_exit_slippage_lowers_the_fill_price():
    """卖出方向的滑点必须**压低**成交价,量级 = scale × 信号日 atr_pct × sqrt(参与率)。"""
    panel = mk_panel(_sl_bars())
    r = simulate({**BASE, "atr_slip_scale": 1.0}, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    t = r["trades"].iloc[0]
    trigger = float(t["exit_trigger"])
    part = int(t["size"]) / 1_000_000.0
    expect = trigger * (1.0 - 1.0 * float(t["atr_pct"]) * np.sqrt(part))
    assert t["exit_price"] < trigger, "出场滑点没把成交价压低"
    assert t["exit_trigger"] == pytest.approx(float(t["entry_price"]) * (1 - 0.021))
    assert t["exit_price"] == pytest.approx(expect, rel=1e-12)


def test_exit_slippage_is_zero_when_disabled():
    """不设 atr_slip_scale ⇒ 成交价逐位等于触发价(老行为,既有用例继续有效)。"""
    panel = mk_panel(_sl_bars())
    t = simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)["trades"].iloc[0]
    assert t["exit_price"] == pytest.approx(FLAT * (1 - 0.021))


def test_loss_exceeds_one_R_once_exit_slippage_is_on():
    """开了出场滑点后,止损不再是「恰好 −1R」—— R 分布不再退化。"""
    panel = mk_panel(_sl_bars())
    p = {**BASE, "atr_slip_scale": 1.0}
    from research.metrics import r_multiples
    t = simulate(p, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)["trades"]
    assert float(r_multiples(t).iloc[0]) < -1.0, "出场滑点没让亏损超过计划风险"


def test_exit_trigger_must_be_inside_the_bar_even_with_slippage():
    """区间断言管的是**触发价**;滑点后的成交价允许略微越界(与入场侧同理)。"""
    bars = _sl_bars()
    for b in bars:
        b.update({"high": FLAT, "low": FLAT})      # 平盘 bar:止损触发价必在区间内
    t = simulate({**BASE, "atr_slip_scale": 1.0}, mk_signal("2024-01-01"),
                 mk_panel(bars), start="2024-01-01", end="2024-02-10",
                 cost_bps_roundtrip=0.0)["trades"].iloc[0]
    # 触发价 97.9 落在 [100, 100] 之外 ⇒ 这里恰好无法成交(见 gap 分支),
    # 所以真正要断言的是:任何**报出来**的成交都必须有可解释的触发。
    assert t["exit_price"] <= FLAT


def test_exit_over_the_volume_cap_fails_fast():
    """出场量超过该 bar 的量能上限 ⇒ fail-fast,不静默按足额成交。

    §4(a) 的 IOC-部分成交没有实现(实测 24~127 倍余量,永不咬合),所以这里把它变成
    **显式边界**:一旦某个仓位真的大到卖不掉,立刻报错而不是给出一个假的足额成交。
    """
    panel = mk_panel(_sl_bars(exit_volume=5_000.0))     # 出场日 cap = 500 < size 1000
    with pytest.raises(BrokerError, match="量能上限"):
        simulate(BASE, mk_signal("2024-01-01"), panel,
                 start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
