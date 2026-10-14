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
    """bars: [{date, open, high, low, close, volume, atr_pct?}] → broker 需要的 panel。"""
    rows = []
    for b in bars:
        rows.append({
            "symbol": b.get("symbol", "AAA"),
            "date": pd.Timestamp(b["date"]),
            "open": b.get("open", FLAT), "high": b.get("high", FLAT),
            "low": b.get("low", FLAT), "close": b.get("close", FLAT),
            "volume": b.get("volume", 1_000_000.0),
            "atr_pct": b.get("atr_pct", 0.02),
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
        "intraday_tiebreak": "sl_first", "initial_cash": 100_000.0}


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
