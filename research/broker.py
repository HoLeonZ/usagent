"""执行代理 —— hiagent/uptrend_pullback/portfolio.py 的**美股子集**。

保留的语义(逐条对应 CLAUDE.md §2/§4):
  * 信号 T 收盘 → **T+1 开盘**入场
  * `intraday_tiebreak = "sl_first"`:同一根 bar 同时触碰 SL 与 TP 时,按**最坏情况**判 SL
  * gap-through-stop:开盘已越过止损 → 按**开盘价**成交(不按止损价)
  * 成交量上限 `Bar_Volume × max_volume_participation`,超出部分 IOC 丢弃、不排队
  * `max_positions` 上限;超容量按 score 降序取,**当 T 丢弃不排队**
  * all-in:`position_fraction = 1.0`,目标股数 `floor(cash / (px × (1+slip)))`

**明确删除**的 A 股件(理由见设计文档 §3):
  T+1 可卖性 / 涨跌停不可成交 / 100 股整手 / 印花税 / 三池保证金。

一处必须小心的历史坑(hiagent 自己踩过,见 preset_search §1):
  **equity 曲线必须用 `cash + qty × close`。** 曾把建仓名义额计入两次,导致
  Sharpe / 回撤 / 敞口全部由一条物理上不可能的日收益序列算出(一个 0% 胜率的窗口
  报出 Sharpe +3.00,修正后是 -4.71)。test_broker 用「一次性建仓后价格不变 ⇒
  权益必须等于初始现金减去成本」把这条钉死。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADE_COLS = [
    "entry_date", "exit_date", "symbol", "exit_reason",
    "entry_price", "exit_price", "size", "hold_days",
    "gross_pnl", "fees", "net_pnl", "net_return", "atr_pct",
    # 每笔的**计划风险**宽度与金额。hiagent 的 tail_objective 用固定 sl_pct 推 R,
    # 但 ATR 归一化后每笔的止损宽度不同,所以必须逐笔记录,否则 E[R] 算不出来。
    "sl_width", "tp_width", "risk_amount",
]

VALID_TIEBREAK = ("sl_first",)

EXIT_REASONS = ("sl", "sl_gap", "tp", "tp_gap", "max_hold", "eod")


class BrokerError(RuntimeError):
    """物理上不可能的状态 —— fail-fast(CLAUDE.md §0)。"""


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    size: int
    sl_price: float
    tp_price: float
    bars_held: int = 0
    atr_pct: float = float("nan")
    sl_width: float = float("nan")
    tp_width: float = float("nan")
    # H2:移动止损(ratchet)。trail_width=None 表示不移动(原始行为)。
    trail_width: float | None = None
    # 截至**前一交易日**的最高 high。只用 ≤D-1 的数据决定 D 日的止损位 ——
    # 若用 D 日自身的 high 抬高止损再拿 D 日的 low 去触发,就是同 bar 前视。
    high_water: float = float("nan")


@dataclass
class _Book:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)


def _bar_index(panel: pd.DataFrame) -> tuple[dict[str, dict[str, int]], list[pd.Timestamp],
                                             dict[str, dict[str, np.ndarray]]]:
    """把 panel 预处理成 {symbol: {date: **该标的数组内**的下标}} + 全局交易日轴。

    ⚠️ 两条索引轴不可混用:全局轴(day_pos)用来找"上一交易日",
    标的内下标(per_sym)用来取该标的的 OHLCV。日线数据里各标的的日期覆盖
    并不相同(晚间上市、停牌、退市),混用会直接越界或静默取错行。
    """
    days = list(pd.Index(sorted(panel["date"].unique())))
    day_pos = {d: i for i, d in enumerate(days)}
    per_sym: dict[str, dict[str, int]] = {}
    cols = ("open", "high", "low", "close", "volume", "atr_pct")
    data: dict[str, dict[str, np.ndarray]] = {}
    for sym, sub in panel.groupby("symbol", sort=False):
        sub = sub.sort_values("date")
        per_sym[sym] = {d: i for i, d in enumerate(sub["date"])}
        data[sym] = {c: sub[c].to_numpy(dtype=float) for c in cols}
    return per_sym, days, data


def simulate(
    preset: dict,
    signals: pd.DataFrame,
    panel_ind: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_bps_roundtrip: float = 5.0,
) -> dict:
    """在 [start, end) 上跑一次回测。

    signals: select_entries 的输出(信号日 = date);
    入场发生在信号日的**下一个交易日**开盘。
    """
    if cost_bps_roundtrip < 0:
        raise BrokerError(f"cost_bps_roundtrip 必须 >= 0,得到 {cost_bps_roundtrip}")

    tiebreak = preset.get("intraday_tiebreak", "sl_first")
    if tiebreak not in VALID_TIEBREAK:
        raise BrokerError(f"intraday_tiebreak 必须是 {VALID_TIEBREAK},得到 {tiebreak!r}")

    max_positions = int(preset.get("max_positions", 1))
    position_fraction = float(preset.get("position_fraction", 1.0))
    max_participation = float(preset.get("max_volume_participation", 0.10))
    max_hold = int(preset.get("max_hold", 18))
    # 退出宽度二选一:固定百分比(hiagent 基线口径)或 ATR 归一化(覆盖固定值)。
    # ATR 归一化的动机是机制性的,不是调参偏好 —— 见 presets.py 与 BASELINE.md:
    # 2.1% 的固定止损在 atr_pct 中位 6.8% 的美股池里只有 0.31 ATR,落在噪声之内。
    # ⚠️ hiagent preset_search §5 M3 曾在 A 股语境下证伪 ATR 归一化;那次证伪的前提
    #    (「2.1% 固定止损在 A 股是对的」)在这里不成立,故这是一次**新检验**。
    atr_sl_mult = preset.get("atr_sl_mult")
    atr_tp_mult = preset.get("atr_tp_mult")
    sl_pct = preset.get("sl_pct")
    tp_pct = preset.get("tp_pct")
    atr_slip_scale = float(preset.get("atr_slip_scale", 0.0))
    initial_cash = float(preset.get("initial_cash", 100_000.0))
    # 仓位政策二选一:
    #   position_fraction = 1.0  → all-in(hiagent §2 的政策)
    #   risk_per_trade           → 风险预算式,qty = equity × risk / (止损宽度 × 价格)
    # 为什么需要后者:all-in 与「窄止损」是**耦合**的。hiagent §2 的原话是
    #   "Tail risk is managed entirely by exit logic (TP/SL), never by reserving
    #    cash at entry" —— 这句话只在止损很窄时成立。A 股 2.1% 止损下 all-in
    # 每次亏 2.1% 权益;把止损加宽到 1 ATR(美股中位 6.79%)后,每次都亏 6.79%,
    # 连亏 20 次从 −34.6% 变成 −75.5%。风险预算式让单笔亏损与止损宽度**脱钩**。
    risk_per_trade = preset.get("risk_per_trade")
    if risk_per_trade is not None:
        risk_per_trade = float(risk_per_trade)
        if not (0 < risk_per_trade < 1):
            raise BrokerError(f"risk_per_trade ∈ (0,1),得到 {risk_per_trade}")

    # H2:移动止损。True 时止损随最高价上移(棘轮),距离沿用初始止损宽度
    # (= atr_sl_mult × atr_pct),因此**不引入新的待调参数**。
    # 动机是机制性的:BASELINE.md §1D 实测 ONDS 入场后 +44.4%、ADTN +27.1%、
    # VERI +25.0% 最后全部亏着出场 —— 固定价位止损 + 不可达 TP = 只止损不止盈。
    trailing_sl = bool(preset.get("trailing_sl", False))

    if max_positions < 1:
        raise BrokerError(f"max_positions 必须 >= 1,得到 {max_positions}")
    if not (0 < position_fraction <= 1.0):
        raise BrokerError(f"position_fraction ∈ (0,1],得到 {position_fraction}")

    use_atr_widths = atr_sl_mult is not None or atr_tp_mult is not None
    if use_atr_widths:
        if atr_sl_mult is None or atr_tp_mult is None:
            raise BrokerError("atr_sl_mult 与 atr_tp_mult 必须同时给出(或都不给)")
        if atr_sl_mult <= 0 or atr_tp_mult <= 0:
            raise BrokerError(
                f"atr_*_mult 必须 > 0,得到 sl={atr_sl_mult} tp={atr_tp_mult}")
    elif sl_pct is None or tp_pct is None or sl_pct <= 0 or tp_pct <= 0:
        raise BrokerError(f"需要 sl_pct/tp_pct > 0,或给出 atr_sl_mult/atr_tp_mult;"
                          f"得到 sl={sl_pct} tp={tp_pct}")

    def exit_widths(atr_row: float) -> tuple[float, float]:
        """返回 (sl_width, tp_width)。ATR 归一化时逐笔不同。"""
        if use_atr_widths:
            if not (np.isfinite(atr_row) and atr_row > 0):
                raise BrokerError(
                    f"ATR 归一化需要有效的 atr_pct,得到 {atr_row!r} —— "
                    f"信号日 atr_pct 缺失(不该发生:signals 已断言 notna)"
                )
            return atr_sl_mult * atr_row, atr_tp_mult * atr_row
        return float(sl_pct), float(tp_pct)

    per_sym, days, data = _bar_index(panel_ind)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    axis = [d for d in days if lo <= d < hi]
    if not axis:
        raise BrokerError(f"窗口 {start}..{end} 内没有任何交易日")

    cost_half = cost_bps_roundtrip / 2.0 / 10_000.0

    # 每个信号日 → 按 score 降序的候选(入场发生在 date 的下一交易日)
    pending: dict[pd.Timestamp, list[dict]] = {}
    if not signals.empty:
        for d, grp in signals.groupby("date"):
            grp = grp.sort_values("score", ascending=False)
            pending[pd.Timestamp(d)] = grp.to_dict("records")

    book = _Book(cash=initial_cash)
    trades: list[dict] = []
    equity_rows: list[dict] = []
    day_pos = {d: i for i, d in enumerate(days)}

    def mark_to_market(day: pd.Timestamp) -> float:
        """equity = cash + Σ qty × close。这是唯一正确的口径(见模块 docstring)。"""
        total = book.cash
        for pos in book.positions.values():
            k = per_sym.get(pos.symbol, {}).get(day)
            if k is None:
                # 当日无该标的数据(停牌/退市):按最近一次已知收盘 mark
                k = max((j for dd, j in per_sym[pos.symbol].items() if dd <= day),
                        default=None)
            if k is None:
                raise BrokerError(f"{pos.symbol} 在 {day.date()} 前无任何数据,无法 mark")
            total += pos.size * data[pos.symbol]["close"][k]
        return total

    def close_position(pos: Position, day: pd.Timestamp, px: float, reason: str) -> None:
        notional_exit = px * pos.size
        exit_fee = notional_exit * cost_half
        entry_notional = pos.entry_price * pos.size
        entry_fee = entry_notional * cost_half
        gross = (px - pos.entry_price) * pos.size
        fees = entry_fee + exit_fee
        net = gross - fees
        cost_basis = entry_notional + entry_fee
        book.cash += notional_exit - exit_fee
        trades.append({
            "entry_date": pos.entry_date, "exit_date": day, "symbol": pos.symbol,
            "exit_reason": reason,
            "entry_price": pos.entry_price, "exit_price": px, "size": pos.size,
            # bars_held 是**已走完**的 bar 数:入场当日盘中出场 → 0,持满 max_hold → max_hold。
            # 不要写 +1 —— 那会让 max_hold=18 的仓位报出 hold_days=19(实测踩过)。
            "hold_days": pos.bars_held,
            "gross_pnl": gross, "fees": fees, "net_pnl": net,
            "net_return": net / cost_basis if cost_basis > 0 else float("nan"),
            "atr_pct": pos.atr_pct,
            # 计划风险:入场时即固定(止损宽度 × 股数),与最终出场无关。
            # E[R] = net_pnl / risk_amount —— 分母必须是计划风险,不能是事后结果
            # (hiagent tail_objective 的原话)。
            "sl_width": pos.sl_width, "tp_width": pos.tp_width,
            "risk_amount": pos.sl_width * pos.entry_price * pos.size,
        })
        del book.positions[pos.symbol]

    for day in axis:
        # ---------- 1) 持仓:开盘 gap → 盘中 SL/TP ----------
        for sym in list(book.positions):
            pos = book.positions[sym]
            # H2:棘轮。用截至**前一日**的最高价抬高止损位;只用 ≤D-1 的数据,
            # 所以与当日 low 的比较不存在同 bar 前视。止损只会收紧,不会放松。
            if pos.trail_width is not None and np.isfinite(pos.high_water):
                pos.sl_price = max(pos.sl_price, pos.high_water - pos.trail_width)
            k = per_sym.get(sym, {}).get(day)
            if k is None:                      # 当日无数据 → 无法成交,顺延
                continue
            o = data[sym]["open"][k]
            h = data[sym]["high"][k]
            low_ = data[sym]["low"][k]
            if not (o > 0 and h > 0 and low_ > 0):
                continue

            if o <= pos.sl_price:              # gap 跳过止损 → 按开盘成交
                close_position(pos, day, o, "sl_gap")
                continue
            if o >= pos.tp_price:              # gap 跳过止盈 → 按开盘成交
                close_position(pos, day, o, "tp_gap")
                continue

            hit_sl = low_ <= pos.sl_price
            hit_tp = h >= pos.tp_price
            if hit_sl and hit_tp:
                # 盘中盲区:两根阈值同 bar 触及 → 按最坏情况(CLAUDE.md §4)
                close_position(pos, day, pos.sl_price, "sl")
                continue
            if hit_sl:
                close_position(pos, day, pos.sl_price, "sl")
                continue
            if hit_tp:
                close_position(pos, day, pos.tp_price, "tp")
                continue

        # ---------- 2) 入场:前一交易日(T)的信号在今日(T+1)开盘成交 ----------
        prev_idx = day_pos[day] - 1
        if prev_idx >= 0:
            sig_day = days[prev_idx]
            for sig in pending.get(sig_day, []):
                if len(book.positions) >= max_positions:
                    break                       # 超容量当 T 丢弃,不排队
                sym = sig["symbol"]
                if sym in book.positions:
                    continue
                k = per_sym.get(sym, {}).get(day)
                if k is None:
                    continue
                o = data[sym]["open"][k]
                bar_vol = data[sym]["volume"][k]
                if not (o > 0) or not (bar_vol > 0):
                    continue                    # 无法成交(停牌/零量)

                sym_atr = float(sig.get("atr_pct", np.nan))
                slip = 0.0
                if atr_slip_scale > 0 and np.isfinite(sym_atr):
                    slip = sym_atr * 1.0 * atr_slip_scale
                entry_px = o * (1.0 + slip)
                sl_w, tp_w = exit_widths(sym_atr)

                if risk_per_trade is not None:
                    # 风险预算式:计划亏损 = 当前权益 × risk_per_trade,与止损宽度无关
                    risk_budget = mark_to_market(day) * risk_per_trade
                    qty = int(np.floor(risk_budget / (sl_w * entry_px)))
                else:
                    budget = book.cash * position_fraction
                    qty = int(np.floor(budget / (entry_px * (1.0 + cost_half))))
                if qty <= 0:
                    continue

                cap = int(np.floor(bar_vol * max_participation))
                qty = min(qty, cap)             # IOC:超出部分取消,不排队
                # 现金闸门(两种口径都要):风险预算可能算出一个买不起的股数
                affordable = int(np.floor(book.cash / (entry_px * (1.0 + cost_half))))
                qty = min(qty, affordable)
                if qty <= 0:
                    continue

                notional = entry_px * qty
                fee = notional * cost_half
                book.cash -= (notional + fee)
                book.positions[sym] = Position(
                    symbol=sym, entry_date=day, entry_price=entry_px, size=qty,
                    sl_price=entry_px * (1.0 - sl_w),
                    tp_price=entry_px * (1.0 + tp_w),
                    atr_pct=sym_atr, sl_width=sl_w, tp_width=tp_w,
                    # 注意:trail_width 存的是**价格距离**(entry_px × sl_w),
                    # 不是分数 —— 棘轮里是 `high_water - trail_width`,量纲必须对。
                    trail_width=(entry_px * sl_w if trailing_sl else None),
                    high_water=entry_px,     # 从入场价起步:止损先不抬高,直到价格创新高
                )

                # 入场当日同样接受盘中 SL/TP 检验(悲观,且与持仓路径一致)
                pos = book.positions[sym]
                h = data[sym]["high"][k]
                low_ = data[sym]["low"][k]
                if low_ <= pos.sl_price and h >= pos.tp_price:
                    close_position(pos, day, pos.sl_price, "sl")
                elif low_ <= pos.sl_price:
                    close_position(pos, day, pos.sl_price, "sl")
                elif h >= pos.tp_price:
                    close_position(pos, day, pos.tp_price, "tp")

        # ---------- 3) 收盘:更新最高价(H2)、持仓天数与 max_hold ----------
        for sym in list(book.positions):
            pos = book.positions[sym]
            k = per_sym.get(sym, {}).get(day)
            if k is not None and pos.trail_width is not None:
                # 收盘后记录今日最高价 —— 供**下一交易日**抬高止损(见 §1 的棘轮)
                pos.high_water = max(pos.high_water, data[sym]["high"][k])
            pos.bars_held += 1
            if pos.bars_held >= max_hold:
                if k is None:
                    continue
                close_position(pos, day, data[sym]["close"][k], "max_hold")

        equity_rows.append({"date": day, "equity": mark_to_market(day),
                            "cash": book.cash, "n_pos": len(book.positions),
                            "exposure": (mark_to_market(day) - book.cash)
                                        / mark_to_market(day)})

    equity = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades, columns=TRADE_COLS)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    metrics = _window_metrics(equity, trades_df, max_hold=max_hold)
    return {"trades": trades_df, "equity": equity, "metrics": metrics}


def _window_metrics(equity: pd.DataFrame, trades: pd.DataFrame, *, max_hold: int) -> dict:
    """单窗口指标。total_return / sharpe / max_dd 全部来自**正确的**权益曲线。"""
    if equity.empty:
        raise BrokerError("权益曲线为空")
    e = equity["equity"].to_numpy(dtype=float)
    if not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise BrokerError("权益曲线含非有限值或非正值 —— 记账错了")

    ret = equity["equity"].pct_change().fillna(0.0).to_numpy(dtype=float)
    total_return = e[-1] / e[0] - 1.0
    std = float(np.std(ret, ddof=1)) if len(ret) > 1 else float("nan")
    sharpe = (float(np.mean(ret)) / std * np.sqrt(252)
              if std and np.isfinite(std) and std > 1e-12 else float("nan"))
    peak = np.maximum.accumulate(e)
    max_dd = float(np.min(e / peak - 1.0))

    out = {
        "start": equity["date"].iloc[0], "end": equity["date"].iloc[-1],
        "trades": int(len(trades)), "total_return": float(total_return),
        "sharpe": float(sharpe), "max_dd": max_dd,
        "avg_exposure": float(equity["exposure"].mean()),
        "days": int(len(equity)),
    }
    if trades.empty:
        out.update({"win_rate": float("nan"), "avg_hold_days": float("nan"),
                    "tp_count": 0, "sl_count": 0, "time_count": 0})
        return out

    wins = trades["net_pnl"] > 0
    reasons = trades["exit_reason"]
    out.update({
        "win_rate": float(wins.mean()),
        "avg_hold_days": float(trades["hold_days"].mean()),
        "tp_count": int((reasons.isin(["tp", "tp_gap"])).sum()),
        "sl_count": int((reasons.isin(["sl", "sl_gap"])).sum()),
        "time_count": int((reasons == "max_hold").sum()),
    })
    return out
