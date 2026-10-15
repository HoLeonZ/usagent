"""执行代理 —— hiagent/uptrend_pullback/portfolio.py 的**美股子集**。

保留的语义(逐条对应 CLAUDE.md §2/§4):
  * 信号 T 收盘 → **T+1 开盘**入场
  * `intraday_tiebreak = "sl_first"`:同一根 bar 同时触碰 SL 与 TP 时,按**最坏情况**判 SL
  * gap-through-stop:开盘已越过止损 → 按**开盘价**成交(不按止损价)
  * 成交量上限 `Bar_Volume × max_volume_participation`,超出部分 IOC 丢弃、不排队
  * `max_positions` 上限;超容量按 score 降序取,**当 T 丢弃不排队**
  * **结算隔离(§2(d))**:当日入场的预算与容量取自**上一交易日收盘**的账本快照 ——
    同日卖出所得(以及同日出场释放的仓位槽)当日一律不可用。主循环是「先平仓(盘中价)
    再开仓(开盘价)」,开盘在时间上早于盘中,若不隔离就相当于用几小时后才产生的钱
    付几小时前的账。
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

import weakref

from core import costs
from core.preset_contract import resolve

logger = logging.getLogger(__name__)

TRADE_COLS = [
    "entry_date", "exit_date", "symbol", "exit_reason",
    "entry_price", "exit_price", "size", "hold_days",
    # dividends = 持有期内实际领到的现金分红(§3(c))。它必须进 net_pnl,否则
    # Σ net_pnl 与账本现金流对不上,双式记账(§2)断裂。
    "gross_pnl", "dividends", "fees", "net_pnl", "net_return", "atr_pct",
    # exit_trigger = 滑点**前**的触发价。区间不变量(触发价必须落在该 bar 的
    # [raw_low, raw_high] 内)只能盯它 —— 成交价因卖出滑点允许略微越界(§4(c))。
    "exit_trigger",
    # 每笔的**计划风险**宽度与金额。hiagent 的 tail_objective 用固定 sl_pct 推 R,
    # 但 ATR 归一化后每笔的止损宽度不同,所以必须逐笔记录,否则 E[R] 算不出来。
    "sl_width", "tp_width", "risk_amount",
]

VALID_TIEBREAK = ("sl_first",)

EXIT_REASONS = ("sl", "sl_gap", "tp", "tp_gap", "max_hold", "eod")


# 锁「交易列表行为」用的列(CLAUDE.md §6)。sl_width/tp_width 必须在内:它们是
# 「计划风险」的载体且**不是**派生量 —— 实测三倍化 atr_tp_mult 时,若不含这两列,
# v1 的交易列表与哈希逐字不变(因为 tp 从未被触发),护栏对 tp 几何完全盲。
TRADE_LOCK_COLS = ["entry_date", "exit_date", "symbol", "exit_reason",
                   "entry_price", "exit_price", "size", "hold_days",
                   "sl_width", "tp_width", "exit_trigger", "net_pnl"]


def trade_fingerprint(trades: pd.DataFrame) -> str:
    """交易列表的规范 sha256 —— §6 的行为锁。空表 → "EMPTY" 的哈希。

    规范化:按 (entry_date, symbol) 做**稳定**排序(mergesort),浮点固定 6 位。
    浮点不比较容差而是固定格式:这是一把**锁**,要的是逐位可比,不是统计相等。
    """
    import hashlib
    import math
    if trades is None or len(trades) == 0:
        return hashlib.sha256(b"EMPTY").hexdigest()
    t = trades[TRADE_LOCK_COLS].sort_values(["entry_date", "symbol"], kind="mergesort")
    lines = []
    for row in t.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                cells.append(pd.Timestamp(v).strftime("%Y-%m-%d"))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                f = float(v)
                cells.append("nan" if not math.isfinite(f) else f"{f:.6f}")
            else:
                cells.append(str(v))
        lines.append(",".join(cells))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


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
    # 持有期内累计领到的现金分红(§3(c))。出场时计入该笔的净盈亏。
    dividends: float = 0.0
    # H2:移动止损(ratchet)。trail_width=None 表示不移动(原始行为)。
    trail_width: float | None = None
    # 截至**前一交易日**的最高 high。只用 ≤D-1 的数据决定 D 日的止损位 ——
    # 若用 D 日自身的 high 抬高止损再拿 D 日的 low 去触发,就是同 bar 前视。
    high_water: float = float("nan")


@dataclass
class _Book:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)


# _bar_index 的缓存。它以 `id()` 为键会有「对象被回收后 id 被复用」的风险,所以这里用
# **弱引用 + 身份比较**:面板一被回收,弱引用变 None,缓存自动失效,不可能命中错的面板。
# 效果(实测):一次 pooled 调用从 1.43s 降到 ~0.05s(425k 行的 groupby+取数组是纯重复劳动,
# 而搜索里同一份面板要被用几千次)。
_BAR_INDEX_CACHE: list[tuple[weakref.ref, tuple, tuple]] = []


def _bar_index_cached(panel: pd.DataFrame):
    for ref, key, val in _BAR_INDEX_CACHE:
        if ref() is panel:
            return val
    val = _bar_index(panel)
    key = ()
    _BAR_INDEX_CACHE.insert(0, (weakref.ref(panel), key, val))
    del _BAR_INDEX_CACHE[4:]                     # 只留最近 4 个面板
    return val


def _bar_index(panel: pd.DataFrame) -> tuple[dict[str, dict[str, int]], list[pd.Timestamp],
                                             dict[str, dict[str, np.ndarray]]]:
    """把 panel 预处理成 {symbol: {date: **该标的数组内**的下标}} + 全局交易日轴。

    ⚠️ 两条索引轴不可混用:全局轴(day_pos)用来找"上一交易日",
    标的内下标(per_sym)用来取该标的的 OHLCV。日线数据里各标的的日期覆盖
    并不相同(晚间上市、停牌、退市),混用会直接越界或静默取错行。

    ⚠️ 取的是 raw_ 前缀的那一组列(§3(b)):止损止盈触发、跳空判定、逐日市值、现金记账
       一律用 as-traded 价,复权价只喂给数学指标。缺列直接抛错,不静默退回复权价。

    ⚠️ 调用方请走 `_bar_index_cached`:本函数是 O(全部行) 的,而搜索里同一份面板要用几千次。
       缓存的正确性由**弱引用 + 身份比较**保证,行为一致性由 §6 的 golden 哈希基线钉住。
    """
    days = list(pd.Index(sorted(panel["date"].unique())))
    day_pos = {d: i for i, d in enumerate(days)}
    per_sym: dict[str, dict[str, int]] = {}
    cols = ("raw_open", "raw_high", "raw_low", "raw_close",
            "volume", "atr_pct", "dividends")
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise BrokerError(
            f"panel 缺列 {missing} —— 执行与记账必须用 as-traded 价(CLAUDE.md §3(b));"
            f"请用 panel.load_full_panel() 载入,不要手拼面板"
        )
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
    cost_bps_roundtrip: float | None = None,
) -> dict:
    """在 [start, end) 上跑一次回测。

    signals: select_entries 的输出(信号日 = date);
    入场发生在信号日的**下一个交易日**开盘。

    参数一律经 `core.preset_contract.resolve()` 读入 —— 引擎之外没有任何另一条读取
    路径,于是「preset 里声明了却没人读」在结构上不可能(§2(b) 单一来源)。
    `cost_bps_roundtrip` 显式传值只用于成本敏感性分析;默认取 preset 里的口径。
    """
    spec = resolve(preset)
    cost_bps_roundtrip = (float(spec["cost_bps_roundtrip"])
                          if cost_bps_roundtrip is None else float(cost_bps_roundtrip))
    if cost_bps_roundtrip < 0:
        raise BrokerError(f"cost_bps_roundtrip 必须 >= 0,得到 {cost_bps_roundtrip}")

    tiebreak = spec["intraday_tiebreak"]
    if tiebreak not in VALID_TIEBREAK:
        raise BrokerError(f"intraday_tiebreak 必须是 {VALID_TIEBREAK},得到 {tiebreak!r}")

    max_positions = int(spec["max_positions"])
    position_fraction = float(spec["position_fraction"])
    max_participation = float(spec["max_volume_participation"])
    max_hold = int(spec["max_hold"])
    # 退出宽度二选一:固定百分比(hiagent 基线口径)或 ATR 归一化(覆盖固定值)。
    # ATR 归一化的动机是机制性的,不是调参偏好 —— 见 presets.py 与 BASELINE.md:
    # 2.1% 的固定止损在 atr_pct 中位 6.8% 的美股池里只有 0.31 ATR,落在噪声之内。
    # ⚠️ hiagent preset_search §5 M3 曾在 A 股语境下证伪 ATR 归一化;那次证伪的前提
    #    (「2.1% 固定止损在 A 股是对的」)在这里不成立,故这是一次**新检验**。
    atr_sl_mult = spec["atr_sl_mult"]
    atr_tp_mult = spec["atr_tp_mult"]
    sl_pct = spec["sl_pct"]
    tp_pct = spec["tp_pct"]
    atr_slip_scale = float(spec["atr_slip_scale"])
    initial_cash = float(spec["initial_cash"])
    # 仓位政策二选一:
    #   position_fraction = 1.0  → all-in(hiagent §2 的政策)
    #   risk_per_trade           → 风险预算式,qty = equity × risk / (止损宽度 × 价格)
    # 为什么需要后者:all-in 与「窄止损」是**耦合**的。hiagent §2 的原话是
    #   "Tail risk is managed entirely by exit logic (TP/SL), never by reserving
    #    cash at entry" —— 这句话只在止损很窄时成立。A 股 2.1% 止损下 all-in
    # 每次亏 2.1% 权益;把止损加宽到 1 ATR(美股中位 6.79%)后,每次都亏 6.79%,
    # 连亏 20 次从 −34.6% 变成 −75.5%。风险预算式让单笔亏损与止损宽度**脱钩**。
    risk_per_trade = spec["risk_per_trade"]
    if risk_per_trade is not None:
        risk_per_trade = float(risk_per_trade)
        if not (0 < risk_per_trade < 1):
            raise BrokerError(f"risk_per_trade ∈ (0,1),得到 {risk_per_trade}")

    # H2:移动止损。True 时止损随最高价上移(棘轮),距离沿用初始止损宽度
    # (= atr_sl_mult × atr_pct),因此**不引入新的待调参数**。
    # 动机是机制性的:BASELINE.md §1D 实测 ONDS 入场后 +44.4%、ADTN +27.1%、
    # VERI +25.0% 最后全部亏着出场 —— 固定价位止损 + 不可达 TP = 只止损不止盈。
    trailing_sl = bool(spec["trailing_sl"])

    if max_positions < 1:
        raise BrokerError(f"max_positions 必须 >= 1,得到 {max_positions}")
    if not (0 < position_fraction <= 1.0):
        raise BrokerError(f"position_fraction ∈ (0,1],得到 {position_fraction}")
    if atr_slip_scale < 0:
        raise BrokerError(f"atr_slip_scale 必须 >= 0,得到 {atr_slip_scale}")
    if not (0 < max_participation <= 1.0):
        # >1 等于允许买超整根 bar;<0 会让滑点公式里的 sqrt 产生 NaN(§0 Fail-Fast)
        raise BrokerError(
            f"max_volume_participation 必须 ∈ (0,1],得到 {max_participation}")

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

    per_sym, days, data = _bar_index_cached(panel_ind)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    axis = [d for d in days if lo <= d < hi]
    if not axis:
        raise BrokerError(f"窗口 {start}..{end} 内没有任何交易日")

    cost_half = costs.half_rate(cost_bps_roundtrip)

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
    # 上一交易日收盘的权益 —— 当日入场的风险预算只能用它(见步骤 0 的快照)。
    prev_equity = initial_cash

    def mark_to_market(day: pd.Timestamp) -> float:
        """equity = cash + Σ qty × close。这是唯一正确的口径(见模块 docstring)。

        市值用 **as-traded** 收盘价(§3(b)):账本里的现金是物理现金,它只能被物理价格
        估值。旧实现用复权价,于是「现金 + 市值」这条恒等式里的价格与现金不同源。
        """
        total = book.cash
        for pos in book.positions.values():
            k = per_sym.get(pos.symbol, {}).get(day)
            if k is None:
                # 当日无该标的数据(停牌/退市):按最近一次已知收盘 mark
                k = max((j for dd, j in per_sym[pos.symbol].items() if dd <= day),
                        default=None)
            if k is None:
                raise BrokerError(f"{pos.symbol} 在 {day.date()} 前无任何数据,无法 mark")
            total += pos.size * data[pos.symbol]["raw_close"][k]
        return total

    def close_position(pos: Position, day: pd.Timestamp, px: float, reason: str,
                       *, bar_lo: float | None = None, bar_hi: float | None = None,
                       bar_vol: float | None = None) -> None:
        """平仓。`px` 是**触发价**,本函数负责把它变成成交价。

        三条纪律:
        * **触发价**必须落在该 bar 的 as-traded [low, high] 内(§0 Fail-Fast)。各条路径
          其实**可证**:gap 分支按开盘价成交(开盘必在区间内);盘中分支的触发条件本身
          保证 low ≤ sl_price 或 tp_price ≤ high;max_hold 按收盘价。滑点**后**的成交价
          允许略微越界 —— 滑点就是「成交比触发差」这件成本本身,与入场侧同理。
        * **卖出滑点**(§4(c))= scale × 信号日 atr_pct × sqrt(止损股数 / 该 bar 成交量),
          **压低**成交价。不建模它会让每笔止损恒等于 −1R,而 R 分布退化会让搜索
          系统性偏爱窄止损(窄止损的损失被机械封顶,真实成交的相对损耗反而更大)。
        * **出场量超过该 bar 的量能上限** ⇒ 报错。§4(a) 的 IOC 部分成交**未实现**
          (实测余量 24~127 倍,永不咬合),所以这里做成**显式边界**:真发生就 fail-fast,
          绝不给出一个假的足额成交。
        """
        if bar_lo is not None and bar_hi is not None:
            if not (bar_lo - 1e-9 <= px <= bar_hi + 1e-9):
                raise BrokerError(
                    f"{pos.symbol} {day.date()} 触发价 {px:.6f} 落在该 bar 的 "
                    f"[{bar_lo}, {bar_hi}] 之外 —— 物理上不可能({reason})"
                )
        # §4(a) 量能上限:**与滑点无关**,是独立的一条要求。
        if bar_vol is not None and np.isfinite(bar_vol) and bar_vol > 0:
            cap = int(np.floor(float(bar_vol) * max_participation))
            if pos.size > cap:
                raise BrokerError(
                    f"{pos.symbol} {day.date()} 要卖 {pos.size} 股 > 该 bar 量能上限 {cap} 股 —— "
                    f"§4(a) 的 IOC 部分成交**未实现**(实测余量 24~127 倍,永不咬合);"
                    f"此处 fail-fast 而不是给出一个足额的假成交({reason})"
                )
        fill_px = px
        if atr_slip_scale > 0:
            ok = (bar_vol is not None and np.isfinite(bar_vol) and bar_vol > 0
                  and np.isfinite(pos.atr_pct) and pos.atr_pct > 0)
            if not ok:
                raise BrokerError(
                    f"atr_slip_scale={atr_slip_scale} 需要有效的 atr_pct 与该 bar 成交量,"
                    f"得到 atr={pos.atr_pct!r} vol={bar_vol!r} —— 滑点不得静默退化为 0(§0)"
                )
            fill_px = px * (1.0 - atr_slip_scale * pos.atr_pct
                            * float(np.sqrt(pos.size / float(bar_vol))))
        notional_exit = fill_px * pos.size
        exit_fee = notional_exit * cost_half
        entry_notional = pos.entry_price * pos.size
        entry_fee = entry_notional * cost_half
        gross = (fill_px - pos.entry_price) * pos.size
        fees = entry_fee + exit_fee
        # 分红是这笔交易的**已实现收入**(§3(c)),必须进净盈亏。
        net = gross + pos.dividends - fees
        cost_basis = entry_notional + entry_fee
        book.cash += notional_exit - exit_fee
        trades.append({
            "entry_date": pos.entry_date, "exit_date": day, "symbol": pos.symbol,
            "exit_reason": reason,
            "entry_price": pos.entry_price, "exit_price": fill_px, "size": pos.size,
            "exit_trigger": px,
            # bars_held 是**已走完**的 bar 数:入场当日盘中出场 → 0,持满 max_hold → max_hold。
            # 不要写 +1 —— 那会让 max_hold=18 的仓位报出 hold_days=19(实测踩过)。
            "hold_days": pos.bars_held,
            "gross_pnl": gross, "dividends": pos.dividends, "fees": fees, "net_pnl": net,
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
        # ---------- 0) 当日入场的账本快照(CLAUDE.md §2(d) Settlement Isolation)----------
        # 铁律:do NOT assume funds from a sell order at T are available for a buy order at T.
        # 当日入场的预算与容量只能来自**上一交易日收盘**的账本状态。快照必须在步骤 1 之前取:
        # 步骤 1 用当日**盘中**价平仓,而步骤 2 用当日**开盘**价建仓 —— 开盘在时间上早于
        # 盘中,所以步骤 1 释放出来的现金与仓位槽,在开盘那一刻都不存在。
        # 2026-10-15 审计实测:旧实现让 12m preset 的 3/13 笔靠这笔钱成交(其中一笔用
        # 一小时后才发生的盘中止损回款,付了当日开盘的入场款)。
        cash_at_open = book.cash                  # 上一交易日收盘的现金
        held_at_open = set(book.positions)        # 上一交易日收盘的持仓
        equity_at_open = prev_equity              # 上一交易日收盘的权益
        slots_at_open = max_positions - len(held_at_open)
        spent_today = 0.0
        filled_today = 0

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
            # 现金分红显式入账(§3(c) Event-Sourced Corporate Actions)。位置放在步骤 1
            # 的**开头**,领息资格以「开盘时在场」为准,两个边界因此都对:
            #   * 除息日当天才建仓 ⇒ 不领(仓位在步骤 2 才产生)
            #   * 除息日开盘就 gap 止损 ⇒ **要领**(卖在除息日 = 含息卖出)
            # 拆股**不在这里处理**:库里的价本身已是拆股后价位
            # (src/series.py::source_is_split_adjusted 已实测),再乘一次就是重复缩放。
            div = float(data[sym]["dividends"][k])
            if div > 0:
                cash_div = pos.size * div
                book.cash += cash_div
                pos.dividends += cash_div
            o = data[sym]["raw_open"][k]
            h = data[sym]["raw_high"][k]
            low_ = data[sym]["raw_low"][k]
            v = data[sym]["volume"][k]
            if not (o > 0 and h > 0 and low_ > 0):
                continue

            if o <= pos.sl_price:              # gap 跳过止损 → 按开盘成交
                close_position(pos, day, o, "sl_gap", bar_lo=low_, bar_hi=h, bar_vol=v)
                continue
            if o >= pos.tp_price:              # gap 跳过止盈 → 按开盘成交
                close_position(pos, day, o, "tp_gap", bar_lo=low_, bar_hi=h, bar_vol=v)
                continue

            hit_sl = low_ <= pos.sl_price
            hit_tp = h >= pos.tp_price
            if hit_sl and hit_tp:
                # 盘中盲区:两根阈值同 bar 触及 → 按最坏情况(CLAUDE.md §4)
                close_position(pos, day, pos.sl_price, "sl", bar_lo=low_, bar_hi=h, bar_vol=v)
                continue
            if hit_sl:
                close_position(pos, day, pos.sl_price, "sl", bar_lo=low_, bar_hi=h, bar_vol=v)
                continue
            if hit_tp:
                close_position(pos, day, pos.tp_price, "tp", bar_lo=low_, bar_hi=h, bar_vol=v)
                continue

        # ---------- 2) 入场:前一交易日(T)的信号在今日(T+1)开盘成交 ----------
        # 预算与容量都取自步骤 0 的快照,不读当日的出场所得与释放的槽位(§2(d))。
        prev_idx = day_pos[day] - 1
        if prev_idx >= 0:
            sig_day = days[prev_idx]
            cash_avail = cash_at_open
            for sig in pending.get(sig_day, []):
                if filled_today >= slots_at_open:
                    break                       # 超容量当 T 丢弃,不排队
                sym = sig["symbol"]
                if sym in held_at_open:
                    # 开盘时仍持有(它当天盘中才出场)⇒ 不得在当日再次建仓,
                    # 否则等于用一根尚未走完的 bar 同时做两件事。
                    continue
                k = per_sym.get(sym, {}).get(day)
                if k is None:
                    continue
                o = data[sym]["raw_open"][k]
                bar_vol = data[sym]["volume"][k]
                if not (o > 0) or not (bar_vol > 0):
                    continue                    # 无法成交(停牌/零量)
                h_entry = float(data[sym]["raw_high"][k])
                low_entry = float(data[sym]["raw_low"][k])

                sym_atr = float(sig.get("atr_pct", np.nan))
                if atr_slip_scale > 0 and not (np.isfinite(sym_atr) and sym_atr > 0):
                    raise BrokerError(
                        f"atr_slip_scale={atr_slip_scale} 需要有效的信号日 atr_pct,"
                        f"得到 {sym_atr!r} —— 滑点不得静默退化为 0(CLAUDE.md §0 Fail-Fast)"
                    )
                sl_w, tp_w = exit_widths(sym_atr)
                cap = int(np.floor(bar_vol * max_participation))

                def target_qty(px: float) -> int:
                    """给定成交价下的目标股数(风险预算 / all-in 两种口径)。"""
                    if risk_per_trade is not None:
                        budget_ = equity_at_open * risk_per_trade
                        return int(np.floor(budget_ / (sl_w * px)))
                    return int(np.floor(cash_avail * position_fraction / (px * (1.0 + cost_half))))

                # 滑点 = f(ATR, 参与率)(CLAUDE.md §4(c)):平方根冲击律
                #     slip = scale × atr_pct × sqrt(参与率)
                # 参与率 = 本单股数 / 本 bar 成交量。冲击是参与率的函数、参与率又取决于
                # 股数,这里做**一阶**近似:先用未含冲击的开盘价估一版股数算参与率,
                # 再据此定价、按定价重算股数。滑点只让股数变小 ⇒ 二阶不会反过来放大参与率,
                # 所以一阶是保守的。参与率先按量能上限截断,因此滑点天然有上界
                # scale × atr_pct × sqrt(max_volume_participation)。
                qty_pre = min(target_qty(o), cap)
                participation = (qty_pre / bar_vol) if bar_vol > 0 else 0.0
                slip = (atr_slip_scale * sym_atr * float(np.sqrt(max(participation, 0.0)))
                        if atr_slip_scale > 0 else 0.0)
                # ⚠️ 已知建模简化:成交价是「开盘价 × (1+滑点)」,**不按当日 bar 区间钳制**。
                #    当滑点加成大到超过当日最高价时(实测 v1 有 2/41 笔、12m 有 3/15 笔,
                #    超出量 ≤0.13%),报出的入场价会略高于该 bar 的最高价。
                #    不钳制的理由:钳制等于在窄幅 bar 上把 §4(c) 要求的滑点**整体抹掉**
                #    (high==open 时滑点归零),那是用一个更大的错误换一个小错误。
                #    另一侧的 gap 修复(见开仓当日的 §4(f) 分支)保证了**触发型**成交
                #    (止损/止盈/跳空)永远落在 bar 区间内 —— 那才是会产生「不存在的成交」的一侧。
                entry_px = o * (1.0 + slip)

                # 现金闸门(两种口径都要):风险预算可能算出一个买不起的股数
                qty = min(target_qty(entry_px), cap,
                          int(np.floor(cash_avail / (entry_px * (1.0 + cost_half)))))
                if qty <= 0:
                    continue

                notional = entry_px * qty
                fee = notional * cost_half
                book.cash -= (notional + fee)
                spent_today += notional + fee
                filled_today += 1
                cash_avail = cash_at_open - spent_today      # 当日预算的剩余额度
                if book.cash < -1e-9:
                    raise BrokerError(
                        f"{day.date()} 入场 {sym} 后现金为负({book.cash:.6f})—— "
                        f"闸门漏了,物理上不可能"
                    )
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
                h, low_ = h_entry, low_entry
                # 与步骤 1 对称:开盘已越过阈值 ⇒ 按**开盘价**成交(§4(f) gap-through-stop)。
                # 入场价含滑点,所以这里要拿**未含滑点**的当日开盘价 o 去比。
                # 2026-10-15 复核实测:此前这条路径没有 gap 分支,于是在 [100,100] 的 bar 上
                # 报出 102.13 的成交价 —— 高于最高价,物理上不存在。
                if low_ <= pos.sl_price and o <= pos.sl_price:
                    close_position(pos, day, o, "sl_gap", bar_lo=low_, bar_hi=h, bar_vol=bar_vol)
                elif h >= pos.tp_price and o >= pos.tp_price:
                    close_position(pos, day, o, "tp_gap", bar_lo=low_, bar_hi=h, bar_vol=bar_vol)
                elif low_ <= pos.sl_price and h >= pos.tp_price:
                    close_position(pos, day, pos.sl_price, "sl", bar_lo=low_, bar_hi=h, bar_vol=bar_vol)
                elif low_ <= pos.sl_price:
                    close_position(pos, day, pos.sl_price, "sl", bar_lo=low_, bar_hi=h, bar_vol=bar_vol)
                elif h >= pos.tp_price:
                    close_position(pos, day, pos.tp_price, "tp", bar_lo=low_, bar_hi=h, bar_vol=bar_vol)

        # ---------- 3) 收盘:更新最高价(H2)、持仓天数与 max_hold ----------
        for sym in list(book.positions):
            pos = book.positions[sym]
            k = per_sym.get(sym, {}).get(day)
            if k is not None and pos.trail_width is not None:
                # 收盘后记录今日最高价 —— 供**下一交易日**抬高止损(见 §1 的棘轮)
                pos.high_water = max(pos.high_water, data[sym]["raw_high"][k])
            pos.bars_held += 1
            if pos.bars_held >= max_hold:
                if k is None:
                    continue
                close_position(pos, day, data[sym]["raw_close"][k], "max_hold",
                               bar_lo=data[sym]["raw_low"][k],
                               bar_hi=data[sym]["raw_high"][k],
                               bar_vol=data[sym]["volume"][k])

        eq_today = mark_to_market(day)
        prev_equity = eq_today            # 供**下一交易日**的入场定预算(§2(d) 快照)
        equity_rows.append({"date": day, "equity": eq_today,
                            "cash": book.cash, "n_pos": len(book.positions),
                            "exposure": (eq_today - book.cash) / eq_today})

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
