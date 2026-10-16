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
from core import spread as spread_mod
from core.preset_contract import resolve
from research import metrics

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
    # sizing_cap = 最终决定股数的那个约束（policy / volume_cap / max_weight /
    # max_heat / cash）。2026-10-17 审计 D3：现金闸门曾在**无日志**的情况下把
    # 声明风险改写成另一个值（12m preset 第二仓从声明 5% 权益掉到 ~0.6%），
    # 使"单笔风险预算"变成记账副作用。记下来之后这条路径是可审计的。
    "sizing_cap",
    # risk_frac = 该笔的计划风险 / 入场时的权益。声明值与实际值并排可读。
    "risk_frac",
    # volume_cap = 入场当日由 §4(a) 量能上限算出的**可成交股数上限**。
    # 它让"这套策略能装多少钱"变成可读的数：capacity_headroom = volume_cap / size，
    # 即"在当前单笔规模上还能放大几倍才会撞到量能上限"。审计 E4/D5 之前，
    # 库和报告里都没有任何量能/容量读数。
    "volume_cap",
    # sl_penetration = 触发止损时"bar 的最低点比止损价低多少"（相对止损价）。
    # 它衡量的是**这条假定有多要紧**：我们按触发价成交，而当日实际到过更低的位置。
    # 只在盘中止损（sl）上有意义；跳空（sl_gap）按开盘价成交，谈不上"穿价"。
    "sl_penetration",
    # spread_cost = 本笔因跨越买卖价差付出的钱（入场侧 + 出场侧）。
    # 它体现在成交价里、不在 fees 里，所以必须单独记一列 —— 否则"成本"这一栏
    # 只显示规费（占建模成本的两成上下），读者会以为滑点/价差不存在。
    "spread_cost",
]

VALID_TIEBREAK = ("sl_first",)

# 相关性闸门的尾部窗口（交易日）。60 个交易日 ≈ 一个季度：
# 短于此会把噪声当信号，长于此会让"最近才变高相关"的两个名字漏过去。
# 它是**实现参数**而不是策略旋钮（preset 只声明 max_pair_corr 这个阈值）。
CORR_LOOKBACK = 60

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
    # 最终压住股数的那个约束（见 TRADE_COLS 里的说明）。
    sizing_cap: str = "policy"
    volume_cap: int = 0
    # 入场时的权益 —— "计划风险占权益多少"的分母。用入场时刻的值，
    # 而不是事后某一天的权益（否则这个数会随时间漂移，无法回答"我当时冒了多大风险"）。
    equity_at_entry: float = float("nan")
    # 入场时（信号日）估到的**单边**价差费率。出场沿用同一个值：
    # 决策时只能知道当时的估计，用出场当日的估计属于偷看未来。
    half_spread: float = 0.0


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
        # 买卖价差估计（Corwin-Schultz，逐日、因果）。用 **as-traded** high/low ——
        # 价差是物理盘口属性，复权价会把它缩放成另一个量。与面板同生命周期缓存，
        # 所以搜索里几千次模拟只算一次。
        # 尾部收益（相关性闸门用；因果：只用 ≤T 的价格）
        rc = data[sym]["raw_close"]
        with np.errstate(divide="ignore", invalid="ignore"):
            data[sym]["ret1"] = np.concatenate([[np.nan], rc[1:] / rc[:-1] - 1.0])
        data[sym]["spread_bps"] = spread_mod.rolling_median(
            spread_mod.corwin_schultz(sub["raw_high"].to_numpy(dtype=float),
                                      sub["raw_low"].to_numpy(dtype=float)))
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

    # 组合层风险预算（2026-10-17 审计 D1/D2）。三者都是"不设 = 关闭"，
    # 但引擎无论如何都会**输出观测值**（单名权重峰值、组合 heat）——
    # 没设上限这件事不许是隐形的。
    max_weight = spec["max_weight"]
    max_heat = spec["max_heat"]
    max_dd_halt = spec["max_drawdown_halt"]
    for _nm, _v in (("max_weight", max_weight), ("max_heat", max_heat)):
        if _v is not None and not (0.0 < float(_v) <= 1.0):
            raise BrokerError(f"{_nm} ∈ (0,1]，得到 {_v!r}")
    if max_dd_halt is not None and not (0.0 < float(max_dd_halt) < 1.0):
        raise BrokerError(f"max_drawdown_halt ∈ (0,1)，得到 {max_dd_halt!r}")
    liquidate_at_end = bool(spec["liquidate_at_end"])
    sl_overshoot = float(spec["sl_overshoot"])
    if not (0.0 <= sl_overshoot <= 1.0):
        raise BrokerError(f"sl_overshoot ∈ [0,1]，得到 {sl_overshoot!r}")
    spread_model = spec["spread_model"]
    if spread_model not in ("none", "corwin_schultz"):
        raise BrokerError(
            f"spread_model 必须是 'none' 或 'corwin_schultz'，得到 {spread_model!r}")
    max_weight = None if max_weight is None else float(max_weight)
    max_heat = None if max_heat is None else float(max_heat)
    max_dd_halt = None if max_dd_halt is None else float(max_dd_halt)

    # 组合内两两相关性上限（审计 D6）。用**尾部相关**而不是外部行业分类：
    #   * 数据驱动、因果（只用 ≤ 信号日的收益）、不需要我们没有的行业数据；
    #   * 只与**已在场**的标的做两两比较（≤2 笔），不是全矩阵 ⇒ 每次入场 O(在持数)，
    #     不用每天算 3028² 的相关矩阵。
    # 语义：相关性高的两个名字在经济上是**同一个仓位**，同时持有等于加倍下注。
    max_pair_corr = spec["max_pair_corr"]
    if max_pair_corr is not None and not (-1.0 <= float(max_pair_corr) <= 1.0):
        raise BrokerError(f"max_pair_corr ∈ [-1,1]，得到 {max_pair_corr!r}")
    max_pair_corr = None if max_pair_corr is None else float(max_pair_corr)

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
    # 组合层风险的状态（D1/D2）：
    #   planned_risk(pos) = 入场时即固定的计划风险；它是 heat 与 E[R] 共同的分子，
    #   所以只能有一个定义（此处），不能在别处再写一遍。
    #   peak_equity 是**已实现路径**的权益峰值，停机判定只用 ≤ 上一收盘的信息。
    def planned_risk(pos: Position) -> float:
        return pos.sl_width * pos.entry_price * pos.size

    def trailing_corr(a: str, b: str, end_idx: int, dates: list) -> float:
        """a 与 b 在**截至信号日**的尾部窗口里的相关系数（对齐到共同日期）。

        对齐必须按**日期**而不是按下标：面板里各标的的日期覆盖不同（晚间上市、
        停牌、退市），按下标对齐会把不同日期的收益配成一对 —— 那是错的相关系数，
        而且错得看不出来。共同观测不足窗口一半 ⇒ 返回 nan（不判定）。
        """
        ka = per_sym.get(a, {})
        kb = per_sym.get(b, {})
        xa, xb = [], []
        d0 = end_idx - CORR_LOOKBACK
        for j in range(max(d0, 0), end_idx + 1):
            d = dates[j]
            ia, ib = ka.get(d), kb.get(d)
            if ia is None or ib is None:
                continue
            ra, rb = data[a]["ret1"][ia], data[b]["ret1"][ib]
            if np.isfinite(ra) and np.isfinite(rb):
                xa.append(ra)
                xb.append(rb)
        if len(xa) < CORR_LOOKBACK // 2:
            return float("nan")
        xa_, xb_ = np.array(xa), np.array(xb)
        if xa_.std() <= 0 or xb_.std() <= 0:
            return float("nan")
        return float(np.corrcoef(xa_, xb_)[0, 1])

    peak_equity = initial_cash
    stats_halted_days = 0
    # 因相关性超限被挡掉的入场次数（审计 D6：这类约束此前完全没有）
    stats_blocked_corr = 0
    # 窗口末尾因"当日无 bar"而**无法**清算的仓位个数（见步骤 3b 的注释）。
    stats_unliquidated = 0
    # 持仓**当日无 bar** 的次数（停牌/退市/数据中断）。审计 A3：这些日子权益是按
    # "最近一次已知收盘"重估的（陈价），此前没有任何读数 —— 于是"这只票到底停了多久"
    # 只能靠人肉看数据。累计位-日与占比进指标。
    stats_stale_days = 0
    # §4(b) 部分成交（出场侧卖不完）的次数。入场侧的对应量是 sizing_cap="volume_cap"。
    partial_exits = 0

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
                       bar_vol: float | None = None,
                       sl_penetration: float | None = None) -> bool:
        """平仓。返回**是否真的成交了**（False = 这一根 bar 根本没法交易）。"""
        nonlocal partial_exits          # §4(b) 部分成交计数（审计 E6）
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
        # §4(a) 量能上限 + §4(b) **部分成交是合法状态**（审计 E6）。
        #
        # 此前的行为是**不对称**的：入场侧超量能就静默缩量（等于 IOC 撤单，现金留在账上），
        # 出场侧却直接 fail-fast。而准则 §4(b) 说得很清楚:部分成交是合法状态,
        # 未成交部分不许排队、也不许抛异常。出场侧的经营含义也不一样 ——
        # 入场可以"decline"（少买点），出场**不能**（止损触发就是必须卖）。
        # 所以出场侧的正确处置是:**卖掉能卖的，剩下的继续持有**，而不抛错。
        # 零量 bar = 那天根本没有成交 ⇒ **不得造出成交**（与入场侧对称：
        # 入场侧早有 `if not (o > 0) or not (bar_vol > 0): continue`）。
        # 此前出场侧漏了这条：bar_vol=0 会让上面的量能上限分支整个被跳过，
        # 于是"一根零量 bar"也能足额卖出 —— 实测用一条测试打出来的（E6 的邻居缺陷）。
        if bar_vol is not None and (not np.isfinite(bar_vol) or bar_vol <= 0):
            return False
        exit_qty = pos.size
        if bar_vol is not None and np.isfinite(bar_vol) and bar_vol > 0:
            cap = int(np.floor(float(bar_vol) * max_participation))
            if pos.size > cap:
                if cap <= 0:
                    # 这一根 bar 完全卖不动（停牌前的零量）—— 那不是"部分成交"，
                    # 是物理上的"卖不掉"。这时**不造假的足额成交**，也不硬崩；
                    # 由调用方按"当日无法出场"处理（顺延到下一根 bar）。
                    raise BrokerError(
                        f"{pos.symbol} {day.date()} 该 bar 量能上限为 0 股,"
                        f"无法卖出任何一股（{reason}）—— 这不是部分成交,是卖不掉")
                exit_qty = cap
                partial_exits += 1
        fill_px = px
        # 卖出同样要跨价差：成交价再低半个价差（只在这笔建仓时估到的那个值）。
        # 放在滑点之前是等价位置 —— 两者都是"成交比触发差"这件成本。
        if pos.half_spread > 0:
            fill_px = fill_px * (1.0 - pos.half_spread)
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
        # ⚠️ 全部按 **exit_qty**（本次真正卖掉的股数）记账，而不是 pos.size ——
        # 部分成交时剩下的股数继续持有（§4(b)），它们的分红与成本不能算进这一笔。
        _frac = exit_qty / pos.size
        div_out = pos.dividends * _frac
        notional_exit = fill_px * exit_qty
        exit_fee = notional_exit * cost_half
        entry_notional = pos.entry_price * exit_qty
        entry_fee = entry_notional * cost_half
        gross = (fill_px - pos.entry_price) * exit_qty
        fees = entry_fee + exit_fee
        # 价差成本单独记账：入场侧付在更高的买入价上，出场侧付在更低的卖出价上。
        # 它已经体现在 gross 里，这里只是把它**显式化**供归因（不重复计费）。
        spread_cost = pos.half_spread * pos.entry_price * exit_qty \
            + pos.half_spread * fill_px * exit_qty
        # 分红是这笔交易的**已实现收入**(§3(c)),必须进净盈亏。
        net = gross + div_out - fees
        cost_basis = entry_notional + entry_fee
        book.cash += notional_exit - exit_fee
        trades.append({
            "entry_date": pos.entry_date, "exit_date": day, "symbol": pos.symbol,
            "exit_reason": reason,
            "entry_price": pos.entry_price, "exit_price": fill_px, "size": exit_qty,
            "exit_trigger": px,
            # bars_held 是**已走完**的 bar 数:入场当日盘中出场 → 0,持满 max_hold → max_hold。
            # 不要写 +1 —— 那会让 max_hold=18 的仓位报出 hold_days=19(实测踩过)。
            "hold_days": pos.bars_held,
            "gross_pnl": gross, "dividends": div_out, "fees": fees, "net_pnl": net,
            "net_return": net / cost_basis if cost_basis > 0 else float("nan"),
            "atr_pct": pos.atr_pct,
            # 计划风险:入场时即固定(止损宽度 × 股数),与最终出场无关。
            # 部分成交时按**卖出的股数**取比例 —— 否则一笔半仓出场的 R 会被系统性放大。
            "volume_cap": int(pos.volume_cap),
            "sl_penetration": (float("nan") if sl_penetration is None
                               else float(sl_penetration)),
            "sl_width": pos.sl_width, "tp_width": pos.tp_width,
            "risk_amount": pos.sl_width * pos.entry_price * exit_qty,
            "spread_cost": spread_cost,
            "sizing_cap": pos.sizing_cap,
            "risk_frac": ((pos.sl_width * pos.entry_price * exit_qty)
                          / pos.equity_at_entry if pos.equity_at_entry > 0
                          else float("nan")),
        })
        if exit_qty < pos.size:
            # §4(b)：剩下的继续持有 —— 不排队（下一次该出场时按当时的 bar 再判）、
            # 也不抛异常。账本里剩下的股数连同它们未领的分红一起留下。
            pos.size -= exit_qty
            pos.dividends -= div_out
            return True          # 部分成交：这次卖出成功，剩下的继续持有
        del book.positions[pos.symbol]
        return True              # 全额成交

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

        # ---- 组合层风险快照（D1/D2）----
        # heat = Σ 计划风险（入场时固定的 止损宽度×入场价×股数）。它衡量的是
        # "若同时全部止损，权益会掉多少" —— 机构里这是组合层的第一约束，
        # 而本引擎此前**完全没有**这个概念（只有 max_positions 数个数）。
        # 与现金/仓位槽一样取**上一交易日收盘**的快照（§2(d) 结算隔离）。
        heat_at_open = sum(planned_risk(book.positions[s_]) for s_ in held_at_open)
        heat_used = heat_at_open
        # 停机开关（D2）：按**上一交易日收盘**权益判回撤，当日盘中回撤最早也要等到
        # 收盘才进入下一日的判定 —— 不偷看当日盘中最低点。
        halted_today = (max_dd_halt is not None
                        and equity_at_open <= peak_equity * (1.0 - max_dd_halt))
        if halted_today:
            stats_halted_days += 1

        # ---------- 1) 持仓:开盘 gap → 盘中 SL/TP ----------
        for sym in list(book.positions):
            pos = book.positions[sym]
            # H2:棘轮。用截至**前一日**的最高价抬高止损位;只用 ≤D-1 的数据,
            # 所以与当日 low 的比较不存在同 bar 前视。止损只会收紧,不会放松。
            if pos.trail_width is not None and np.isfinite(pos.high_water):
                pos.sl_price = max(pos.sl_price, pos.high_water - pos.trail_width)
            k = per_sym.get(sym, {}).get(day)
            if k is None:                      # 当日无数据 → 无法成交,顺延
                stats_stale_days += 1          # 该持仓这一天是按陈价重估的（A3）
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
            # 盘中触发止损的**成交触发价**：sl_price 之下按 θ 向当日最低点靠。
            # θ=0 ⇒ 就是 sl_price（既有行为）。穿透深度同时被记进交易列表，
            # 于是"这条假定值多少"是可读的，而不是只能被相信。
            pen = (pos.sl_price - low_) / pos.sl_price if pos.sl_price > 0 else 0.0
            sl_fill_at = pos.sl_price - sl_overshoot * max(0.0, pos.sl_price - low_)
            if hit_sl and hit_tp:
                # 盘中盲区:两根阈值同 bar 触及 → 按最坏情况(CLAUDE.md §4)
                close_position(pos, day, sl_fill_at, "sl", bar_lo=low_, bar_hi=h, bar_vol=v,
                               sl_penetration=pen)
                continue
            if hit_sl:
                close_position(pos, day, sl_fill_at, "sl", bar_lo=low_, bar_hi=h, bar_vol=v,
                               sl_penetration=pen)
                continue
            if hit_tp:
                close_position(pos, day, pos.tp_price, "tp", bar_lo=low_, bar_hi=h, bar_vol=v)
                continue

        # ---------- 2) 入场:前一交易日(T)的信号在今日(T+1)开盘成交 ----------
        # 预算与容量都取自步骤 0 的快照,不读当日的出场所得与释放的槽位(§2(d))。
        prev_idx = day_pos[day] - 1
        if prev_idx >= 0 and not halted_today:
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
                if max_pair_corr is not None and book.positions:
                    # 与**已在场**的每个标的比尾部相关（审计 D6）。超限就不建并计数 ——
                    # 相关性高的两个名字同时持有，经济上等于加倍下注同一件事。
                    # 窗口右端取**信号日**（prev_idx）：入场日的收益在开盘那一刻还不知道。
                    if any((lambda c: np.isfinite(c) and c > max_pair_corr)(
                            trailing_corr(sym, other, prev_idx, days))
                           for other in list(book.positions)):
                        stats_blocked_corr += 1
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
                # 买入要跨价差：从买方角度看，开盘价之上再付半个价差。
                # 用**信号日**（≤T）的估计 —— 用成交当日的估计属于偷看未来。
                sym_spread = data[sym]["spread_bps"][k - 1] if k >= 1 else float("nan")
                half_spread = (spread_mod.half_spread_rate(sym_spread)
                               if spread_model != "none" else 0.0)
                entry_px = o * (1.0 + slip) * (1.0 + half_spread)

                # ---- 组合层上限（D1/D3）----
                # 顺序施加，并**记下最后压住它的那个约束**。为什么必须记：
                # 此前这里只有一句 min(..., 现金闸门)，于是"声明 5% 权益的单笔风险"
                # 会被静默改写成 0.6%（实测 12m preset 第二仓），而报告里看不出来。
                # 记录之后，"声明风险 vs 实际风险"在交易列表里逐笔可查。
                qty = target_qty(entry_px)
                cap_by = "policy"
                volume_cap = cap
                if qty > volume_cap:
                    qty, cap_by = volume_cap, "volume_cap"
                if max_weight is not None:
                    w_cap = int(np.floor(
                        max_weight * equity_at_open / (entry_px * (1.0 + cost_half))))
                    if qty > w_cap:
                        qty, cap_by = w_cap, "max_weight"
                if max_heat is not None:
                    # 剩余风险额度：heat 上限 - 已用（上一收盘持仓 + 今日已建仓）
                    room = max_heat * equity_at_open - heat_used
                    h_cap = int(np.floor(room / (sl_w * entry_px))) if room > 0 else 0
                    if qty > h_cap:
                        qty, cap_by = h_cap, "max_heat"
                cash_cap = int(np.floor(cash_avail / (entry_px * (1.0 + cost_half))))
                if qty > cash_cap:
                    qty, cap_by = cash_cap, "cash"
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
                # 建仓后立即占用风险额度：同日后续信号的 heat 判定要看得见它。
                heat_used += sl_w * entry_px * qty
                book.positions[sym] = Position(
                    symbol=sym, entry_date=day, entry_price=entry_px, size=qty,
                    sizing_cap=cap_by, equity_at_entry=equity_at_open,
                    half_spread=half_spread, volume_cap=int(volume_cap),
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
                    close_position(pos, day, pos.sl_price - sl_overshoot * max(0.0, pos.sl_price - low_),
                                   "sl", bar_lo=low_, bar_hi=h, bar_vol=bar_vol,
                                   sl_penetration=(pos.sl_price - low_) / pos.sl_price)
                elif low_ <= pos.sl_price:
                    close_position(pos, day, pos.sl_price - sl_overshoot * max(0.0, pos.sl_price - low_),
                                   "sl", bar_lo=low_, bar_hi=h, bar_vol=bar_vol,
                                   sl_penetration=(pos.sl_price - low_) / pos.sl_price)
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

        # ---------- 3b) 窗口末尾清算（C5）----------
        # 为什么必须清算：不清算时，**头条收益/Sharpe/maxDD 来自权益曲线（含未实现
        # 浮盈）**，而 **E[R]/胜率/盈亏比只覆盖已平仓成交** —— 同一份报告里的两组数
        # 落在两个不同的样本上，且窗口末尾刚好浮盈时会系统性地抬高头条。
        # 这里按最后一根收盘价成交（正常收出场成本），于是两组数同源；
        # 代价是"本会继续持有的仓位也付了一次出场成本"，方向保守。
        if liquidate_at_end and day == axis[-1] and book.positions:
            exit_day = axis[-1]
            for sym in list(book.positions):
                k2 = per_sym.get(sym, {}).get(exit_day)
                if k2 is None:
                    # ⚠️ 当日没有该标的的 bar（停牌/退市/数据中断）⇒ **不得**造一个成交。
                    # 准则 §1 的原文:Suspended names are marked at the last raw_close
                    # and **MUST NOT generate fills** until trading resumes。
                    # 所以这里什么都不做，把它留作未平仓，并由 `_window_metrics` 把
                    # "仍有未平仓 ⇒ 两个样本未对齐"**显式报出来** —— 而不是假装清算干净。
                    stats_unliquidated += 1
                    continue
                pos = book.positions[sym]
                if not close_position(pos, exit_day, data[sym]["raw_close"][k2], "eod",
                                      bar_lo=data[sym]["raw_low"][k2],
                                      bar_hi=data[sym]["raw_high"][k2],
                                      bar_vol=data[sym]["volume"][k2]):
                    stats_unliquidated += 1      # 零量 bar：没成交，仍算未清算

        eq_today = mark_to_market(day)
        prev_equity = eq_today            # 供**下一交易日**的入场定预算(§2(d) 快照)
        peak_equity = max(peak_equity, eq_today)
        # 组合层风险的**逐日观测值**：不设上限时它们是唯一的可见性来源，
        # 设了上限时它们是事后复核（上限有没有真的生效）。
        open_risk = sum(planned_risk(p) for p in book.positions.values())
        notional = {}
        for p in book.positions.values():
            k = per_sym.get(p.symbol, {}).get(day)
            px = (data[p.symbol]["raw_close"][k] if k is not None
                  else p.entry_price)
            notional[p.symbol] = p.size * px
        equity_rows.append({"date": day, "equity": eq_today,
                            "cash": book.cash, "n_pos": len(book.positions),
                            "exposure": (eq_today - book.cash) / eq_today,
                            "heat": open_risk / eq_today if eq_today > 0 else float("nan"),
                            "top_weight": (max(notional.values()) / eq_today
                                           if notional and eq_today > 0 else 0.0),
                            "halted": bool(halted_today)})

    equity = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades, columns=TRADE_COLS)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    metrics = _window_metrics(equity, trades_df, max_hold=max_hold)
    # 显式暴露"样本是否对齐"：只有全部仓位都在窗口末尾被真实清算时，
    # 权益口径的头条与交易口径的 E[R] 才落在同一个样本上。
    metrics["n_unliquidated"] = int(stats_unliquidated)
    # 陈旧重估占"持仓-日"的比例：>0 才说明有停牌/退市真的进了回测。
    pos_days = int(equity["n_pos"].sum())
    metrics["stale_position_days"] = int(stats_stale_days)
    metrics["n_blocked_by_corr"] = int(stats_blocked_corr)
    metrics["n_partial_exits"] = int(partial_exits)
    metrics["pct_position_days_stale"] = (stats_stale_days / pos_days) if pos_days else 0.0
    # 判据只有一条：窗口结束时账本里还剩不剩仓位。
    # 剩 ⇒ 权益口径含未实现浮盈、而交易口径只覆盖已平仓 ⇒ 两组数不同源。
    metrics["samples_aligned"] = bool(equity["n_pos"].iloc[-1] == 0)
    return {"trades": trades_df, "equity": equity, "metrics": metrics}


def _window_metrics(equity: pd.DataFrame, trades: pd.DataFrame, *, max_hold: int) -> dict:
    """单窗口指标。total_return / sharpe / max_dd 全部来自**正确的**权益曲线。"""
    if equity.empty:
        raise BrokerError("权益曲线为空")
    e = equity["equity"].to_numpy(dtype=float)
    if not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise BrokerError("权益曲线含非有限值或非正值 —— 记账错了")

    # Sharpe 只有一份定义（research.metrics.annualized_sharpe）—— 此前 broker 自己
    # 又写了一遍（`pct_change().fillna(0)` + √252），与 DSR 里那条序列算出来的
    # 不是同一个数（首日那个填出来的 0 让 n 差 1、均值被稀释）。同一份报告里
    # 两个 Sharpe 并列是"同指标两口径"的老毛病，这里收成一处。
    ret = equity["equity"].pct_change().dropna().to_numpy(dtype=float)
    total_return = e[-1] / e[0] - 1.0
    sharpe = metrics.annualized_sharpe(ret, metrics.TRADING_DAYS)
    # ⚠️ 口径双报（审计 B7）：`sharpe` 把**空仓日**（收益恰好为 0）也算进去 ——
    # 那是"这个策略作为一个账户"的真实收益流，也是 DSR 应当用的序列；
    # 而 `sharpe_invested` 只用**在场日**，回答的是另一个问题：
    # "它的信号在它真在场时有多好"。本仓库的策略敞口常低到个位数百分比，
    # 两者会差很多；只报一个就会被读成另一个。
    expo = equity["exposure"].to_numpy(dtype=float)[1:]
    ret_inv = ret[expo > 0] if len(expo) == len(ret) else ret
    sharpe_invested = metrics.annualized_sharpe(ret_inv, metrics.TRADING_DAYS)
    peak = np.maximum.accumulate(e)
    max_dd = float(np.min(e / peak - 1.0))

    out = {
        "start": equity["date"].iloc[0], "end": equity["date"].iloc[-1],
        "trades": int(len(trades)), "total_return": float(total_return),
        "sharpe": float(sharpe), "max_dd": max_dd,
        "sharpe_invested": float(sharpe_invested),
        "n_days_invested": int((expo > 0).sum()) if len(expo) == len(ret) else int(len(ret)),
        "avg_exposure": float(equity["exposure"].mean()),
        "days": int(len(equity)),
        # ---- 组合层风险的观测值（D1/D2）----
        # 这三项让"组合层有没有失控"变成可读的量，而不是靠人肉推演：
        #   peak_heat        单日同时在场仓位的计划风险峰值（占权益）
        #   peak_name_weight 单名名义额峰值（占权益）
        #   pct_days_halted  停机开关生效的交易日占比（未声明上限时为 0）
        "peak_heat": (float(equity["heat"].max())
                      if "heat" in equity.columns else float("nan")),
        "peak_name_weight": (float(equity["top_weight"].max())
                             if "top_weight" in equity.columns else float("nan")),
        "pct_days_halted": (float(equity["halted"].mean())
                            if "halted" in equity.columns else 0.0),
    }
    if trades.empty:
        out.update({"win_rate": float("nan"), "avg_hold_days": float("nan"),
                    "tp_count": 0, "sl_count": 0, "time_count": 0})
        return out

    # 「声明风险 vs 实际风险」的可审计性（D3）：有多少笔的股数其实是被
    # 量能/单名权重/heat/现金压下来的（即 declaration 与 realized 不一致）。
    # 容量余量（D5/E4）：volume_cap/size = "当前单笔规模还能放大几倍才会撞到量能上限"。
    # 这是"这套策略能装多少钱"唯一可读的答案；小于 1 表示已经被量能上限截断。
    if "volume_cap" in trades.columns and (trades["size"] > 0).all():
        hr = trades["volume_cap"].astype(float) / trades["size"].astype(float)
        out["capacity_headroom_min"] = float(hr.min())
        out["capacity_headroom_median"] = float(hr.median())

    # 止损穿价的分布（E2）：这是「按触发价成交」这条假定的**代价读数**。
    # 只统计盘中止损（sl）：跳空（sl_gap）按开盘价成交，谈不上"穿价"。
    if "sl_penetration" in trades.columns:
        sl = trades.loc[trades["exit_reason"] == "sl", "sl_penetration"].dropna()
        out["median_sl_penetration_bps"] = (float(sl.median()) * 1e4 if len(sl)
                                            else float("nan"))
        out["p90_sl_penetration_bps"] = (float(sl.quantile(0.9)) * 1e4 if len(sl)
                                         else float("nan"))
        out["n_sl"] = int(len(sl))

    if "sizing_cap" in trades.columns:
        capped = trades["sizing_cap"] != "policy"
        out["capped_trades"] = int(capped.sum())
        out["capped_share"] = float(capped.mean())
        if "risk_frac" in trades.columns:
            out["max_risk_frac"] = float(trades["risk_frac"].max())

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
