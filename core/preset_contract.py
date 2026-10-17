"""preset 的执行契约 —— 「引擎会读哪些键」的唯一声明(CLAUDE.md §2(a)/§0)。

为什么需要这一层:2026-10-15 审计发现同一条回测链路的两个入口给 broker 喂了不同的键 ——
render_report 用一份 7 键白名单静默丢掉 broker 实际会读的 5 个执行键,同一个 preset 因此
跑出两条曲线(实测差 45.5pp / 4 笔成交)。根因不是那行白名单写错,而是仓库里**没有一份
「引擎读哪些键」的声明**:丢键、拼错键、写了没人读的键,全都无声通过。

契约把 preset 的键分成三类:
  * `ENGINE_KEYS`  —— 引擎(broker.simulate)会读的执行参数。`resolve()` 返回且**只**
    返回这些键 ⇒「声明了却没人读」在结构上不可能发生;其中 `REQUIRED_KEYS` 缺一即
    fail-fast(§2(a) 明确把「preset 让 max_positions / position_fraction 隐式化」定义为
    contract violation)。
  * `METADATA_KEYS` —— 由编排层或文档消费的键(signal / regime / universe / verdict …)。
  * 其它 —— 一律拒绝。典型是拼错的执行键(`atr_slip_scal`),静默忽略它正是 H10 类缺陷。

⚠️ 已知的诚实缺口:`universe` 是元数据 —— **引擎不读它**。回测股票池实际等于「DB 里
   恰好有什么」(panel.load_full_panel 的 SELECT DISTINCT symbol)。所以改 preset 的
   universe 不会改变任何数字,而 S3(a) 要求的 PIT 取池接口在本仓库尚不存在。
"""
from __future__ import annotations

from core import costs

# 必填:缺一即 contract violation(§2(a) 点了 max_positions / position_fraction;
# 其余几个是 §4 明令必须显式建模的项)。
REQUIRED_KEYS: tuple[str, ...] = (
    "max_positions",
    "position_fraction",
    "atr_slip_scale",
    "intraday_tiebreak",
    "max_hold",
    "cost_bps_roundtrip",
)

# 可选:有缺省来源(见 DEFAULTS),或本身就是「不设 = 关闭」的开关。
OPTIONAL_KEYS: tuple[str, ...] = (
    "initial_cash",
    "risk_per_trade",
    "max_volume_participation",
    "atr_sl_mult", "atr_tp_mult",
    "sl_pct", "tp_pct",
    "trailing_sl",
    # ---- 组合层风险（2026-10-17 审计 D1/D2，机构风控的核心）----
    # 三者都是「不设 = 关闭」，但**不设必须被看见**：引擎会在报告里输出
    # 实际观测到的单名权重峰值与组合 heat，所以"没设上限"不会变成一个隐形事实。
    "max_weight",          # 单名名义额 / 权益 上限
    "max_heat",            # 组合计划风险（Σ 止损宽度×价格×股数）/ 权益 上限
    "max_drawdown_halt",   # 自峰值回撤超过它 ⇒ 停止开新仓（停机开关）
    # 组合内**两两相关性**上限（数据驱动，无外部行业分类）：新仓与任一在持标的的
    # 尾部相关超过它就不建。见 broker 里的实现说明（审计 D6：此前完全没有相关性约束）。
    "max_pair_corr",
    # regime 翻转时**出场侧**怎么办（审计 D4）。"none" = 只挡入场（既有行为）；
    # "close" = 在 regime 失效后的下一个开盘平掉全部持仓（组合层降杠杆）。
    "regime_exit_policy",
    # ---- 止损穿价深度（2026-10-17 审计 E2）----
    # 盘中触发止损时的成交位置：fill_trigger = sl_price − θ × (sl_price − bar_low)。
    # θ=0 是既有行为（假定止损价上无限流动性）；θ=1 是"按当日最低价成交"，
    # 那是**系统性造错**（最低价可能出现在触发之后），只作为敏感性的悲观端点。
    # 与 atr_slip_scale 不重复：后者是**我们自己的冲击**，这里描述的是
    # **市场在我们下单那一刻的路径**（止损单穿过价位时的成交质量）。
    "sl_overshoot",
    # ---- 评价口径（2026-10-17 审计 C5）----
    # 窗口末尾是否按最后一根收盘价清算未平仓仓位（True/False）。
    # 见 broker.py 的说明：不平仓会让"权益口径的头条收益"与"交易口径的 E[R]"
    # 落在两个不同的样本上。
    "liquidate_at_end",
    # 入场价是否钳制到当日 bar 区间内（审计 E7）。默认 False = 保留滑点（见 broker 里
    # 那段"已知建模简化"的取舍说明）；打开则把"不可能的成交价"消掉，代价是窄幅 bar 上
    # 滑点被抹平。**两种都可用，但必须显式选**，而不是默默留着不可能的价格。
    "clamp_entry_to_bar",
    # ---- 成本口径（2026-10-17 审计 E1）----
    # "none"（默认）| "corwin_schultz"。默认 none 只为不改动既有数字，
    # 不是因为它更对：不建模价差等于把**尺寸无关**的那一半成本当成零。
    "spread_model",
    # ---- 事件维度（2026-10-17，见 docs/2026-10-17-event-preset-presearch.md）----
    # 这三个由**信号层**读（不是 broker），但放在这里：它们会改变**信号集**，
    # 而"报告里看不见的旋钮"正是这一层要根除的东西（审计 E5）。放在 ENGINE_KEYS
    # 而不是 METADATA_KEYS 的唯一理由就是**它必须出现在口径头里** —— 读者要能看出
    # 这份数字是在"剔除了财报窗"还是"没剔除"的信号上跑出来的。
    # 此前 `avoid_earnings` 只能由 CLI 的 --avoid-earnings 打开，preset 里**无法声明**，
    # 而 `avoid_earnings_forward_calendar_days` 却是合法的元数据键 —— 即"声明了却没人读"
    # 的又一处残留。
    "avoid_earnings",                        # 持有窗内撞上财报 ⇒ 不出信号
    "avoid_earnings_forward_calendar_days",  # 前瞻窗长度（自然日），应覆盖 max_hold
    "require_earnings_within_trading_days",  # 只做「距上一份财报 ≤ N 个交易日」的信号
)

ENGINE_KEYS: tuple[str, ...] = REQUIRED_KEYS + OPTIONAL_KEYS

# 缺省值只在 core/ 里出现一次(§2(b))。
DEFAULTS: dict[str, object] = {
    "initial_cash": costs.INITIAL_CASH,
    "max_volume_participation": costs.MAX_VOLUME_PARTICIPATION,
    "trailing_sl": False,
    # 组合层风控默认**关闭**：缺省关闭是为了不改动既有 preset 的行为，
    # 不是因为"没有上限更好"。观测值（单名权重峰值/heat）无论如何都会输出。
    "max_weight": None,
    "max_heat": None,
    "max_drawdown_halt": None,
    "max_pair_corr": None,
    "regime_exit_policy": "none",
    "spread_model": "none",
    # 默认清算：它让头条收益与 E[R]/胜率/盈亏比落在**同一个样本**上，
    # 而且方向保守（多付一次出场成本）。不想要就显式写 False。
    "liquidate_at_end": True,
    "clamp_entry_to_bar": False,
    "sl_overshoot": 0.0,
    # ---- 事件维度：默认**全关**（不改动既有 4 条 preset 的行为与 7 个哈希）----
    "avoid_earnings": False,
    # 26 自然日 ≈ max_hold=18 个交易日。用自然日是因为日历是按自然日组织的；
    # 直接用交易日数当自然日会漏掉窗口尾部一周（见 signals.filter_earnings_window）。
    "avoid_earnings_forward_calendar_days": 26,
    "require_earnings_within_trading_days": None,
}

# 指标价轴是**引擎约定,不是 preset 旋钮**:CLAUDE.md §3(b) 规定数学指标必须用
# 前复权(分红回补)价,as-traded 价只用于执行与记账。曾把 `price_adjust` 放进 preset,
# 结果是「一个入口读它、另外四个入口静默忽略」——正好是这套契约要根除的那类缺陷,
# 所以这个键已被删除(2026-10-15 第三层复核)。
INDICATOR_PRICE_ADJUST: str = "total"

# 引擎不读、但链路里有明确消费者的键。
METADATA_KEYS: frozenset[str] = frozenset({
    "signal", "regime",                      # 编排层(signals.select_entries / regime)
    # 走 families.py 的族(不走 signals.select_entries)。编排层会读它们;
    # broker 侧仍然只吃 ENGINE_KEYS,所以执行契约不变。
    "family", "family_params",
    "universe", "universe_note",             # 口径声明(引擎不读,见模块 docstring)
    "n_comparisons", "validated", "verdict", "notes",   # §5 纪律与负面证据
    "position_sizing",                       # 派生量,声明值必须与键一致(见 validate)
    "earnings_filter_validated",              # 声明"这个过滤器的口径已被人工核对过"
    "trailing_sl_validated",
})


# 每个引擎键在"口径头"里怎么显示（标签, 值格式化函数）。
# ⚠️ 这张表必须**覆盖全部 ENGINE_KEYS** —— research/tests/test_contract.py 会断言
# 这一点。加了一个新键却忘了写进这里，就等于又造出一个"报告里看不见的旋钮"，
# 而那正是这一层要根除的东西（审计 E5：`atr_slip_scale=0` 可以合法地把两侧滑点
# 静默关掉，读者从报告里看不出执行口径到底是什么）。
SPEC_LABELS: dict[str, tuple[str, "callable"]] = {
    "cost_bps_roundtrip": ("规费", lambda v: f"{float(v):g}bp 双边"),
    "spread_model": ("价差", lambda v: "Corwin-Schultz 日线估计" if v == "corwin_schultz"
                     else f"{v}（**未建模买卖价差**）"),
    "atr_slip_scale": ("冲击滑点", lambda v: (f"{float(v):g}×ATR×√参与率"
                        if float(v) > 0 else "**关闭**（两侧滑点均为 0）")),
    "sl_overshoot": ("止损穿价", lambda v: (f"按触发价（θ=0，穿价仅记录）"
                      if float(v) == 0 else f"θ={float(v):g} 向当日最低点靠")),
    "max_volume_participation": ("量能上限", lambda v: f"单 bar {float(v):.0%}"),
    "initial_cash": ("初始权益", lambda v: f"${float(v):,.0f}"),
    "risk_per_trade": ("单笔风险预算", lambda v: ("—（走 position_fraction）"
                        if v is None else f"{float(v):.2%} 权益")),
    "position_fraction": ("单笔仓位", lambda v: f"{float(v):.0%} 可用现金"),
    "max_positions": ("最大持仓", lambda v: f"{int(v)} 只"),
    "max_weight": ("单名上限", lambda v: ("**未设**" if v is None else f"{float(v):.0%} 权益")),
    "max_heat": ("组合热风险上限", lambda v: ("**未设**" if v is None else f"{float(v):.0%} 权益")),
    "max_drawdown_halt": ("回撤停机", lambda v: ("**未设**" if v is None else f"{float(v):.0%}")),
    "clamp_entry_to_bar": ("入场价钳制",
                           lambda v: "钳到 bar 区间内（滑点在窄 bar 上被抹平）" if v
                           else "不钳制（保留滑点，可能报出高于 bar 高点的价）"),
    "regime_exit_policy": ("regime 翻转时", lambda v: "只挡入场" if v == "none"
                           else "平掉全部持仓（组合层降杠杆）"),
    "max_pair_corr": ("组合内相关性上限",
                      lambda v: ("**未设**" if v is None else f"两两 ≤ {float(v):.2f}")),
    "atr_sl_mult": ("止损宽度", lambda v: "—" if v is None else f"{float(v):g}×ATR"),
    "atr_tp_mult": ("止盈宽度", lambda v: "—" if v is None else f"{float(v):g}×ATR"),
    "sl_pct": ("止损宽度", lambda v: "—" if v is None else f"{float(v):.2%}"),
    "tp_pct": ("止盈宽度", lambda v: "—" if v is None else f"{float(v):.2%}"),
    "max_hold": ("最长持有", lambda v: f"{int(v)} 个交易日"),
    "intraday_tiebreak": ("同 bar 双触发", lambda v: f"{v}（按最坏情况）"),
    "trailing_sl": ("移动止损", lambda v: "开" if v else "关"),
    "liquidate_at_end": ("窗口末尾", lambda v: ("按收盘清算（两组数同源）" if v
                          else "**不清算**（头条含未实现、与 E[R] 不同源）")),
    "avoid_earnings": ("事件过滤", lambda v: ("剔除持有窗内撞上财报的信号"
                        if v else "关（不剔除）")),
    "avoid_earnings_forward_calendar_days": ("事件前瞻窗", lambda v: f"{int(v)} 自然日"),
    "require_earnings_within_trading_days": (
        "事件刚过", lambda v: "关" if v is None else f"距上一份财报 ≤ {int(v)} 个交易日"),
}


def describe_engine_spec(preset: dict) -> list[tuple[str, str]]:
    """把生效的执行口径翻成 [(标签, 值)] —— **每个报告入口都打印同一份**。

    存在的理由（审计 E5 及其同类）:报告只报数字，不报"这些数字是在什么成本/风险假设
    下产生的"。读者要么去读 preset 源码，要么假设默认值 —— 而默认值恰好可能意味着
    "价差没建模""滑点被关掉""组合层没有上限"。口径必须**自报**，而且只能有一份
    （否则又是一次"两个入口两套口径"）。
    """
    spec = resolve(preset)
    out: list[tuple[str, str]] = []
    for k in ENGINE_KEYS:
        if k not in SPEC_LABELS:
            raise ValueError(f"引擎键 {k!r} 没有在 SPEC_LABELS 里登记 ⇒ 它会是报告里"
                             f"看不见的旋钮。请补上标签与格式化。")
        label, fmt = SPEC_LABELS[k]
        v = spec.get(k)
        if k in ("sl_pct", "tp_pct") and v is None:
            continue                     # 与 atr_* 互斥，未用的那套不显示
        if k in ("atr_sl_mult", "atr_tp_mult") and v is None:
            continue
        out.append((label, fmt(v)))
    return out


def spec_header(preset: dict) -> str:
    """口径头的一行文本版（CLI 用；HTML 报告用 describe_engine_spec 自行排版）。"""
    return " | ".join(f"{lab} {val}" for lab, val in describe_engine_spec(preset))


def exit_scheme(preset: dict) -> str:
    """这份 preset 用的是哪套退出几何：`"atr"`（ATR 归一化）还是 `"pct"`（固定百分比）。

    为什么要有一个函数（审计 G5）：契约**允许**两套（`validate` 强制恰好一套），
    但下游有些地方**只认 ATR 那套**（`render_report` 直接读 `preset["atr_sl_mult"]`
    ⇒ 一个合法的 pct preset 会让它 KeyError）。把"哪一套"变成显式查询之后，
    下游就不可能再默默假设只有一套。
    """
    spec = resolve(preset)
    if spec["atr_sl_mult"] is not None:
        return "atr"
    if spec["sl_pct"] is not None:
        return "pct"
    raise ValueError("preset 的退出几何既不是 atr_* 也不是 pct_* —— 契约校验漏了")


def derived_position_sizing(preset: dict) -> str:
    """实际生效的仓位政策 —— 由**键**派生,不由声明字段决定。

    这正是审计抓到的那处隐性耦合:改 `position_sizing` 的值毫无效果,
    改 `risk_per_trade` 这个键才会换政策。

    ⚠️ 必须同时看 `position_fraction`:只看 risk_per_trade 会让
       `{position_fraction: 0.5}` 这种配置被标成 "all_in" —— 标签撒谎。
    """
    if preset.get("risk_per_trade") is not None:
        return "risk_budget"
    return "all_in" if float(preset.get("position_fraction") or 0.0) == 1.0 else "fixed_fraction"


def validate(preset: dict) -> None:
    """校验一份 preset 是否满足执行契约。违规 ⇒ ValueError(fail-fast,§0)。"""
    missing = [k for k in REQUIRED_KEYS if preset.get(k) is None]
    if missing:
        raise ValueError(
            f"preset 缺必填的执行键 {missing} —— CLAUDE.md §2(a) 把「让 max_positions / "
            f"position_fraction 隐式化」定义为 contract violation;§4 另要求滑点被显式建模"
        )

    unknown = sorted(k for k in preset if k not in ENGINE_KEYS and k not in METADATA_KEYS)
    if unknown:
        raise ValueError(
            f"preset 含未知键 {unknown} —— 拼错的执行键不会被静默忽略(§0 Fail-Fast)。"
            f"若这是新的元数据键,请加进 core.preset_contract.METADATA_KEYS"
        )

    atr_scheme = preset.get("atr_sl_mult") is not None
    pct_scheme = preset.get("sl_pct") is not None
    if atr_scheme == pct_scheme:
        raise ValueError(
            "退出宽度必须**恰好**给一套:atr_sl_mult/atr_tp_mult 或 sl_pct/tp_pct"
        )

    declared = preset.get("position_sizing")
    if declared is not None:
        want = derived_position_sizing(preset)
        if declared != want:
            raise ValueError(
                f"position_sizing={declared!r} 与实际生效的键不一致:按 risk_per_trade "
                f"是否给出来看应当是 {want!r}(该字段是派生量,不是旋钮)"
            )


def resolve(preset: dict) -> dict:
    """校验并展开成引擎参数 —— 返回且只返回 ENGINE_KEYS。"""
    validate(preset)
    return {k: preset.get(k, DEFAULTS.get(k)) for k in ENGINE_KEYS}
