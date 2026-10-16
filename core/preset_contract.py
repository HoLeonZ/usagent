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
    # ---- 评价口径（2026-10-17 审计 C5）----
    # 窗口末尾是否按最后一根收盘价清算未平仓仓位（True/False）。
    # 见 broker.py 的说明：不平仓会让"权益口径的头条收益"与"交易口径的 E[R]"
    # 落在两个不同的样本上。
    "liquidate_at_end",
    # ---- 成本口径（2026-10-17 审计 E1）----
    # "none"（默认）| "corwin_schultz"。默认 none 只为不改动既有数字，
    # 不是因为它更对：不建模价差等于把**尺寸无关**的那一半成本当成零。
    "spread_model",
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
    "spread_model": "none",
    # 默认清算：它让头条收益与 E[R]/胜率/盈亏比落在**同一个样本**上，
    # 而且方向保守（多付一次出场成本）。不想要就显式写 False。
    "liquidate_at_end": True,
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
    "avoid_earnings_forward_calendar_days", "earnings_filter_validated",
    "trailing_sl_validated",
})


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
