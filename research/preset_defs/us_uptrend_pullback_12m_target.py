"""preset 定义体：us_uptrend_pullback_12m_target（见 research/preset_registry.py 的装载契约）。
"""
from __future__ import annotations

from core import costs
from research import universe as mod_universe
from research.preset_common import (DEFAULT_RISK_PER_TRADE, RISK_LIMITS,
                                    V21_PAYOFF_GEOMETRY, V21_SIGNAL_PARAMS)

NAME = "us_uptrend_pullback_12m_target"

NOTES = """
【2026-10-15 全量重测】每一次引擎口径修正(§2(d) 结算隔离、§3(b) 双价格、§3(c) 分红、
§4(c) 入场/出场滑点)都会改变成交价,所以这里的数字必须跟着重测。下面每一个数字都能由
「run_baseline + 直接调 search_12m.random_control」再生;数据 vintage 见 run_baseline 输出
里的 data_vintage 字段。

结论:**目标已不再达成。** 本 preset 是为「最近 12 个月 CAGR >= 30%」搜出来的,当时报 +44.5%;
现在同一个 preset、同一个窗口 2025-10-14..2026-10-14:
    信号 21   成交 15   CAGR +9.51%   总收益 +9.48%   maxDD -24.1%   E[R] +0.212
    同期等权池基准 +15.60%  ⇒  超额 -6.12pp
低于它自称达到的 30%,也不及基准。复现:
    python -m research.run_baseline --preset us_uptrend_pullback_12m_target --start 2025-10-14 --end 2026-10-14

为什么失效:达标数字的相当一部分来自一次**物理上不可能**的记账 —— 旧引擎允许用「当日盘中
止损的回款」支付「当日开盘的入场款」(主循环先按盘中价平仓、再按开盘价开仓,而开盘在时间上
早于盘中;CLAUDE.md §2(d) 明令禁止同日复用卖出所得)。只把这一条**回退**、其余全部保留当前
代码,同一窗口即回到 **+45.51%**(14 笔)。也就是说那份「达标」不是信号做出来的。
修法见 research/broker.py 的「步骤 0:当日入场的账本快照」。

仍然成立的定性结论:**CAGR 由杠杆驱动,不由信号驱动。** 仅改 risk_per_trade(信号与止损止盈
一字不改),同一窗口:
    risk=1%  ->  CAGR  +3.23%   回撤  -6.4%   E[R] +0.225
    risk=2%  ->  CAGR  +6.00%   回撤 -12.4%   E[R] +0.219
    risk=5%  ->  CAGR  +9.51%   回撤 -24.1%   E[R] +0.212
    risk=25% ->  CAGR +11.87%   回撤 -29.6%   E[R] -0.014
前三档 E[R] 几乎不变 ⇒ 杠杆放大的只是仓位,不是边缘。⚠️ 第四档(25%)E[R] 转负:仓位大到
被现金闸门与量能上限截断,定尺寸退化成「能买多少买多少」,此时 CAGR 与 E[R] 都不可比 ——
这正是「CAGR 无约束最大化」会撞上的那面墙。

同杠杆随机入场:本配置落在第 **83** 百分位。同样止损止盈、同样笔数、同样杠杆,**从策略自己的
机会集**里随机选入场点(指标就绪 + amount60 ≥ $3e7 + regime 可做多)200 次(seed=0):
随机中位 -8.1%、p90 +16.3%、p95 +29.6%、最大 +87.0%、达到 >=30% 的概率 5%;本配置 +9.51%
位于第 83.0 百分位 ⇒ **约 17% 的随机策略做得和它一样好或更好**。
(旧记录写的是第 71~95 百分位 —— 那几版 null 从**全池**抽,含策略永不碰的小盘股;修正抽样池
后这个百分位**更差**,即「不构成边缘证据」在新口径下更强,不是更弱。修正因由:全池抽样还会让
200 次里 33 次在出场时撞上 §4(a) 的量能上限,分布被筛成成功子集。)

换一个 12 个月窗口(2024-10-14..2025-10-14,选择时未看):
    本 preset   CAGR +51.57%   E[R] +0.611   12 笔
    v21 基线    CAGR  +8.30%   E[R] +0.611   14 笔
    等权池基准  +64.97%   ⇒ 本 preset 超额 -13.40pp、v21 基线超额 -56.67pp
两个窗口合看:过滤层相对基线的贡献在当前窗口是 **+29.8pp**(+9.51% vs v21 的 -20.32%),
在前一个窗口是 **+43.3pp** —— 换窗口后**没有消失**,这一点比旧记录的结论要好。
但两个窗口都跑输基准,而两个窗口的基准相差 4 倍(+15.60% vs +64.97%):
变的是市场,不是信号。

记账:n_comparisons 的规则写在 v1 那一侧的注释里。
⚠️ 已声明的边界:①`search_12m --control` 的 cfg 仍写死成 {},要复算上面那张随机对照表得
直接调 `search_12m.random_control`(cfg 口径见 render_report 里那段反推);②§4(a) 的**出场**
IOC 部分成交未实现 —— 实测余量 24~127 倍(策略池全是 amount60 ≥ $3e7 的票,56 笔出场一次都
没咬合),一旦真咬合就 fail-fast,绝不给出一个足额的假成交。
"""

PRESET = {
    "universe": dict(mod_universe.DEFAULT_SPEC),
    "intraday_tiebreak": "sl_first",
    "max_volume_participation": costs.MAX_VOLUME_PARTICIPATION,
    "cost_bps_roundtrip": costs.COST_BPS_ROUNDTRIP,
    # 价差是**尺寸无关**的那一半成本，此前完全没建模（审计 E1）。
    # 用 Corwin-Schultz 从日线 high/low 估计（逐日、因果、系统性低估 ⇒ 是下限）。
    "spread_model": "corwin_schultz",
    # 止损穿价系数：0 = 按触发价成交（既有行为）。**为什么不取 0.5**：实测穿价中位
    # 约 1% / p90 约 1.8%（见 broker 的 sl_penetration 读数），θ 从 0 到 1 会让
    # 头条差 1~3.5pp —— 也就是说这条假定值钱。但 θ 的正确值需要**实盘成交**来标定
    # （台账 G7 的前向验证闭环），在没有标定数据之前拍一个 0.5 就是凭空造假设。
    # 所以：默认 0、穿价**逐笔记录并进指标**，读者自己按 θ∈{0,0.5,1} 跑敏感性。
    "sl_overshoot": 0.0,
    # 与 v1 同一条滑点律(f(ATR, 参与率),平方根冲击,scale=1.0 取经验区间保守端)。
    # 实测(2026-10-15 重测,本 preset 15 笔):**入场**参与率中位 0.103%(最大 0.32%)、
    # 信号日 atr_pct 中位 3.73% ⇒ 入场滑点中位 0.111%/边;出场参与率中位 0.048%
    # (最大 0.42%)⇒ 出场滑点中位 0.094%/边。开与不开的 CAGR 是 +9.51% ↔ +11.05%,
    # 即 **1.54pp** —— 比只有入场滑点时大得多(出场滑点作用在每一次止损/止盈上)。
    # (初版注释写的 0.2pp / 8.64%↔8.49% 是 amount 口径修好之前测的,复核实测已纠正。)
    "atr_slip_scale": costs.ATR_SLIP_SCALE,
    "position_sizing": "risk_budget",
    # §2(a) 要求显式声明(派生量:真正生效的是 risk_per_trade 这个键在不在)。
    "position_fraction": 1.0,
    "risk_per_trade": 0.05,          # <= 这一个是 CAGR 的真正来源(1% -> 5% 杠杆)
    "max_positions": 2,
    **RISK_LIMITS,                     # 组合层风险政策（见上方 RISK_LIMITS）
    "atr_sl_mult": 1.5,
    "atr_tp_mult": 6.0,
    "max_hold": 18,
    "signal": {
        **V21_SIGNAL_PARAMS,
        "min_dist_high252": -0.1,
        "max_stoch_k": 60.0,
    },
    "regime": {"ma_window": 20, "breadth_window": 20, "min_breadth": 0.45,
               "require_rising": False},
    # 2(流动性)+ 3(成本) + 37(stage1) + 45(stage2) + 15(分散化) + 3(对照)
    # +1: 前一个 12 个月的 OOS 检验
    # 2026-10-15 审计后:+1(执行滑点口径的选定,理由与计不计规则见 v1 的注释)。
    "n_comparisons": 107,
    "validated": False,
        "notes": NOTES,
}

VERDICT = (
    '**目标已不再达成**(2026-10-15 修正记账与口径后重测):本 preset 在它自己的窗口 2025-10-14..2026-10-14 上成交 15 笔、CAGR +9.51%、maxDD -24.1%、E[R] +0.212,低于它自称达到的 30%,也低于同期等权池基准(+15.60%)6.12pp。此前的 +44.5% 有相当一部分来自一次物理上不可能的记账:用当日盘中止损的回款支付当日开盘的入场款(§2(d) 明令禁止);只回退那一条、其余不变,同窗即回到 +45.51%。同杠杆、同机会集的随机入场 200 次(seed=0),本配置落在随机分布第 83.0 百分位 —— 约 17% 的随机策略不差于它,不构成边缘证据。**不得用于实盘,不得当作选股能力的证据。** 复现:python -m research.run_baseline --preset us_uptrend_pullback_12m_target --start 2025-10-14 --end 2026-10-14'
)

LOCK_WINDOW = ("2025-10-14", "2026-10-14")
LOCK_IS_CLI_DEFAULT = False

EXPORTS = {"US_12M_SEARCH_NOTE": NOTES}
FIXTURE_WINDOW = ("2024-01-01", "2025-01-01")
