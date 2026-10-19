"""曝光匹配口径（overlay）的**唯一定义** —— preset 与族共用一把尺。

为什么需要它（原文见 `family_search.overlay`，这里是它的抽取版）：

    一个平均敞口 7.7% 的策略与 100% 敞口的指数比原始收益是**结构性不公平**的
    （它大部分时间持有零息现金）。真正该问的是「这套信号在它**在场**的那些天里，
    是否优于随便持有」:

        portfolio_ret(t) = w(t−1) × 策略日收益(t) + (1 − w(t−1)) × 池指数日收益(t)

    w 取**前一交易日**的敞口：当日敞口要到收盘才知道，用它会引入同日前视。

判读：overlay CAGR > 池指数 CAGR ⇒ 这套信号创造了价值；≈ 相等 ⇒ 只是把敞口换成了
一张更曲折的指数；明显更低 ⇒ 它的在场是负贡献。

**为什么抽出来**：这个重构（2026-10-17）之前，只有 `family_search` 有这份实现，
而且它 `family_id=None` 的分支是**死的**（`broker.simulate` 不接受 `signals=None`）
⇒ preset 级的 overlay 只能另写一份公式。两把尺量出来的"更好"没有意义，
所以这里收成一处：两条路径都调 `overlay_from_equity`。护栏是
`tests/test_overlay.py::test_family_search_delegates_to_the_single_definition`。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 一年按 365.25 天折算：加密/股票日历差异在这里不重要，重要的是**两条路径用同一个数**
DAYS_PER_YEAR = 365.25


class OverlayError(ValueError):
    """输入不满足口径的硬要求（缺列、样本太短）—— fail-fast，不静默返回 nan。"""


def overlay_from_equity(eq: pd.DataFrame, bench_ret: pd.Series, *,
                        days_per_year: float = DAYS_PER_YEAR) -> dict:
    """按曝光匹配口径折算：返回 {cagr, exposure, n_days}。

    `eq` 必须是 broker 的权益曲线（索引 = 日期，含 `equity` 与 `exposure` 两列）；
    `bench_ret` 是**同一时间轴上的池指数日收益**（调用方负责对齐，通常是
    `reg.set_index("date")["ew_index"].pct_change()`）。
    """
    for col in ("equity", "exposure"):
        if col not in eq.columns:
            raise OverlayError(f"权益曲线缺列 {col!r} —— 曝光匹配口径需要前一日敞口")
    if len(eq) < 3:
        raise OverlayError(f"样本太短（{len(eq)} 个交易日）—— 折算年化没有意义")

    sret = eq["equity"].pct_change().fillna(0.0)
    w = eq["exposure"].shift(1).fillna(0.0)          # 前一日敞口（因果）
    b = bench_ret.reindex(eq.index).fillna(0.0).copy()
    # ⚠️ **第一天的池子收益必须置零**：`pct_change()` 的第一项是"窗口开始**之前**一天 →
    # 窗口首日"的收益，而组合在这段窗口里根本赚不到它（第一天它还没持有任何东西）。
    # 不置零的后果是**可测的**（2026-10-17 实测，台账 H1）：0 信号 / 0 成交 / 0 敞口的
    # 策略会显示 **+0.44pp** 的"超额"，而 win 1.5 年的窗口上那个数就是首日池子收益的
    # 年化 —— 于是"什么都不做"看起来赢了基准，任何候选也白得同样的分。
    # 置零之后，"零敞口 ⇒ 超额恰好 0" 在**窗口完全落在池子覆盖范围内**时成立
    # （tests/test_overlay.py 钉住）。⚠️ 窗口若伸到池子覆盖之外（池子要预热，起得比面板晚，
    # 实测面板 2023-10-17 起 / 池子 2024-01-12 起），`excess_vs_pool` 会把两条腿裁到共同覆盖的
    # 那些日子上再比 —— 但此时**首日那一天仍有不对称**：这里把首日池子收益置零（因为它是
    # "窗口之前 → 窗口首日"的那一段，组合赚不到），而 `window_benchmark_cagr` 是 last/first、
    # 不置零 ⇒ 残余可达几个百分点（实测 6 个月窗口上 +3.73pp）。这条残余比 H1 那处（+0.44pp）
    # 大，**没有修**（改它会动到所有窗口的 baseline，属要单独决策的事）—— 如实记在这里，
    # 别让"不变量"这句话读起来像无条件的。
    if len(b):
        b.iloc[0] = 0.0
    ov = w * sret + (1.0 - w) * b

    days = (eq.index[-1] - eq.index[0]).days
    if days <= 0:
        raise OverlayError("权益曲线的首尾日期相同 —— 无法折算年化")
    cagr = float((1.0 + ov).prod() ** (days_per_year / days) - 1.0)
    return {"cagr": cagr, "exposure": float(w.mean()), "n_days": int(len(eq))}


def pool_daily_returns(reg: pd.DataFrame) -> pd.Series:
    """等权池指数日收益（旧→新）。`reg` 是 `regime.compute_regime` 的输出。"""
    return reg.set_index("date")["ew_index"].pct_change()


def window_bench_cagr(reg: pd.DataFrame, lo: str, hi: str) -> float:
    """同窗等权池 CAGR —— 委托给 `regime.window_benchmark_cagr`（它的唯一定义）。"""
    from research.regime import window_benchmark_cagr
    return window_benchmark_cagr(reg, lo, hi)


def excess_vs_pool(eq: pd.DataFrame, reg: pd.DataFrame, lo: str, hi: str) -> dict:
    """一条权益曲线在一个窗口上的 overlay 超额（= overlay CAGR − 同窗池 CAGR）。

    ⚠️ **两条腿必须在同一个区间上比**（2026-10-19 边界实测发现的静默失真）：
    `overlay_from_equity` 会把池子收益 reindex 到权益的日期轴上、缺失的日子补 `0.0`，
    而"池子那天没有数据"根本不是"那天没涨"。池子（reg 的等权指数）要预热，起得比面板晚
    （实测面板 2023-10-17 起，池子 2024-01-12 起），所以**任何起点早于池子覆盖的窗口**
    都会同时犯两个错：
      1. 现金腿被当成"59 天零收益" ⇒ 年化被低估（实测 13.76% vs 池子真实的 28.85%）；
      2. overlay 的年化按权益的 120 天折算，而 `window_bench_cagr` 用的是池子自己的 61 天
         —— 两条腿在不同跨度上比较。
    后果是**一个什么都不做的窗口会报出 −15.09pp 的"曝光匹配超额"**，而这正是平台面板上
    最显眼的那个数。所以这里先把权益裁到"池子真有数据"的那些日子上，再算两条腿；比较过
    多少天、从哪天起，一并放回结果里让调用方**说明白**（不静默截断）。
    """
    # ⚠️ 交集要按**水平值**（levels）的覆盖来取，不能按日收益序列取：日收益的第一项天然是
    # NaN（pct_change），dropna 会连带把"覆盖的第一天"也丢掉 ⇒ 首日被错当成"窗口首日"而置零，
    # 于是两条腿又差一天（实测：连那条不变式测试都会红，+0.6988%）。
    pool_ret = pool_daily_returns(reg)
    levels = reg.set_index("date")["ew_index"].dropna()
    idx = eq.index.intersection(levels.index)
    if len(idx) < 3:
        raise OverlayError(
            f"窗口 {lo}..{hi} 里权益与池子**共同覆盖**的交易日只有 {len(idx)} 天 —— 太少，"
            f"折算年化没有意义（池子指数覆盖 "
            f"{levels.index.min() if len(levels) else '?'} 起，窗口请落在它的覆盖范围内）")
    eq_c = eq.loc[idx]
    ov = overlay_from_equity(eq_c, pool_ret)
    bench = window_bench_cagr(reg, lo, hi)
    return {**ov, "bench": bench, "excess": ov["cagr"] - bench,
            "compared_from": str(idx.min()), "compared_to": str(idx.max()),
            "clamped": bool(len(idx) != len(eq))}


def mean_excess_over_windows(results: list[tuple[dict, dict]], reg: pd.DataFrame,
                             windows: list[tuple[str, str]]) -> dict:
    """多窗口折叠：每个窗口算一次 overlay 超额，返回均值与逐窗明细。

    为什么按**窗口**折叠而不是先拼净值再折算：跨窗口拼接净值会把窗口之间的空档
    也算成持有（每个窗口之间可能有大段空档），那是另一套口径。
    """
    per = []
    for eq, w in zip(results, windows):
        per.append({**excess_vs_pool(eq, reg, w[0], w[1]), "window": f"{w[0]}..{w[1]}"})
    exc = [p["excess"] for p in per]
    return {"excess_mean": float(np.mean(exc)) if exc else float("nan"),
            "per_window": per}
