"""信号层 —— hiagent/uptrend_pullback/signals.py::select_entries_v33_long_mirror 的移植。

v33_long_mirror 是 short_reversal v33 的做多镜像,五个条件严格对应:

    short_reversal v33 (做空)          →  mirror (本模块, 做多)
    A: close < MA60 ∧ below_ratio≥60%  →  A: close > MA60 ∧ above_ratio ≥ 阈值
    B: up_streak ∈ [3, 10]             →  B: down_streak ∈ [3, 10]
    C: ret1 ∈ [+2%, +7%]               →  C: ret1 ∈ [-8.8%, -2%]
    D: MACD dif<0 ∧ dea<0 ∧ |bar| 缩短 →  D: MACD dif>0 ∧ dea>0 ∧ |bar| 缩短
    E: am60 ∈ [3e7, 3e8]               →  E: amount60 ∈ [3e7, 3e8](见下)

语义:T 日收盘满足全部条件 → T+1 开盘买入(由 broker.py 执行)。
排序分 score = mom120,broker 按分数降序抢占仓位。

**E 条件的上界是这里唯一的口径分歧点。** amount60 是代理量(volume ×
**as-traded** close,见 panel.to_panel_frame —— 不能用复权价,否则 E 判定依赖未来分红),
见 panel.py),hiagent 的 [3e7, 3e8] 是 A 股主板口径。美股实测 NVDA 的
amount60 ≈ $26.0B/天,上界 $300M 几乎全被击穿。max_amount=None 表示移除上界。
两者都跑、都记 n_trials,让结果回答"这条 A 股筛选能不能搬"。
"""
from __future__ import annotations

import logging

import pandas as pd

from research import universe as mod_universe

logger = logging.getLogger(__name__)

ENTRY_COLS = [
    "date", "symbol", "score",
    "sig_close", "sig_ma20", "sig_ma60", "sig_down_streak",
    "sig_pullback", "sig_mom120", "sig_amount60", "atr_pct",
]

# v33_long_reverse_v21 的原始信号参数(基线原样值,一个都不改)
V21_SIGNAL_PARAMS: dict[str, object] = {
    "min_down_streak": 3,
    "max_down_streak": 10,
    "pct_chg_low": -0.088,
    "pct_chg_high": -0.02,
    "min_amount": 3e7,
    "max_amount": 3e8,
    "min_above_ma60_ratio": 0.6693,
    "close_ma60_buffer": 0.0121,
    "min_mom120": 0.18,
}


class SignalError(RuntimeError):
    """参数非法 —— fail-fast。"""


def select_entries(
    panel_ind: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    min_down_streak: int = 3,
    max_down_streak: int = 10,
    pct_chg_low: float = -0.088,
    pct_chg_high: float = -0.02,
    min_amount: float = 3e7,
    max_amount: float | None = 3e8,
    min_above_ma60_ratio: float = 0.6693,
    min_mom120: float = 0.18,
    close_ma60_buffer: float = 0.0,
    regime_df: pd.DataFrame | None = None,
    # ---- 可选技术指标过滤层(2026-10-14 新增)--------------------------
    # 全部默认 None/关闭 ⇒ 原有行为一字不变(既有测试因此仍然有效)。
    # 这些是**额外的收窄条件**,叠加在 A~F 之上,不改变 uptrend_pullback 的骨架。
    max_rsi14: float | None = None,          # RSI 超卖:回调到低位
    max_bb_pctb: float | None = None,        # 布林位置:贴近下轨
    max_stoch_k: float | None = None,        # KDJ 低位
    min_adx14: float | None = None,          # 趋势强度:只要趋势够陡的
    min_dist_high252: float | None = None,   # 52 周位置:不要接在深跌里
    min_xs_mom120: float | None = None,      # 横截面:相对同池要够强
) -> pd.DataFrame:
    """上升趋势中的回调做多信号(左闭右开 [start_date, end_date))。

    max_amount=None → 移除流动性上界(见模块 docstring)。
    regime_df 非空 → 只保留 regime_ok 的交易日(F 条件)。
    """
    if min_down_streak < 0 or max_down_streak < min_down_streak:
        raise SignalError(
            f"连阴区间非法: [{min_down_streak}, {max_down_streak}]"
        )
    if not (pct_chg_low < pct_chg_high):
        raise SignalError(f"跌幅区间非法: [{pct_chg_low}, {pct_chg_high}]")
    if max_amount is not None and max_amount <= min_amount:
        raise SignalError(f"流动性区间非法: [{min_amount}, {max_amount}]")

    df = panel_ind

    mask = (
        # A —— 上升趋势
        (df["close"] > df["ma60"])
        & (df["above_ma60_ratio"] >= min_above_ma60_ratio)
        & (df["mom120"] >= min_mom120)
        # B —— 连阴(上升趋势中的回调)
        & (df["down_streak"] >= min_down_streak)
        & (df["down_streak"] <= max_down_streak)
        # C —— 当日跌幅
        & (df["ret1"] >= pct_chg_low)
        & (df["ret1"] <= pct_chg_high)
        # D —— MACD 动能衰减(上升趋势中柱状缩短 = 回调买点)
        & (df["macd_dif"] > 0)
        & (df["macd_dea"] > 0)
        & (df["macd_bar"].abs() < df["macd_bar_prev"].abs())
        # E —— 流动性
        & (df["amount60"] >= min_amount)
    )
    if max_amount is not None:
        mask = mask & (df["amount60"] <= max_amount)

    # 池资格（§3 PIT Mandate）：只在**可投资集**里找信号。缺列 fail-fast ——
    # 不允许存在"忘了带池子"却照样出数字的路径。
    mask = mask & mod_universe.require_eligible_mask(df)

    # 指标就绪(不靠 bfill 补预热期的空洞)
    for c in ("ma60", "above_ma60_ratio", "amount60", "macd_bar",
              "macd_bar_prev", "mom120"):
        mask = mask & df[c].notna()

    # A' —— 趋势质量增强(可选):close 需高出 MA60 一定缓冲
    if close_ma60_buffer > 0:
        mask = mask & (df["close"] >= df["ma60"] * (1 + close_ma60_buffer))

    # ---- 可选技术指标过滤层(2026-10-14)------------------------------
    # 每个都是"再 ∧ 一个条件",且显式要求该指标非空 —— 指标预热期不得因为
    # NaN 与比较运算的怪癖而被静默放进或踢出。
    for col, thr, op in (
        ("rsi14", max_rsi14, "<="),
        ("bb_pctb", max_bb_pctb, "<="),
        ("stoch_k", max_stoch_k, "<="),
        ("adx14", min_adx14, ">="),
        ("dist_high252", min_dist_high252, ">="),
        ("xs_mom120", min_xs_mom120, ">="),
    ):
        if thr is None:
            continue
        if col not in df.columns:
            raise SignalError(
                f"过滤层用到 {col},但面板里没有这一列 —— "
                f"需要先用 indicators.compute_all() 而不是 compute_indicators()"
            )
        cond = (df[col] <= thr) if op == "<=" else (df[col] >= thr)
        mask = mask & df[col].notna() & cond

    # 信号窗口
    mask = mask & (df["date"] >= pd.Timestamp(start_date)) & (df["date"] < pd.Timestamp(end_date))

    hits = df[mask].copy()

    # F —— 大盘择时
    if regime_df is not None and not hits.empty:
        ok_days = set(pd.to_datetime(
            regime_df.loc[regime_df["regime_ok"], "date"]).tolist())
        hits = hits[hits["date"].isin(ok_days)]

    if hits.empty:
        return pd.DataFrame(columns=ENTRY_COLS)

    hits["score"] = hits["mom120"]
    hits["sig_close"] = hits["close"]
    hits["sig_ma20"] = hits["ma20"]
    hits["sig_ma60"] = hits["ma60"]
    hits["sig_down_streak"] = hits["down_streak"]
    hits["sig_pullback"] = hits["pullback"]
    hits["sig_mom120"] = hits["mom120"]
    hits["sig_amount60"] = hits["amount60"]

    out = hits[ENTRY_COLS].sort_values(
        ["date", "score"], ascending=[True, False]
    ).reset_index(drop=True)
    # ⚠️ 这里**不**对空结果抛错。曾有一句 `if out.empty: raise SignalError("信号集为空…")`,
    #    但上面 `hits.empty` 已经 return 了,所以那句话不可达 —— 而 README 与三处 docstring
    #    都据此声称「信号集为空要抛错不静默」。真相是:空信号是**合法结果**(参数把信号
    #    杀光),该由调用方按「无结果」处理,而不是当异常吞掉或记成 0 收益。
    #    2026-10-15 审计实测:search_12m 因此把空转配置记成 cagr=+0.0%,在 argmax(CAGR)
    #    的搜索里会赢过所有负收益配置。
    logger.info("select_entries: %d 个候选信号", len(out))
    return out


def filter_earnings_window(
    sig: pd.DataFrame,
    by_symbol: dict[str, list[pd.Timestamp]],
    *,
    forward_days: int,
) -> pd.DataFrame:
    """剔除「持有窗口内夹着财报」的信号。

    预注册假设 H1(hiagent preset_search §6 指的事件维度):
        财报是不可预测的二值事件,且以**跳空**形式释放。本策略用固定价位止损,
        跳空会直接穿透它 —— 止损价形同虚设。持有窗口里夹着财报,等于把一个无法
        定价的事件风险塞进一笔"技术面回调"里。这不是价格派生条件能修的。

    forward_days 用**日历日**:max_hold=18 个交易日 ≈ 26 个自然日,别用交易日数
    直接当自然日,那会漏掉窗口尾部一周的财报。
    """
    if sig.empty:
        return sig
    from research.earnings import has_earnings_between

    keep = []
    for _, r in sig.iterrows():
        lo = pd.Timestamp(r["date"]) + pd.Timedelta(days=1)
        hi = pd.Timestamp(r["date"]) + pd.Timedelta(days=forward_days)
        keep.append(not has_earnings_between(by_symbol, str(r["symbol"]), lo, hi))
    out = sig[keep].reset_index(drop=True)
    logger.info("filter_earnings_window: %d -> %d 个信号(剔除 %d)",
                len(sig), len(out), len(sig) - len(out))
    return out
