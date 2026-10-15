"""筛选器 —— 把 596 只收窄成人能过目的候选,并**同时报告这组条件的历史基准率**。

为什么这个工具长这样(读之前先读这段)
======================================
2026-10-14 的实测结论:hiagent 那套 A~F 条件在美股这份数据上**历史期望为负**
(E[R] −0.29,胜率 13.6%,超额 −119pp)。所以:

  * 一个只输出"候选清单"的筛选器,等于给你一份**历史上亏钱的名单**;
  * 而"给我选股清单"比"给我策略"更容易骗人 —— 人看到 20 个里涨了 6 个,
    会记住那 6 个,不会记住基准。

因此本工具是两个东西拼在一起:
  1. **候选清单**(人过目用,带每个条件的取值);
  2. **同一组条件的历史基准率**,与股票池基准并列。

第 2 项是判据。如果某一层的远期收益与股票池基准没有差别,那这一层就是噪声 ——
无论它筛出来的票"看起来多像"。**先看基准率,再看清单。**

用法
----
    # 看各层条件到底有没有携带信息(先做这个)
    python -m research.screen --measure

    # 换持有期
    python -m research.screen --measure --horizon 5
    python -m research.screen --measure --horizon 60

    # 出今天的候选(默认在输出里带上基准率)
    python -m research.screen --tier trend_pullback --top 20
    python -m research.screen --tier v21_full --as-of 2026-10-13

口径
----
  价格      series.daily(adjust="total")  —— 拆股+分红复权(总收益口径)。**这是指标口径**;
            执行与记账用的是 as-traded 价(raw_*)。下面印出来的「收盘(复权)」属于前者,
            不是当日盘面可成交价 —— 要核当日价请用 raw_close。
  amount    代理量 = volume × **as-traded** close(usagent 无成交额列)
  远期收益  入场 = T+1 开盘,出场 = T+h 收盘(h 默认 18 个交易日)
            ⚠️ 远期收益**只用于测量**,绝不参与选股 —— 选股只用 ≤T 的数据。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research import indicators as I
from research.metrics import bonferroni_t_threshold
from research import panel as P
from research import presets as PS

logger = logging.getLogger("screen")

# 条件层级:从最宽到最窄。每层加一个条件,这样能看出**边际贡献**。
# 顺序即"信息增量",不是"优劣"。
TIER_ORDER = ["base", "trend", "trend_pullback", "trend_pullback_macd", "v21_full"]

# 真正被**检验**的层级:base 是全池基准本身,不构成一次比较(§5(c) 的族规模按它算)。
COMPARED_TIERS = [t for t in TIER_ORDER if t != "base"]

TIER_DESC = {
    "base": "全池(基准)",
    "trend": "A 趋势:close>MA60 ∧ MA20>MA60 ∧ above_ma60_ratio≥0.6693 ∧ mom120≥0.18",
    "trend_pullback": "A + B 回调:down_streak∈[3,10] ∧ ret1∈[-8.8%,-2%]",
    "trend_pullback_macd": "A + B + D MACD 衰减:|bar|<|bar_prev| ∧ dif>0 ∧ dea>0",
    "v21_full": "A + B + D + E 流动性 + close≥MA60×1.0121(v21 原样)",
}


def _masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """逐层掩码。每层都是上一层再 ∧ 一个条件。"""
    sp = PS.V21_SIGNAL_PARAMS

    ready = (
        df["ma60"].notna() & df["ma120"].notna() & df["mom120"].notna()
        & df["amount60"].notna() & df["macd_bar"].notna()
        & df["macd_bar_prev"].notna() & df["above_ma60_ratio"].notna()
        & df["prev_close"].notna()
    )

    trend = ready & (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"]) & (
        df["above_ma60_ratio"] >= sp["min_above_ma60_ratio"]) & (
        df["mom120"] >= sp["min_mom120"])

    pullback = (
        (df["down_streak"] >= sp["min_down_streak"])
        & (df["down_streak"] <= sp["max_down_streak"])
        & (df["ret1"] >= sp["pct_chg_low"])
        & (df["ret1"] <= sp["pct_chg_high"])
    )

    macd = (
        (df["macd_bar"].abs() < df["macd_bar_prev"].abs())
        & (df["macd_dif"] > 0) & (df["macd_dea"] > 0)
    )

    liq = (df["amount60"] >= sp["min_amount"]) & (df["amount60"] <= sp["max_amount"])
    buf = df["close"] >= df["ma60"] * (1 + sp["close_ma60_buffer"])

    return {
        "base": ready,
        "trend": trend,
        "trend_pullback": trend & pullback,
        "trend_pullback_macd": trend & pullback & macd,
        "v21_full": trend & pullback & macd & liq & buf,
    }


def forward_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    """入场 T+1 开盘、出场 T+h 收盘的收益。

    ⚠️ **只用来看结果,不得参与选股。** 选股掩码里没有任何 shift(-x)。
    """
    g = df.groupby("symbol", group_keys=False)
    entry = g["open"].shift(-1)
    exit_ = g["close"].shift(-horizon)
    return exit_ / entry - 1.0


def measure(ind: pd.DataFrame, horizon: int, *, lo: str, hi: str,
            family_size: int | None = None) -> pd.DataFrame:
    """每一层的远期收益分布,与全池基准并列。

    ⚠️ 判读用**与基准的差**,不是"均值是否 > 0"。
       在一个上涨的市场里,任何纯多头筛选的均值都是正的 —— 用"vs 0"检验会把
       **每一层**都判成"优于基准",包括实际低于基准的那些。这正是本函数第一版
       犯的错。

    ⚠️ 重叠采样校正:h 日持有期、逐日采样 → 相邻观测高度重叠。独立观测数约为
       n/h,故均值的标准误被低估约 sqrt(h) 倍。迟到不报的 t 值会虚高
       (用户报告 §3.5 列过这张膨胀系数表:h=10 → 3.32 倍,h=20 → 4.59 倍)。
       这里用 sqrt(h) 这个保守近似,并同时给出原值。

    ⚠️ family_size:本次扫描实际做了多少次比较。判词门槛按 Bonferroni 折减
       (§5(c))—— 4 个层级各下一次判词就是 4 次比较,门槛从 1.96 抬到 2.498。
       默认取层级数;换 horizon / 换窗口再跑一次,族就更大,调用方应当把它加进去。
    """
    k = int(family_size) if family_size is not None else len(COMPARED_TIERS)
    t_crit = bonferroni_t_threshold(k)
    df = ind[(ind["date"] >= pd.Timestamp(lo)) & (ind["date"] < pd.Timestamp(hi))].copy()
    df["fwd"] = forward_returns(ind, horizon).loc[df.index]

    masks = {k: v.loc[df.index] for k, v in _masks(ind).items()}
    base = df.loc[masks["base"], "fwd"].dropna()
    base_mean, base_n, base_sd = float(base.mean()), len(base), float(base.std(ddof=1))

    rows = []
    for tier in TIER_ORDER:
        f = df.loc[masks[tier], "fwd"].dropna()
        if len(f) < 30 or tier == "base":
            rows.append({"tier": tier, "n": int(len(f)), "mean": float(f.mean()) if len(f) else np.nan,
                         "median": float(f.median()) if len(f) else np.nan,
                         "win": float((f > 0).mean()) if len(f) else np.nan,
                         "vs_base": 0.0 if tier == "base" else np.nan,
                         "t_raw": np.nan, "t_adj": np.nan, "ann": np.nan})
            continue
        sd = float(f.std(ddof=1))
        se = np.sqrt(sd ** 2 / len(f) + base_sd ** 2 / base_n) if sd > 0 else np.nan
        diff = float(f.mean()) - base_mean
        t_raw = diff / se if se and np.isfinite(se) and se > 0 else np.nan
        # 重叠校正:n_eff ≈ n/h → SE 放大 sqrt(h)
        t_adj = t_raw / np.sqrt(horizon) if np.isfinite(t_raw) else np.nan
        rows.append({
            "tier": tier, "n": int(len(f)),
            "mean": float(f.mean()), "median": float(f.median()),
            "win": float((f > 0).mean()),
            "vs_base": diff,
            # 年化后的差值,便于判断"值不值得"
            "ann": diff * (252.0 / horizon),
            "t_raw": float(t_raw), "t_adj": float(t_adj),
        })
    out = pd.DataFrame(rows)
    out.attrs.update(base_mean=base_mean, base_n=int(base_n), horizon=horizon,
                     family_size=k, t_crit=t_crit)
    return out


def _verdict(r: pd.Series, t_crit: float) -> str:
    """判读只看**重叠校正后**的 vs-基准 t,且门槛已按族规模折减(§5(c))。

    阈值**不再硬编码 2**:那等价于 α=0.05 单次比较,而本函数每次 `--measure` 要对
    4 个层级各下一次判词。t_crit 由 metrics.bonferroni_t_threshold(族规模) 给出。
    """
    if not np.isfinite(r.get("t_adj", np.nan)):
        return "样本不足"
    t = r["t_adj"]
    ann = r["ann"] * 100 if np.isfinite(r["ann"]) else 0.0
    if t <= -t_crit:
        return f"劣于基准 {ann:+.1f}pp/年 ← 注意"
    if t >= t_crit:
        return f"优于基准 {ann:+.1f}pp/年"
    return f"与基准无差别({ann:+.1f}pp/年)"


def candidates(ind: pd.DataFrame, tier: str, as_of: str, top: int) -> pd.DataFrame:
    """as_of 当日的候选,按 mom120 降序。"""
    ts = pd.Timestamp(as_of)
    day = ind[ind["date"] == ts]
    if day.empty:
        raise SystemExit(f"as_of={as_of} 不是库里的交易日(可用区间 "
                         f"{ind['date'].min().date()} .. {ind['date'].max().date()})")
    mask = {k: v.loc[day.index] for k, v in _masks(ind).items()}
    hit = day[mask[tier]].copy()
    if hit.empty:
        return hit
    hit["score"] = hit["mom120"]
    cols = ["symbol", "close", "mom120", "pullback", "down_streak", "ret1",
            "atr_pct", "amount60", "above_ma60_ratio", "score"]
    return hit[cols].sort_values("score", ascending=False).head(top).reset_index(drop=True)


def _print_measure(m: pd.DataFrame) -> None:
    base_mean, base_n, h = m.attrs["base_mean"], m.attrs["base_n"], m.attrs["horizon"]
    print(f"\n持有期 {h} 个交易日(入场 T+1 开盘 / 出场 T+{h} 收盘)")
    print(f"全池基准: 均值 {base_mean*100:+.2f}%   n={base_n:,}")
    print(f"  ↑ 这个基准本身很大,是因为样本覆盖了一段上涨市。**所有比较都要相对它看** ——"
          f"\n    绝对均值是正的,不能说明任何一层有信息。\n")
    print(f"{'层级':<22}{'n':>8}{'均值':>8}{'中位':>8}{'胜率':>7}"
          f"{'vs基准':>9}{'年化':>9}{'t(校正)':>9}   判读")
    print("-" * 104)
    for _, r in m.iterrows():
        if not np.isfinite(r["mean"]):
            print(f"{r['tier']:<22}{r['n']:>8}   (样本不足)")
            continue
        vs = "    基准" if r["tier"] == "base" else f"{r['vs_base']*100:>8.2f}%"
        ann = f"{r['ann']*100:>8.2f}pp" if np.isfinite(r.get("ann", np.nan)) else "        —"
        tt = f"{r['t_adj']:>9.2f}" if np.isfinite(r.get("t_adj", np.nan)) else "        —"
        note = ("← 这就是基准" if r["tier"] == "base"
                else _verdict(r, m.attrs["t_crit"]))
        print(f"{r['tier']:<22}{r['n']:>8}{r['mean']*100:>7.2f}%{r['median']*100:>7.2f}%"
              f"{r['win']*100:>6.1f}%{vs}{ann}{tt}   {note}")
    print(f"\n族规模 k={m.attrs['family_size']} ⇒ Bonferroni 校正门槛 |t| >= "
          f"{m.attrs['t_crit']:.3f}(§5(c):报告多参数扫描必须做多重比较折减)")
    print(f"判读依据:'vs基准' 是相对全池的收益差;t(校正) 是把重叠采样按 sqrt({h}) 折减后的")
    print( "          t 值(h 日持有期逐日采样,独立观测数约为 n/h,不折减会虚高)。")
    print(f"          |t| < {m.attrs['t_crit']:.3f} = 与随机选票无统计差别(已按族规模折减)。")
    print("\n逐层定义:")
    for t in TIER_ORDER:
        print(f"  {t:<22} {TIER_DESC[t]}")


def _print_candidates(c: pd.DataFrame, tier: str, as_of: str) -> None:
    if c.empty:
        print(f"\n{as_of} 层级 {tier} 无候选。")
        return
    show = c.copy()
    show["close"] = show["close"].round(2)
    show["mom120"] = (show["mom120"] * 100).round(1)
    show["pullback"] = (show["pullback"] * 100).round(1)
    show["ret1"] = (show["ret1"] * 100).round(2)
    show["atr_pct"] = (show["atr_pct"] * 100).round(1)
    show["amount60"] = (show["amount60"] / 1e6).round(0)
    show["above_ma60_ratio"] = (show["above_ma60_ratio"] * 100).round(0)
    show["score"] = (show["score"] * 100).round(1)
    show = show.rename(columns={
        "close": "收盘(复权)", "mom120": "动量120%", "pullback": "距20日高%",
        "down_streak": "连阴", "ret1": "当日%", "atr_pct": "ATR%",
        "amount60": "额60($M)", "above_ma60_ratio": "MA60上占比%", "score": "分"})
    print(f"\n{as_of} 候选(层级 {tier},共 {len(c)} 只,按 mom120 降序)")
    print(show.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="候选筛选 + 历史基准率测量")
    ap.add_argument("--measure", action="store_true",
                    help="只输出各层条件的历史基准率(先跑这个)")
    ap.add_argument("--tier", default="trend_pullback", choices=TIER_ORDER)
    ap.add_argument("--as-of", default=None, help="默认用库里最后一个交易日")
    ap.add_argument("--horizon", type=int, default=18, help="远期收益的持有交易日数")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--family-size", type=int, default=None,
                    help="本次比较的族规模(默认=层级数);换 horizon/窗口再跑要自己加进去")
    ap.add_argument("--start", default="2024-04-09", help="测量起点(预热之后)")
    ap.add_argument("--end", default="2026-10-14")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    p = P.load_full_panel()
    ind = I.compute_all(p)

    if a.measure:
        m = measure(ind, a.horizon, lo=a.start, hi=a.end, family_size=a.family_size)
        _print_measure(m)
        return 0

    as_of = a.as_of or str(ind["date"].max().date())
    c = candidates(ind, a.tier, as_of, a.top)
    _print_candidates(c, a.tier, as_of)

    # 清单下面**永远**跟着基准率 —— 这是这个工具与"选股清单"的区别
    m = measure(ind, a.horizon, lo=a.start, hi=a.end, family_size=a.family_size)
    row = m[m["tier"] == a.tier]
    base = m.attrs["base_mean"]
    print(f"\n这层的实测表现(持有 {a.horizon} 个交易日,相对全池基准):")
    if row.empty or not np.isfinite(row.iloc[0]["mean"]):
        print("  样本不足,无法评估")
    else:
        r = row.iloc[0]
        print(f"  该层均值 {r['mean']*100:+.2f}%   全池基准 {base*100:+.2f}%   "
              f"差 {r['vs_base']*100:+.2f}%   年化 {r['ann']*100:+.1f}pp   "
              f"n={int(r['n'])}   t(校正) {r['t_adj']:.2f}")
        t = r["t_adj"]
        t_crit = m.attrs["t_crit"]          # 与 measure 的判词同一门槛(§5(c))
        if np.isfinite(t) and t <= -t_crit:
            print("  ⚠️ 该层**历史上劣于**股票池基准。清单只是缩小范围,"
                  "不要把它当作看涨理由。")
        elif not np.isfinite(t) or abs(t) < t_crit:
            print("  ⚠️ 该层与股票池基准**无统计差别** —— 筛出来的票不比随机票更好。"
                  "\n     它的价值在于把 596 只收窄到几十只供你过目,不在于预测方向。"
                  "\n     方向判断得由你补。(见 README「这个工具不做什么」)")
        else:
            print("  ℹ️ 该层历史上优于基准,但注意:n/h 的重叠采样已按 sqrt(h) 折减,"
                  "\n     且样本只覆盖一段上涨市,不代表未来。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
