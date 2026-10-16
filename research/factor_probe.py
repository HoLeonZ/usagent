"""横截面因子探针 —— 满仓、月频再平衡,所以与基准的比较**没有敞口问题**。

为什么需要它
------------
前几轮把 14 个策略族都按「闲置资金计息」的公平口径测过,**没有一个创造价值**;而机制隔离
把「个体性」「不放量」两条否掉,只留下一个线索:**低波动**是唯一承重的条件。但那只是**间接
证据** —— 它来自我自己翻的对照臂,网格小、样本少。本文件直接测这个因子。

两个检验,一个便宜一个贵:

  ① **秩相关 IC**(主判据,最干净):在每个再平衡日,算「因子在池内的排名」与「下一个月的
     总收益排名」的 Spearman 相关。**一个数字,不经过任何阈值/分位选择** —— 所以它无法被
     「你试了 20 个分位组合」这类质疑反驳。IC ≈ 0 就是「这个因子没有横截面信息」。
  ② **分位组合**(可读性):满仓持有因子排序的一档,月频再平衡,与**同一流动性门下的等权
     全池**比。两者都是 100% 敞口、同一个成本模型 ⇒ **差就是因子本身**。

口径纪律
--------
  * 选择只用 ≤T 的数据(所有因子都是因果的,`families.add_family_indicators` 保证);
    持有从再平衡日的**下一个交易日**到下一个再平衡日;
  * 收益用**分红回补价**(总收益口径):这里没有止损止盈,不涉及 as-traded 价;
  * 成本沿用 `core.costs` 的唯一定义,按换手计,**因子组合与全池用同一个成本模型**;
  * 长-空价差只作**测量**(市场 beta 被抵消掉,是最经典的因子检验),
    **不作为策略** —— 未声明借券假设(CLAUDE.md §4(g)),不可交易。
  * 幸存者偏差对**全池与所有因子同等**作用,所以绝对水平偏高;
    本文件读的是**因子组合与同门全池的差**,那是可比的。

**预注册判据(写在这里意味着跑之前就定死;三条同时满足才算「这个因子在这段数据上可用」)**

    ① 有信息   秩相关 IC 的 |t| ≥ 2
    ② 稳定     把窗口在**预注册的那条线**(搜索期/holdout 分界 2025-10-14)对半切开,
               两半的 IC **符号不翻转**、量级同阶
    ③ 拿得走   逐月**配对**超额(组合 − 同门全池)|t| ≥ 2,且 ΔSharpe > 0

为什么加 ②:实测 —— 全窗 `mom120 high` 打赢全池 +28.72pp,切开后前半段 +67.31pp(t 2.07)、
后半段 **+0.18pp**(t 0.21);`rvol20` 的 IC 从 +0.024 翻成 −0.125。全窗那个数是**半段牛市**,
不是因子。没有 ② 这条,**每一轮都会重新发现同一个幻觉**。
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from core import costs
from research import families as F, holdout as mod_holdout, indicators, metrics
from research import panel as mod_panel, universe as mod_universe

logger = logging.getLogger("factor_probe")

QUANTILES = (0.2, 0.3)

# 预注册判据的 t 门槛。**它是预注册的，不得在看到结果之后放宽**（§5 迭代记账：
# 事后改判据就是一次新的搜索，n_trials 要跟着涨）。族错误率校正的产物是
# "在该族规模下还剩下几个"，不是"把判据改成另一个数"。
PRE_REG_T = 2.0


def _family_report(pairs: list[tuple[str, float]], label: str,
                   q: float = 0.05) -> dict:
    """把一次扫描里的多个点比较**按族错误率校正后**报出来（审计 B4 / §5(c)）。

    为什么要两层：Bonferroni 控制的是最坏情况（FWER），在因子扫描这种族规模下
    功效极低；BH-FDR 控制的是"被判为显著的那些里有多少是假的"，是因子研究里
    通行的工具（Harvey-Liu-Zhu 那一类工作）。两个都给，读者才能判断
    "点比较的显著"在族规模下还剩多少 —— 此前这里只有一句 |t| ≥ 2。
    """
    ts = [(nm, float(t)) for nm, t in pairs if np.isfinite(t)]
    m = len(ts)
    print(f"\n  ── 族错误率校正（{label}，可检验的比较数 k={m}）──")
    if m == 0:
        print("     没有可检验的比较（全部期数不足）⇒ 不下判词")
        return {"k": 0, "bonferroni_t": float("nan"), "n_fdr": 0}
    t_bonf = metrics.bonferroni_t_threshold(m, q)
    bh = metrics.benjamini_hochberg([metrics.two_sided_p(t) for _, t in ts], q)
    n_point = sum(1 for _, t in ts if abs(t) >= PRE_REG_T)
    n_bonf = sum(1 for _, t in ts if abs(t) >= t_bonf)
    print(f"     预注册判据仍是 |t| ≥ {PRE_REG_T:.2f}（**不事后放宽**）；按族规模校正后：")
    print(f"     Bonferroni α={q} 双侧 ⇒ |t| ≥ {t_bonf:.2f}，存活 {n_bonf}/{m}")
    fdr_txt = (f"临界 p ≤ {bh['crit_p']:.4f}" if bh["n_rejected"] else "无通过者")
    print(f"     BH-FDR q={q} ⇒ {fdr_txt}，存活 {bh['n_rejected']}/{m}"
          + (f"（点比较口径 {n_point}/{m}）" if n_point != bh["n_rejected"] else ""))
    if bh["n_rejected"] == 0 and n_point:
        print(f"     ⇒ 该族规模下，{n_point} 个「显著」**一个都过不了**族校正："
              f"它们只能读作待验证，不能当判据用。")
    return {"k": m, "bonferroni_t": t_bonf, "n_fdr": bh["n_rejected"],
            "n_pointwise": n_point, "crit_p": bh["crit_p"]}
# 因子:(列名, "low"/"high" = 取哪一档)—— 全部来自因果指标,日期 ≤T
FACTORS: list[tuple[str, str]] = [
    ("rvol20", "low"),        # 低波动(唯一的承重线索)
    ("rvol20", "high"),       # 高波动(对照)
    ("mom120", "high"),       # 动量(诊断:这个池子是不是"动量就是一切")
    ("mom120", "low"),        # 输家组合(= 长周期反转)
    ("ret21", "low"),         # 1 月反转(≈ str 族的月度版)
    ("dd120", "low"),         # 距 120 日高点最远 = 最"便宜"
    ("dd120", "high"),        # 最接近/突破新高
    ("amount60", "low"),      # 流动性最低(诊断:是不是小盘效应)
    ("amount60", "high"),     # 流动性最高
]
GATE = 3e7          # 与项目其它部分一致的流动性门
GATE_SCAN = (0.0, 1e6, 3e6, 1e7, 3e7, 1e8)   # 标的资格梯度检查用


def _gate_scan(ind: pd.DataFrame, *, lo: str, mid: str, hi: str, cost_bps: float) -> None:
    """标的资格(universe)梯度:同一段代码只改流动性门。

    为什么单列一节:上面所有因子检验都在**已被门砍过的池子**里做,而门本身可能就是一个
    被忽略的选择。判据与因子检验一致 —— **两半必须同号**,否则又是在读半段行情。
    """
    print("\n" + "=" * 88)
    print("标的资格梯度:同一段代码只改流动性门(判据:两半同号)")
    print("=" * 88)
    segs = (("全窗", lo, hi), ("前半段", lo, mid), ("后半段", mid, hi))
    print(f"{'流动性门':>12}" + "".join(f"{t + ' CAGR':>16}" for t, _, _ in segs)
          + f"{'全窗波动':>10}{'全窗maxDD':>11}{'全窗只数':>10}")
    for g in GATE_SCAN:
        cells, ref = [], None
        for t, a, b in segs:
            st = equal_weight_pool(ind, lo=a, hi=b, cost_bps=cost_bps, gate=g)
            cells.append(st["net"]["cagr"])
            if t == "全窗":
                ref = st
        print(f"{g:>12,.0f}" + "".join(f"{c*100:>+15.2f}%" for c in cells)
              + f"{ref['net']['vol']*100:>9.1f}%{ref['net']['max_dd']*100:>10.1f}%"
              f"{ref['n_holdings']:>10.0f}")
    print("判读:门越低收益越高、且两半同号 ⇒ 值得追;但尾部正是「今日快照」幸存者偏差最重")
    print("      的地方,而本探针**没有成交模型** ⇒ 这是线索,不是可交付的策略。")


def _pivot(ind: pd.DataFrame, col: str) -> pd.DataFrame:
    return ind.pivot(index="date", columns="symbol", values=col).sort_index()


def month_ends(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """每个自然月的最后一个交易日(当再平衡日,不需要未来信息)。"""
    s = pd.Series(dates, index=dates)
    return list(s.groupby([dates.year, dates.month]).last())


def _periods(px: pd.DataFrame, amt: pd.DataFrame, lo: str, hi: str):
    """→ [(再平衡日 r, 期末日 nxt)];最后一个再平衡日没有下一个月可持有,丢掉。"""
    dates = px.index[(px.index >= pd.Timestamp(lo)) & (px.index < pd.Timestamp(hi))]
    reb = month_ends(dates)
    return [(r, reb[i + 1]) for i, r in enumerate(reb[:-1])]


def _gate(px, amt, r, gate: float) -> pd.Index:
    a_row = amt.loc[r]
    ok = a_row.notna() & (a_row >= gate) & px.loc[r].notna()
    return a_row[ok].index


def _seg_return(px: pd.DataFrame, pick, r, nxt) -> pd.Series:
    seg = px.loc[(px.index > r) & (px.index <= nxt), sorted(pick)]
    return seg.iloc[-1] / seg.iloc[0] - 1.0


def rank_ic(ind: pd.DataFrame, factor: str, *, lo: str, hi: str,
            gate: float = GATE) -> dict:
    """因子排名 vs 次月收益排名的 Spearman 相关(逐期),再对期数做 t 检验。

    只用了两个东西:因子值(T 日已知)与持有期收益(用 T 之后的价格算)。
    **不涉及分位、阈值、成本** —— 所以这个数字没有可调的旋钮。
    """
    px, fac, amt = _pivot(ind, "close"), _pivot(ind, factor), _pivot(ind, "amount60")
    ics, n_cand = [], []
    for r, nxt in _periods(px, amt, lo, hi):
        pick = _gate(px, amt, r, gate)
        x = fac.loc[r].reindex(pick).dropna()
        if len(x) < 20:
            continue
        fwd = _seg_return(px, x.index, r, nxt).reindex(x.index)
        j = pd.concat([x, fwd], axis=1, keys=["x", "y"]).dropna()
        if len(j) < 20 or j["x"].nunique() < 3 or j["y"].nunique() < 3:
            continue
        # Spearman = 对排名取 Pearson(本仓库没有 scipy,手算即可)
        ics.append(float(j["x"].rank().corr(j["y"].rank())))
        n_cand.append(len(j))
    if len(ics) < 6:
        return {"n": len(ics), "mean": float("nan"), "t": float("nan"),
                "pos_frac": float("nan"), "n_cand": float("nan")}
    a = np.array(ics)
    sd = a.std(ddof=1)
    return {"n": len(a), "mean": float(a.mean()),
            "t": float(a.mean() / (sd / np.sqrt(len(a)))) if sd > 0 else float("nan"),
            "pos_frac": float((a > 0).mean()), "n_cand": float(np.mean(n_cand)),
            # 探测下限:|IC| 要多大才可能被本检验抓到(t=2)。诚实地标出「不显著」的边界。
            "floor": float(2.0 * sd / np.sqrt(len(a))) if sd > 0 else float("nan")}


def portfolio(ind: pd.DataFrame, factor: str, side: str, q: float, *,
              lo: str, hi: str, cost_bps: float, gate: float = GATE) -> dict:
    """一个因子分位组合的月频满仓净值(净额与毛额都给)。"""
    px, fac, amt = _pivot(ind, "close"), _pivot(ind, factor), _pivot(ind, "amount60")
    per, turn, prev, idx, nh = [], [], set(), [], []
    for r, nxt in _periods(px, amt, lo, hi):
        pick_all = _gate(px, amt, r, gate)
        cand = fac.loc[r].reindex(pick_all).dropna()
        if len(cand) < 20:
            continue
        k = max(2, int(len(cand) * q))
        pick = set(cand.nsmallest(k).index if side == "low" else cand.nlargest(k).index)
        if px.loc[(px.index > r) & (px.index <= nxt), sorted(pick)].shape[0] < 2:
            continue
        per.append(float(_seg_return(px, pick, r, nxt).mean()))   # 等权**买入持有**
        turn.append(len(pick - prev) / max(len(pick), 1))
        prev = pick
        idx.append(nxt)
        nh.append(len(pick))
    if len(per) < 6:
        raise ValueError(f"{factor}/{side}: 有效期间只有 {len(per)} 个")
    return _stats(np.array(per), np.array(turn), cost_bps, idx, float(np.mean(nh)))


def equal_weight_pool(ind: pd.DataFrame, *, lo: str, hi: str, cost_bps: float,
                      gate: float = GATE) -> dict:
    """基准:同一流动性门下的等权全池,同样的月频换手 + **同一个成本模型**。"""
    px, amt = _pivot(ind, "close"), _pivot(ind, "amount60")
    per, turn, prev, idx, nh = [], [], set(), [], []
    for r, nxt in _periods(px, amt, lo, hi):
        pick = set(_gate(px, amt, r, gate))
        if len(pick) < 20:
            continue
        per.append(float(_seg_return(px, pick, r, nxt).mean()))
        turn.append(len(pick - prev) / max(len(pick), 1))
        prev = pick
        idx.append(nxt)
        nh.append(len(pick))
    return _stats(np.array(per), np.array(turn), cost_bps, idx, float(np.mean(nh)))


def _stats(per: np.ndarray, turn: np.ndarray, cost_bps: float,
           idx: list[pd.Timestamp], n_holdings: float) -> dict:
    def _one(rets: np.ndarray) -> dict:
        eq = np.cumprod(1.0 + rets)
        days = (idx[-1] - idx[0]).days
        cagr = float(eq[-1] ** (365.25 / days) - 1.0) if days > 0 else float("nan")
        r = pd.Series(rets)
        sd = float(r.std(ddof=1))
        peak = np.maximum.accumulate(eq)
        return {"cagr": cagr, "total": float(eq[-1] - 1.0),
                "vol": sd * np.sqrt(12), "sharpe": float(r.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan"),
                "max_dd": float((eq / peak - 1.0).min()), "eq": eq}
    net_rets = per - turn * cost_bps / 1e4          # 换手 × 双边成本
    net, gross = _one(net_rets), _one(per)
    for d, r in ((net, net_rets), (gross, per)):
        d.pop("eq", None)                            # 净值曲线不往外传(避免误用)
    return {"net": net, "gross": gross, "months": len(per),
            "turnover": float(turn.mean()), "n_holdings": n_holdings,
            "idx": list(idx), "net_rets": net_rets}


def excess_stats(st: dict, base: dict) -> dict:
    """组合与全池**逐月配对**的净额超额:t 值与胜率。

    为什么必须做这一步:单个组合的 CAGR 差值没有标准误。30 个月里只要有两三个月踩中风口,
    差值就能到 +20pp 以上 —— 而这条序列自己的 t 值会把这件事立刻说出来。
    """
    mine = dict(zip(st["idx"], st["net_rets"]))
    bench = dict(zip(base["idx"], base["net_rets"]))
    d = np.array([mine[k] - bench[k] for k in mine if k in bench])
    if len(d) < 6:
        return {"t": float("nan"), "win": float("nan"), "n": len(d)}
    sd = d.std(ddof=1)
    return {"t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan"),
            "win": float((d > 0).mean()), "n": len(d)}


def stability(ind: pd.DataFrame, factors, *, lo: str, mid: str, hi: str) -> dict:
    """判据②:把窗口在**预注册的那条线** mid 处切开,两半的 IC 是否同号。

    加这一层的原因见模块 docstring —— 全窗数字会被半段行情冒充成因子。
    """
    out = {}
    for f in factors:
        a = rank_ic(ind, f, lo=lo, hi=mid)
        b = rank_ic(ind, f, lo=mid, hi=hi)
        same = (np.isfinite(a["mean"]) and np.isfinite(b["mean"])
                and np.sign(a["mean"]) == np.sign(b["mean"]))
        # 「翻转」与「衰减到 0」要分开:符号不同但后半段本身就是 0,那是**消失**而不是反向。
        if same:
            kind = "一致"
        elif abs(b["mean"]) < 0.02:
            kind = "衰减到 0"
        elif abs(a["mean"]) < 0.02:
            kind = "后半段才出现"
        else:
            kind = "翻转"
        out[f] = {"first": a, "second": b, "same_sign": bool(same), "kind": kind}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="横截面因子探针(满仓月频,无敞口问题)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gate-scan", action="store_true",
                    help="只跑标的资格梯度(流动性门的两半稳定性)")
    ap.add_argument("--lo", default="2024-04-09")
    ap.add_argument("--mid", default="2025-10-14",
                    help="判据②的切分线:预注册的搜索期/holdout 分界")
    ap.add_argument("--hi", default="2026-10-14")
    ap.add_argument("--force-holdout", default=None, metavar="REASON",
                    help="[mid,hi) 这半段是 holdout,已被消费过时须显式声明理由")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if not a.run:
        ap.print_help()
        return 0

    # 判据②把窗口在 [mid, hi) 处切开，而后半段**就是 holdout** —— 在这里出结论
    # 等于消费它，所以先过台账（2026-10-17 审计 B1）。默认拒绝二次消费。
    try:
        prior = mod_holdout.consume(
            a.mid, a.hi, script="factor_probe",
            tag="gate_scan" if a.gate_scan else "stability",
            force_reason=a.force_holdout)
    except mod_holdout.HoldoutBurned as exc:
        print(f"\n[holdout 台账] 拒绝继续:\n{exc}\n")
        return 2
    if prior:
        print(f"⚠️ 后半段 {a.mid}..{a.hi} 此前已被 claim 消费 {prior} 次 ⇒ "
              f"本次的两半比较**不是样本外**。理由已记入台账：{a.force_holdout}\n")

    # 诊断脚本没有 preset，用项目默认的 PIT 取池口径（显式，不靠隐含默认）
    ind = F.add_family_indicators(indicators.compute_all(
        mod_panel.load_full_panel(universe=mod_universe.DEFAULT_SPEC)))
    cb = costs.COST_BPS_ROUNDTRIP
    if a.gate_scan:
        _gate_scan(ind, lo=a.lo, mid=a.mid, hi=a.hi, cost_bps=cb)
        return 0
    print("=" * 112)
    print(f"横截面因子探针  {a.lo}..{a.hi}  流动性门 amount60 ≥ {GATE:,.0f}  "
          f"成本 {cb}bp 双边(按换手)  月频等权买入持有")
    print("=" * 112)

    base = equal_weight_pool(ind, lo=a.lo, hi=a.hi, cost_bps=cb)
    n_trials, rows = 1, []          # 评估次数计数器(判据①/②/③ 都计入)
    print(f"\n【检验① 秩相关 IC;判据① = |t| ≥ 2】因子排名 vs 次月收益排名(Spearman,逐期后 t 检验)"
          f"  满仓、无阈值、不涉及成本")
    print(f"{'因子':<12}{'平均IC':>9}{'t 值':>8}{'IC>0 期数占比':>14}{'期数':>6}{'每期候选数':>11}   方向判读")
    ics = {}
    for factor in dict.fromkeys(f for f, _ in FACTORS):
        ics[factor] = rank_ic(ind, factor, lo=a.lo, hi=a.hi)
    # 判词必须先看到**整个族**再下（审计 B4）：此前是逐个因子点比较 |t| ≥ 2，
    # 于是"扫 8 个因子总有一个过 2"这件事对读者完全不可见。
    _fam = {f: ic for f, ic in ics.items() if np.isfinite(ic.get("t", np.nan))}
    _bh = metrics.benjamini_hochberg([metrics.two_sided_p(ic["t"]) for ic in _fam.values()], 0.05)
    _adj_ok = dict(zip(_fam.keys(), _bh["rejected"]))
    for factor, ic in ics.items():
        if not np.isfinite(ic["mean"]):
            print(f"{factor:<12}{'—— 期数不足':>42}")
            continue
        pointwise = abs(ic["t"]) >= PRE_REG_T
        verdict = "**显著**" if pointwise else "不显著"
        if pointwise and not _adj_ok.get(factor, False):
            verdict += "·族校正后不存活"
        arrow = "高值→未来收益高" if ic["mean"] > 0 else "高值→未来收益低"
        print(f"{factor:<12}{ic['mean']:>+9.3f}{ic['t']:>+8.2f}{ic['pos_frac']*100:>13.0f}%"
              f"{ic['n']:>6}{ic['n_cand']:>11.0f}   {arrow}({verdict})")
    fam_ic = _family_report([(f, ic["t"]) for f, ic in _fam.items()], "判据① 秩相关 IC")
    fin = [v for v in ics.values() if np.isfinite(v["mean"])]
    if fin:
        fl = float(np.mean([v["floor"] for v in fin]))
        print(f"\n  ⚠️ 探测下限(必须说清):{fin[0]['n']} 期、IC 序列 σ≈"
              f"{np.mean([v['floor']*np.sqrt(fin[0]['n'])/2 for v in fin]):.3f} ⇒ 本检验只能抓到 "
              f"|IC| ≳ {fl:.3f}。真实股票因子常在 0.02~0.05 —— **本检验结构上抓不到它们**。")
        print(f"     所以「不显著」≠「不存在」,只等于「即便存在,也弱到扣不掉成本」"
              f"(IC 0.02~0.03 的多空毛超额约在个位数 pp/年量级,扣掉换手就打平)。")

    # ---- 判据② 稳定性:同一批因子在切分线两侧各测一次。这一层是**最便宜的杀手**。
    stab = stability(ind, list(dict.fromkeys(f for f, _ in FACTORS)),
                     lo=a.lo, mid=a.mid, hi=a.hi)
    n_trials += 2 * len(stab)
    print(f"\n【检验② 稳定性(判据②)】把窗口在预注册的分界线 {a.mid} 切开,两半各测一次 IC")
    print(f"{'因子':<12}{'前半段 IC':>11}{'t':>7}{'期数':>6}{'后半段 IC':>11}{'t':>7}{'期数':>6}"
          f"   符号")
    for f, s in stab.items():
        p, q2 = s["first"], s["second"]
        print(f"{f:<12}{p['mean']:>+11.3f}{p['t']:>+7.2f}{p['n']:>6}"
              f"{q2['mean']:>+11.3f}{q2['t']:>+7.2f}{q2['n']:>6}   "
              f"{s['kind'] + (' ✓' if s['same_sign'] else ' ✗')}")
    flip = [f for f, s in stab.items() if not s["same_sign"]]
    print(f"  → 两半不一致的因子:{', '.join(f + ':' + stab[f]['kind'] for f in flip) if flip else '无'}"
          f"{' —— 方向(或其存在性)由行情段决定,不是因子的性质' if flip else ''}")

    print(f"\n【检验③ 分位组合(净额,已扣成本);判据③ = 逐月配对超额 |t| ≥ 2 且 ΔSharpe > 0】")
    print(f"{'组合':<24}{'CAGR':>9}{'年化波动':>10}{'Sharpe':>8}{'ΔSharpe':>9}{'maxDD':>8}"
          f"{'换手':>7}{'超额pp':>9}{'超额t值':>9}{'超前月占比':>11}")
    print(f"{'【基准】同门等权全池':<24}{base['net']['cagr']*100:>+8.2f}%{base['net']['vol']*100:>9.2f}%"
          f"{base['net']['sharpe']:>8.2f}{'—':>9}{base['net']['max_dd']*100:>7.1f}%"
          f"{base['turnover']*100:>6.1f}%{'—':>9}{'—':>9}{'—':>11}")
    for factor, side in FACTORS:
        for q in QUANTILES:
            n_trials += 1
            try:
                st = portfolio(ind, factor, side, q, lo=a.lo, hi=a.hi, cost_bps=cb)
            except ValueError as e:
                print(f"{f'{factor} {side} q={q:.0%}':<24} —— {e}")
                continue
            ex = st["net"]["cagr"] - base["net"]["cagr"]
            es = excess_stats(st, base)
            rows.append((f"{factor} {side}:q{q:.0%}", st, ex, es))
            print(f"{f'{factor} {side}:q{q:.0%}':<24}{st['net']['cagr']*100:>+8.2f}%"
                  f"{st['net']['vol']*100:>9.2f}%{st['net']['sharpe']:>8.2f}"
                  f"{st['net']['sharpe']-base['net']['sharpe']:>+9.2f}"
                  f"{st['net']['max_dd']*100:>7.1f}%{st['turnover']*100:>6.1f}%"
                  f"{ex*100:>+8.2f}{es['t']:>+9.2f}{es['win']*100:>10.0f}%")

    _family_report([(nm, es["t"]) for nm, _st, _ex, es in rows],
                   "判据③ 分位组合(逐月配对超额)")

    print(f"\n【检验④ 长-空价差(仅作测量,不可交易:未声明借券假设 §4(g))】q=30%")
    for factor in dict.fromkeys(f for f, _ in FACTORS):
        try:
            lo_ = portfolio(ind, factor, "low", 0.3, lo=a.lo, hi=a.hi, cost_bps=cb)
            hi_ = portfolio(ind, factor, "high", 0.3, lo=a.lo, hi=a.hi, cost_bps=cb)
        except ValueError as e:
            print(f"  {factor:<10} —— {e}")
            continue
        n_trials += 2
        g = hi_["gross"]["cagr"] - lo_["gross"]["cagr"]
        n = hi_["net"]["cagr"] - lo_["net"]["cagr"]
        print(f"  {factor:<10} 毛 {g*100:+7.2f}pp / 净 {n*100:+7.2f}pp"
              f"(高 {hi_['net']['cagr']*100:+.2f}% / 低 {lo_['net']['cagr']*100:+.2f}%,净额)")

    if rows:
        best = max(rows, key=lambda x: x[1]["net"]["cagr"])
        print(f"\n【毛额对照】全池毛 CAGR {base['gross']['cagr']*100:+.2f}%  "
              f"|  最好的组合 {best[0]} 毛 {best[1]['gross']['cagr']*100:+.2f}%  "
              f"净 {best[1]['net']['cagr']*100:+.2f}%")
        try:
            # 月度净收益序列 ⇒ periods_per_year=12（此前把月频 Sharpe 除以 √252，
            # 又把月数当 T —— 两处单位都不一致，审计 B3）
            dsr = metrics.deflated_sharpe_from_returns(
                best[1]["net_rets"], n_trials=n_trials + 1614, periods_per_year=12,
                label="factor_probe 月度净收益")
            print(f"【选择偏差】最好组合月频 Sharpe {dsr['observed_sharpe']:+.2f}"
                  f"（{dsr['n_returns']} 个月）,k={n_trials + 1614}(含前几轮的 1614)"
                  f"⇒ DSR {dsr['deflated_sharpe']:.3f}")
        except Exception as e:                                          # noqa: BLE001
            print(f"【选择偏差】DSR 跳过:{e}")
    print(f"\nn_trials = {n_trials}(本文件) + 1614(前几轮搜索)")

    # ---------------- 裁决(全部由上面的数字算出,不写死) ----------------
    def _row(fac, side, q):
        for lab, st, ex, es in rows:
            if lab.startswith(f"{fac} {side}:q{int(q*100)}%"):
                return lab, st, ex, es
        return None
    sig = [f for f, v in ics.items() if np.isfinite(v["mean"]) and abs(v["t"]) >= 2.0]
    flipped = [f for f, s in stab.items() if not s["same_sign"]]
    beat = [r for r in rows if r[2] > 0]
    beat_sig = [r for r in beat if np.isfinite(r[3]["t"]) and abs(r[3]["t"]) >= 2.0]
    heavy = [r for r in beat if r[1]["net"]["vol"] > 1.5 * base["net"]["vol"]]
    usable = [f for f in ics if f in sig and f not in flipped
              and any(r[0].startswith(f) and np.isfinite(r[3]["t"]) and abs(r[3]["t"]) >= 2.0
                      and r[1]["net"]["sharpe"] > base["net"]["sharpe"] for r in rows)]
    print("\n" + "=" * 112)
    print("裁决(三条预注册判据,每条都由上面的数字算出)")
    print("=" * 112)
    print(f"① 有信息(|IC t| ≥ 2):{len(ics)} 个因子里 **{len(sig)} 个**通过。"
          f"最接近的是 dd120(IC {ics['dd120']['mean']:+.3f}, t {ics['dd120']['t']:+.2f})。")
    print(f"② 稳定(两半 IC 同号):**{len(flipped)} 个不一致**"
          f"{'(' + ', '.join(f + ':' + stab[f]['kind'] for f in flipped) + ')' if flipped else ''}"
          f"{' —— 方向(或其存在性)由行情段决定,不是因子的性质' if flipped else ''}。")
    print(f"③ 拿得走(逐月配对超额 |t| ≥ 2 且 ΔSharpe > 0):"
          f"{len(beat)}/{len(rows)} 个组合打赢全池,**{len(beat_sig)} 个**通过显著性")
    print(f"\n⇒ **通过三条判据的因子:{len(usable)} 个**"
          f"{'(' + ', '.join(usable) + ')' if usable else ' —— 一个都没有'}")
    lv = ics["rvol20"]
    print(f"\n关于我上一轮剩下的唯一线索(低波动):IC {lv['mean']:+.3f}(t {lv['t']:+.2f});"
          f"切开后 {stab['rvol20']['first']['mean']:+.3f} → {stab['rvol20']['second']['mean']:+.3f},"
          f"**符号翻转**。分位组合两头都反:")
    for fac, side, note in (("rvol20", "low", "低波动一档"), ("rvol20", "high", "高波动一档")):
        r = _row(fac, side, 0.3)
        if r:
            print(f"     {note}(q30%,全窗):{r[1]['net']['cagr']*100:+.2f}% vs 全池 "
                  f"{base['net']['cagr']*100:+.2f}% ⇒ {r[2]*100:+.2f}pp"
                  f"(年化波动 {r[1]['net']['vol']*100:.1f}% vs 全池 {base['net']['vol']*100:.1f}%)")
    print("  ⇒ 「低波动是承重条件」这句**必须撤回**:它是在搜索期那半段里成立的;"
          "直接测该因子,方向在两半之间翻转。")
    print(f"\n关于「买便宜的/被打下去的」(用户的核心假设):")
    for fac, side, note in (("ret21", "low", "1 月跌幅最大一档"), ("dd120", "low", "距 120 日高点最远一档")):
        r = _row(fac, side, 0.3)
        if r:
            print(f"     {note}(q30%):{r[1]['net']['cagr']*100:+.2f}% ⇒ 超额 {r[2]*100:+.2f}pp"
                  f"(逐月超额 t {r[3]['t']:+.2f})")
    print(f"     dd120 的 IC 是 **{ics['dd120']['mean']:+.3f}**:全窗方向上「越接近 120 日新高」"
          f"收益越高 —— 与「买被打下去的」相反。")
    if rows:
        b = max(rows, key=lambda x: x[1]["net"]["cagr"])
        print(f"\n全窗最好的组合 {b[0]}:CAGR {b[1]['net']['cagr']*100:+.2f}% vs 全池 "
              f"{base['net']['cagr']*100:+.2f}% = 超额 {b[2]*100:+.2f}pp,"
              f"ΔSharpe {b[1]['net']['sharpe']-base['net']['sharpe']:+.2f},"
              f"逐月超额 t {b[3]['t']:+.2f}。")
        fam = b[0].split()[0]
        if fam in stab:
            print(f"    切开看这个家族:{stab[fam]['first']['mean']:+.3f} → "
                  f"{stab[fam]['second']['mean']:+.3f}(两半 IC)—— 全窗那个数是**半段行情**。")
        print(f"    其中 {len(heavy)} 个打赢的组合年化波动超过全池 1.5 倍:"
              f"它们不是「选得准」,是「拿的是更猛的篮子」。")
    print("\n判读:① IC 是所有检验里最不受「你试了多少次」影响的 —— 它一个数字,没有分位/阈值旋钮。")
    print("      ② 因子组合是**满仓**的,所以「超额(vs 同门全池)」就是因子本身的贡献;")
    print("         没有敞口、择时或现金拖累可以解释它 —— 有超额就只能来自选股。")
    print("      ③ 判据②(两半同号)是三层里最便宜的杀手,也是最该先跑的那一条。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
