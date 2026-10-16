"""多策略族横向比较 —— 同一套 WFV 协议下,把「价格行为里挖低估」的几个族与
uptrend-pullback 的**当前最佳**放在一起比(CLAUDE.md §5)。

**预注册的「战胜」判据(四条同时满足才算战胜,写在这里意味着跑之前就定死)**

    ① 池化**测试窗**平均 CAGR  >  uptrend-pullback 的同一统计量
    ② **连续净值**口径 CAGR     >  uptrend-pullback 的同一统计量
    ③ 4 折里 ≥3 折的测试窗 CAGR > 0
    ④ 4 折里 ≥3 折打赢「同止损止盈、同笔数、同机会集」的随机入场对照(中位以上)

为什么是四条:上轮实测过 —— 单看 ① 会被单折主导(第一折 +49.2% 把均值抬起来),
单看 ③/④ 又太松(随便什么族在牛市里都能有 3 折为正)。四条**同时**为假的概率远低于任一条;
而且每一条都能被单独扳倒。这是 3 年数据 + 已消费 holdout 的条件下能做的最强约束。

⚠️ holdout 已在上一轮用过一次(2025-10-14..2026-10-13,单次评估了一个预注册配置)。
   所以新族的**绝对水平不能再声称「样本外验证过」**;本文件只报搜索期 + 那一窗的对照。

协议(与 wfv_search.py 完全一致,保证可比):
    搜索期 2024-04-09..2025-10-13,4 折(train 6m / test 3m / step 3m)
    A) 族自己的参数网格 → 池化训练窗 argmax
    B) 退出几何(与上轮同一个 48 格) → 池化训练窗 argmax
    报告的测试窗/连续净值数字**只来自未参与选择的部分**
    每个族的三阶段评估全部计入 n_trials;末尾给 DSR
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys

import numpy as np
import pandas as pd

from research import broker, families as F, indicators, metrics, panel as mod_panel
from research import presets, regime, search_12m as S, wfv_search as W

logger = logging.getLogger("family_search")

# ---- 要跨过的那条线:uptrend-pullback 的当前最佳(由本文件用**同一套代码路径**重算,
#      这里的常数只作自检 —— 实测重算值必须与它们吻合)。
UPB = {"signal": {"min_dist_high252": -0.1, "max_stoch_k": 70.0},
       "exit": {"sl_mult": 2.0, "tp_mult": 3.0, "max_hold": 10}}
# 记录值（**自检用**：本文件会用同一条代码路径重算，实测值必须与它们吻合）。
# ⚠️ 2026-10-17 口径大修后**重算过**：取池换成 PIT 流动性池（去掉幸存者/前视）、
#    加上价差成本、组合层风险上限、窗口末尾清算之后，这条基准线**由正转负**：
#        池化测试窗 +14.19% → -7.71%   连续净值 +8.15% → -4.57%
#        且**逐折全负**（旧口径下是正的）。
#    含义：10 个族要"战胜"的这条线本身已经是个亏钱策略了 ——
#    「战胜 uptrend-pullback」在旧口径下是一个**实质**主张，在新口径下要弱得多
#    （打败一条负的线不等于创造价值；真正的判据是曝光匹配那一条）。
#    这条常数只是**自检**：口径一变它就该跟着变，变了要说明原因（§6）。
TARGET_POOLED_TEST = -0.077061
TARGET_CONTINUOUS = -0.045735

EXIT_GRID = W.EXIT_GRID            # 与上轮同一个退出网格
SIZE_FIXED = W.SIZE_FIXED          # risk 1% / mp 2
N_TRIALS = {"n": 0}


def windows() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    fs = W.folds()
    return ([(a, b) for a, b, _, _ in fs], [(c, d) for _, _, c, d in fs])


def pooled(ind, reg, windows_, *, family_id=None, fparams=None, cfg=None,
           signals_df=None) -> dict:
    """在一个族/配置上跑一组窗口。family_id 给了就走族(_signals),否则走 cfg(uptrend-pullback)。"""
    N_TRIALS["n"] += 1
    cagrs, dds, sharpes, tcounts, no_sig, ers = [], [], [], [], 0, []
    bench, expo = [], []
    # ⚠️ 保留**按窗口对齐**的版本(None = 该窗无信号)。只留紧凑列表会让调用方
    #    (随机对照按窗口索引、基准取平均)错位 —— 实测在 lhr(3/4 窗无信号)上越界崩过。
    by_win: list[float | None] = []
    per_win_rets: list[tuple[tuple[str, str], list[float]]] = []
    for lo, hi in windows_:
        sig = (F.select(ind, family_id, fparams or {}, lo, hi, reg=reg)
               if family_id is not None else None)
        r = S.evaluate(ind, reg, cfg or {}, lo, hi, signals_df=sig)
        # §5(e) 归因强制:头条收益必须与**基准相对超额**一起报。基准 = 同一个等权池指数
        # (与 uptrend-pullback 用的同一把尺)。没有它,「买被打下去的票」在牛市里跑出的
        # 高 CAGR 会被误读成选股能力。
        bench.append(_bench_cagr(reg, lo, hi))
        if r.get("daily_ret") is not None:
            per_win_rets.append(((lo, hi), r["daily_ret"]))
        if r["cagr"] is None:
            no_sig += 1
            tcounts.append(0)
            by_win.append(None)
            continue
        by_win.append(float(r["cagr"]))
        cagrs.append(r["cagr"])
        dds.append(r["max_dd"])
        sharpes.append(r["sharpe"])
        ers.append(r["e_r"])
        expo.append(r["exposure"])
        tcounts.append(r["trades"])
    out = {"n_windows": len(windows_), "n_no_signals": no_sig, "trades_per_window": tcounts,
           "n_trades": int(sum(tcounts)), "n_bench": len(bench),
           "cagrs_by_window": by_win,
           "pooled_daily_ret": metrics.pool_windows_rets(per_win_rets)}
    if not cagrs:
        out.update({"mean_cagr": None, "worst_dd": None, "mean_dd": None,
                    "mean_sharpe": None, "mean_e_r": None, "cagrs": [],
                    "mean_bench": float("nan"), "excess": float("nan"),
                    "mean_exposure": float("nan")})
        return out
    fin_e = [e for e in ers if np.isfinite(e)]
    fin_s = [s for s in sharpes if np.isfinite(s)]
    # 基准也要**按窗口对齐**取平均:用 bench[-len(cagrs):] 在有窗无信号时会取错窗口
    # (实测 lhr 这种 3/4 窗空信号的情形)。这里只对有信号的那些窗口取基准。
    _b = [bench[i] for i in range(len(windows_)) if by_win[i] is not None]
    bm = float(np.mean(_b)) if _b else float("nan")
    out.update({"mean_cagr": float(np.mean(cagrs)), "cagrs": [float(c) for c in cagrs],
                "mean_dd": float(np.mean(dds)), "worst_dd": float(np.min(dds)),
                "mean_sharpe": float(np.mean(fin_s)) if fin_s else float("nan"),
                "mean_e_r": float(np.mean(fin_e)) if fin_e else float("nan"),
                "mean_bench": bm, "excess": float(np.mean(cagrs)) - bm if cagrs else float("nan"),
                "mean_exposure": float(np.mean(expo)) if expo else float("nan")})
    return out


def overlay(ind, reg, windows_, *, family_id=None, fparams=None, cfg=None) -> dict:
    """**闲置资金按等权池计息**后的收益 —— 对低敞口策略唯一公平的口径。

    为什么需要它:一个平均敞口 7.7% 的策略与 100% 敞口的指数比原始收益是**结构性不公平**的
    (它大部分时间在持有零息现金)。但那不等于没法比 —— 真正该问的是「这套信号在它**在场**
    的那些天里,是否优于随便持有」。做法:

        portfolio_ret(t) = w(t-1) × 策略日收益(t) + (1 − w(t-1)) × 池指数日收益(t)

    其中 w 取**前一交易日**的敞口:当日敞口要到收盘才知道,用它会引入同日前视。
    这是仓库设计文档要求的「曝光匹配」口径(metrics.beta_attribution 的
    bench_on_exposed_days 是它的一个变体)。

    判读:overlay CAGR > 池指数 CAGR ⇒ 这套信号**创造了价值**(择时/选股);
    ≈ 相等 ⇒ 它只是把敞口换成了一张更曲折的指数;明显更低 ⇒ 它的在场是负贡献。
    """
    idx = reg.set_index("date")["ew_index"]
    bret = idx.pct_change()
    outs, benchs, expos = [], [], []
    for lo, hi in windows_:
        sig = (F.select(ind, family_id, fparams or {}, lo, hi, reg=reg)
               if family_id is not None else None)
        if sig is not None and sig.empty:
            benchs.append(_bench_cagr(reg, lo, hi))
            continue
        res = broker.simulate(S.make_preset(cfg or {}), sig, ind, start=lo, end=hi)
        eq = res["equity"].set_index("date")
        if len(eq) < 3:
            benchs.append(_bench_cagr(reg, lo, hi))
            continue
        sret = eq["equity"].pct_change().fillna(0.0)
        w = eq["exposure"].shift(1).fillna(0.0)          # 前一日敞口(因果)
        b = bret.reindex(eq.index).fillna(0.0)
        ov = (w * sret + (1.0 - w) * b)
        days = (eq.index[-1] - eq.index[0]).days
        c = float((1.0 + ov).prod() ** (365.25 / days) - 1.0) if days > 0 else float("nan")
        outs.append(c)
        benchs.append(_bench_cagr(reg, lo, hi))
        expos.append(float(w.mean()))
    if not outs:
        return {"cagr": None, "bench": None, "excess": None, "n": 0}
    cb = float(np.nanmean(benchs)) if benchs else float("nan")
    co = float(np.mean(outs))
    return {"cagr": co, "bench": cb, "excess": co - cb, "n": len(outs),
            "exposure": float(np.mean(expos)) if expos else float("nan")}


def _bench_cagr(reg: pd.DataFrame, lo: str, hi: str) -> float:
    """同一窗口上等权池指数的 CAGR。委托给 `regime.window_benchmark_cagr`（唯一定义）。"""
    from research.regime import window_benchmark_cagr
    return window_benchmark_cagr(reg, lo, hi)


def continuous(ind, reg, lo, hi, *, family_id=None, fparams=None, cfg=None) -> dict:
    """**连续净值**口径:一个账户从 lo 跑到 hi,不按折重置资金。连同基准一起返回。"""
    N_TRIALS["n"] += 1
    sig = (F.select(ind, family_id, fparams or {}, lo, hi, reg=reg)
           if family_id is not None else None)
    r = S.evaluate(ind, reg, cfg or {}, lo, hi, signals_df=sig)
    return {"cagr": float("nan") if r["cagr"] is None else r["cagr"],
            "bench": _bench_cagr(reg, lo, hi),
            "exposure": r.get("exposure", float("nan"))}


def pool_on_signal_days(ind, reg, family_id, fparams, lo, hi) -> pd.DataFrame:
    """聚簇对照的抽样池 = **该族自己的信号日** × 那些日子上可交易的标的。

    为什么必须这样:信号在时间上聚簇(实测 str 单日最多 142 个),从全窗口所有日随机抽
    得到的对照是**分散**的,而策略是聚簇的 —— 两者比的就不是「选股」而是「选日子」。
    """
    days = set(pd.to_datetime(
        F.select(ind, family_id, fparams, lo, hi, reg=reg)["date"]).tolist())
    # ⚠️ 这里**不能**传 reg:regime_ok 是 uptrend-pullback 的门。合成复核实测:
    #    注册池上 regime_ok 日的 5 日前瞻均值 +0.353%,非 regime_ok 日 +1.015% ——
    #    给一个 100% 在非 regime 日开火的族套上 regime 对照,会把它系统性抬高一档。
    #    每族的池只保留「该族自己口径下可交易的标的 × 该族自己的信号日」。
    pool = S._control_pool(ind, lo, hi, reg=None)
    return pool[pool["date"].isin(days)]


def _pick_exit(ind, reg, train, *, family_id, fparams, label: str) -> tuple[dict, dict]:
    grid = [{k: v for k, v in zip(EXIT_GRID, c)} for c in itertools.product(*EXIT_GRID.values())]
    best, best_info = None, None
    for ex in grid:
        cfg = {**ex, **SIZE_FIXED}
        info = pooled(ind, reg, train, family_id=family_id, fparams=fparams, cfg=cfg)
        if info["mean_cagr"] is None:
            continue
        if best is None or info["mean_cagr"] > best_info["mean_cagr"]:
            best, best_info = dict(ex), info
    if best is None:
        raise RuntimeError(f"{label}:退出网格里没有任何有效配置")
    return best, best_info


def _fmt(tag: str, i: dict) -> str:
    if i.get("mean_cagr") is None:
        return f"  {tag:<30} 无有效窗口(空信号窗 {i['n_no_signals']})"
    med = float(np.median(i["cagrs"])) if i.get("cagrs") else float("nan")
    return (f"  {tag:<26} CAGR {i['mean_cagr']*100:+7.2f}%  中位 {med*100:+7.2f}%  "
            f"基准 {i.get('mean_bench', float('nan'))*100:+6.2f}%  "
            f"超额 {i.get('excess', float('nan'))*100:+7.2f}pp  "
            f"maxDD {i['mean_dd']*100:+5.1f}%  敞口 {i.get('mean_exposure', float('nan'))*100:4.1f}%  "
            f"成交 {i['n_trades']:3d}  空信号窗 {i['n_no_signals']}/{i['n_windows']}")


def c1_on_common_windows(f_by: list, b_by: list, bar_pooled: float
                         ) -> tuple[bool, str]:
    """判据① 的取值：**只在共同窗口集上比**，集合不同就退回共同窗重算并出声。

    为什么（审计 B5）：本族的 `mean_cagr` 只在**它有信号的那些窗**上取均值，基准线同理。
    两批样本不一样时，"均值比均值"比的就不只是策略 —— 一个只在 2 个窗出信号的族，
    可能在另外 2 个基准线亏钱的窗里根本没参与过。集合不同 ⇒ 退回交集重算。
    纯函数，便于单独测（内联在裁决循环里的话没法测）。
    """
    n = max(len(f_by), len(b_by))
    f = list(f_by) + [None] * (n - len(f_by))
    b = list(b_by) + [None] * (n - len(b_by))
    common = [i for i in range(n) if f[i] is not None and b[i] is not None]
    only_f = [i for i in range(n) if f[i] is not None and b[i] is None]
    only_b = [i for i in range(n) if f[i] is None and b[i] is not None]
    if not (only_f or only_b):
        return bool(np.mean(f) > float(bar_pooled)), ""
    if not common:
        return False, "⚠️ 与基准线**没有一个共同窗** ⇒ ①无法判定（按不通过处理）"
    m_f = float(np.mean([f[i] for i in common]))
    m_b = float(np.mean([b[i] for i in common]))
    note = (f"  ⚠️ 与基准线的有效窗口集不同（本族独有 {len(only_f)} 窗 / 基准独有 "
            f"{len(only_b)} 窗）⇒ ①改用**共同 {len(common)} 窗**重算："
            f"本族 {m_f*100:+.2f}% vs 基准 {m_b*100:+.2f}%")
    return bool(m_f > m_b), note


def report_family(ind, reg, family_id: str, train, test, *,
                  lo: str, hi: str) -> dict:
    name = (F.MASK[family_id].__doc__ or "").strip().splitlines()[0]
    print("\n" + "=" * 100)
    print(f"族:{family_id}  {name}")
    print(f"  预注册网格大小 {F.grid_size(family_id)}(× {len(train)} 个训练窗)")
    keys = list(F.GRIDS[family_id])
    grid = [{k: v for k, v in zip(keys, c)} for c in itertools.product(*F.GRIDS[family_id].values())]
    scored = []
    for fp in grid:
        scored.append((fp, pooled(ind, reg, train, family_id=family_id, fparams=fp,
                                  cfg=dict(SIZE_FIXED))))
    ok = [(fp, i) for fp, i in scored if i["mean_cagr"] is not None]
    if not ok:
        print("  ✗ 该族全部配置在训练窗上都没有信号 —— 直接判不可用")
        return {"family_id": family_id, "verdict": "no_signals"}
    fb, fb_info = max(ok, key=lambda x: x[1]["mean_cagr"])
    for fp, i in sorted(ok, key=lambda x: -x[1]["mean_cagr"])[:5]:
        print(_fmt(str(fp), i))
    print(f"  ⇒ A 胜者 {fb}(训练窗平均 CAGR {fb_info['mean_cagr']*100:+.2f}%)")

    ex, ex_info = _pick_exit(ind, reg, train, family_id=family_id, fparams=fb, label=family_id)
    print(f"  ⇒ B 退出几何 {ex}(训练窗平均 CAGR {ex_info['mean_cagr']*100:+.2f}%)")

    cfg = {**ex, **SIZE_FIXED}
    tr = pooled(ind, reg, train, family_id=family_id, fparams=fb, cfg=cfg)
    te = pooled(ind, reg, test, family_id=family_id, fparams=fb, cfg=cfg)
    cont = continuous(ind, reg, lo, hi, family_id=family_id, fparams=fb, cfg=cfg)
    print(f"  连续净值 CAGR {cont['cagr']*100:+.2f}%  基准 {cont['bench']*100:+.2f}%  "
          f"超额 {(cont['cagr']-cont['bench'])*100:+.2f}pp  平均敞口 {cont['exposure']*100:.1f}%")
    print(_fmt("训练窗(选择用)", tr))
    print(_fmt("测试窗(样本外统计量)", te))
    print("  逐折测试窗 CAGR:" + "".join(
        f"  [{W.folds()[i][2][:7]}] " + ("(无信号)" if c is None else f"{c*100:+.1f}%")
        for i, c in enumerate(te["cagrs_by_window"])))


    # 随机对照:逐窗
    pcts, nulls = [], []
    for idx, (wlo, whi) in enumerate(test):
        n = te["trades_per_window"][idx]
        win_cagr = te["cagrs_by_window"][idx]
        if n <= 0 or win_cagr is None:
            continue
        ctl = S.random_control(ind, cfg, n, n_runs=200, lo=wlo, hi=whi, seed=0, reg=reg,
                               pool=pool_on_signal_days(ind, reg, family_id, fb, wlo, whi))
        pct = float((ctl["cagrs"] < win_cagr).mean() * 100.0)
        pcts.append(pct)
        nulls.append({"lo": wlo, "pct": pct, "median": ctl["median"], "failed": ctl["n_failed"]})
        print(f"    随机对照 {wlo[:7]}:策略 {win_cagr*100:+.1f}%  "
              f"随机中位 {ctl['median']*100:+.1f}%  ⇒ 第 {pct:.1f} 百分位  失败 {ctl['n_failed']}")
    ov = overlay(ind, reg, test, family_id=family_id, fparams=fb, cfg=cfg)
    ov_cont = overlay(ind, reg, [(lo, hi)], family_id=family_id, fparams=fb, cfg=cfg)
    print(f"  闲置资金计息(曝光匹配,平滑口径):测试窗 overlay CAGR "
          f"{0 if ov['cagr'] is None else ov['cagr']*100:+.2f}% vs 基准 "
          f"{0 if ov['bench'] is None else ov['bench']*100:+.2f}% ⇒ 超额 "
          f"{0 if ov['excess'] is None else ov['excess']*100:+.2f}pp | 连续 "
          f"{ov_cont['cagr']*100:+.2f}% vs 基准 {ov_cont['bench']*100:+.2f}% ⇒ "
          f"{ov_cont['excess']*100:+.2f}pp")
    win50 = sum(1 for x in pcts if x >= 50)
    win90 = sum(1 for x in pcts if x >= 90)
    print(f"  随机对照:≥中位 {win50}/{len(pcts)} 窗   ≥p90 {win90}/{len(pcts)} 窗"
          f"(后者是**描述性**披露:合成复核指出按中位口径 ④ 形同虚设,但判据不事后退让)")
    # DSR 从**一条收益序列**算（审计 B3）：此前的口径是"逐窗 Sharpe 的均值 +
    # 成交笔数当 T"，那不是任何一条序列的 DSR。拿不到可拼接序列就不报。
    _rets = te.get("pooled_daily_ret")
    if _rets:
        dsr = metrics.deflated_sharpe_from_returns(
            _rets, n_trials=max(N_TRIALS["n"], 1), label=f"{family_id} 测试期拼接")
        print(f"  DSR({dsr['n_returns']} 个日收益,年化 Sharpe {dsr['observed_sharpe']:+.2f},"
              f"偏度 {dsr['skewness']:+.2f},峰度 {dsr['kurtosis']:.1f},k={N_TRIALS['n']}) = "
              f"{dsr['deflated_sharpe']:.3f}")
    else:
        dsr = {"deflated_sharpe": float("nan"), "n_trials": max(N_TRIALS["n"], 1),
               "reason": "no_poolable_returns"}
        print("  DSR：**不报** —— 无可拼接的收益序列（窗口重叠或全窗无信号）")
    return {"family_id": family_id, "family_params": fb, "exit": ex, "overlay": ov,
            "overlay_cont": ov_cont,
            "train": tr, "test": te, "continuous": cont, "null_pcts": pcts,
            "win90": win90, "dsr": dsr, "n_trials_at_done": N_TRIALS["n"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多策略族横向 WFV 比较")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--families", default=None, help="逗号分隔;默认全部")
    ap.add_argument("--k-offset", type=int, default=0,
                    help="把之前几轮的评估次数累加进 n_trials(分轮跑时必须给,否则 DSR 会虚高)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if not a.run:
        ap.print_help()
        return 0
    N_TRIALS["n"] = int(a.k_offset)
    if a.k_offset:
        print(f"(n_trials 起点 = {a.k_offset},来自之前的轮次;本轮的 DSR 按累计 k 计算)")

    preset = presets.get_preset("us_uptrend_pullback_v1")
    p = mod_panel.load_panel_for_preset(preset)
    ind = F.add_family_indicators(indicators.compute_all(p))    # 族指标只算一次
    reg = regime.compute_regime(ind, **(preset.get("regime") or {}))
    train, test = windows()
    lo, hi = W.SEARCH_LO, W.SEARCH_HI

    print("=" * 100)
    print(f"横向比较:搜索期 {lo}..{hi},{len(train)} 折(train 6m / test 3m / step 3m)")
    print(f"要跨过的线:uptrend-pullback 的当前最佳(由本文件用同一条代码路径**重算**)")
    print("=" * 100)

    # ---- 基准线:uptrend-pullback 的 WFV 配置,走 cfg 路径(不带 signals_df)
    upb_cfg = {**UPB["signal"], **UPB["exit"], **SIZE_FIXED}
    upb_te = pooled(ind, reg, test, cfg=upb_cfg)
    upb_cont = continuous(ind, reg, lo, hi, cfg=upb_cfg)
    print(f"\n[基准] uptrend-pullback WFV 配置 {UPB}")
    print(_fmt("测试窗", upb_te))
    print(f"  逐折:" + "".join(f"  {c*100:+.1f}%" for c in upb_te["cagrs"]))
    print(f"  连续净值 CAGR {upb_cont['cagr']*100:+.2f}%  基准 {upb_cont['bench']*100:+.2f}%  "
          f"超额 {(upb_cont['cagr']-upb_cont['bench'])*100:+.2f}pp"
          f"(记录值 {TARGET_CONTINUOUS*100:+.2f}%)")
    print(f"  ⚠️ 注意这条线自己的超额是**负的**:{upb_te['excess']*100:+.2f}pp(池化) / "
          f"{(upb_cont['cagr']-upb_cont['bench'])*100:+.2f}pp(连续)。")
    if abs(upb_te["mean_cagr"] - TARGET_POOLED_TEST) > 0.02:
        print(f"  ⚠️ 重算的池化测试窗 {upb_te['mean_cagr']*100:+.2f}% 与记录值 "
              f"{TARGET_POOLED_TEST*100:+.2f}% 差得超过 2pp。"
              f"\n     若你确实改了口径/取池/成本：更新 TARGET_* 并说明成因（§6）；"
              f"\n     若不知道为什么会变：**先查基准线**，再拿它去比其它族。")
    bar_pooled, bar_cont = upb_te["mean_cagr"], upb_cont["cagr"]
    print("\n  判据用的是**同窗口同口径**的比较(§5):池化测试窗与连续净值。"
          "\n  参考:该家族在任何窗口上的最高数是 12m_target 在 2024-10-14..2025-10-14 上的"
          "+51.57%(那是另一个窗口,拿它当线会变成比窗口而不是比策略)。")

    ids = ([x.strip() for x in a.families.split(",")] if a.families else list(F.NEW_FAMILIES))
    results = [report_family(ind, reg, fid, train, test, lo=lo, hi=hi) for fid in ids]

    # ---- 最终裁决
    print("\n" + "=" * 100)
    print(f"裁决(四条判据;基准线:池化测试窗 {bar_pooled*100:+.2f}% / 连续净值 {bar_cont*100:+.2f}%)")
    # 判据的**零边缘假阳性率**（审计 B6）：不写清楚，"四条同时满足"会被读成
    # 一件比它更强的事 —— 其中两条各自都有 5/16 的概率在毫无边缘时为真。
    _nr = metrics.judgement_null_rates(len(test))
    print("  判据的零边缘假阳性率（各折独立近似，仅给数量级）:")
    for _k, _v in _nr["rates"].items():
        print(f"    {_k:<28s} ≈ {_v*100:.1f}%")
    print(f"    ⇒ 四条**同时**为真 ≈ {_nr['joint']*100:.1f}%"
          f"（独立近似；实际因判据间相关而不同）—— 这不是 α=5% 的证据标准")
    print("    ⚠️ ③④ 用的是**固定次数**（≥3 折）⇒ 折数越多越松"
          "（P(≥3|4 折)=31.2% vs P(≥3|6 折)=65.6%）。要收紧就该改成**比例**或固定 α，"
          "而不是「多切几折」—— 那是反方向。")
    print("=" * 100)
    for r in results:
        if r.get("verdict") == "no_signals":
            print(f"  {r['family_id']:<12} ✗ 训练窗上无信号 —— 不可用")
            continue
        te, cont = r["test"], r["continuous"]
        # 判据③的分母必须是**有信号的窗数**,不是 4 —— 否则无信号窗被当成「不为正」
        n_eff = sum(1 for c in te["cagrs_by_window"] if c is not None)
        pos = sum(1 for c in te["cagrs"] if c > 0)
        won_null = sum(1 for p in r["null_pcts"] if p >= 50)
        # ① 必须在**共同窗口集**上比（审计 B5）：本族的 mean_cagr 只在它有信号的那些窗上取均值，
        # 基准线同理 —— 两批样本不一样时，"均值比均值"比的就不只是策略。
        # 集合不同 ⇒ 退回共同窗重算，并把这件事说出来。
        c1, note = c1_on_common_windows(te["cagrs_by_window"], upb_te["cagrs_by_window"],
                                        bar_pooled)
        c2 = cont["cagr"] > bar_cont
        c3 = (n_eff >= 3) and (pos >= 3)
        c4 = won_null >= 3
        verdict = "**战胜**" if all([c1, c2, c3, c4]) else "未战胜"
        print(f"  {r['family_id']:<12} 池化 {te['mean_cagr']*100:+7.2f}% (超额 {te['excess']*100:+7.2f}pp) "
              f"连续 {cont['cagr']*100:+7.2f}% (超额 {(cont['cagr']-cont['bench'])*100:+7.2f}pp) "
              f"| overlay超额 {r['overlay_cont']['excess']*100:+7.2f}pp "
              f"| ①{_tick(c1)} ②{_tick(c2)} ③{pos}/{n_eff} ④{won_null}/{len(r['null_pcts'])} "
              f"(p90 {r['win90']}/{len(r['null_pcts'])})"
              f"\n               → {verdict}  DSR {r['dsr']['deflated_sharpe']:.3f}")
        if note:
            print(note)
    print("\n归因说明(§5(e) 要求报超额,但两种口径都要读懂):")
    print("  * 所有族与基准线跑的是**同一套 sizing**(risk 1% / mp 2)⇒ 敞口在同一个量级")
    print("    (实测线 7.4%),所以**原始 CAGR 的比较本身就是敞口匹配的** —— 这是预注册 ①② 的口径。")
    print("  * 「相对全池指数(100% 敞口、日度再平衡)的原始超额」对低敞口策略**不构成能力判据**:")
    print("    5% 敞口的策略永远跑不过 100% 敞口的指数。这一列只用来回答「这里面有多少是 beta」。")
    print("  * 判 beta 还是选股,看的是**随机对照臂**(同日随机票 + 同样的止损止盈/笔数/机会集)")
    print("    与**对照臂**(如 str 的买涨臂、cap 的 clv 反向臂),不是看那列超额。")
    print(f"\n实际评估次数 {N_TRIALS['n']}(所有族的所有阶段都计入 n_trials)")
    return 0


def _tick(b: bool) -> str:
    return "✓" if b else "✗"


if __name__ == "__main__":
    sys.exit(main())
