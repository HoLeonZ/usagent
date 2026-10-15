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
TARGET_POOLED_TEST = 0.1419        # 记录值(pooled test mean CAGR)
TARGET_CONTINUOUS = 0.0815         # 记录值(连续净值 CAGR)

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
    for lo, hi in windows_:
        sig = (F.select(ind, family_id, fparams or {}, lo, hi, reg=reg)
               if family_id is not None else None)
        r = S.evaluate(ind, reg, cfg or {}, lo, hi, signals_df=sig)
        # §5(e) 归因强制:头条收益必须与**基准相对超额**一起报。基准 = 同一个等权池指数
        # (与 uptrend-pullback 用的同一把尺)。没有它,「买被打下去的票」在牛市里跑出的
        # 高 CAGR 会被误读成选股能力。
        bench.append(_bench_cagr(reg, lo, hi))
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
           "cagrs_by_window": by_win}
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


def _bench_cagr(reg: pd.DataFrame, lo: str, hi: str) -> float:
    """同一窗口上等权池指数的 CAGR(与 uptrend-pullback 的基准是同一把尺)。"""
    idx = reg.set_index("date")["ew_index"]
    b = idx[(idx.index >= pd.Timestamp(lo)) & (idx.index < pd.Timestamp(hi))]
    if len(b) < 2:
        return float("nan")
    days = (b.index[-1] - b.index[0]).days
    return float((b.iloc[-1] / b.iloc[0]) ** (365.25 / days) - 1.0) if days > 0 else float("nan")


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
    win50 = sum(1 for x in pcts if x >= 50)
    win90 = sum(1 for x in pcts if x >= 90)
    print(f"  随机对照:≥中位 {win50}/{len(pcts)} 窗   ≥p90 {win90}/{len(pcts)} 窗"
          f"(后者是**描述性**披露:合成复核指出按中位口径 ④ 形同虚设,但判据不事后退让)")
    dsr = metrics.deflated_sharpe_ratio(te["mean_sharpe"], n_trials=max(N_TRIALS["n"], 1),
                                        n_returns=max(2, te["n_trades"]))
    print(f"  DSR(Sharpe {te['mean_sharpe']:+.2f}, k={N_TRIALS['n']}) = "
          f"{dsr['deflated_sharpe']:.3f}")
    return {"family_id": family_id, "family_params": fb, "exit": ex,
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

    p = mod_panel.load_full_panel()
    ind = F.add_family_indicators(indicators.compute_all(p))    # 族指标只算一次
    preset = presets.get_preset("us_uptrend_pullback_v1")
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
              f"{TARGET_POOLED_TEST*100:+.2f}% 差得超过 2pp —— 先查基准线,再比其它族")
    bar_pooled, bar_cont = upb_te["mean_cagr"], upb_cont["cagr"]
    print("\n  判据用的是**同窗口同口径**的比较(§5):池化测试窗与连续净值。"
          "\n  参考:该家族在任何窗口上的最高数是 12m_target 在 2024-10-14..2025-10-14 上的"
          "+51.57%(那是另一个窗口,拿它当线会变成比窗口而不是比策略)。")

    ids = ([x.strip() for x in a.families.split(",")] if a.families else list(F.NEW_FAMILIES))
    results = [report_family(ind, reg, fid, train, test, lo=lo, hi=hi) for fid in ids]

    # ---- 最终裁决
    print("\n" + "=" * 100)
    print(f"裁决(四条判据;基准线:池化测试窗 {bar_pooled*100:+.2f}% / 连续净值 {bar_cont*100:+.2f}%)")
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
        c1 = te["mean_cagr"] > bar_pooled
        c2 = cont["cagr"] > bar_cont
        c3 = (n_eff >= 3) and (pos >= 3)
        c4 = won_null >= 3
        verdict = "**战胜**" if all([c1, c2, c3, c4]) else "未战胜"
        print(f"  {r['family_id']:<12} 池化 {te['mean_cagr']*100:+7.2f}% (超额 {te['excess']*100:+7.2f}pp) "
              f"连续 {cont['cagr']*100:+7.2f}% (超额 {(cont['cagr']-cont['bench'])*100:+7.2f}pp) "
              f"| ①{_tick(c1)} ②{_tick(c2)} ③{pos}/{n_eff} ④{won_null}/{len(r['null_pcts'])} "
              f"(p90 {r['win90']}/{len(r['null_pcts'])})"
              f"\n               → {verdict}  DSR {r['dsr']['deflated_sharpe']:.3f}")
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
