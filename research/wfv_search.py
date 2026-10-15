"""WFV 搜索 —— 带预注册网格与一次性 holdout 的 preset 挖掘(CLAUDE.md §5)。

**预注册声明**:本文件的网格、折窗、选参判据与 holdout 划分在**运行之前**写定,不在看到
结果之后修改。跑的每一次配置都计入 n_trials;最终结论在**未参与选择的 holdout** 上单独
验证一次,并另报一次全量重跑(标 in-sample)。

为什么要有这个文件(而不是继续用 stage1/stage2)
------------------------------------------------
stage1/stage2 是「在同一窗口上取 argmax,再对胜者在同一窗口精调退出参数」。2026-10-15 的
审计与两轮复核把它的后果都实测出来了:

  * 胜者的**五个参数全部落在网格边界**上 —— 边界最优意味着「最优」由网格取景范围决定,
    不由数据决定;
  * 空信号配置会被记成 cagr=+0.0%,在 argmax(CAGR) 里赢过所有负收益配置。

本文件改成:
  * **池化 WFV**:训练窗取参、测试窗报数;每折独立跑,池化后给一个数;
  * 网格预注册,每一次评估计入 n_trials,末尾给 DSR;
  * 零对照从**策略自己的机会集**抽(与 search_12m._control_pool 同源);
  * 一次性 holdout(2025-10-14..2026-10-13)**全程不参与任何选择**。

预注册的网格与协议
------------------
数据划分(库内 2023-10-16..2026-10-13,vintage 见 run_baseline 输出):

    搜索期  2024-04-09 .. 2025-10-13
    holdout 2025-10-14 .. 2026-10-13      (只在最后评估一次)

折窗:train 6m / test 3m / step 3m,test 不越过搜索期末 ⇒ **4 折**(见 folds())

选参:三阶段**池化训练窗** argmax(每阶段网格逐个评估,全部计入 n_trials)
    A 单轴筛选 19 次(6 轴 × 非 None 取值 + 基线 1 次)
    B 三轴组合 64 次:min_dist_high252 × max_stoch_k × min_xs_mom120
    C 退出几何 48 次:sl_mult × tp_mult × max_hold(在 B 的胜者上精调)
    D 杠杆 8 次:risk_per_trade ∈ {1,2,3,5,8,12,20,25}%(**只用于画前沿,不用于选信号**)

A~C 一律固定 sizing(risk_per_trade=1%、max_positions=2),使「信号质量」与「杠杆」不再混在
同一个 argmax 里 —— 这正是旧搜索的归因错误。判据 = 池化训练窗平均 CAGR;头条数字只来自
池化测试窗。
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys

import numpy as np
import pandas as pd

from research import indicators, metrics, panel as mod_panel, presets, regime, search_12m as S

logger = logging.getLogger("wfv_search")

# ---- 预注册常量(改动即等于一次新搜索,n_trials 要跟着涨)------------------------
SEARCH_LO, SEARCH_HI = "2024-04-09", "2025-10-13"
HOLDOUT_LO, HOLDOUT_HI = "2025-10-14", "2026-10-13"
TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS = 6, 3, 3

SIZE_FIXED: dict = {"risk_per_trade": 0.01, "max_positions": 2}

AXES: dict[str, list] = {
    "max_rsi14": [None, 40.0, 50.0, 60.0],
    "max_bb_pctb": [None, 0.2, 0.4],
    "max_stoch_k": [None, 30.0, 50.0, 70.0],
    "min_adx14": [None, 15.0, 20.0, 25.0],
    "min_dist_high252": [None, -0.4, -0.25, -0.1],
    "min_xs_mom120": [None, 0.3, 0.5, 0.7],
}
COMBO: dict[str, list] = {"min_dist_high252": [None, -0.4, -0.25, -0.1],
                          "max_stoch_k": [None, 30.0, 50.0, 70.0],
                          "min_xs_mom120": [None, 0.3, 0.5, 0.7]}
EXIT_GRID: dict[str, list] = {"sl_mult": [1.0, 1.5, 2.0, 3.0],
                              "tp_mult": [2.0, 3.0, 4.0, 6.0],
                              "max_hold": [10, 18, 30]}
RISK_LADDER: list[float] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.25]

N_A = 1 + sum(len(v) - 1 for v in AXES.values())          # 19
N_B = int(np.prod([len(v) for v in COMBO.values()]))      # 64
N_C = int(np.prod([len(v) for v in EXIT_GRID.values()]))  # 48
N_D = len(RISK_LADDER)                                    # 8
N_TRIALS = N_A + N_B + N_C + N_D

EVAL_COUNT = {"n": 0}          # 实际评估次数 —— 与 N_TRIALS 对上才算没偷偷多跑


def folds() -> list[tuple[str, str, str, str]]:
    return S.make_folds(SEARCH_LO, SEARCH_HI, train_months=TRAIN_MONTHS,
                        test_months=TEST_MONTHS, step_months=STEP_MONTHS)


def pooled(ind: pd.DataFrame, reg: pd.DataFrame, cfg: dict,
           windows: list[tuple[str, str]]) -> dict:
    """在一个配置上跑一组窗口,返回逐窗明细与汇总。

    ⚠️ 空信号窗口计进 n_no_signals,**不算作 0 收益**(否则 argmax(CAGR) 会把空转配置选成
       最优 —— 见 search_12m.evaluate 的注释)。每个窗口的成交笔数逐窗记下,供零对照对齐。
    """
    EVAL_COUNT["n"] += 1
    cagrs, dds, ers, sharpes, tcounts, no_sig = [], [], [], [], [], 0
    for lo, hi in windows:
        r = S.evaluate(ind, reg, cfg, lo, hi)
        if r["cagr"] is None:
            no_sig += 1
            tcounts.append(0)
            continue
        cagrs.append(r["cagr"])
        dds.append(r["max_dd"])
        ers.append(r["e_r"])
        sharpes.append(r["sharpe"])
        tcounts.append(r["trades"])
    out = {"n_windows": len(windows), "n_no_signals": no_sig,
           "trades_per_window": tcounts, "n_trades": int(sum(tcounts))}
    if not cagrs:
        out.update({"mean_cagr": None, "mean_dd": None, "worst_dd": None,
                    "mean_e_r": None, "mean_sharpe": None, "cagrs": []})
        return out
    finite_e = [e for e in ers if np.isfinite(e)]
    finite_s = [s for s in sharpes if np.isfinite(s)]
    out.update({
        "mean_cagr": float(np.mean(cagrs)), "cagrs": [float(c) for c in cagrs],
        "mean_dd": float(np.mean(dds)), "worst_dd": float(np.min(dds)),
        "mean_e_r": float(np.mean(finite_e)) if finite_e else float("nan"),
        "mean_sharpe": float(np.mean(finite_s)) if finite_s else float("nan"),
    })
    return out


def scan(ind, reg, grid: list[dict], windows) -> list[tuple[dict, dict]]:
    """逐个评估网格,返回 [(cfg, info)] —— **只评估一次**,调用方复用结果(重评估会虚增 n_trials)。"""
    return [(dict(c), pooled(ind, reg, c, windows)) for c in grid]


def pick(results: list[tuple[dict, dict]], label: str) -> tuple[dict, dict]:
    """池化训练窗 argmax(空信号配置不参与竞争)。"""
    ok = [(c, i) for c, i in results if i["mean_cagr"] is not None]
    if not ok:
        raise RuntimeError(f"{label}:训练窗上没有任何有效配置")
    return max(ok, key=lambda ci: ci[1]["mean_cagr"])


def _fmt_row(tag: str, i: dict) -> str:
    if i["mean_cagr"] is None:
        return f"  {tag:<34} 无有效窗口(空信号窗 {i['n_no_signals']})"
    return (f"  {tag:<34} 平均CAGR {i['mean_cagr']*100:+7.2f}%  "
            f"平均maxDD {i['mean_dd']*100:+6.1f}%  最差 {i['worst_dd']*100:+6.1f}%  "
            f"E[R] {i['mean_e_r']:+.3f}  成交 {i['n_trades']:3d}  "
            f"空信号窗 {i['n_no_signals']}/{i['n_windows']}")


def report(ind: pd.DataFrame, reg: pd.DataFrame) -> dict:
    fs = folds()
    train = [(a, b) for a, b, _, _ in fs]
    test = [(c, d) for _, _, c, d in fs]

    print("=" * 104)
    print(f"预注册搜索:折窗 {len(fs)} 个(train {TRAIN_MONTHS}m / test {TEST_MONTHS}m / "
          f"step {STEP_MONTHS}m),搜索期 {SEARCH_LO}..{SEARCH_HI}")
    print(f"holdout(全程不参与选择):{HOLDOUT_LO}..{HOLDOUT_HI}")
    print(f"n_trials 计划 = {N_TRIALS}(A 单轴 {N_A} + B 组合 {N_B} + C 退出 {N_C} + D 杠杆 {N_D})")
    print("=" * 104)

    # ---------------- A 单轴筛选 ----------------
    print("\n[A] 单轴筛选(池化训练窗,固定 sizing risk=1% / mp=2)")
    a_res = scan(ind, reg, [{}] + [{k: v} for k, vs in AXES.items() for v in vs if v is not None],
                 train)
    for cfg, info in a_res:
        tag = "基线(即 v21)" if not cfg else ", ".join(f"{k}={v!r}" for k, v in cfg.items())
        print(_fmt_row(tag, info))

    # ---------------- B 三轴组合 ----------------
    keys = list(COMBO)
    combos = [{k: v for k, v in zip(keys, c) if v is not None}
              for c in itertools.product(*(COMBO[k] for k in keys))]
    print(f"\n[B] 三轴组合 {len(combos)} 个(池化训练窗 argmax)")
    b_res = scan(ind, reg, combos, train)
    b_best, b_info = pick(b_res, "B")
    for cfg, info in sorted(b_res, key=lambda ci: -ci[1]["mean_cagr"] or 0)[:8]:
        print(_fmt_row(str(cfg), info))
    print(f"  ⇒ B 胜者:{b_best}")

    # ---------------- C 退出几何 ----------------
    exits = [{k: v for k, v in zip(EXIT_GRID, c)}
             for c in itertools.product(*EXIT_GRID.values())]
    c_grid = [{**b_best, **e} for e in exits]
    print(f"\n[C] 退出几何 {len(exits)} 个(信号层固定为 B 的胜者)")
    c_res = scan(ind, reg, c_grid, train)
    c_best, c_info = pick(c_res, "C")
    for cfg, info in sorted(c_res, key=lambda ci: -ci[1]["mean_cagr"] or 0)[:8]:
        print(_fmt_row(f"sl={cfg.get('sl_mult')} tp={cfg.get('tp_mult')} "
                       f"hold={cfg.get('max_hold')}", info))
    print(f"  ⇒ C 胜者:{c_best}")

    sig_exit = {k: v for k, v in c_best.items() if k not in SIZE_FIXED}
    chosen = {**sig_exit, **SIZE_FIXED}

    # ---------------- 测试窗报告(头条数字只来自这里)----------------
    tr = pooled(ind, reg, chosen, train)
    te = pooled(ind, reg, chosen, test)
    te_base = pooled(ind, reg, {}, test)
    print("\n" + "=" * 104)
    print("池化结果(选择只看训练窗;下面**测试窗**那两行才是样本外)")
    print(_fmt_row("训练窗(选择用)", tr))
    print(_fmt_row("测试窗(样本外,头条)", te))
    print(_fmt_row("测试窗 · v21 基线的同窗同 sizing", te_base))
    print(f"  逐折测试窗 CAGR:" +
          "".join(f"  [{fs[i][2][:7]}] {c*100:+.1f}%" for i, c in enumerate(te["cagrs"])))

    # ---------------- D 杠杆前沿(只放大仓位,不重选信号)----------------
    print(f"\n[D] 风险-收益前沿(同一配置,仅改 risk_per_trade;{len(test)} 个测试窗池化)")
    front = []
    for r in RISK_LADDER:
        i = pooled(ind, reg, {**chosen, "risk_per_trade": r}, test)
        front.append((r, i))
        print(_fmt_row(f"risk={r:.0%}", i))
    base_front = []
    for r in RISK_LADDER:
        i = pooled(ind, reg, {k: v for k, v in SIZE_FIXED.items() if k != "risk_per_trade"}
                   | {"risk_per_trade": r}, test)
        base_front.append((r, i))
    print("  对照:v21 基线在同杠杆下")
    for r, i in base_front:
        print(_fmt_row(f"  v21 risk={r:.0%}", i))

    # ---------------- 零对照(从策略自己的机会集抽)----------------
    print("\n[零对照] 同止损止盈、同笔数、同机会集的随机入场 200 次/窗(seed=0)")
    nulls = []
    for idx, (lo, hi) in enumerate(test):
        n = te["trades_per_window"][idx]
        if n <= 0:
            nulls.append(None)
            print(f"  {lo[:7]}:该窗 0 笔,跳过")
            continue
        ctl = S.random_control(ind, chosen, n, n_runs=200, lo=lo, hi=hi, seed=0, reg=reg)
        pct = float((ctl["cagrs"] < te["cagrs"][idx]).mean() * 100.0)
        nulls.append({"lo": lo, "n": n, "pct": pct, "median": ctl["median"],
                      "p90": ctl["p90"], "max": ctl["max"], "failed": ctl["n_failed"]})
        print(f"  {lo[:7]}:策略 {te['cagrs'][idx]*100:+.1f}%  随机中位 {ctl['median']*100:+.1f}%  "
              f"p90 {ctl['p90']*100:+.1f}%  最大 {ctl['max']*100:+.1f}%  "
              f"⇒ 第 {pct:.1f} 百分位  失败 {ctl['n_failed']}")

    # ---------------- DSR ----------------
    dsr = metrics.deflated_sharpe_ratio(te["mean_sharpe"], n_trials=N_TRIALS,
                                        n_returns=max(2, te["n_trades"]))
    print(f"\nDSR:测试窗平均 Sharpe {te['mean_sharpe']:+.2f},k={N_TRIALS} ⇒ "
          f"{dsr['deflated_sharpe']:.3f} (p={dsr['dsr_p_value']:.3f})")

    # ---------------- 一次性 holdout ----------------
    print("\n" + "=" * 104)
    print(f"一次性 holdout(全程未参与任何选择):{HOLDOUT_LO}..{HOLDOUT_HI}")
    ho = pooled(ind, reg, chosen, [(HOLDOUT_LO, HOLDOUT_HI)])
    ho_base = pooled(ind, reg, {}, [(HOLDOUT_LO, HOLDOUT_HI)])
    print(_fmt_row("本次选中配置", ho))
    print(_fmt_row("v21 基线(同窗同 sizing)", ho_base))
    print("\n全量重跑(该配置 over 2024-04-09..2026-10-13;⚠️ 这一段**参与过**上面的选择过程,")
    print("        所以这个数字是 in-sample,不得与上面的样本外数字混用)")
    full = pooled(ind, reg, chosen, [(SEARCH_LO, "2026-10-14")])
    print(_fmt_row("in-sample 全量", full))

    print(f"\n实际评估次数 {EVAL_COUNT['n']} (计划 {N_TRIALS} + 前沿对照 {N_D} + 基线对照 "
          f"{len(test)} + holdout 2 + 全量 1 = {N_TRIALS + N_D + len(test) + 3})")
    return {"chosen": chosen, "train": tr, "test": te, "test_base": te_base,
            "frontier": front, "base_frontier": base_front, "nulls": nulls,
            "dsr": dsr, "holdout": ho, "holdout_base": ho_base, "full": full,
            "folds": fs, "n_trials": N_TRIALS, "eval_count": EVAL_COUNT["n"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WFV 搜索(预注册网格 + 一次性 holdout)")
    ap.add_argument("--run", action="store_true", help="真的跑(默认只打印帮助)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if not a.run:
        ap.print_help()
        return 0
    p = mod_panel.load_full_panel()
    ind = indicators.compute_all(p)
    preset = presets.get_preset("us_uptrend_pullback_v1")
    reg = regime.compute_regime(ind, **(preset.get("regime") or {}))
    report(ind, reg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
