"""最近 12 个月的 preset 搜索 —— 带**运气对照**。

目的(用户口径):在 uptrend_pullback 骨架不变的前提下,用新增指标做过滤层,
找出最近 12 个月 CAGR > 30% 的 preset,并把止损止盈控制好。

⚠️ 三件必须一起读的事
=====================
1. **这个目标可被运气通过。** 最近 12 个月里,596 只票中有 **27%(160 只)**
   自身上涨超过 30% CAGR,而**中位数是 −5.8%**。一个约 20 笔交易的单持仓策略
   只要碰上其中一只就破 30%。所以"搜到了"本身不构成证据。

2. **因此本脚本内置对照:`random_control`。** 用**同样的止损止盈、同样的交易
   笔数**,随机选入场点跑 N 次,给出"纯运气能达到的 CAGR 区间"。
   搜索出来的 preset **如果落在那个区间内,它就是运气产物** —— 报告里会直说。

3. **n_trials 如实计数。** 每评估一个配置就加一。DSR 会因此塌到近 0,照报。

用法
----
    python -m research.search_12m --stage1              # 单轴扫描,看哪个轴有信息
    python -m research.search_12m --stage2              # 组合 + 止损止盈精调
    python -m research.search_12m --control             # 运气对照(必做)
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker, indicators, metrics, panel as panel_mod, presets, regime, signals

logger = logging.getLogger("search")

# 最近 12 个月(库内最后一个交易日回推一年)
WIN_LO, WIN_HI = "2025-10-14", "2026-10-14"
FULL_LO = "2024-04-09"          # 全期起点(预热之后),用于 OOS 反查
TARGET_CAGR = 0.30

# 单轴候选。None = 不使用该过滤层(即 v21 原口径)。
AXES: dict[str, list] = {
    "max_rsi14": [None, 30.0, 40.0, 50.0, 60.0],
    "max_bb_pctb": [None, 0.1, 0.3, 0.5],
    "max_stoch_k": [None, 20.0, 40.0, 60.0],
    "min_adx14": [None, 15.0, 20.0, 25.0],
    "min_dist_high252": [None, -0.5, -0.3, -0.1],
    "min_xs_mom120": [None, 0.3, 0.5, 0.7],
    "sl_mult": [1.0, 1.5, 2.0, 3.0],
    "tp_mult": [2.0, 3.0, 4.0, 6.0, 10.0],
    "max_hold": [10, 18, 30],
    # 敞口轴。**加这一轴的目的是证明一件事**:CAGR 由敞口驱动,不由过滤层驱动。
    # 期望看到 CAGR 随 risk_per_trade 单调上升,而 E[R] 几乎不变。
    "risk_per_trade": [0.01, 0.03, 0.05, 0.10, 0.25],
}
SIGNAL_AXES = {"max_rsi14", "max_bb_pctb", "max_stoch_k", "min_adx14",
               "min_dist_high252", "min_xs_mom120"}
EXIT_AXES = {"sl_mult", "tp_mult", "max_hold"}
SIZE_AXES = {"risk_per_trade"}

BASE_EXIT = {"sl_mult": 1.5, "tp_mult": 4.0, "max_hold": 18}


class SearchError(RuntimeError):
    pass


def make_preset(cfg: dict) -> dict:
    """把搜索配置翻成 broker 能吃的 preset。"""
    return {
        "atr_sl_mult": cfg.get("sl_mult", BASE_EXIT["sl_mult"]),
        "atr_tp_mult": cfg.get("tp_mult", BASE_EXIT["tp_mult"]),
        "max_hold": cfg.get("max_hold", BASE_EXIT["max_hold"]),
        "max_positions": cfg.get("max_positions", 1),
        "risk_per_trade": cfg.get("risk_per_trade", 0.01),
        "max_volume_participation": 0.10,
        "intraday_tiebreak": "sl_first",
        "initial_cash": 100_000.0,
    }


def signal_kwargs(cfg: dict) -> dict:
    kw = dict(signals.V21_SIGNAL_PARAMS)
    for k in SIGNAL_AXES:
        if cfg.get(k) is not None:
            kw[k] = cfg[k]
    return kw


def evaluate(ind: pd.DataFrame, reg: pd.DataFrame, cfg: dict,
             lo: str = WIN_LO, hi: str = WIN_HI, *, use_regime: bool = True) -> dict:
    """跑一个配置,返回窗口指标。信号集为空 → 返回 cagr=None(不静默当 0)。"""
    kw = signal_kwargs(cfg)
    try:
        sig = signals.select_entries(ind, start_date=lo, end_date=hi,
                                     regime_df=reg if use_regime else None, **kw)
    except signals.SignalError:
        return {"cagr": None, "trades": 0, "reason": "no_signals", "cfg": cfg}

    preset = make_preset(cfg)
    try:
        res = broker.simulate(preset, sig, ind, start=lo, end=hi,
                              cost_bps_roundtrip=5.0)
    except broker.BrokerError as exc:
        return {"cagr": None, "trades": 0, "reason": str(exc)[:40], "cfg": cfg}

    eq, tr = res["equity"], res["trades"]
    cagr = metrics.cagr(eq["equity"], eq["date"])
    ts = metrics.tail_stats(tr) if len(tr) else {"expectancy_r": float("nan")}
    return {
        "cagr": float(cagr), "trades": int(len(tr)),
        "total": float(res["metrics"]["total_return"]),
        "max_dd": float(res["metrics"]["max_dd"]),
        "win": float(res["metrics"].get("win_rate", float("nan"))),
        "e_r": float(ts.get("expectancy_r", float("nan"))),
        "exposure": float(res["metrics"]["avg_exposure"]),
        "reason": "ok", "cfg": cfg,
    }


# ---------------------------------------------------------------- 运气对照

def random_control(ind: pd.DataFrame, cfg: dict, n_trades: int, *,
                   n_runs: int = 200, lo: str = WIN_LO, hi: str = WIN_HI,
                   seed: int = 0) -> dict:
    """同止损止盈、同交易笔数的**随机入场**对照。

    做法:从窗口内随机抽 n_trades 个 (symbol, date) 作为入场点,套用与候选
    完全相同的 preset 跑一遍。重复 n_runs 次 → 得到"纯运气"的 CAGR 分布。

    这是本脚本里唯一能区分"策略"与"运气"的东西。
    """
    rng = np.random.default_rng(seed)
    win = ind[(ind["date"] >= pd.Timestamp(lo)) & (ind["date"] < pd.Timestamp(hi))]
    # 只在有信号资格的日期里抽(指标就绪),否则随机点会落在预热期
    ok = win[win["ma120"].notna() & win["atr_pct"].notna()]
    pool = ok[["symbol", "date", "atr_pct"]].to_numpy(dtype=object)
    if len(pool) == 0:
        raise SearchError("随机对照的候选池为空")

    preset = make_preset(cfg)
    cagrs = []
    for _ in range(n_runs):
        idx = rng.choice(len(pool), size=min(n_trades, len(pool)), replace=False)
        # 必须补全 ENTRY_COLS 的全部列(broker 只用 date/symbol/score/atr_pct,
        # 其余留 NaN 即可)—— 少列会在 select 时 KeyError。
        rows = []
        for i in idx:
            row = {c: np.nan for c in signals.ENTRY_COLS}
            row.update({"date": pool[i][1], "symbol": pool[i][0],
                        "score": 1.0, "atr_pct": float(pool[i][2])})
            rows.append(row)
        sig = pd.DataFrame(rows)[signals.ENTRY_COLS]
        try:
            res = broker.simulate(preset, sig, ind, start=lo, end=hi,
                                  cost_bps_roundtrip=5.0)
            c = metrics.cagr(res["equity"]["equity"], res["equity"]["date"])
            if np.isfinite(c):
                cagrs.append(float(c))
        except broker.BrokerError:
            continue
    a = np.array(cagrs)
    if len(a) == 0:
        raise SearchError("随机对照全部失败")
    return {
        "n_runs": int(len(a)), "n_trades": int(n_trades),
        "median": float(np.median(a)), "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)), "max": float(a.max()),
        "p_ge_target": float((a >= TARGET_CAGR).mean()),
        # 原始分布一并返回 —— 调用方要用它算"策略落在第几百分位"。
        # 早先 render_report 把 p90 硬编码成常数,独立验证指出那是不可复现的。
        "cagrs": a,
    }


# ---------------------------------------------------------------- 扫描

def stage1(ind, reg) -> tuple[pd.DataFrame, int]:
    """单轴扫描:每次只动一个轴,其余保持基线。用于看**哪个轴携带信息**。"""
    rows, n = [], 0
    base = evaluate(ind, reg, {})
    n += 1
    base.update(axis="BASE(v21①)", value="—")
    rows.append(base)

    for axis, vals in AXES.items():
        for v in vals:
            if v is None:
                continue                       # None == 基线,已算过
            cfg = {axis: v}
            r = evaluate(ind, reg, cfg)
            n += 1
            r.update(axis=axis, value=v)
            rows.append(r)
    return pd.DataFrame(rows), n


def _fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    d["cagr"] = d["cagr"].map(lambda v: "—" if v is None or not np.isfinite(v)
                              else f"{v*100:+.1f}%")
    d["total"] = d["total"].map(lambda v: "—" if v is None or not np.isfinite(v)
                                else f"{v*100:+.1f}%")
    d["max_dd"] = d["max_dd"].map(lambda v: "—" if v is None or not np.isfinite(v)
                                  else f"{v*100:+.1f}%")
    d["win"] = d["win"].map(lambda v: "—" if v is None or not np.isfinite(v)
                            else f"{v*100:.0f}%")
    d["e_r"] = d["e_r"].map(lambda v: "—" if v is None or not np.isfinite(v)
                            else f"{v:+.2f}")
    d["exposure"] = d["exposure"].map(lambda v: "—" if v is None or not np.isfinite(v)
                                      else f"{v*100:.0f}%")
    return d[["axis", "value", "trades", "cagr", "total", "max_dd",
              "win", "e_r", "exposure"]].to_string(index=False)



# ---------------------------------------------------------------- stage 2

# 有界组合网格。**故意小**:stage 1 已经表明没有哪个轴能把 CAGR 推近 30%,
# 组合搜索的意义是"把这个问题问完",不是真的指望捞到 30%。
# 网格越大 n_trials 越大、DSR 越接近 0 —— 那本身就是成本,不该滥花。
STAGE2_SIGNAL_GRID = {
    "min_dist_high252": [None, -0.3, -0.2, -0.1],
    "max_stoch_k": [None, 40.0, 60.0],
    "max_rsi14": [None, 60.0, 70.0],
}
STAGE2_EXIT_GRID = {"sl_mult": [1.5, 2.0, 3.0], "tp_mult": [3.0, 4.0, 6.0]}


def stage2(ind, reg) -> tuple[pd.DataFrame, int]:
    """两步贪心:先扫信号层(退出固定),再对胜者精调退出层。

    比全网格省得多(45 次 vs 324 次),代价是漏掉"信号与退出交互"的组合。
    这个代价我选择接受并写在这里,因为 n_trials 每翻一倍,DSR 就更没意义一分。
    """
    rows, n = [], 0

    keys = list(STAGE2_SIGNAL_GRID)
    for combo in itertools.product(*(STAGE2_SIGNAL_GRID[k] for k in keys)):
        cfg = {k: v for k, v in zip(keys, combo) if v is not None}
        r = evaluate(ind, reg, cfg)
        n += 1
        r.update(axis="sig:" + ",".join(f"{k}={v}" for k, v in cfg.items()) or "sig:base",
                 value=len(cfg))
        rows.append(r)

    df = pd.DataFrame(rows)
    ok = df[df["cagr"].notna()]
    if ok.empty:
        return df, n
    best = ok.sort_values("cagr", ascending=False).iloc[0]
    best_cfg = dict(best["cfg"])

    for sl in STAGE2_EXIT_GRID["sl_mult"]:
        for tp in STAGE2_EXIT_GRID["tp_mult"]:
            cfg = {**best_cfg, "sl_mult": sl, "tp_mult": tp}
            r = evaluate(ind, reg, cfg)
            n += 1
            r.update(axis="exit", value=f"sl={sl},tp={tp}")
            rows.append(r)

    return pd.DataFrame(rows), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="单轴扫描")
    ap.add_argument("--stage2", action="store_true", help="有界组合搜索")
    ap.add_argument("--control", action="store_true", help="运气对照")
    ap.add_argument("--lo", default=WIN_LO)
    ap.add_argument("--hi", default=WIN_HI)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    p = panel_mod.load_full_panel()
    ind = indicators.compute_all(p)
    reg = regime.compute_regime(ind)

    if a.control:
        print("（--control 需要先跑 --stage1 选定候选;此处仅演示基线对照）")
        cfg = {}
        r = evaluate(ind, reg, cfg, a.lo, a.hi)
        ctl = random_control(ind, cfg, r["trades"], n_runs=a.runs, lo=a.lo, hi=a.hi)
        print(f"候选 CAGR {r['cagr']*100:+.1f}%  成交 {r['trades']} 笔")
        print(f"随机对照({ctl['n_runs']} 次,每次 {ctl['n_trades']} 笔): "
              f"中位 {ctl['median']*100:+.1f}%  p90 {ctl['p90']*100:+.1f}%  "
              f"p95 {ctl['p95']*100:+.1f}%  最大 {ctl['max']*100:+.1f}%")
        print(f"随机达到 >=30% 的概率: {ctl['p_ge_target']*100:.1f}%")
        return 0

    if a.stage2:
        df, n = stage2(ind, reg)
        ok = df[df["cagr"].notna()].sort_values("cagr", ascending=False)
        print(f"\n最近 12 个月 {a.lo} .. {a.hi}   目标 CAGR >= {TARGET_CAGR*100:.0f}%")
        print(f"stage2 n_trials = {n}(不含 stage1 的 37)\n")
        print(_fmt(ok.head(18)))
        print(f"\n达到 >=30% 的配置数: {(ok['cagr'] >= TARGET_CAGR).sum()} / {len(ok)}")
        print(f"最好成绩: {ok.iloc[0]['cagr']*100:+.2f}% CAGR")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            df.drop(columns=["cfg"]).to_json(a.out, orient="records",
                                             force_ascii=False, indent=1)
            print(f"已写入 {a.out}")
        return 0

    if a.stage1:
        df, n = stage1(ind, reg)
        print(f"\n最近 12 个月 {a.lo} .. {a.hi}   目标 CAGR >= {TARGET_CAGR*100:.0f}%")
        print(f"n_trials 累计 = {n}\n")
        print(_fmt(df.sort_values("cagr", ascending=False, na_position="last")))
        print("\n各轴定义:")
        for k, v in AXES.items():
            print(f"  {k:<18} {v}")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            df.drop(columns=["cfg"]).to_json(a.out, orient="records", force_ascii=False, indent=1)
            print(f"\n已写入 {a.out}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
