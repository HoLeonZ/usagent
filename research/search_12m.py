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

from core import costs
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
        "max_volume_participation": costs.MAX_VOLUME_PARTICIPATION,
        "intraday_tiebreak": "sl_first",
        "initial_cash": costs.INITIAL_CASH,
        "position_fraction": 1.0,          # §2(a):派生量,风险预算下不生效
        "atr_slip_scale": costs.ATR_SLIP_SCALE,
        "cost_bps_roundtrip": costs.COST_BPS_ROUNDTRIP,
    }


def signal_kwargs(cfg: dict) -> dict:
    kw = dict(signals.V21_SIGNAL_PARAMS)
    for k in SIGNAL_AXES:
        if cfg.get(k) is not None:
            kw[k] = cfg[k]
    return kw


def evaluate(ind: pd.DataFrame, reg: pd.DataFrame, cfg: dict,
             lo: str = WIN_LO, hi: str = WIN_HI, *, use_regime: bool = True) -> dict:
    """跑一个配置,返回窗口指标。**三种结果必须可区分** —— 这是 argmax(CAGR) 的前提:

      * 正常      → cagr 是有限数
      * 信号为空  → cagr=None, reason="no_signals",**且不去跑回测**
                    (跑了会得到一条平直权益曲线 → cagr=+0.0%,而 +0.0% 在 argmax(CAGR)
                     里会赢过所有负收益配置,搜索会系统性朝过度过滤的方向跑)
      * 参数非法 / 记账断言 → **直接抛**(§0 Fail-Fast)。搜索空间是预注册的,里面不该有
                    非法配置;若出现说明空间写错了,必须立刻知道 —— 不能静默出局却照计
                    n_trials(审计实测旧版把 BrokerError 降级成一行 cagr=None)。
    """
    kw = signal_kwargs(cfg)
    sig = signals.select_entries(ind, start_date=lo, end_date=hi,
                                 regime_df=reg if use_regime else None, **kw)
    if sig.empty:
        return {"cagr": None, "trades": 0, "reason": "no_signals", "cfg": cfg}

    # 不覆盖成本口径 —— 让引擎读 preset 里的(§2(b) 单一来源)
    res = broker.simulate(make_preset(cfg), sig, ind, start=lo, end=hi)

    eq, tr = res["equity"], res["trades"]
    cagr = metrics.cagr(eq["equity"], eq["date"])
    ts = metrics.tail_stats(tr) if len(tr) else {"expectancy_r": float("nan")}
    return {
        "cagr": float(cagr), "trades": int(len(tr)),
        # Sharpe 也一并给出 —— 判据可以换,搜索空间不必重跑。
        # 注意 broker 的 sharpe 是**年化**的(mean/std × sqrt(252))。
        "sharpe": float(res["metrics"]["sharpe"]),
        "total": float(res["metrics"]["total_return"]),
        "max_dd": float(res["metrics"]["max_dd"]),
        "win": float(res["metrics"].get("win_rate", float("nan"))),
        "e_r": float(ts.get("expectancy_r", float("nan"))),
        "exposure": float(res["metrics"]["avg_exposure"]),
        "reason": "ok", "cfg": cfg,
    }


# ---------------------------------------------------------------- 运气对照

def _control_pool(ind: pd.DataFrame, lo: str, hi: str, *,
                  reg: pd.DataFrame | None = None) -> pd.DataFrame:
    """随机对照的抽样池 = **策略自己够得着的机会集**(只去掉信号本身)。

    口径与信号层同源的三个门:指标就绪、E 条件的流动性下界(amount60 ≥ min_amount)、
    regime 可做多(给了 reg 时)。

    2026-10-15 实测:旧实现只要求「指标就绪」,于是从全池 596 只里抽 —— 含被 E 的流动性门
    挡在外面、策略永远不会碰的小盘股。后果有两个:(a) null 与策略不在同一个机会集上,
    对照不成立;(b) 200 次里 33 次在出场时撞上 §4(a) 的量能上限而 fail-fast,
    失败重抽把分布筛成了成功子集。
    """
    sp = signals.V21_SIGNAL_PARAMS
    win = ind[(ind["date"] >= pd.Timestamp(lo)) & (ind["date"] < pd.Timestamp(hi))]
    ok = (win["ma120"].notna() & win["atr_pct"].notna()
          & win["amount60"].notna() & (win["amount60"] >= sp["min_amount"]))
    pool = win[ok]
    if reg is not None:
        ok_days = set(pd.to_datetime(reg.loc[reg["regime_ok"], "date"]).tolist())
        pool = pool[pool["date"].isin(ok_days)]
    return pool[["symbol", "date", "atr_pct"]]


def random_control(ind: pd.DataFrame, cfg: dict, n_trades: int, *,
                   n_runs: int = 200, lo: str = WIN_LO, hi: str = WIN_HI,
                   seed: int = 0, reg: pd.DataFrame | None = None) -> dict:
    """同止损止盈、同交易笔数、**同机会集**的随机入场对照。

    做法:从 _control_pool 里随机抽 n_trades 个 (symbol, date) 作为入场点,套用与候选
    完全相同的 preset 跑一遍。重复 n_runs 次 → 得到"纯运气"的 CAGR 分布。

    这是本脚本里唯一能区分"策略"与"运气"的东西。
    """
    rng = np.random.default_rng(seed)
    pool = _control_pool(ind, lo, hi, reg=reg).to_numpy(dtype=object)
    if len(pool) == 0:
        raise SearchError("随机对照的候选池为空")

    preset = make_preset(cfg)
    cagrs: list[float] = []
    n_failed = 0
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
        # 失败必须计数并报出 —— 「200 次」不能只是名义值。静默 continue 会让
        # n_runs 只记成功次数,而分布已被成功子集筛过(§ 无 silent cap)。
        try:
            res = broker.simulate(preset, sig, ind, start=lo, end=hi)
            c = metrics.cagr(res["equity"]["equity"], res["equity"]["date"])
            if np.isfinite(c):
                cagrs.append(float(c))
            else:
                n_failed += 1
        except broker.BrokerError as exc:
            n_failed += 1
            logger.warning("随机对照第 %d 次失败:%s", n_failed, str(exc)[:60])
    a = np.array(cagrs)
    if len(a) == 0:
        raise SearchError("随机对照全部失败")
    if n_failed:
        logger.warning("随机对照:%d 次成功 / %d 次失败(合计 %d) —— "
                       "分布只由成功子集给出", len(a), n_failed, len(a) + n_failed)
    return {
        "n_runs": int(len(a)), "n_failed": int(n_failed), "n_trades": int(n_trades),
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


def _wf_fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (没有任何折窗有结果)"
    d = df.copy()
    d["train_cagr"] = d["train_cagr"].map(lambda v: f"{v*100:+.1f}%")
    d["test_cagr"] = d["test_cagr"].map(lambda v: f"{v*100:+.1f}%")
    return d[["train_lo", "train_hi", "train_cagr", "train_trades",
              "test_lo", "test_hi", "test_cagr", "test_trades", "cfg"]].to_string(index=False)


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


# ---------------------------------------------------------------- Walk-Forward(§5(b))
# 铁律原文:Optimization logic MUST enforce Walk-Forward Validation (WFV) with strict
# out-of-sample (OOS) testing windows.
#
# 为什么必须有:stage1/stage2 是在**同一个窗口**上取 argmax,而且 stage2 还对胜者在同
# 一窗口继续精调退出参数。2026-10-15 审计复跑显示胜者的**五个参数全部落在网格边界**上
# —— 边界最优意味着「最优」由网格取景范围决定,不由数据决定。WFV 不能消除过拟合,
# 但它让「选出来的配置在没见过的数据上是什么样」变成一个必须报出来的数。


def cfg_key(cfg: dict) -> str:
    """配置的规范字符串 —— 折窗结果里用它记住「这一折选了哪一个」。"""
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True, default=str)


def make_folds(start: str, end: str, *, train_months: int, test_months: int,
               step_months: int) -> list[tuple[str, str, str, str]]:
    """滚动折窗 [(train_lo, train_hi, test_lo, test_hi), ...],全部左闭右开。

    训练窗紧邻其测试窗之前;窗与窗之间不重叠;整体不越过 end。
    """
    if min(train_months, test_months, step_months) < 1:
        raise ValueError("train_months / test_months / step_months 必须 >= 1")
    out: list[tuple[str, str, str, str]] = []
    lo, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    while True:
        tr_hi = lo + pd.DateOffset(months=train_months)
        te_hi = tr_hi + pd.DateOffset(months=test_months)
        if te_hi > end_ts:
            break
        out.append((lo.strftime("%Y-%m-%d"), tr_hi.strftime("%Y-%m-%d"),
                    tr_hi.strftime("%Y-%m-%d"), te_hi.strftime("%Y-%m-%d")))
        lo = lo + pd.DateOffset(months=step_months)
    if not out:
        raise ValueError(
            f"在 {start}..{end} 里排不下一个 fold(train={train_months}m + "
            f"test={test_months}m)—— WFV 需要比单窗口更长的历史(§5(b))"
        )
    return out


def walk_forward(ind, reg, *, grid: list[dict], folds: list[tuple],
                 criterion: str = "cagr", evaluate_fn=None) -> pd.DataFrame:
    """训练窗选参 → **测试窗**验证。

    ⚠️ 选择只看训练窗。测试窗在每个 fold 里只被评估**一次**,且在选完参数之后 ——
       这是 §5(b) 的全部内容。判据是「同一份网格在训练窗上的 argmax」,不做任何
       测试窗反馈(那正是 stage2 的错)。

    返回每折一行:train_lo/hi、train_cagr、test_lo/hi、test_cagr、test_trades、cfg。
    `evaluate_fn` 只用于测试注入(默认走模块级 `evaluate`)。
    """
    if not folds:
        raise ValueError("folds 不能为空 —— WFV 至少需要一个训练/测试折(§5(b))")
    if not grid:
        raise ValueError("grid 不能为空 —— 没有候选就没有要验证的东西")
    ev = evaluate_fn or evaluate

    rows: list[dict] = []
    for tr_lo, tr_hi, te_lo, te_hi in folds:
        best: dict | None = None
        best_val = float("-inf")
        best_trades = 0
        for cfg in grid:                          # ← 只碰训练窗
            r = ev(ind, reg, cfg, tr_lo, tr_hi)
            v = r.get(criterion)
            if v is None or not np.isfinite(v):
                continue
            if v > best_val:
                best, best_val = dict(cfg), float(v)
                # 训练窗里「胜者」成交了几笔 —— 只看 CAGR 会漏掉「零交易空转」那种胜者
                best_trades = int(r.get("trades") or 0)
        if best is None:
            logger.warning("fold %s..%s:训练窗上没有任何有效配置", tr_lo, tr_hi)
            continue
        te = ev(ind, reg, best, te_lo, te_hi)     # ← 测试窗只读一次
        rows.append({
            "train_lo": tr_lo, "train_hi": tr_hi,
            "train_cagr": best_val, "train_trades": best_trades,
            "test_lo": te_lo, "test_hi": te_hi,
            "test_cagr": te.get("cagr"),
            "test_trades": te.get("trades"),
            "test_e_r": te.get("e_r"),
            "cfg": cfg_key(best),
        })
    return pd.DataFrame(rows)


def wf_grid() -> list[dict]:
    """WFV 用的网格 —— 与 stage2 的信号层同源,保证「被检验的东西」就是被交付的那类。"""
    keys = list(STAGE2_SIGNAL_GRID)
    return [{k: v for k, v in zip(keys, combo) if v is not None}
            for combo in itertools.product(*(STAGE2_SIGNAL_GRID[k] for k in keys))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="单轴扫描")
    ap.add_argument("--stage2", action="store_true", help="有界组合搜索")
    ap.add_argument("--control", action="store_true", help="运气对照")
    ap.add_argument("--walk-forward", action="store_true",
                    help="训练窗选参 → 测试窗验证(CLAUDE.md §5(b),优化逻辑必须 enforce)")
    ap.add_argument("--train-months", type=int, default=6)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--step-months", type=int, default=3)
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

    if a.walk_forward:
        folds = make_folds(FULL_LO, WIN_HI, train_months=a.train_months,
                           test_months=a.test_months, step_months=a.step_months)
        grid = wf_grid()
        df = walk_forward(ind, reg, grid=grid, folds=folds)
        print(f"\nWalk-Forward:{len(folds)} 折(train {a.train_months}m / "
              f"test {a.test_months}m / step {a.step_months}m),每折在训练窗上扫 "
              f"{len(grid)} 个配置(§5(b):选择只看训练窗)")
        print(_wf_fmt(df))
        if df.empty:
            return 1
        # 诚实对照:同一批**测试窗**上,不做任何过滤的 v21 基线是什么样。
        base = [evaluate(ind, reg, {}, r["test_lo"], r["test_hi"])["cagr"]
                for _, r in df.iterrows()]
        win = float(np.mean(df["test_cagr"]))
        bl = float(np.nanmean(base))
        print(f"\n测试窗平均 CAGR:  WFV 选出的配置 {win*100:+.2f}%   "
              f"v21 基线(不做任何过滤) {bl*100:+.2f}%   差 {(win-bl)*100:+.2f}pp")
        print(f"测试窗 CAGR 为正的折: {int((df['test_cagr'] > 0).sum())}/{len(df)}")
        if win <= bl:
            print("⚠️ 训练窗选参在测试窗上**没有跑赢**什么都不做的基线 —— "
                  "这就是 §5(b) 要暴露的东西,不要把训练窗上的数字当结论。")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            df.to_json(a.out, orient="records", force_ascii=False, indent=1)
            print(f"已写入 {a.out}")
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
