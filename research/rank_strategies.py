"""把**所有策略**放到同一台仪器上回测并排名 —— 一个可复现的工具，不是一次性命令。

它回测三层：
  1. **6 条 preset**（交付物）——在**共同窗**上比（跨窗比是本仓库明令禁止的），
     同时给出各自**声明的锁窗口**上的读数；
  2. **10 个策略族**——每条族取**网格中点**那一组参数（**不是**搜索出来的最优配置：
     用最优就是在报 in-sample 的 argmax）；
  3. **对照**：什么都不做（零敞口 ⇒ 超额恰好 0）。

判据用它自己的既有定义：`research/overlay.py`（曝光匹配超额，唯一定义）。
DSR 用 `metrics.deflated_sharpe_from_returns`，k 取该条 preset 自己声明的 `n_comparisons`
（族取网格规模）——不同行的 k 不同，所以**列名里写着 k**，别跨行直接比 DSR。

⚠️ 共同窗 2024-04-09..2026-10-14 **含 holdout 天数**（2025-10-14 起）。这一窗此前被读过
（台账「重跑」一节、以及各 preset 的行为锁），所以它不是样本外 —— 读数里的 legend 会标出来。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import obs                                                            # noqa: E402  凭证头（§6 归因）
from research import broker, families, golden_baseline as G, indicators   # noqa: E402
from research import metrics, overlay, panel as panel_mod, presets, regime as regime_mod  # noqa: E402
from research.panel import data_vintage                               # noqa: E402  数据是哪一批

COMMON = ("2024-04-09", "2026-10-14")
OUT = ROOT / "research" / "results" / "strategy_ranking.json"
# 「怎么读这份文件」——**同一份字符串**印在终端、也写进落盘的 JSON（此前只有终端有）。
READING_NOTES = (
    "读法提醒：① 「超额」是曝光匹配口径（闲置资金按等权池计息）⇒ 择时得零分，"
    "只有'在场时打赢池子'才算数；② 共同窗「不是样本外」；"
    "③ DSR 的 k 逐行不同，别跨行比；④ 零敞口的对照那条线是 0.00pp。",
    "① 「超额」是曝光匹配口径（闲置资金按等权池计息）⇒ 择时得零分，只有'在场时打赢池子'才算数。",
    "② 共同窗不是样本外（这一行数字不能当样本外证据）。",
    "③ DSR 的 k 逐行不同（各自策略的试验次数不同）⇒ 别跨行比 DSR。",
    "④ 零敞口的那条对照线是 0.00pp（定义如此：两条腿完全相同），不是'跑平了'。",
)


def load_chain():
    p = presets.get_preset("us_uptrend_pullback_v1")
    panel = panel_mod.load_panel_for_preset(p)
    ind = indicators.compute_all(panel)
    reg = regime_mod.compute_regime(ind, **(p.get("regime") or {}))
    return panel, ind, reg


def mid_grid(fid: str) -> dict:
    """族参数的**中点**配置（每轴取中间那一档）。确定性、不看结果。"""
    g = families.GRIDS.get(fid) or {}
    return {k: v[len(v) // 2] for k, v in g.items()}


def measure(name: str, res: dict, reg, window, k: int, *, note: str = "",
            trades=None, n_signals=None) -> dict:
    eq = res["equity"].set_index("date")
    ov = overlay.excess_vs_pool(eq, reg, window[0], window[1])
    ba = metrics.beta_attribution(eq["equity"].pct_change().fillna(0.0),
                                  overlay.pool_daily_returns(reg), eq["exposure"])
    rets = eq["equity"].pct_change().dropna()
    folds = np.array_split(rets, 4)
    fpos = int(sum(1 for f in folds if len(f) and float((1 + f).prod() - 1) > 0))
    dd = float((eq["equity"] / eq["equity"].cummax() - 1).min())
    tr = res["trades"]
    r = metrics.r_multiples(tr) if len(tr) else pd.Series(dtype=float)
    dsr_d = metrics.deflated_sharpe_from_returns(rets, n_trials=k)
    return {
        "name": name, "window": list(window), "note": note,
        "n_signals": None if n_signals is None else int(n_signals),
        "n_trades": int(len(tr)),
        "own_ret": float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0),
        "cagr": float(metrics.cagr(eq["equity"], pd.Series(eq.index))),
        "exposure": float(eq["exposure"].mean()),
        "overlay_cagr": ov["cagr"], "bench_cagr": ov["bench"], "excess": ov["excess"],
        "excess_per_exposure": (ov["excess"] / ov["exposure"]) if ov["exposure"] else float("nan"),
        "pool_on_exposed": float(ba["bench_on_exposed_days"]),
        "beta": float(ba["beta"]), "alpha_annual": float(ba["alpha_annual"]),
        "t_alpha": float(ba["t_alpha"]),
        "folds_positive": fpos, "maxdd": dd,
        "r_median": float(r.median()) if len(r) else None,
        "r_mean": float(r.mean()) if len(r) else None,
        "win_rate": float((tr["net_pnl"].astype(float) > 0).mean()) if len(tr) else None,
        "stop_share": float((tr["exit_reason"] == "sl").mean()) if len(tr) else None,
        "dsr": float(dsr_d.get("deflated_sharpe", float("nan"))), "dsr_k": int(k),
    }


def _row(r: dict) -> str:
    """⚠️ 比率型字段（收益/超额/回撤/alpha）**存的是分数**，打印时必须 ×100。

    这里踩过一次：`'{:+8.2f}%'.format(0.2931)` 会打出 `+0.29%` 而不是 `+29.31%` ——
    百分表里每一列都小 100 倍，而笔数/敞口/beta/t/DSR 那些**分数列**照旧正确，
    于是表格"看起来合理"却全错。所以格式化只走这两个函数，不再手写 `%`。
    """
    def num(v, fmt="{:+8.2f}", scale=1.0, dash="        —"):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return dash
        return fmt.format(float(v) * scale)

    def pct(v, fmt="{:+8.2f}%", dash="       —"):
        return num(v, fmt, 100.0, dash)

    return (f"{r['name']:32s} {r['n_trades']:4d} {pct(r['own_ret'])} "
            f"{pct(r['cagr'], '{:+7.2f}%')} {num(r['exposure'], '{:6.2f}%', 100.0, '     —')} "
            f"{pct(r['excess'], '{:+8.2f}pp')} {pct(r['excess_per_exposure'], '{:+9.1f}%')} "
            f"{num(r['beta'], '{:+5.2f}', 1.0, '    —')} {pct(r['alpha_annual'])} "
            f"{num(r['t_alpha'], '{:+5.2f}', 1.0, '    —')} "
            f"{r['folds_positive']}/4 {pct(r['maxdd'], '{:+7.1f}%')} "
            f"{num(r['r_median'], '{:+5.2f}', 1.0, '    —')} "
            f"{num(r['dsr'], '{:.3f}', 1.0, '    —')}@{r['dsr_k']}")


HDR = (f"{'策略':32s} {'笔':>4s} {'自身收益':>9s} {'CAGR':>8s} {'敞口':>7s} {'超额':>9s} "
       f"{'超额/敞口':>10s} {'beta':>5s} {'alpha年化':>9s} {'t':>6s} {'折正':>4s} "
       f"{'maxDD':>8s} {'R中位':>6s} {'DSR@k':>9s}")


def payload(rows: list[dict], panel: pd.DataFrame) -> dict:
    """落盘的那一份：**凭证头 + 数据 vintage + 共同窗 + 读法提醒 + 行**。

    为什么不是一裸列 rows（2026-10-22 实测）：脚本读到这份文件时，不知道它是哪份代码、
    哪份数据算的；人几个月后打开它，看到一个「超额」列，不知道那是曝光匹配口径，
    也不知道共同窗**不是样本外** —— 那四条"读法提醒"此前只印在终端里。终端里的话既然是
    "读这份文件"的规矩，就该写进文件（与 run_baseline 的 JSON 同一套：manifest +
    data_vintage）。
    """
    return {
        "manifest": obs.manifest(),                 # ts + git sha + 脏树（§6 归因）
        "data_vintage": data_vintage(panel),        # 这份数据是哪一批（复现历史数字要用）
        "window": list(COMMON),
        "notes": list(READING_NOTES[1:]),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="回测所有策略并排名（preset / 族 / 对照）")
    ap.add_argument("--families", action="store_true", help="连 10 个族一起跑（慢一些）")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    panel, ind, reg = load_chain()
    rows: list[dict] = []

    print(f"\n{'='*118}\n【一】6 条 preset —— **共同窗** {COMMON[0]}..{COMMON[1]}（含 holdout 天数，非样本外）\n{'='*118}")
    print(HDR)
    for name, rec in sorted(presets.REGISTRY.items()):
        res, sig = G.run_preset({"name": name, "preset": name, "start": COMMON[0], "end": COMMON[1]},
                                panel, ind)
        r = measure(name, res, reg, COMMON, int(presets.get_preset(name).get("n_comparisons", 1)),
                    note="共同窗", n_signals=len(sig))
        rows.append(r)
        print(_row(r))
    # 对照：什么都不做（零信号 ⇒ 零敞口 ⇒ 超额恰好 0）
    dead = {**presets.get_preset("us_uptrend_pullback_wfv"),
            "signal": {**presets.get_preset("us_uptrend_pullback_wfv")["signal"],
                       "min_dist_high252": 0.95}}
    presets.PRESETS["DO_NOTHING"] = dead
    res0, sig0 = G.run_preset({"name": "DO_NOTHING", "preset": "DO_NOTHING",
                               "start": COMMON[0], "end": COMMON[1]}, panel, ind)
    r0 = measure("DO_NOTHING（对照）", res0, reg, COMMON, 1, note="共同窗", n_signals=len(sig0))
    rows.append(r0)
    print(_row(r0))

    print(f"\n{'='*118}\n【二】每条 preset 在**它自己声明的锁窗口**上\n{'='*118}")
    print(HDR)
    for name, rec in sorted(presets.REGISTRY.items()):
        w = rec["window"]
        res, sig = G.run_preset({"name": name, "preset": name, "start": w[0], "end": w[1]}, panel, ind)
        r = measure(name, res, reg, w, int(presets.get_preset(name).get("n_comparisons", 1)),
                    note="自窗", n_signals=len(sig))
        rows.append(r)
        print(_row(r))

    if a.families:
        print(f"\n{'='*118}\n【三】10 个策略族（各自网格**中点**配置 —— 不是搜索最优；同窗 {COMMON[0]}..{COMMON[1]}）\n{'='*118}")
        print(HDR)
        base = presets.get_preset("us_reversal_dip_v1")      # 用同一条退出/仓位口径，A/B 才可比
        for fid in sorted(families.MASK):
            cfg = {**base, "family": fid, "family_params": mid_grid(fid)}
            presets.PRESETS["_FAM"] = cfg
            k = int(np.prod([len(v) for v in (families.GRIDS.get(fid) or {}).values()])) if fid in families.GRIDS else 1
            res, sig = G.run_preset({"name": f"family:{fid}", "preset": "_FAM",
                                     "start": COMMON[0], "end": COMMON[1]}, panel, ind)
            r = measure(f"family:{fid}", res, reg, COMMON, max(k, 1), n_signals=len(sig),
                        note="族·网格中点")
            rows.append(r)
            print(_row(r))

    # ⚠️ 落盘的 JSON 以前是**一裸列 rows** —— 没有代码版本、没有数据 vintage、
    # 也没有那四条"读法提醒"（提醒只印在终端里）。见 payload() 的说明。
    Path(a.out).write_text(json.dumps(payload(rows, panel), ensure_ascii=False, indent=1,
                                     default=str),
                           encoding="utf-8")
    print(f"\n已写 {a.out}")
    print(READING_NOTES[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
