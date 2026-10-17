"""用**修好的** overlay 口径重测四条 preset（各自声明的锁窗口）+ 两个对照。

为什么按"各自声明的窗口"：每条 preset 的 `LOCK_WINDOW` 就是它的评测坐标（行为锁钉的那个），
拿别的窗口比是"跨窗比"，本仓库明令禁止（§5(e)：头条必须同窗比）。

两个对照：
  * DO_NOTHING —— 信号条件恒不满足（0 信号/0 成交/0 敞口）。修好偏置后它的超额必须**恰好 0**，
    这是"任何'更好'的主张必须先过的那条线"。
  * 池子本身 —— 由 DO_NOTHING 等价（w=0 ⇒ overlay=池子）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]        # 模块相对 —— 不许写死本机绝对路径
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from research import golden_baseline as G, indicators, metrics, overlay   # noqa: E402
from research import panel as panel_mod, presets, regime as regime_mod    # noqa: E402

OUT = ROOT / "research" / "results" / "preset_ranking_corrected.json"


def load_chain():
    p = presets.get_preset("us_uptrend_pullback_v1")
    panel = panel_mod.load_panel_for_preset(p)
    ind = indicators.compute_all(panel)
    reg = regime_mod.compute_regime(ind, **(p.get("regime") or {}))
    return panel, ind, reg


def report(panel, ind, reg, name, window):
    case = {"name": name, "preset": name, "start": window[0], "end": window[1]}
    res, sig = G.run_preset(case, panel, ind)
    eq = res["equity"].set_index("date")
    ov = overlay.excess_vs_pool(eq, reg, window[0], window[1])
    ba = metrics.beta_attribution(eq["equity"].pct_change().fillna(0.0),
                                  overlay.pool_daily_returns(reg), eq["exposure"])
    rets = eq["equity"].pct_change().dropna()
    folds = np.array_split(rets, 4)
    fpos = int(sum(1 for f in folds if len(f) and float((1 + f).prod() - 1) > 0))
    trades = res["trades"]
    r = metrics.r_multiples(trades) if len(trades) else pd.Series(dtype=float)
    return {
        "preset": name, "window": list(window), "n_signals": int(len(sig)),
        "n_trades": int(len(trades)),
        "own_ret": float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0),
        "exposure": float(eq["exposure"].mean()),
        "overlay_cagr": ov["cagr"], "bench_cagr": ov["bench"], "excess": ov["excess"],
        "excess_per_exposure": ov["excess"] / ov["exposure"] if ov["exposure"] else float("nan"),
        "pool_on_exposed": ba["bench_on_exposed_days"],
        "beta": ba["beta"], "alpha_annual": ba["alpha_annual"], "t_alpha": ba["t_alpha"],
        "folds_positive": fpos,
        "r_median": float(r.median()) if len(r) else None,
        "win_rate": (float((trades["net_pnl"].astype(float) > 0).mean())
                     if len(trades) else None),
    }


def main() -> int:
    panel, ind, reg = load_chain()
    rows = []
    for name, rec in sorted(presets.REGISTRY.items()):
        rows.append(report(panel, ind, reg, name, rec["window"]))
    base = presets.get_preset("us_uptrend_pullback_wfv")
    for label in ("DO_NOTHING",):
        dead = {**base, "signal": {**base["signal"], "min_dist_high252": 0.95}}
        presets.PRESETS[label] = dead
        rows.append(report(panel, ind, reg, label, ("2024-04-09", "2025-10-13")))

    print(f"{'preset':32s} {'窗口':25s} {'笔数':>4s} {'自身':>8s} {'敞口':>7s} "
          f"{'超额':>8s} {'超额/敞口':>10s} {'在场日池':>9s} {'折正':>4s} {'R中位':>7s}")
    for r in rows:
        print(f"{r['preset']:32s} {'..'.join(r['window']):25s} {r['n_trades']:4d} "
              f"{r['own_ret']*100:+7.2f}% {r['exposure']*100:6.2f}% {r['excess']*100:+7.2f}% "
              f"{r['excess_per_exposure']*100:+9.1f}% {r['pool_on_exposed']*100:+8.1f}% "
              f"{r['folds_positive']}/4 {(r['r_median'] if r['r_median'] is not None else float('nan')):+7.2f}")
    dn = next(r for r in rows if r["preset"] == "DO_NOTHING")
    print(f"\n不变量检查：DO_NOTHING 的超额 = {dn['excess']:+.10%}（修好之后必须恰好 0）")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
