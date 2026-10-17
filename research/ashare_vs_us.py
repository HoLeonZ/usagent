"""A 股 vs 美股同策略对照：**同窗口、同指标**（2024-09..2026-09）。

动机：hiagent 的 v21 基线记录该窗口 E[R]=+1.182 / 胜率 24.6% / PF 2.02 /
top-decile 占毛利 R = 103%，并自述"表现为约 1.6× 市场 beta"。usagent 里
`us_uptrend_pullback_v1` **就是**这条策略的忠实移植（pct 2.1%/26.28%、mp=1、全仓 all-in）。
于是可以在**同一天历窗口**上算出同一批指标，直接把"哪一步不一样"拆开：

  ① 忠实移植（pct 止损、mp=1、**全仓**）      ← hiagent 的原配置
  ② 同信号 + 风险预算（1%/笔 ⇒ 敞口 ~8%）    ← usagent 的现行做法
  ③ 忠实移植 + **低流动性池**（= A 股式池子：参与者结构不同）

预测（先写下来再看）：①的**绝对收益**会明显高于②（因为它在赚 beta，而 beta × 全仓
才有量级），但两者的**曝光匹配超额**都应≤池子；③用来检验"池子/参与者结构"这条机制。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research import broker, indicators, metrics, overlay                    # noqa: E402
from research import panel as panel_mod, presets, regime as regime_mod       # noqa: E402
from research import signals, universe as mod_universe                       # noqa: E402

WIN = ("2024-09-01", "2026-09-01")          # 与 A 股基线**同一天历窗口**
OUT = ROOT / "research" / "results" / "ashare_vs_us.json"


def run_variant(P, panel, ind, reg, *, label, window=WIN):
    sig = signals.select_entries(ind, start_date=window[0], end_date=window[1],
                                 regime_df=reg if P.get("regime") else None, **P["signal"])
    res = broker.simulate(P, sig, ind, start=window[0], end=window[1])
    eq = res["equity"].set_index("date")
    ts = metrics.tail_stats(res["trades"]) if len(res["trades"]) else {"n": 0}
    ov = overlay.excess_vs_pool(eq, reg, window[0], window[1])
    ba = metrics.beta_attribution(eq["equity"].pct_change().fillna(0.0),
                                  overlay.pool_daily_returns(reg), eq["exposure"])
    return {
        "label": label, "n_trades": int(len(res["trades"])),
        "own_ret": float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0),
        "exposure": float(eq["exposure"].mean()),
        "beta": float(ba["beta"]), "alpha_annual": float(ba["alpha_annual"]),
        "t_alpha": float(ba["t_alpha"]),
        "excess": ov["excess"], "bench_cagr": ov["bench"],
        "E_R": float(ts.get("expectancy_r", float("nan"))),
        "t_E_R": float(ts.get("t_expectancy", float("nan"))),
        "win_pct": float(ts.get("win_pct", float("nan"))),
        "profit_factor": float(ts.get("profit_factor", float("nan"))),
        "top_decile_share": float(ts.get("top_decile_share", float("nan"))),
        "p_r_ge_5": float(ts.get("p_r_ge_5", float("nan"))),
        "p_r_le_minus_1": float(ts.get("p_r_le_minus_1", float("nan"))),
    }


def main() -> int:
    base = presets.get_preset("us_uptrend_pullback_v1")     # = 忠实移植（pct 止损/mp=1/全仓）
    print("=" * 112)
    print(f"【A 股 vs 美股：同策略、同窗口 {WIN[0]}..{WIN[1]}、同指标】")
    print("  A 股基线（hiagent 记录）：E[R]=+1.182  胜率 24.6%  PF 2.02  "
          "top-decile 占毛利 R=103%  n=57  beta≈1.6×（它自述）")
    print("=" * 112)

    panel = panel_mod.load_panel_for_preset(base)
    ind = indicators.compute_all(panel)
    reg = regime_mod.compute_regime(ind, **(base.get("regime") or {}))
    rows = []
    # ⚠️ "忠实移植"必须**同时**改 position_sizing：契约里 `position_sizing` 是**派生量**，
    # 去掉 risk_per_trade 后它会派生成 all_in —— 只删键不改声明会被 validate() 当场拦下
    # （实测踩过）。这一组才是 hiagent 的原配置：全仓 + mp=1 + pct 2.1%/26.28%。
    allin = {**base, "risk_per_trade": None, "position_sizing": "all_in",
             "position_fraction": 1.0}
    rows.append(run_variant(allin, panel, ind, reg, label="① 忠实移植（全仓/mp=1/pct止损）"))
    rows.append(run_variant(base, panel, ind, reg, label="② 同信号 + 风险预算 1%/笔"))

    # ③ 低流动性池（A 股式：参与者更散户、名字更小）
    spec = {**mod_universe.DEFAULT_SPEC, "min_dollar_volume": 1e6, "note": "A股式低流动性池"}
    p2 = panel_mod.load_full_panel(universe=spec)
    i2 = indicators.compute_all(p2)
    r2 = regime_mod.compute_regime(i2, **(base.get("regime") or {}))
    rows.append(run_variant(base, p2, i2, r2, label="③ 忠实移植 + 低流动性池($1e6)"))

    hdr = (f"{'变体':34s} {'笔':>4s} {'自身':>9s} {'敞口':>7s} {'beta':>5s} {'alpha':>8s} "
           f"{'t':>5s} {'超额':>9s} {'E[R]':>7s} {'t(E[R])':>8s} {'胜率':>6s} {'PF':>6s} "
           f"{'top10%':>7s} {'P(R>=5)':>8s} {'P(R<=-1)':>9s}")
    print(hdr)
    for r in rows:
        print(f"{r['label']:34s} {r['n_trades']:4d} {r['own_ret']*100:+8.2f}% "
              f"{r['exposure']*100:6.2f}% {r['beta']:+5.2f} {r['alpha_annual']*100:+7.2f}% "
              f"{r['t_alpha']:+5.2f} {r['excess']*100:+8.2f}pp {r['E_R']:+7.3f} "
              f"{r['t_E_R']:+8.2f} {r['win_pct']*100:5.1f}% {r['profit_factor']:6.2f} "
              f"{r['top_decile_share']*100:6.1f}% {r['p_r_ge_5']*100:7.1f}% "
              f"{r['p_r_le_minus_1']*100:8.1f}%")
    print(f"\n池子（同窗等权）CAGR: {rows[0]['bench_cagr']*100:+.2f}%")
    print("\n【把差异落到一个数上】同样的胜率，差的是「赢的时候赢多少」：")
    print(f"{'':34s} {'胜率':>6s} {'PF':>6s} {'平均盈利/平均亏损':>16s} {'该胜率的不亏线':>14s} {'判':>6s}")
    for r in rows:
        w, pf = r["win_pct"], r["profit_factor"]
        if not (0 < w < 1) or not np.isfinite(pf):
            continue
        mult, need = pf * (1 - w) / w, (1 - w) / w
        print(f"{r['label']:34s} {w*100:5.1f}% {pf:6.2f} {mult:15.2f}× {need:13.2f}× "
              f"{'过线' if mult > need else '**不过线**':>6s}")
    Path(OUT).write_text(json.dumps({"window": list(WIN), "rows": rows},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
