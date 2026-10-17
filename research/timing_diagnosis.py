"""择时的**稳定性体检** —— 「中性篮子」候选必须先过这一关（跑之前定判据）。

背景（2026-10-17 实测）：`us_reversal_dip_v1` 在场那些天池子 **+131.4%**，而它自己只有
+29.31% ⇒ 看起来"择时很强、选股很差"。但那是**一个窗口**的读数：如果那些在场日恰好
落在 2.5 年里的牛市段，+131% 就只是"它在这段里碰巧在场"，不是择时能力。

判据（**先写死**）：
  ① 两半**同号**：把窗口对半切开，两半里"在场日池子收益 − 全窗池子收益"的方向必须一致；
  ② **超过随机块**：观察值在"同样段数、同样段长的随机块"分布里的分位，两半都要 ≥ 95%
     （用块而不是独立抽样：持仓是连续段，池子收益有自相关，独立抽样会低估方差）。

两条都过 ⇒ 择时是真信号，值得做「同择时 + 中性篮子」的 preset（下一轮，预先登记 1 组 + 1 对照）。
任一条不过 ⇒ 那就是**单窗运气**，篮子候选不必做 —— 如实记为负面证据。

⚠️ 本脚本只做**诊断**（不产生参数组合、不计 n_trials），且只看每半段内的相对关系。
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

from research import golden_baseline as G, indicators, overlay          # noqa: E402
from research import panel as panel_mod, presets, regime as regime_mod   # noqa: E402

N_DRAWS = 2000
OUT = ROOT / "research" / "results" / "timing_stability.json"


def load_chain():
    p = presets.get_preset("us_uptrend_pullback_v1")
    panel = panel_mod.load_panel_for_preset(p)
    ind = indicators.compute_all(panel)
    reg = regime_mod.compute_regime(ind, **(p.get("regime") or {}))
    return panel, ind, reg


def runs_of(mask: np.ndarray) -> list[int]:
    """把布尔序列压成连续段的长度列表（持仓是连续段，随机对照必须保持这个形状）。"""
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def pool_prod(b: np.ndarray, idx: np.ndarray) -> float:
    return float(np.prod(1.0 + b[idx]) - 1.0)


def block_control(b: np.ndarray, lengths: list[int], n_draws: int, rng) -> np.ndarray:
    """同样段数、同样段长的随机块 —— 返回 n_draws 个对照值。"""
    n = len(b)
    total = sum(lengths)
    if total == 0 or total >= n:
        return np.full(n_draws, np.nan)
    starts_hi = n - max(lengths) + 1
    out = np.empty(n_draws)
    for d in range(n_draws):
        pos, idx = 0, np.zeros(total, dtype=int)
        for ln in lengths:
            s = int(rng.integers(0, max(1, starts_hi)))
            idx[pos:pos + ln] = np.arange(s, min(s + ln, n))
            pos += ln
        out[d] = pool_prod(b, idx[idx < n])
    return out


def analyse(b: pd.Series, expo: pd.Series, label: str, rng) -> dict:
    """一段窗口上的：在场日池子收益 vs 全窗，与随机块对照。"""
    bb = b.reindex(expo.index).to_numpy(float)
    # ⚠️ 掩码不能用 `exposure > 0`：低敞口策略可能**几乎每天都有小仓位**
    # （实测 dip：620/694 天有仓位，但平均权重只有 22%），那样"在场日"≈全部天数，
    # 这个体检就退化成空转。用"敞口达到自己均值的一半"当"真在场"。
    thr = 0.5 * float(expo.mean())
    m = (expo.reindex(b.index).fillna(0.0) >= thr).to_numpy()
    lengths = runs_of(m)
    obs = pool_prod(bb, np.where(m)[0])
    allv = pool_prod(bb, np.arange(len(bb)))
    ctrl = block_control(bb, lengths, N_DRAWS, rng) if lengths else np.array([np.nan])
    pct = float(np.mean(ctrl <= obs)) if np.isfinite(ctrl).any() else float("nan")
    return {"段": label, "在场天数": int(m.sum()), "总天数": int(len(m)),
            "在场段数": len(lengths), "最长段": int(max(lengths)) if lengths else 0,
            "在场日池子收益": obs, "全窗池子收益": allv, "差": obs - allv,
            "随机块分位": pct, "n_draws": N_DRAWS}


def main() -> int:
    ap = argparse.ArgumentParser(description="择时稳定性体检（诊断，不产生 n_trials）")
    ap.add_argument("--preset", default="us_reversal_dip_v1")
    ap.add_argument("--start", default="2024-04-09")
    ap.add_argument("--end", default="2026-10-14")
    ap.add_argument("--mid", default="2025-07-01")
    a = ap.parse_args()

    panel, ind, reg = load_chain()
    b = overlay.pool_daily_returns(reg)
    case = {"name": a.preset, "preset": a.preset, "start": a.start, "end": a.end}
    res, _ = G.run_preset(case, panel, ind)
    expo = res["equity"].set_index("date")["exposure"]

    rng = np.random.default_rng(20261017)
    rows = [analyse(b.loc[a.start:a.end], expo, "全窗", rng)]
    for lo, hi, lab in ((a.start, a.mid, "前半段"), (a.mid, a.end, "后半段")):
        rows.append(analyse(b.loc[lo:hi], expo.loc[lo:hi], lab, rng))

    print(f"\n=== 择时稳定性体检（{a.preset}，{a.start}..{a.end}，切分线 {a.mid}）===")
    print(f"{'段':8s} {'在场天数':>8s} {'段数':>5s} {'最长段':>6s} {'在场日池子':>10s} "
          f"{'全窗池子':>9s} {'差':>9s} {'随机块分位':>10s}")
    for r in rows:
        print(f"{r['段']:8s} {r['在场天数']:8d} {r['在场段数']:5d} {r['最长段']:6d} "
              f"{r['在场日池子收益']*100:+9.1f}% {r['全窗池子收益']*100:+8.1f}% "
              f"{r['差']*100:+8.1f}% {r['随机块分位']*100:9.1f}%")

    halves = rows[1:]
    same_sign = (halves[0]["差"] > 0) == (halves[1]["差"] > 0)
    strong = all(h["随机块分位"] >= 0.95 for h in halves)
    print(f"\n判据① 两半同号 : {'✓' if same_sign else '✗'}"
          f"（前 {halves[0]['差']*100:+.1f}% / 后 {halves[1]['差']*100:+.1f}%）")
    print(f"判据② 两半都 ≥95% 分位 : {'✓' if strong else '✗'}"
          f"（前 {halves[0]['随机块分位']*100:.1f}% / 后 {halves[1]['随机块分位']*100:.1f}%）")
    print(f"\n结论：{'择时是真信号，值得做「同择时 + 中性篮子」' if (same_sign and strong) else '**择时不成立**（单窗运气）⇒ 篮子候选不必做'}")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
