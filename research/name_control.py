"""**名字随机对照** —— 同一天入场、同样的持有结构，只把「选哪只」换成随机合格标的。

为什么需要它（这是本仓库唯一还没做过的对照）：现有的"同日随机入场对照"
（`search_12m.random_control`）随机的是**日子**，用来回答"择时有没有信息"；
它**不回答**"选股有没有信息"，而这两件事必须分开 —— 一个是时机、一个是标的。

判据（先写死）：
  * 若真实策略的读数落在对照分布的**下尾（≤5 分位）** ⇒ **选股是反预测的**
    （"随便买都更好"），那么任何"改进选股"的方向都只剩**反着用**这一条，
    而反着用的对照臂（`str_inv` 一族）已经在族搜索里被否过；
  * 若落在**中段** ⇒ 选股只是**没有信息**（与随机不可区分），负超额来自结构与成本，
    不是"挑得差"；此时"改进选股"没有可改进的对象，只能换信号源。

口径：随机标的自**当日合格集**（`universe.require_eligible_mask`）里均匀抽取，
每个信号日抽的数量与真实策略在该日的信号数相同；排序分沿用该族的默认口径（mom120），
免得对照里混进"随机挑选 + 另一个排序规则"两个变量。持有结构（止损/止盈/持有期）
与真实策略**一字不差**。

用法：
    python3 -m research.name_control --preset us_reversal_dip_v1 --draws 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research import broker, families, golden_baseline as G, indicators   # noqa: E402
from research import overlay, panel as panel_mod, presets, regime as regime_mod  # noqa: E402
from research import signals, universe as mod_universe                     # noqa: E402

logger = logging.getLogger("name_control")


def load_chain():
    p = presets.get_preset("us_uptrend_pullback_v1")
    panel = panel_mod.load_panel_for_preset(p)
    ind = indicators.compute_all(panel)
    reg = regime_mod.compute_regime(ind, **(p.get("regime") or {}))
    return panel, ind, reg


def random_rows(ind: pd.DataFrame, date: pd.Timestamp, k: int, eligible: pd.Series,
                rng) -> pd.DataFrame:
    """在某日**合格集**里随机抽 k 只，按 `families.select` 的同一套列名补齐。"""
    day = ind[(ind["date"] == date) & eligible]
    if day.empty or k <= 0:
        return pd.DataFrame(columns=signals.ENTRY_COLS)
    take = rng.choice(len(day), size=min(k, len(day)), replace=False)
    hits = day.iloc[np.sort(take)].copy()
    hits["score"] = hits["mom120"]                 # 与族的默认排序口径一致
    hits["sig_close"] = hits["close"]
    hits["sig_ma20"] = hits["ma20"]
    hits["sig_ma60"] = hits["ma60"]
    hits["sig_down_streak"] = hits["down_streak"]
    hits["sig_pullback"] = hits["pullback"]
    hits["sig_mom120"] = hits["mom120"]
    hits["sig_amount60"] = hits["amount60"]
    return hits[signals.ENTRY_COLS]


def main() -> int:
    ap = argparse.ArgumentParser(description="名字随机对照（同择时，随机标的）")
    ap.add_argument("--preset", default="us_reversal_dip_v1")
    ap.add_argument("--start", default="2024-04-09")
    ap.add_argument("--end", default="2026-10-14")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "name_control.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    panel, ind, reg = load_chain()
    eligible = mod_universe.require_eligible_mask(ind)
    preset = presets.get_preset(a.preset)          # 别覆盖 panel（同名会静默传错东西）
    case = {"name": a.preset, "preset": a.preset, "start": a.start, "end": a.end}
    res, sig = G.run_preset(case, panel, ind)
    real_eq = res["equity"].set_index("date")
    real = overlay.excess_vs_pool(real_eq, reg, a.start, a.end)
    real_own = float(real_eq["equity"].iloc[-1] / real_eq["equity"].iloc[0] - 1.0)
    per_day = sig.groupby("date").size().to_dict()
    logger.warning("真实：%d 个信号，%d 笔成交，自身 %.2f%%，超额 %+.2fpp",
                   len(sig), len(res["trades"]), real_own * 100, real["excess"] * 100)

    rng = np.random.default_rng(a.seed)
    rows = []
    for d in range(a.draws):
        parts = [random_rows(ind, pd.Timestamp(dt), int(k), eligible, rng)
                 for dt, k in per_day.items()]
        rsig = (pd.concat(parts).sort_values(["date", "score"], ascending=[True, False])
                .reset_index(drop=True) if parts else
                pd.DataFrame(columns=signals.ENTRY_COLS))
        r = broker.simulate(presets.get_preset(a.preset), rsig, ind,
                            start=a.start, end=a.end,
                            cost_bps_roundtrip=G.preset_cost(a.preset))
        eq = r["equity"].set_index("date")
        ov = overlay.excess_vs_pool(eq, reg, a.start, a.end)
        rows.append({"draw": d, "n_trades": int(len(r["trades"])),
                     "own": float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0),
                     "excess": ov["excess"], "exposure": ov["exposure"]})
        if (d + 1) % 25 == 0:
            logger.warning("  %d/%d 次对照完成", d + 1, a.draws)

    df = pd.DataFrame(rows)
    pct_exc = float((df["excess"] <= real["excess"]).mean())
    pct_own = float((df["own"] <= real_own).mean())
    print(f"\n=== 名字随机对照（{a.preset}，{a.start}..{a.end}，{a.draws} 次）===")
    print(f"真实             ：自身 {real_own*100:+8.2f}%  超额 {real['excess']*100:+7.2f}pp")
    print(f"随机标的中位     ：自身 {df['own'].median()*100:+8.2f}%  超额 {df['excess'].median()*100:+7.2f}pp")
    print(f"随机标的 p5/p95  ：自身 {df['own'].quantile(.05)*100:+8.2f}% / {df['own'].quantile(.95)*100:+8.2f}%"
          f"   超额 {df['excess'].quantile(.05)*100:+7.2f}pp / {df['excess'].quantile(.95)*100:+7.2f}pp")
    print(f"真实所处的分位   ：自身 {pct_own*100:5.1f}%   超额 {pct_exc*100:5.1f}%")
    verdict = ("**反预测**（≤5 分位）：随便买都更好 —— 只剩「反着用」这一条，而那已被族搜索否过"
               if pct_exc <= 0.05 else
               "**无信息**（5~95 分位内）：与随机不可区分 ⇒ 负超额来自结构与成本，不是「挑得差」"
               if pct_exc < 0.95 else "**正预测**（≥95 分位）：选股有信息 —— 与族搜索的结论冲突，需复核")
    print(f"判读（按先写死的判据）：{verdict}")
    Path(a.out).write_text(json.dumps({"preset": a.preset, "window": [a.start, a.end],
                                       "draws": a.draws, "real": {**real, "own": real_own},
                                       "control": df.describe().to_dict(),
                                       "pct_excess": pct_exc, "pct_own": pct_own,
                                       "rows": rows}, ensure_ascii=False, indent=1))
    print(f"已写 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
