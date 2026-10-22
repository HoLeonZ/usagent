"""事件维度归因 —— 协议第 5 节的**步 1**（`docs/2026-10-17-event-preset-presearch.md`）。

只回答一个问题：**现有最佳 preset 的成交里，"持有窗内撞上财报"的那批笔，是否比其余笔差？**
它的用途是决定"事件维度值不值得做成 preset"。它**不是搜索**（不产生参数组合、不计
n_trials），也**不看 holdout** —— 窗口落在训练窗之外时直接拒绝跑（除非显式放行）。

为什么写成脚本而不是一次性命令：①窗口纪律要在**代码里**挡住，不是靠自觉；
②结论要被能复算；③"当前最佳"必须由**程序**按 §2 的判据选出，不由人挑。

两个分组口径（都要报，它们回答的是两件事）：

  * **前瞻窗**（`[信号日, 信号日+forward]`）：这是**过滤器看得见的**集合 —— 开着
    `avoid_earnings` 时被剔除的就是它。它的两组差异**预测**过滤器的效果。
  * **实现窗**（`[入场日, 出场日]`）：这是**实际撞上**事件的笔（H1 的机制断言：
    跳空穿透止损）。它检验机制，但它不是过滤器能精确挑出的集合（入场前不知道出场日）。

用法：

    python3 -m research.event_attribution                    # 训练窗，自动选当前最佳
    python3 -m research.event_attribution --preset us_reversal_dip_v1
    python3 -m research.event_attribution --allow-holdout     # 只在协议通过后才用
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

from research import earnings as mod_earnings          # noqa: E402
from research import golden_baseline as G              # noqa: E402
from research import indicators, metrics, overlay, panel as panel_mod, presets  # noqa: E402
# ⚠️ 训练窗 / holdout 的**分界线只许有一处定义**（`wfv_search` 的预注册常量）。
# 此前这里自己又写了一份 `TRAIN`/`HOLDOUT` 字面量：`HOLDOUT` 压根没人用（**死常量**），
# 而 `TRAIN` 与 `wfv_search.SEARCH_LO/HI` 是同两条日期 —— 改了预注册却漏改这里，
# 「归因阶段不许看 holdout」这道闸门就会**静默失效**（它正是为防这件事而存在的）。
from research.wfv_search import HOLDOUT_LO, HOLDOUT_HI, SEARCH_LO, SEARCH_HI  # noqa: E402

logger = logging.getLogger("event_attr")

# 训练窗 = 18 个月里去掉最后一年；这里取"数据起点 + 预热"到 holdout 前一天。
# 值**来自预注册常量**（上面的 import），本文件不再抄一份日期。
TRAIN = (SEARCH_LO, SEARCH_HI)
# （原先这里还定义过一个 `HOLDOUT = (...)` 字面量，但全仓库没人用它 —— 删掉：
#  一个没人用、又长得像纪律边界的东西，只会让下一个改窗口的人以为改了它就有用。）
FORWARD_CALENDAR_DAYS = 26          # 与 preset 的 max_hold=18 个交易日相称
OUT_DEFAULT = ROOT / "research" / "results" / "event_attribution.json"


def _assert_window_is_allowed(lo: str, hi: str, allow_holdout: bool) -> None:
    """窗口纪律写进代码：跑 holdout 必须**显式**放行，且会被打印出来。

    这不是形式主义：一旦有人（或某轮 cron）顺手把默认窗口改成 holdout，
    "一次性 holdout" 这条纪律就无声失效了 —— 与 `has_earnings_between` 的
    "未知不等于没有" 是同一类防线。
    """
    if allow_holdout:
        logger.warning("⚠️ 显式放行 holdout 窗口：%s..%s —— 这次读数必须登记 claim", lo, hi)
        return
    t_lo, t_hi = pd.Timestamp(TRAIN[0]), pd.Timestamp(TRAIN[1])
    if pd.Timestamp(lo) < t_lo or pd.Timestamp(hi) > t_hi:
        raise SystemExit(
            f"窗口 {lo}..{hi} 落在训练窗 {TRAIN[0]}..{TRAIN[1]} 之外 —— 归因阶段不许看\n"
            f"  holdout（协议 §5）。确要用就显式加 --allow-holdout，并登记 claim。")


def load_chain() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """真数据链路：面板 → 指标 → regime（与 golden 载体同一条路径）。"""
    if not G.db_available():
        raise SystemExit(f"本地 DB 不在（{G.db_path()}）—— 归因需要真数据")
    preset = presets.get_preset("us_uptrend_pullback_v1")   # 取池口径（universe）取自它
    panel = panel_mod.load_panel_for_preset(preset)
    ind = indicators.compute_all(panel)
    from research import regime as regime_mod
    reg = regime_mod.compute_regime(ind, **(preset.get("regime") or {}))
    return panel, ind, reg


def rank_presets(panel: pd.DataFrame, ind: pd.DataFrame, reg: pd.DataFrame,
                 window: tuple[str, str]) -> list[dict]:
    """按协议 §2 的判据（曝光匹配 overlay 超额）给四条 preset 排名。

    "当前最佳"由这个函数选出，不由人挑 —— 否则"比当前最佳更好"这句话取决于谁挑的。

    ⚠️ 同时报 **超额/敞口** 与 `bench_on_exposed_days`（只在策略有敞口的那些天持有池子）：
    本窗池子 +32.9% 而策略只有 26~31%，于是 `超额 = w ×(策略−池) < 0` ——
    **任何降低敞口的做法都会让"超额"自动变好看**。也就是说 P1 单独是可被"少做"刷高的，
    所以判读必须同时看这两列（这不是改判据，是把判据的退化面写出来）。
    """
    rows = []
    bench = overlay.pool_daily_returns(reg)
    for name in sorted(presets.PRESETS):
        case = {"name": name, "preset": name, "start": window[0], "end": window[1]}
        try:
            res, _ = G.run_preset(case, panel, ind)
        except Exception as exc:                            # noqa: BLE001
            rows.append({"preset": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        eq = res["equity"].set_index("date")
        ov = overlay.excess_vs_pool(eq, reg, window[0], window[1])
        ba = metrics.beta_attribution(eq["equity"].pct_change().fillna(0.0), bench,
                                      eq["exposure"])
        rows.append({"preset": name, "n_trades": int(len(res["trades"])),
                     "overlay_cagr": ov["cagr"], "bench": ov["bench"],
                     "excess": ov["excess"], "exposure": ov["exposure"],
                     "excess_per_exposure": (ov["excess"] / ov["exposure"]
                                             if ov["exposure"] else float("nan")),
                     "bench_on_exposed_days": ba["bench_on_exposed_days"],
                     "own_total_return": float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)})
    ranked = sorted([r for r in rows if "excess" in r],
                    key=lambda r: r["excess"], reverse=True)
    return ranked + [r for r in rows if "excess" not in r]


def _trade_groups(trades: pd.DataFrame, by_symbol: dict, coverage, forward_days: int,
                  trading_days, recent_grid=(3, 5, 10)) -> dict:
    """把成交分组（前瞻 / 实现 / **距上一份财报 ≤ k 个交易日**）。

    前瞻与实现两组回答 H1（过滤器看得见的集合 / 实际撞上的笔）；
    `rec{k}` 组回答 **H2**（事件刚过 = 不确定性刚消解）—— 按**信号日**判定，
    因为过滤器的决策只能发生在信号日（因果）。
    """
    r_mult = metrics.r_multiples(trades)
    out: dict[str, list[int]] = {"fwd_yes": [], "fwd_no": [], "real_yes": [], "real_no": []}
    for k in recent_grid:
        out[f"rec{k}_yes"] = []
        out[f"rec{k}_no"] = []
    gap_none = 0
    for i, row in trades.iterrows():
        sym = str(row["symbol"])
        sig_d = pd.Timestamp(row.get("signal_date", row["entry_date"]))
        ent, ext = pd.Timestamp(row["entry_date"]), pd.Timestamp(row["exit_date"])
        fwd = mod_earnings.has_earnings_between(
            by_symbol, sym, sig_d, sig_d + pd.Timedelta(days=forward_days),
            coverage=coverage)
        real = mod_earnings.has_earnings_between(
            by_symbol, sym, ent, ext, coverage=coverage)
        out["fwd_yes" if fwd else "fwd_no"].append(i)
        out["real_yes" if real else "real_no"].append(i)
        gap = mod_earnings.last_earnings_gap_trading_days(by_symbol, sym, sig_d,
                                                          trading_days)
        if gap is None:
            gap_none += 1
        for k in recent_grid:
            recent = gap is not None and gap <= k
            out[f"rec{k}_yes" if recent else f"rec{k}_no"].append(i)
    return {"groups": out, "r_mult": r_mult, "gap_unknown": gap_none}


def summarize(trades: pd.DataFrame, idx: pd.Series, group: list[int],
              r_mult: pd.Series) -> dict:
    """一组的读数：尾部指标为主（H1 是**尾部**假设，不是均值假设）。"""
    if not group:
        return {"n": 0}
    t = trades.loc[group]
    r = r_mult.loc[group].astype(float)
    # 每笔相对池子的超额（同窗、因果：用 asof 取 ≤ 该日期的最近指数值）
    def _pool_ret(row) -> float:
        a = idx.asof(pd.Timestamp(row["entry_date"]))
        b = idx.asof(pd.Timestamp(row["exit_date"]))
        return float(b / a - 1.0) if a and a > 0 else float("nan")
    pool = t.apply(_pool_ret, axis=1)
    notion = (t["entry_price"].astype(float) * t["size"].astype(float))
    own = t["net_pnl"].astype(float) / notion.replace(0.0, np.nan)
    pen = pd.to_numeric(t["sl_penetration"], errors="coerce")
    return {
        "n": int(len(t)),
        "win_rate": float((t["net_pnl"].astype(float) > 0).mean()),
        "r_mean": float(r.mean()), "r_median": float(r.median()),
        "r_p10": float(r.quantile(0.10)), "r_worst": float(r.min()),
        "hold_days_mean": float(pd.to_numeric(t["hold_days"], errors="coerce").mean()),
        "stop_share": float((t["exit_reason"] == "sl").mean()),
        "sl_pen_median": float(pen.median()) if pen.notna().any() else None,
        "own_ret_mean": float(own.mean()),
        "pool_ret_mean": float(pool.mean()),
        "excess_vs_pool_mean": float((own - pool).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="事件维度归因（协议步 1）")
    ap.add_argument("--preset", default=None, help="不给则由程序按 overlay 超额选当前最佳")
    ap.add_argument("--start", default=TRAIN[0])
    ap.add_argument("--end", default=TRAIN[1])
    ap.add_argument("--forward-calendar-days", type=int, default=FORWARD_CALENDAR_DAYS)
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _assert_window_is_allowed(a.start, a.end, a.allow_holdout)

    cal = mod_earnings.load_calendar()
    coverage = mod_earnings.calendar_coverage(cal)
    by_symbol = mod_earnings.earnings_dates_by_symbol(cal)
    logger.info("财报日历：%d 个工作日，覆盖 %s..%s", len(cal), coverage[0].date(),
                coverage[1].date())

    panel, ind, reg = load_chain()
    ranked = rank_presets(panel, ind, reg, (a.start, a.end))
    logger.info("\n=== 四条 preset 在 %s..%s 上的曝光匹配超额（越高越好）===", a.start, a.end)
    for r in ranked:
        if "excess" in r:
            logger.info("  %-32s 超额 %+7.2fpp  超额/敞口 %+7.2f  池 %+7.2fpp  敞口 %5.1f%%  "
                        "%3d 笔  [在场日池 %+6.1f%% vs 自身 %+6.1f%%]",
                        r["preset"], r["excess"] * 100, r["excess_per_exposure"] * 100,
                        r["bench"] * 100, r["exposure"] * 100, r["n_trades"],
                        r["bench_on_exposed_days"] * 100, r["own_total_return"] * 100)
        else:
            logger.info("  %-32s 跑不动：%s", r["preset"], r["error"])
    logger.info("  ⚠️ 池子为正而策略跑输时，超额 = 敞口×(策略−池) **为负** ⇒ 单纯少做也会让"
                "\n     超额变好看。判读必须同时看「超额/敞口」与「在场日池 vs 自身」两列。")

    best = a.preset or (ranked[0]["preset"] if ranked else None)
    if not best:
        raise SystemExit("选不出当前最佳 preset")
    logger.info("\n=== 归因对象：%s ===", best)

    case = {"name": best, "preset": best, "start": a.start, "end": a.end}
    res, _ = G.run_preset(case, panel, ind)
    trades = res["trades"].copy()
    if "signal_date" not in trades.columns:
        trades["signal_date"] = trades["entry_date"]      # 成交表可能没留下信号日
    g = _trade_groups(trades, by_symbol, coverage, a.forward_calendar_days,
                      ind["date"].unique())
    idx = reg.set_index("date")["ew_index"].sort_index()   # asof 要求有序
    report = {"preset": best, "window": [a.start, a.end],
              "forward_calendar_days": a.forward_calendar_days,
              "recent_grid_trading_days": [3, 5, 10],
              "gap_unknown": g["gap_unknown"],
              "ranking": ranked,
              "calendar": {"days": len(cal),
                           "covered": [str(coverage[0].date()), str(coverage[1].date())]},
              "groups": {}}
    logger.info("  （距上一份财报的交易日数无法判定：%d 笔 —— 归入「事件未刚过」）",
                g["gap_unknown"])
    for key in ("fwd_yes", "fwd_no", "real_yes", "real_no",
                "rec3_yes", "rec3_no", "rec5_yes", "rec5_no", "rec10_yes", "rec10_no"):
        s = summarize(trades, idx, g["groups"][key], g["r_mult"])
        report["groups"][key] = s
        logger.info("  %-9s n=%3d  R中位 %+6.2f  R均值 %+6.2f  R最差 %+6.2f  胜率 %4.1f%%  "
                    "止损占比 %4.1f%%  窗内 vs 池 %+6.2fpp",
                    key, s.get("n", 0), s.get("r_median", float("nan")),
                    s.get("r_mean", float("nan")), s.get("r_worst", float("nan")),
                    s.get("win_rate", float("nan")) * 100,
                    s.get("stop_share", float("nan")) * 100,
                    s.get("excess_vs_pool_mean", float("nan")) * 100)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    logger.info("\n已写 %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
