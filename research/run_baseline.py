"""基线运行入口。

    python -m research.run_baseline --preset us_uptrend_pullback_v1
    python -m research.run_baseline --preset us_uptrend_pullback_v1 --windows
    python -m research.run_baseline --preset us_uptrend_pullback_v1 --out research/results/x.json

输出:头条(总收益/CAGR/超额/E[R]/beta 归因/DSR)+ 逐窗口表。
所有数字都来自 broker 的**正确**权益曲线(cash + qty×close)。
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.preset_contract import (INDICATOR_PRICE_ADJUST, describe_engine_spec,
                                   spec_header)
import obs   # src/ 已在 sys.path（research/__init__ 的垫片）
from research import broker, indicators, metrics, panel as panel_mod, presets, regime, signals
from research.broker import trade_fingerprint
from research.panel import data_vintage

logger = logging.getLogger("run_baseline")



@functools.lru_cache(maxsize=4)
def _load_panel_and_indicators(preset_name: str,
                               adjust: str = "total"):
    """载入真库 + 算指标。按 adjust 记忆化:一次进程里 DB 不变,重复载入没有意义
    (审计后新增的入口一致性测试会反复调 run(),不缓存的话套件从 6 秒涨到 100 秒)。

    ⚠️ 返回的对象是共享的,调用方**不得就地修改** —— select_entries / simulate 都只读。
    """
    # 取池口径由 preset 声明（见 panel.load_panel_for_preset）—— 缓存键因此必须
    # 带上 preset 名，否则换个 preset 会命中上一个池子的面板。
    p = panel_mod.load_panel_for_preset(presets.get_preset(preset_name), adjust=adjust)
    return p, indicators.compute_all(p)   # 必须用 compute_all(55列):
                                          # 带新过滤层的 preset 需要 rsi/stoch/bb/dist_high252 等列


def _build(args):
    preset = presets.get_preset(args.preset)
    # 指标价轴是**引擎约定**(§3(b):数学指标必须用分红回补价),不是 preset 旋钮 ——
    # 曾把它做成 preset 键,结果是「一个入口读它、另外四个入口静默忽略」。见
    # core.preset_contract.INDICATOR_PRICE_ADJUST。
    p, ind = _load_panel_and_indicators(args.preset, INDICATOR_PRICE_ADJUST)
    reg = regime.compute_regime(ind, **(preset.get("regime") or {}))
    return p, ind, reg


def _signals(ind, reg, preset, lo, hi, *, avoid_earnings: bool = False):
    if preset.get("family"):
        # 走 families.py 的族(str 反转、volshock、cap…)。**只有一处实现**:
        # 这里不再重写掩码,避免与 families.select 分叉。
        from research import families
        sig = families.select(ind, str(preset["family"]),
                              dict(preset.get("family_params") or {}), lo, hi, reg=reg)
    else:
        kw = dict(preset["signal"])
        sig = signals.select_entries(ind, start_date=lo, end_date=hi,
                                     regime_df=reg if preset.get("regime") else None, **kw)
    # 事件过滤走**唯一入口**（`signals.apply_event_filters`）：preset 自己声明的
    # `avoid_earnings` / `require_earnings_within_trading_days` 在这里生效，
    # CLI 的 --avoid-earnings 只是「强制打开」的覆盖。放两处实现 = 两条入口跑的不是
    # 同一份信号表，而 golden 载体锁的就是这份表（有测试盯住两个入口的一致性）。
    sig = signals.apply_event_filters(sig, preset,
                                      trading_days=np.sort(ind["date"].unique()),
                                      force_avoid_earnings=avoid_earnings)
    return sig


def _fmt(x, nd=4):
    """数值化用于 JSON;非数值(如 Timestamp)转字符串。"""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if not math.isfinite(v) else round(v, nd)


def run(args) -> dict:
    preset = presets.get_preset(args.preset)
    # 生效成本:CLI 覆盖值优先,否则取 preset 口径。**跑数路径必须读 preset**,
    # 否则改 preset 的成本假设不会反映到 CLI 数字上(审计实测过的静默分叉)。
    eff_cost = float(preset["cost_bps_roundtrip"] if args.cost_bps is None
                     else args.cost_bps)
    # 价差口径同理：CLI 覆盖优先，否则取 preset 声明。做成 CLI 旋钮是为了让
    # **成本敏感性**可复现地跑（§0）—— 审计发现此前它只是一个人肉改 preset 的操作。
    eff_spread = preset["spread_model"] if args.spread_model is None else args.spread_model
    preset = {**preset, "spread_model": eff_spread}
    _, ind, reg = _build(args)
    # 预热边界**由数据算出**,不写死常量。此前是 FIRST_SIGNAL_FLOOR = "2024-04-09",
    # 而它恰好等于「当前这一份数据」的第 121 根 K 线 —— 换一份数据(窗口起点漂移)后
    # 仍会拿它当起点,于是预热期被算进 CAGR 的分母,且没有任何断言会报警。
    # panel.first_signal_date 早就算好了这个边界,只是此前是**死代码**(零调用)。
    lo = args.start or str(panel_mod.first_signal_date(ind).date())
    hi = args.end or str(ind["date"].max().date() + pd.Timedelta(days=1))

    sig = _signals(ind, reg, preset, lo, hi, avoid_earnings=args.avoid_earnings)
    res = broker.simulate(preset, sig, ind, start=lo, end=hi,
                          cost_bps_roundtrip=eff_cost,
                          # regime_ok 必须传下去：声明了 regime_exit_policy="close"
                          # 的 preset 缺了它会被 fail-fast（不许静默退化，审计 D4）
                          regime_ok=regime.regime_ok_dates(reg))
    eq, tr = res["equity"], res["trades"]
    ts = metrics.tail_stats(tr)
    cg = metrics.cagr(eq["equity"], eq["date"])

    idx = reg.set_index("date")["ew_index"]
    b = idx[(idx.index >= pd.Timestamp(lo)) & (idx.index < pd.Timestamp(hi))]
    bench_tot = float(b.iloc[-1] / b.iloc[0] - 1.0) if len(b) > 1 else float("nan")

    sret = eq.set_index("date")["equity"].pct_change().dropna()
    bret = idx.pct_change().reindex(eq["date"]).dropna()
    ba = metrics.beta_attribution(sret, bret, eq.set_index("date")["exposure"])

    # DSR 从**收益序列**算（审计 B3）：T/偏度/峰度/自相关全部由序列派生，
    # 调用方只能给序列与试验次数 ⇒ "T 是什么"不再是一个可以各说各话的问题。
    dsr = metrics.deflated_sharpe_from_returns(
        sret.to_numpy(dtype=float), n_trials=preset["n_comparisons"],
        label=f"{args.preset} {lo}..{hi}")

    print(f"\n=== {args.preset} | {lo} .. {hi} ===")
    # 口径自报（审计 E5 及其同类）：数字是在什么成本/风险假设下产生的，必须由报告
    # 自己说清 —— 否则读者只能去读 preset 源码，或者默认"没有价差、没有上限"。
    print(f"生效口径: {spec_header(preset)}")
    print(f"信号 {len(sig)}  成交 {len(tr)}")
    print(f"总收益 {res['metrics']['total_return']*100:+.2f}%   CAGR {cg*100:+.2f}%"
          f"   基准等权 {bench_tot*100:+.2f}%   超额 {(res['metrics']['total_return']-bench_tot)*100:+.2f}pp")
    print(f"Sharpe {res['metrics']['sharpe']:+.2f}  maxDD {res['metrics']['max_dd']*100:+.1f}%"
          f"   平均敞口 {res['metrics']['avg_exposure']*100:.1f}%")
    print(f"E[R] {ts['expectancy_r']:+.3f}  胜率 {ts['win_pct']*100:.1f}%  盈亏比 {ts['payoff']:.2f}"
          f"  PF {ts['profit_factor']:.2f}  前10%贡献 {ts['top_decile_share']:.2f}")
    print(f"退出: SL {res['metrics']['sl_count']}  TP {res['metrics']['tp_count']}"
          f"  max_hold {res['metrics']['time_count']}  平均持有 {res['metrics']['avg_hold_days']:.2f} 天")
    print(f"beta {ba['beta']:+.2f}  alpha年化 {ba['alpha_annual']*100:+.1f}% (t={ba['t_alpha']:+.2f})"
          f"  在场天数 {ba['pct_days_exposed']*100:.0f}%  在场日基准 {ba['bench_on_exposed_days']*100:+.1f}%")
    if dsr.get("reason"):
        print(f"DSR：**不报**（{dsr['reason']}）")
    else:
        print(f"DSR({dsr['n_returns']} 个日收益,年化 Sharpe {dsr['observed_sharpe']:+.2f},"
              f"偏度 {dsr['skewness']:+.2f},峰度 {dsr['kurtosis']:.1f},"
              f"自相关 {dsr['autocorr_lag1']:+.3f},k={preset['n_comparisons']}) "
              f"= {dsr['deflated_sharpe']:.3f}  (p={dsr['dsr_p_value']:.3f})")

    # 双报 Sharpe 口径（B7）：低敞口策略上，"含空仓日"与"只在场日"会差很多。
    _si = res["metrics"].get("sharpe_invested")
    if _si is not None and np.isfinite(_si):
        print(f"Sharpe 口径: 含空仓日 {res['metrics']['sharpe']:+.2f} | "
              f"只在场日 {_si:+.2f}（在场 {res['metrics']['n_days_invested']} 天 / "
              f"{res['metrics']['days']}）—— DSR 用的是**含空仓日**那条序列")

    # 风格因子归因（C4）：超额是不是选股能力，只有归因能回答 —— CAGR 之差对此是瞎的。
    # 默认关闭：它要构造横截面因子（多空三分位），比跑一次回测本身还贵，而测试套件
    # 会反复调 run()。
    if getattr(args, "attribution", False):
        from research import factor_model as FM
        _fac = FM.style_factor_returns(ind.loc[ind["date"].isin(sret.index)], lo=lo, hi=hi)
        _att = FM.attribute(sret, _fac, bret.reindex(sret.index).fillna(0.0))
        print("归因: " + FM.describe(_att))
        print("      ⚠️ 无价值/规模因子（无基本面数据、市值快照非 PIT）——覆盖范围到此为止")

    # 入场价越界（E7）：滑点/价差把成交价顶出当日 bar 区间 ⇒ 记录里出现物理上不存在的
    # 价格。**默认不钳制**（钳制会在窄 bar 上把滑点整体抹平），但读者必须知道有多少笔。
    _nov = res["metrics"].get("n_entry_overflow", 0)
    if _nov:
        print(f"入场价越界(物理上不存在的成交价): {_nov}/{len(tr)} 笔, 最大 "
              f"{res['metrics']['max_entry_overflow_bps']:.0f}bp —— 默认不钳制（保留滑点）；"
              f"要钳制请显式声明 clamp_entry_to_bar")

    # 容量余量（D5/E4）："这套策略能装多少钱"此前在库里和报告里都没有读数。
    _hr = res["metrics"].get("capacity_headroom_min")
    if _hr is not None and np.isfinite(_hr):
        print(f"容量余量: 最小 {_hr:,.0f}× / 中位 "
              f"{res['metrics']['capacity_headroom_median']:,.0f}×"
              f"（= 量能上限 ÷ 本单股数；<1 表示该笔已被 §4(a) 截断）"
              f" ⇒ 当前资金规模下距量能上限还有 {_hr:,.0f} 倍空间")

    if ts["n"] and ts["n"] < 30:
        print(f"⚠️  样本仅 {ts['n']} 笔 —— 任何显著性结论都不可靠")
    if ts["n"] and _degenerate_r(tr):
        print("⚠️  R 分布退化(几乎全部 ≈ −1.00):t 值被机械放大,本行 t 不可用于判显著性")

    out = {"preset": args.preset, "start": lo, "end": hi, "cost_bps": eff_cost,
           "spread_model": eff_spread,
           "engine_spec": dict(describe_engine_spec(preset)),
           # 交易列表指纹(§6):让 CLI 的输出可以被 golden 基线逐一核对
           "trades_sha256": trade_fingerprint(tr),
           # 数据 vintage:让每个被报告的数字可追溯到一份具体的数据快照。
           # 没有它,「当前代码 + 一条命令」复现不出任何历史数字(§0/§7)。
           # 数据 vintage 保持**顶层**（既有接口：测试与下游都按这个键读；不许为了好看挪窝）。
           "data_vintage": data_vintage(ind),
           # 代码侧的凭证（2026-10-17 对抗核验：只有 vintage 答不出"哪份代码算的"）——
           # ts 带时区 + git sha + **工作区是否脏**（脏树上 sha 相同、代码不同）。
           # 两份事实各一处，不互为副本。
           "manifest": obs.manifest(),
           "avoid_earnings": bool(args.avoid_earnings),
           "n_signals": len(sig), "metrics": {k: _fmt(v) for k, v in res["metrics"].items()},
           "cagr": _fmt(cg), "benchmark_total": _fmt(bench_tot),
           "excess": _fmt(res["metrics"]["total_return"] - bench_tot),
           "tail": {k: _fmt(v) for k, v in ts.items()},
           "beta_attribution": {k: _fmt(v) for k, v in ba.items()},
           "dsr": {k: _fmt(v) for k, v in dsr.items()},
           "n_comparisons": preset["n_comparisons"], "validated": preset["validated"]}

    if args.windows:
        rows = []
        for ws, we in metrics.monthly_windows(lo[:7], hi[:7], 2, 1):
            try:
                s = _signals(ind, reg, preset, ws, we,
                             avoid_earnings=args.avoid_earnings)
                r = broker.simulate(preset, s, ind, start=ws, end=we,
                                    cost_bps_roundtrip=eff_cost)
                m = r["metrics"]
                bb = idx[(idx.index >= pd.Timestamp(ws)) & (idx.index < pd.Timestamp(we))]
                bt = float(bb.iloc[-1] / bb.iloc[0] - 1.0) if len(bb) > 1 else float("nan")
                rows.append({"start": ws, "end": we, "trades": m["trades"],
                             "ret": _fmt(m["total_return"]), "bench": _fmt(bt),
                             "excess": _fmt(m["total_return"] - bt),
                             "e_r": _fmt(metrics.tail_stats(r["trades"])["expectancy_r"])})
            except (broker.BrokerError, signals.SignalError) as e:
                rows.append({"start": ws, "end": we, "trades": 0, "ret": None,
                             "bench": None, "excess": None, "e_r": None, "note": str(e)[:60]})
        wdf = pd.DataFrame(rows)
        print("\n--- 逐窗口(2m/1m)---")
        print(wdf.to_string(index=False))
        pos = sum(1 for r in rows if r["excess"] is not None and r["excess"] > 0)
        print(f"\n窗口数 {len(rows)} | 超额为正 {pos}/{len(rows)}")
        out["windows"] = rows

    return out


def _degenerate_r(trades: pd.DataFrame, tol: float = 0.05) -> bool:
    """R 是否退化 —— 超过 2/3 的 R 都贴近 −1.00 时,t 值没有判别力。"""
    r = metrics.r_multiples(trades)
    if r.empty:
        return False
    return float(((r + 1.0).abs() < tol).mean()) > 2 / 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="us_uptrend_pullback_v1")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    # None ⇒ 用 preset 里的成本口径(§2(b):常数只在 core/costs.py 定义一次)。
    # 显式给值只用于成本敏感性分析。
    ap.add_argument("--attribution", action="store_true",
                    help="额外跑风格因子归因（market/mom/lowvol + 跟踪误差/IR，审计 C4）")
    ap.add_argument("--spread-model", default=None,
                    choices=["none", "corwin_schultz"],
                    help="价差口径(默认取 preset):成本模型里**尺寸无关**的那一半")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="覆盖 preset 的成本口径(bp);默认用 preset 里的值")
    ap.add_argument("--windows", action="store_true", help="额外输出逐窗口表")
    ap.add_argument("--avoid-earnings", action="store_true",
                    help="H1:剔除持有窗口内夹财报的信号(需先跑 research.earnings)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    out = run(a)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n已写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
