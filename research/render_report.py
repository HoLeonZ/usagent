"""回测明细 → 自包含 HTML 报告。

每笔交易一张卡片:窗口 K 线 + MA20/MA60 + 止损止盈线 + 入场/出场标记 +
信号日**逐条条件清单**(含实际取值 vs 阈值)+ 成交量。

设计约束
--------
* **不依赖 matplotlib / plotly**(本机都没装)。手写内联 SVG,整份报告自包含、
  离线可看、可直接归档。
* **数值与回测严格同源**:信号日指标读自 compute_all 的同一张表,入场/出场价
  读自 broker 的真实成交(含 sl_gap 按开盘价成交这类悲观口径)。
  报告里的数字与回测不一致 = 报告在说谎,所以 render 不重算任何价格。
* **信号日信息不得含未来**:卡片右侧的条件清单只用信号日及之前的值。
  卡片**画面**会显示出场之后的价格(供复盘),那是展示,不参与信号。

诚实声明会印在报告顶部,不折叠、不淡化 —— 这份报告展示的是 82 次试验里
最好的一个配置,而它 0 个配置达到 30%,且跑输无脑持有。
"""
from __future__ import annotations

import argparse
import html
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker, indicators, metrics, panel as panel_mod, regime, signals

logger = logging.getLogger("render")

# 12 个月搜索里成绩最好的配置(见 search_12m.py 与提交 eed59c1)
BEST_CFG: dict = {
    "min_dist_high252": -0.1,
    "max_stoch_k": 60.0,
    "sl_mult": 1.5,
    "tp_mult": 6.0,
}
WIN_LO, WIN_HI = "2025-10-14", "2026-10-14"
PRE_SIG_BARS = 60          # 卡片左侧多看多少根 K 线
POST_EXIT_BARS = 8         # 卡片右侧多看多少根
OUT_HTML = Path(__file__).resolve().parent / "results" / "trades_12m.html"


class RenderError(RuntimeError):
    pass


# ---------------------------------------------------------------- 条件清单

def condition_rows(ind: pd.DataFrame, symbol: str, sig_date: pd.Timestamp,
                   cfg: dict) -> list[tuple[str, str, str, bool]]:
    """信号日的逐条条件(名称, 实际值, 阈值, 是否满足)。

    **只用信号日那一行** —— 不含任何未来信息。
    """
    row = ind[(ind["symbol"] == symbol) & (ind["date"] == sig_date)]
    if row.empty:
        raise RenderError(f"{symbol} 在 {sig_date.date()} 没有数据")
    r = row.iloc[0]

    def pct(v):
        return "—" if not np.isfinite(v) else f"{v*100:.2f}%"

    def num(v, n=2):
        return "—" if not np.isfinite(v) else f"{v:.{n}f}"

    sp = signals.V21_SIGNAL_PARAMS
    out = [
        # A 趋势
        ("A1", "close > MA60", f"{r['close']:.2f} @{r['ma60']:.2f}",
         f"close>{r['ma60']:.2f}", bool(r["close"] > r["ma60"])),
        ("A2", "MA20 > MA60", f"{r['ma20']:.2f} @{r['ma60']:.2f}",
         f"MA20>{r['ma60']:.2f}", bool(r["ma20"] > r["ma60"])),
        ("A3", "above_ma60_ratio ≥ 阈值", pct(r["above_ma60_ratio"]),
         f"≥{sp['min_above_ma60_ratio']*100:.1f}%",
         bool(r["above_ma60_ratio"] >= sp["min_above_ma60_ratio"])),
        ("A4", "mom120 ≥ 阈值", pct(r["mom120"]),
         f"≥{sp['min_mom120']*100:.1f}%", bool(r["mom120"] >= sp["min_mom120"])),
        ("A5", "close ≥ MA60×(1+buf)", f"{r['close']:.2f}",
         f"≥{r['ma60']*(1+sp['close_ma60_buffer']):.2f}",
         bool(r["close"] >= r["ma60"] * (1 + sp["close_ma60_buffer"]))),
        # B 回调
        ("B1", "连阴 ∈ [min,max]", f"{int(r['down_streak'])}",
         f"[{sp['min_down_streak']},{sp['max_down_streak']}]",
         bool(sp["min_down_streak"] <= r["down_streak"] <= sp["max_down_streak"])),
        ("B2", "当日涨跌幅 ∈ 区间", pct(r["ret1"]),
         f"[{sp['pct_chg_low']*100:.1f}%,{sp['pct_chg_high']*100:.1f}%]",
         bool(sp["pct_chg_low"] <= r["ret1"] <= sp["pct_chg_high"])),
        # D MACD
        ("D1", "MACD dif > 0", num(r["macd_dif"], 4), ">0", bool(r["macd_dif"] > 0)),
        ("D2", "MACD dea > 0", num(r["macd_dea"], 4), ">0", bool(r["macd_dea"] > 0)),
        ("D3", "|bar| < |bar_prev|", f"{abs(r['macd_bar']):.4f}",
         f"<{abs(r['macd_bar_prev']):.4f}",
         bool(abs(r["macd_bar"]) < abs(r["macd_bar_prev"]))),
        # E 流动性
        ("E1", "amount60 ∈ [min,max]", f"${r['amount60']/1e6:.0f}M",
         f"[${sp['min_amount']/1e6:.0f}M,${sp['max_amount']/1e6:.0f}M]",
         bool(sp["min_amount"] <= r["amount60"] <= sp["max_amount"])),
    ]
    # 本次搜索加的过滤层
    for code, col, thr, op, label in (
        ("F1", "dist_high252", cfg.get("min_dist_high252"), ">=", "距 252 日高点"),
        ("F2", "stoch_k", cfg.get("max_stoch_k"), "<=", "KDJ K 值"),
        ("F3", "rsi14", cfg.get("max_rsi14"), "<=", "RSI14"),
        ("F4", "bb_pctb", cfg.get("max_bb_pctb"), "<=", "布林 %B"),
        ("F5", "adx14", cfg.get("min_adx14"), ">=", "ADX14"),
        ("F6", "xs_mom120", cfg.get("min_xs_mom120"), ">=", "横截面动量分位"),
    ):
        if thr is None:
            continue
        v = float(r[col]) if col in r and np.isfinite(r[col]) else float("nan")
        ok = bool(np.isfinite(v) and (v >= thr if op == ">=" else v <= thr))
        fmt = (lambda x: f"{x*100:.1f}%" if col in
               ("dist_high252", "bb_pctb", "xs_mom120") else f"{x:.1f}")
        out.append((code, f"{label} {op} 阈值", fmt(v),
                    f"{op}{fmt(thr)}", ok))
    return out


# ---------------------------------------------------------------- SVG

SVG_W, PAD_L, PAD_R = 760, 46, 12
PRICE_H, VOL_H, GAP = 200, 44, 14


def _scale(vals, lo, hi, y0, y1):
    span = (hi - lo) or 1.0
    return [y1 - (v - lo) / span * (y1 - y0) for v in vals]


def candles_svg(win: pd.DataFrame, tr: pd.Series, sig_date: pd.Timestamp,
                symbol: str) -> str:
    """单笔的 K 线卡:价格面板 + 成交量面板 + 止损止盈线 + 入场/出场标记。"""
    n = len(win)
    if n < 3:
        return '<div class="nodata">数据不足</div>'
    total_h = PRICE_H + GAP + VOL_H
    inner_w = SVG_W - PAD_L - PAD_R
    step = inner_w / n
    bw = max(step * 0.6, 1.0)

    # ⚠️ broker.TRADE_COLS 里**没有** sl_price/tp_price,只有 sl_width/tp_width
    # (分数)。早先这里判 `"sl_price" in tr.index`,条件恒假 → 两条线被静默跳过、
    # 图上什么都没有,而报告其余部分完全正常。现由宽度现算绝对价位。
    sl_px = float(tr["entry_price"]) * (1.0 - float(tr["sl_width"]))
    tp_px = float(tr["entry_price"]) * (1.0 + float(tr["tp_width"]))

    lo = float(min(win["low"].min(), sl_px))
    hi = float(max(win["high"].max(), tp_px))
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad
    y = _scale(win["close"].to_numpy(float), lo, hi, 0, PRICE_H)
    yo = _scale(win["open"].to_numpy(float), lo, hi, 0, PRICE_H)
    yh = _scale(win["high"].to_numpy(float), lo, hi, 0, PRICE_H)
    yl = _scale(win["low"].to_numpy(float), lo, hi, 0, PRICE_H)
    xs = [PAD_L + i * step + step / 2 for i in range(n)]

    # 成交量
    vmax = float(win["volume"].max()) or 1.0
    vy0 = PRICE_H + GAP
    vol = (win["volume"].to_numpy(float) / vmax * VOL_H)

    p = [f'<svg viewBox="0 0 {SVG_W} {total_h}" class="kchart" '
         f'preserveAspectRatio="none" role="img" aria-label="{html.escape(symbol)} K线">']

    # 信号日之前的区域压暗
    dates = list(win["date"])
    try:
        i_sig = dates.index(sig_date)
    except ValueError:
        i_sig = 0
    if i_sig > 0:
        p.append(f'<rect x="{PAD_L}" y="0" width="{i_sig*step:.1f}" '
                 f'height="{PRICE_H}" class="pre-sig"/>')

    # 网格 + 价格轴
    for k in range(5):
        yy = PRICE_H * k / 4
        val = hi - (hi - lo) * k / 4
        p.append(f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{SVG_W-PAD_R}" y2="{yy:.1f}" class="grid"/>')
        p.append(f'<text x="{PAD_L-5}" y="{yy+3:.1f}" class="ax" text-anchor="end">{val:.2f}</text>')

    # 止损 / 止盈线(从入场日起画)
    i_entry = dates.index(pd.Timestamp(tr["entry_date"])) if pd.Timestamp(tr["entry_date"]) in dates else i_sig
    x0 = PAD_L + i_entry * step
    for key, cls, lab in ((sl_px, "sl", f"SL {sl_px:.2f}"),
                          (tp_px, "tp", f"TP {tp_px:.2f}")):
        yy = _scale([key], lo, hi, 0, PRICE_H)[0]
        p.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{SVG_W-PAD_R}" y2="{yy:.1f}" class="{cls}"/>')
        p.append(f'<text x="{SVG_W-PAD_R-2}" y="{yy-3:.1f}" class="ax {cls}-tx" '
                 f'text-anchor="end">{lab}</text>')

    # K 线
    for i in range(n):
        cls = "up" if win["close"].iloc[i] >= win["open"].iloc[i] else "dn"
        x = xs[i]
        p.append(f'<line x1="{x:.1f}" y1="{yh[i]:.1f}" x2="{x:.1f}" y2="{yl[i]:.1f}" class="wick {cls}"/>')
        top, bot = min(yo[i], y[i]), max(yo[i], y[i])
        p.append(f'<rect x="{x-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                 f'height="{max(bot-top,1):.1f}" class="body {cls}"/>')

    # MA20 / MA60
    for col, cls in (("ma20", "ma20"), ("ma60", "ma60")):
        pts, started = [], False
        for i in range(n):
            v = win[col].iloc[i]
            if not np.isfinite(v):
                continue
            yy = _scale([float(v)], lo, hi, 0, PRICE_H)[0]
            pts.append(f"{xs[i]:.1f},{yy:.1f}")
            started = True
        if started and len(pts) > 1:
            p.append(f'<polyline points="{" ".join(pts)}" class="{cls}"/>')

    # 入场 / 出场标记
    if pd.Timestamp(tr["entry_date"]) in dates:
        i = dates.index(pd.Timestamp(tr["entry_date"]))
        yy = _scale([float(tr["entry_price"])], lo, hi, 0, PRICE_H)[0]
        p.append(f'<polygon points="{xs[i]-6:.1f},{yy-16:.1f} {xs[i]+6:.1f},{yy-16:.1f} '
                 f'{xs[i]:.1f},{yy-6:.1f}" class="mk-entry"/>')
        p.append(f'<text x="{xs[i]:.1f}" y="{yy-19:.1f}" class="mk-tx" text-anchor="middle">入 {tr["entry_price"]:.2f}</text>')
    if pd.Timestamp(tr["exit_date"]) in dates:
        i = dates.index(pd.Timestamp(tr["exit_date"]))
        yy = _scale([float(tr["exit_price"])], lo, hi, 0, PRICE_H)[0]
        up = bool(tr["net_pnl"] >= 0)
        p.append(f'<polygon points="{xs[i]-6:.1f},{yy+16:.1f} {xs[i]+6:.1f},{yy+16:.1f} '
                 f'{xs[i]:.1f},{yy+6:.1f}" class="mk-exit {"win" if up else "loss"}"/>')
        p.append(f'<text x="{xs[i]:.1f}" y="{yy+28:.1f}" class="mk-tx" text-anchor="middle">出 {tr["exit_price"]:.2f}</text>')

    # 成交量
    for i in range(n):
        p.append(f'<rect x="{xs[i]-bw/2:.1f}" y="{vy0+VOL_H-vol[i]:.1f}" width="{bw:.1f}" '
                 f'height="{max(vol[i],1):.1f}" class="vol"/>')
    p.append(f'<line x1="{PAD_L}" y1="{vy0+VOL_H}" x2="{SVG_W-PAD_R}" y2="{vy0+VOL_H}" class="grid"/>')
    p.append(f'<text x="{PAD_L-5}" y="{vy0+10}" class="ax" text-anchor="end">VOL</text>')

    # 日期轴(首/信号/末)
    for i, anchor in ((0, "start"), (i_sig, "middle"), (n - 1, "end")):
        if 0 <= i < n:
            p.append(f'<text x="{xs[i]:.1f}" y="{total_h-2}" class="ax" '
                     f'text-anchor="{anchor}">{pd.Timestamp(dates[i]).strftime("%m-%d")}</text>')
    p.append('</svg>')
    return "".join(p)


def equity_svg(eq: pd.DataFrame, bench: pd.Series | None) -> str:
    """权益曲线 + 基准对比。"""
    n = len(eq)
    if n < 2:
        return '<div class="nodata">无权益数据</div>'
    W, H, PL, PR, PT, PB = 900, 220, 58, 14, 12, 22
    s = eq["equity"].to_numpy(float) / eq["equity"].iloc[0]
    series = [(s, "strat")]
    if bench is not None and len(bench) == n:
        b = bench.to_numpy(float)
        series.append((b / b[0], "bench"))
    lo = min(float(x.min()) for x, _ in series)
    hi = max(float(x.max()) for x, _ in series)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    step = (W - PL - PR) / max(n - 1, 1)

    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none" role="img" aria-label="权益曲线">']
    for k in range(5):
        yy = PT + (H - PT - PB) * k / 4
        val = hi - (hi - lo) * k / 4
        p.append(f'<line x1="{PL}" y1="{yy:.1f}" x2="{W-PR}" y2="{yy:.1f}" class="grid"/>')
        p.append(f'<text x="{PL-6}" y="{yy+3:.1f}" class="ax" text-anchor="end">{val:.3f}</text>')
    for arr, cls in series:
        pts = " ".join(f"{PL+i*step:.1f},{PT+(H-PT-PB)-(v-lo)/(hi-lo)*(H-PT-PB):.1f}"
                       for i, v in enumerate(arr))
        p.append(f'<polyline points="{pts}" class="{cls}"/>')
    p.append(f'<text x="{PL+8}" y="{PT+12}" class="ax {"bench-tx"}">— 基准(596只等权,日再平衡)</text>')
    p.append(f'<text x="{PL+8}" y="{PT+26}" class="ax strat-tx">— 策略</text>')
    p.append('</svg>')
    return "".join(p)


# ---------------------------------------------------------------- HTML

CSS = """
:root{--bg:#0f1115;--fg:#e6e8ee;--dim:#9aa3b2;--line:#262b36;--up:#31c48d;--dn:#f05252;
--acc:#5b8def;--sl:#f05252;--tp:#31c48d;--warn:#f0b429;--card:#161a22}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,"PingFang SC","Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:21px;margin:0 0 4px}
h2{font-size:16px;margin:34px 0 10px;border-left:3px solid var(--acc);padding-left:9px}
.sub{color:var(--dim);font-size:13px;margin-bottom:18px}
.warn{background:#241d0e;border:1px solid #4a3a12;border-radius:8px;padding:14px 16px;margin:16px 0}
.warn h3{margin:0 0 8px;font-size:14px;color:var(--warn)}
.warn ul{margin:6px 0 0;padding-left:20px}.warn li{margin:3px 0}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:10px;margin:14px 0}
.kv div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kv b{display:block;color:var(--dim);font-weight:500;font-size:11.5px;letter-spacing:.3px}
.kv span{font-size:17px;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0}
.chead{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;margin-bottom:10px}
.chead .sym{font-size:16px;font-weight:600}
.tag{font-size:11.5px;padding:2px 7px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.tag.win{color:var(--up);border-color:#1e4636}.tag.loss{color:var(--dn);border-color:#4a2222}
.kchart{width:100%;height:264px;display:block}
.grid{stroke:var(--line);stroke-width:.6}
.ax{fill:var(--dim);font-size:10px}
.wick.up,.body.up{stroke:var(--up);fill:var(--up)}
.wick.dn,.body.dn{stroke:var(--dn);fill:var(--dn)}
.body{stroke-width:1}
.ma20{fill:none;stroke:#d9a441;stroke-width:1.2}
.ma60{fill:none;stroke:#7f8ea3;stroke-width:1.2}
.sl{stroke:var(--sl);stroke-width:1;stroke-dasharray:4 3}
.tp{stroke:var(--tp);stroke-width:1;stroke-dasharray:4 3}
.sl-tx{fill:var(--sl)}.tp-tx{fill:var(--tp)}
.pre-sig{fill:#ffffff;opacity:.035}
.mk-entry{fill:var(--acc)}.mk-exit.win{fill:var(--up)}.mk-exit.loss{fill:var(--dn)}
.mk-tx{fill:var(--fg);font-size:10px}
.vol{fill:#39414f}
.chart{width:100%;height:220px;display:block}
.strat{fill:none;stroke:var(--acc);stroke-width:1.6}
.bench{fill:none;stroke:var(--dim);stroke-width:1.2;stroke-dasharray:5 4}
.strat-tx{fill:var(--acc)}.bench-tx{fill:var(--dim)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12.5px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500;text-align:right}
th:first-child,td:first-child{text-align:left}
tr:hover td{background:#1b2029}
.pos{color:var(--up)}.neg{color:var(--dn)}
.conds{font-size:12px;border-collapse:collapse;margin-top:10px;width:100%}
.conds td,.conds th{padding:3px 7px;border-bottom:1px solid #1e222b}
.ok{color:var(--up)}.no{color:var(--dn)}
.nodata{color:var(--dim);padding:16px}
code{background:#1b2029;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def render_html(trades: pd.DataFrame, eq: pd.DataFrame, ind: pd.DataFrame,
                cfg: dict, bench: pd.Series, summary: dict) -> str:
    e = html.escape
    tot = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
    bench_tot = float(bench.iloc[-1] / bench.iloc[0] - 1) if bench is not None and len(bench) else float("nan")

    parts = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             f'<meta name="viewport" content="width=device-width,initial-scale=1">',
             f'<title>回测明细 · {e(summary["preset_name"])}</title>',
             f'<style>{CSS}</style></head><body><div class="wrap">']
    parts.append(f'<h1>回测明细 · {e(summary["preset_name"])}</h1>')
    parts.append(f'<div class="sub">窗口 {e(summary["lo"])} → {e(summary["hi"])} ｜ '
                 f'成交 {len(trades)} 笔 ｜ 未计滑点,成本双边 5bp</div>')

    # ---- 诚实声明(不折叠)----
    parts.append('<div class="warn"><h3>读这份报告之前必须知道的三件事</h3><ul>'
                 f'<li><b>这是 82 个候选里最好的一个</b>,而 <b>0 个</b>达到 30% CAGR 的目标。'
                 f'最好成绩 {summary["cagr"]*100:+.2f}%,同期无脑等权持有 {(bench_tot)*100:+.2f}% —— '
                 f'<b>跑输 {abs((bench_tot - summary["cagr"]))*100:.1f}pp</b>。</li>'
                 f'<li><b>样本只有 {len(trades)} 笔</b>,敞口 {summary["exposure"]*100:.0f}%。'
                 f'运气对照(同止损止盈、同笔数随机入场 200 次)的 p90 是 '
                 f'{summary["ctl_p90"]*100:+.1f}% —— 本配置落在随机分布的这个位置附近,'
                 f'证据很弱。</li>'
                 '<li><b>卡片右侧显示出场之后的价格</b>(供复盘),那是展示,不参与信号。'
                 '信号日清单只用当日及之前的数据。</li>'
                 '</ul></div>')

    # ---- 摘要 ----
    parts.append('<h2>摘要</h2><div class="kv">')
    for lab, val, cls in (
        ("CAGR(12个月)", f'{summary["cagr"]*100:+.2f}%', "pos" if summary["cagr"] > 0 else "neg"),
        ("总收益", f'{tot*100:+.2f}%', "pos" if tot > 0 else "neg"),
        ("无脑等权持有", f'{bench_tot*100:+.2f}%', "pos" if bench_tot > 0 else "neg"),
        ("超额", f'{(tot-bench_tot)*100:+.2f}pp', "pos" if tot > bench_tot else "neg"),
        ("最大回撤", f'{summary["max_dd"]*100:.2f}%', "neg"),
        ("胜率", f'{summary["win"]*100:.0f}%', ""),
        ("E[R]", f'{summary["e_r"]:+.2f}', "pos" if summary["e_r"] > 0 else "neg"),
        ("平均敞口", f'{summary["exposure"]*100:.0f}%', ""),
    ):
        parts.append(f'<div><b>{e(lab)}</b><span class="{cls}">{e(val)}</span></div>')
    parts.append('</div>')

    parts.append('<h2>权益曲线 vs 基准</h2><div class="card">')
    parts.append(equity_svg(eq, bench))
    parts.append('</div>')

    parts.append('<h2>配置</h2><div class="card"><table class="conds">')
    parts.append('<tr><th>项</th><th>值</th></tr>')
    for k, v in cfg.items():
        parts.append(f'<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>')
    for k in ("max_hold", "risk_per_trade", "max_positions",
              "max_volume_participation", "intraday_tiebreak"):
        parts.append(f'<tr><td>{e(k)}</td><td>{e(str(summary["preset"].get(k)))}</td></tr>')
    parts.append('</table></div>')

    # ---- 交易总表 ----
    parts.append('<h2>全部交易</h2><div class="card"><table><thead><tr>')
    for h in ("#", "标的", "信号日", "入场日", "入场价", "止损", "止盈", "股数",
              "出场日", "出场价", "出场原因", "持有", "净收益"):
        parts.append(f'<th>{e(h)}</th>')
    parts.append('</tr></thead><tbody>')
    for i, (_, t) in enumerate(trades.iterrows(), 1):
        cls = "pos" if t["net_pnl"] >= 0 else "neg"
        parts.append(
            f'<tr><td>{i}</td><td><a href="#t{i}">{e(str(t["symbol"]))}</a></td>'
            f'<td>{e(str(pd.Timestamp(summary["sig_dates"][t["symbol"]]).date()))}</td>'
            f'<td>{e(str(pd.Timestamp(t["entry_date"]).date()))}</td><td>{t["entry_price"]:.2f}</td>'
            f'<td class="neg">{t["entry_price"]*(1-t["sl_width"]):.2f}</td>'
            f'<td class="pos">{t["entry_price"]*(1+t["tp_width"]):.2f}</td>'
            f'<td>{int(t["size"])}</td>'
            f'<td>{e(str(pd.Timestamp(t["exit_date"]).date()))}</td><td>{t["exit_price"]:.2f}</td>'
            f'<td>{e(str(t["exit_reason"]))}</td><td>{int(t["hold_days"])}d</td>'
            f'<td class="{cls}">{t["net_return"]*100:+.2f}%</td></tr>')
    parts.append('</tbody></table></div>')

    # ---- 逐笔卡片 ----
    parts.append('<h2>逐笔明细</h2>')
    for i, (_, t) in enumerate(trades.iterrows(), 1):
        sym = str(t["symbol"])
        sig_d = pd.Timestamp(summary["sig_dates"][sym])
        sub = ind[ind["symbol"] == sym].sort_values("date").reset_index(drop=True)
        jj = sub.index[sub["date"] == t["entry_date"]]
        if len(jj) == 0:
            continue
        j = int(jj[0])
        # ⚠️ 窗口右端必须锚在**出场日**,不能锚在入场日 + 固定根数。
        # 早先写 `entry + POST_EXIT_BARS`,于是持有超过 8 根的仓位(含 max_hold=18
        # 那些,往往是最赚的)出场日落在窗口外 → 出场标记静默消失,
        # 9 张卡里有 5 张看不到出场。DOM 查询才发现 —— 静态 HTML 文本里看不出缺失。
        xj = sub.index[sub["date"] == t["exit_date"]]
        k_exit = int(xj[0]) if len(xj) else j
        win = sub.iloc[max(0, j - PRE_SIG_BARS): k_exit + POST_EXIT_BARS + 1].copy()
        cls = "win" if t["net_pnl"] >= 0 else "loss"
        parts.append(f'<div class="card" id="t{i}">')
        parts.append('<div class="chead">'
                     f'<span class="sym">{i}. {e(sym)}</span>'
                     f'<span class="tag {cls}">净收益 {t["net_return"]*100:+.2f}%</span>'
                     f'<span class="tag">出场 {e(str(t["exit_reason"]))}</span>'
                     f'<span class="tag">持有 {int(t["hold_days"])} 交易日</span>'
                     f'<span class="tag">信号日 {e(str(sig_d.date()))}</span>'
                     f'<span class="tag">入场 {e(str(pd.Timestamp(t["entry_date"]).date()))} @ {t["entry_price"]:.2f}</span>'
                     f'<span class="tag">出场 {e(str(pd.Timestamp(t["exit_date"]).date()))} @ {t["exit_price"]:.2f}</span>'
                     '</div>')
        parts.append(candles_svg(win, t, sig_d, sym))
        parts.append('<div class="chead" style="margin-top:8px">'
                     f'<span class="tag">信号日 atr_pct {t["atr_pct"]*100:.2f}%</span>'
                     f'<span class="tag neg">止损宽度 {t["sl_width"]*100:.2f}%</span>'
                     f'<span class="tag pos">止盈宽度 {t["tp_width"]*100:.2f}%</span>'
                     f'<span class="tag">计划风险 ${t["risk_amount"]:,.0f}</span>'
                     f'<span class="tag">毛盈亏 ${t["gross_pnl"]:,.0f}</span>'
                     f'<span class="tag">费用 ${t["fees"]:,.0f}</span>'
                     f'<span class="tag">净盈亏 <b>${t["net_pnl"]:,.0f}</b></span>'
                     '</div>')
        parts.append('<table class="conds"><tr><th>#</th><th>条件</th>'
                     '<th>信号日实际值</th><th>阈值</th><th></th></tr>')
        for code, name, val, thr, ok in condition_rows(ind, sym, sig_d, cfg):
            parts.append(f'<tr><td>{e(code)}</td><td>{e(name)}</td><td>{e(val)}</td>'
                         f'<td>{e(thr)}</td><td class="{"ok" if ok else "no"}">'
                         f'{"✓" if ok else "✗"}</td></tr>')
        parts.append('</table></div>')

    parts.append('</div></body></html>')
    return "".join(parts)


# ---------------------------------------------------------------- 入口

def run(lo: str, hi: str, cfg: dict) -> tuple:
    p = panel_mod.load_full_panel()
    ind = indicators.compute_all(p)
    reg = regime.compute_regime(ind)

    kw = dict(signals.V21_SIGNAL_PARAMS)
    for k in ("min_dist_high252", "max_stoch_k", "max_rsi14",
              "max_bb_pctb", "min_adx14", "min_xs_mom120"):
        if cfg.get(k) is not None:
            kw[k] = cfg[k]
    sig = signals.select_entries(ind, start_date=lo, end_date=hi, regime_df=reg, **kw)

    preset = {
        "atr_sl_mult": cfg["sl_mult"], "atr_tp_mult": cfg["tp_mult"],
        "max_hold": 18, "max_positions": 1, "risk_per_trade": 0.01,
        "max_volume_participation": 0.10, "intraday_tiebreak": "sl_first",
        "initial_cash": 100_000.0,
    }
    res = broker.simulate(preset, sig, ind, start=lo, end=hi, cost_bps_roundtrip=5.0)

    # 每笔对应的信号日:入场日是信号日之后的第一个交易日,取 ≤ 入场日 的最后一个信号
    sig_dates = {}
    for _, r in res["trades"].iterrows():
        c = sig[(sig["symbol"] == r["symbol"]) & (sig["date"] < r["entry_date"])]
        sig_dates[r["symbol"]] = c["date"].max() if len(c) else r["entry_date"]

    eq = res["equity"]
    idx = reg.set_index("date")["ew_index"]
    b = idx[(idx.index >= pd.Timestamp(eq["date"].iloc[0])) &
            (idx.index <= pd.Timestamp(eq["date"].iloc[-1]))]
    bench = (b / b.iloc[0]).reindex(eq["date"]).ffill().bfill()

    ts = metrics.tail_stats(res["trades"])
    summary = {
        "preset_name": "us_uptrend_pullback_12m_best", "preset": preset,
        "lo": lo, "hi": hi,
        "cagr": metrics.cagr(eq["equity"], eq["date"]),
        "max_dd": float(res["metrics"]["max_dd"]),
        "win": float(res["metrics"]["win_rate"]) if len(res["trades"]) else float("nan"),
        "e_r": float(ts["expectancy_r"]), "exposure": float(res["metrics"]["avg_exposure"]),
        "sig_dates": sig_dates, "ctl_p90": 0.059,
    }
    return res, eq, ind, bench, summary


def selfcheck(doc: str, n_trades: int) -> list[str]:
    """渲染后自检:**元素缺失**是最难发现的失败模式。

    2026-10-14 两次实测教训,都是纯文本检查看不见、DOM 查询才暴露:
      1. 止损止盈线因判错列名(`sl_price` 不在 TRADE_COLS 里)整批跳过 —— 图是空的;
      2. 窗口右端锚在入场日+8,持有超过 8 根的仓位出场标记整批消失。
    两者都**不产生任何几何错误**(没有元素就没有 NaN、没有越界),所以"检查
    NaN/越界"那类测试全绿。只有数元素个数才能抓到。
    """
    import re
    problems: list[str] = []
    for name, pat in (
        ("K线卡", r'class="kchart"'),
        ("止损线", r'class="sl"'),
        ("止盈线", r'class="tp"'),
        ("入场标记", r'class="mk-entry"'),
        ("出场标记", r'class="mk-exit'),
    ):
        cnt = len(re.findall(pat, doc))
        if cnt != n_trades:
            problems.append(f"{name} {cnt} 个,应为 {n_trades} 个(每笔一个)")
    bad = re.findall(r'(?:x|y|x1|y1|x2|y2|width|height)="[^"]*(?:nan|inf)', doc, re.I)
    if bad:
        problems.append(f"{len(bad)} 个坐标含 NaN/inf")
    if re.search(r'(?:src|href)="https?://', doc, re.I) or "<script" in doc.lower():
        problems.append("含外部引用或 <script> —— 报告不再自包含")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", default=WIN_LO)
    ap.add_argument("--hi", default=WIN_HI)
    ap.add_argument("--out", default=str(OUT_HTML))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    res, eq, ind, bench, summary = run(a.lo, a.hi, BEST_CFG)
    n = len(res["trades"])
    doc = render_html(res["trades"], eq, ind, BEST_CFG, bench, summary)

    problems = selfcheck(doc, n)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"成交 {n} 笔  CAGR {summary['cagr']*100:+.2f}%")
    print(f"已写入 {out}  ({out.stat().st_size/1024:.0f} KB,自包含、无外部依赖)")
    if problems:
        for p in problems:
            print(f"  [自检失败] {p}")
        return 1
    print(f"  [自检通过] {n} 张卡片各有 K线/止损/止盈/入场/出场 5 个元素,无 NaN,无外部依赖")
    return 0


if __name__ == "__main__":
    sys.exit(main())
