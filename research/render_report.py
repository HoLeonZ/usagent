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

from core.preset_contract import describe_engine_spec, resolve
from research import (broker, indicators, metrics, panel as panel_mod, presets,
                      regime, signals)

logger = logging.getLogger("render")

# 默认渲染哪个 preset —— **从 presets.py 读,不在这里硬编码**。
# 早先这里写死了一份 BEST_CFG(mp=1/risk=1%,CAGR +7.95%),而真正达到
# "12 个月 CAGR >= 30%" 目标的注册 preset 是 mp=2/risk=5%。⚠️ 该目标在 2026-10-15
# 修正记账/口径后**已不再达成**(CAGR +9.51%),报告里的数字一律现算,不写死在这里。
# 报告展示的不是目标 preset —— 这个不一致由 presets.py 做单一事实来源来消除。
DEFAULT_PRESET = "us_uptrend_pullback_12m_target"
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

    # 阈值必须与**引擎实际用的**一致:引擎是 select_entries(**preset["signal"]),
    # 而 select_entries 的缺省值就是 V21_SIGNAL_PARAMS ⇒ 合并两者即可。
    # (此前直接用全局常量:只要某个 preset 改了 A~E 任一阈值,卡片上的「阈值」列
    #  与 ✓/✗ 就会与引擎分叉 —— 与本轮修掉的 7 键白名单是同一类。)
    sp = {**signals.V21_SIGNAL_PARAMS, **cfg}
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
                symbol: str, uid: str = "0") -> str:
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

    # ⚠️ 整张卡只用**一套价**:as-traded(raw_*)。入场/出场/止损/止盈价都是 raw 空间的
    # (broker 的执行与记账都在 raw 空间),蜡烛与均线若画复权价,两套轴会在同一张图里
    # 错位 —— 2026-10-15 复核实测最大错位 14.8%,表现为「图上出现从未发生的止损击穿」。
    lo = float(min(win["raw_low"].min(), sl_px))
    hi = float(max(win["raw_high"].max(), tp_px))
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad
    y = _scale(win["raw_close"].to_numpy(float), lo, hi, 0, PRICE_H)
    yo = _scale(win["raw_open"].to_numpy(float), lo, hi, 0, PRICE_H)
    yh = _scale(win["raw_high"].to_numpy(float), lo, hi, 0, PRICE_H)
    yl = _scale(win["raw_low"].to_numpy(float), lo, hi, 0, PRICE_H)
    xs = [PAD_L + i * step + step / 2 for i in range(n)]

    # 成交量
    vmax = float(win["volume"].max()) or 1.0
    vy0 = PRICE_H + GAP
    vol = (win["volume"].to_numpy(float) / vmax * VOL_H)

    p = [f'<svg viewBox="0 0 {SVG_W} {total_h}" class="kchart" '
         f'preserveAspectRatio="none" role="img" aria-label="{html.escape(symbol)} K线">',
         # MA 折线必须被裁到价格面板内。不裁的话,预热期 MA 值可能落在价格带外,
         # 折线会被画进下方的成交量面板里 —— 看着像"成交量的均线",会误导。
         f'<defs><clipPath id="pc-{uid}">'
         f'<rect x="0" y="0" width="{SVG_W}" height="{PRICE_H}"/></clipPath></defs>']

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
        cls = "up" if win["raw_close"].iloc[i] >= win["raw_open"].iloc[i] else "dn"
        x = xs[i]
        p.append(f'<line x1="{x:.1f}" y1="{yh[i]:.1f}" x2="{x:.1f}" y2="{yl[i]:.1f}" class="wick {cls}"/>')
        top, bot = min(yo[i], y[i]), max(yo[i], y[i])
        p.append(f'<rect x="{x-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                 f'height="{max(bot-top,1):.1f}" class="body {cls}"/>')

    # MA20 / MA60
    # 均线也要在同一条轴(raw)上,否则「价格穿过均线」的视觉是假的。
    for col, cls in (("raw_ma20", "ma20"), ("raw_ma60", "ma60")):
        pts, started = [], False
        for i in range(n):
            v = win[col].iloc[i]
            if not np.isfinite(v):
                continue
            yy = _scale([float(v)], lo, hi, 0, PRICE_H)[0]
            pts.append(f"{xs[i]:.1f},{yy:.1f}")
            started = True
        if started and len(pts) > 1:
            p.append(f'<polyline points="{" ".join(pts)}" class="{cls}" '
                     f'clip-path="url(#pc-{uid})"/>')

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


def _target_note(cagr: float, bench_cagr: float) -> str:
    """警告框第 2 条按**当次算出的** cagr / 基准生成 —— 一个字都不许写死。

    ⚠️ 这条路走过两次、两次翻车,教训同一条:
      * 2026-10-14:写死「0 个达到 30% 的目标」,换 preset 后成了假陈述;
      * 2026-10-15:≥30% 分支写死「15 个杠杆配置里有 5 个达标」,并把基准门槛写死成
        字面量 0.156(同一次渲染算出的 bench_cagr 其实是另一个数)。修正引擎后同一根
        杠杆旋钮 1%/2%/5%/25% → +3.23%/+6.00%/+9.51%/+11.87%,怎么推都到不了 30%
        —— 那句达标叙述已不可复现。
    两个分支现在都由参数派生;要复核自己跑 run_baseline。
    """
    if cagr >= 0.30:
        return ('「达到 30%」本身不作为有效性证据:同一根杠杆旋钮换个取值或换个窗口就能'
                '得到完全不同的数,而风险归一化后的 E[R] 不随杠杆变 —— 变的是倍数,不是'
                '信号质量。判据见本框其余各条;复核:python -m research.run_baseline')
    _rel = ('而且它连无脑持有都没跑赢' if cagr < bench_cagr else '它跑赢了同期等权持有')
    return (f'这个 preset 在<b>本窗口</b>没有达到 30% CAGR 的目标'
            f'({cagr*100:+.2f}% &lt; 30%),{_rel}(基准 {bench_cagr*100:+.2f}%)。'
            f'30% 是目标,不是有效性判据 —— 判据见本框其余各条。')


def render_html(trades: pd.DataFrame, eq: pd.DataFrame, ind: pd.DataFrame,
                cfg: dict, bench: pd.Series, summary: dict) -> str:
    e = html.escape
    tot = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
    bench_tot = float(bench.iloc[-1] / bench.iloc[0] - 1) if bench is not None and len(bench) else float("nan")
    # ⚠️ 必须同口径比较。警告框原来拿策略的 CAGR 去比基准的**总收益**,
    # 一年期两者接近但不相等 —— 那是拿两个不同的量做差。
    _bdays = (pd.Timestamp(bench.index[-1]) - pd.Timestamp(bench.index[0])).days if len(bench) else 0
    bench_cagr = ((1 + bench_tot) ** (365.25 / _bdays) - 1) if _bdays > 0 else float("nan")

    parts = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             f'<meta name="viewport" content="width=device-width,initial-scale=1">',
             f'<title>回测明细 · {e(summary["preset_name"])}</title>',
             f'<style>{CSS}</style></head><body><div class="wrap">']
    parts.append(f'<h1>回测明细 · {e(summary["preset_name"])}</h1>')
    _spec = summary["preset"]
    _slip = (f'含执行滑点(f(ATR,参与率),scale={_spec["atr_slip_scale"]:g})'
             if _spec["atr_slip_scale"] > 0 else '零滑点')
    parts.append(f'<div class="sub">窗口 {e(summary["lo"])} → {e(summary["hi"])} ｜ '
                 f'成交 {len(trades)} 笔 ｜ {_slip},成本双边 '
                 f'{_spec["cost_bps_roundtrip"]:g}bp</div>')
    # 口径自报（审计 E5 及其同类）：与 run_baseline / --json 用的是**同一个函数**，
    # 所以两个入口不可能再各说一套。
    _items = " ・ ".join(f"{e(lab)} <b>{e(val)}</b>"
                         for lab, val in describe_engine_spec(_spec))
    parts.append(f'<div class="sub" style="margin-top:-10px">生效口径：{_items}</div>')
    # 价格口径必须披露。2026-10-15 起是**双轨**(§3(b)):指标用复权价、成交与记账用
    # as-traded 价 —— 旧版这段写的是「回测用复权价是正确的」,在新语义下正好相反。
    parts.append('<div class="sub" style="margin-top:-10px;color:var(--warn)">'
                 '⚠️ 价格口径(双轨,§3(b)):数学指标与条件清单用 <b>分红回补价(total-return)</b>;'
                 '而<b>成交价、止损止盈触发、跳空判定、逐日市值与现金记账一律用当日的 '
                 'as-traded 价</b>;E 条件的成交额代理量 <code>amount60</code> 同样是 '
                 'as-traded 口径(volume × as-traded close)——它比的是绝对美元阈值,'
                 '用复权价会让「能不能进池」取决于未来的分红。K 线卡上的一切(蜡烛、均线、止损止盈线、买卖标记)都画在 '
                 '<b>as-traded</b> 这一条轴上,与真实成交同源。'
                 '现金分红在持有穿过除息日时<b>显式入账</b>并计入该笔净盈亏(§3(c))。'
                 '照这份报告下单时,请以行情软件的实际报价为准。'
                 '</div>')

    # ---- 诚实声明(不折叠)----
    _diff = summary["cagr"] - bench_cagr
    _over = "跑赢" if _diff >= 0 else "跑输"
    if summary.get("ctl_runs"):
        _ctl_line = (
            f'运气对照(同止损止盈、同笔数随机入场 {summary["ctl_runs"]} 次):'
            f'本配置位于随机分布的 <b>{summary["ctl_pct"]:.1f} 百分位</b>,'
            f'即约 {100 - summary["ctl_pct"]:.0f}% 的随机策略做得和它一样好或更好'
            f'(随机 p90 {summary["ctl_p90"]*100:+.1f}%、最大 {summary["ctl_max"]*100:+.1f}%)。'
            + (f'⚠️ 其中 {summary["ctl_failed"]} 次重抽因量能上限未能成交被丢弃。'
               if summary.get("ctl_failed") else ''))
    else:
        _ctl_line = ("运气对照本次未跑(control_runs=0)—— 这份报告因此**不含**"
                     "「位于随机分布第几百分位」那条判词。")
    parts.append('<div class="warn"><h3>读这份报告之前必须知道的五件事</h3><ul>'
                 f'<li>本 preset 最近 12 个月 <b>CAGR {summary["cagr"]*100:+.2f}%</b>,'
                 f'同期无脑等权持有 CAGR {bench_cagr*100:+.2f}% —— '
                 f'<b>{_over} {abs(_diff)*100:.1f}pp</b>(同口径比较)。'
                 f'它是 {summary["n_trials"]} 次试验里选出来的。</li>'
                 f'<li><b>{_target_note(summary["cagr"], bench_cagr)}</b></li>'
                 f'<li><b>样本只有 {len(trades)} 笔</b>,敞口 {summary["exposure"]*100:.0f}%。'
                 f'{_ctl_line}'
                 f'按 §5,{summary["n_trials"]} 次试验的选择偏差折算后门槛还要更高。</li>'
                 '<li><b>卡片右侧显示出场之后的价格</b>(供复盘),那是展示,不参与信号。'
                 '信号日清单只用当日及之前的数据 —— 已由截断重算独立验证。</li>'
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
        parts.append(f'<tr><td>signal.{e(str(k))}</td><td>{e(str(v))}</td></tr>')
    # 执行参数**全部**由已解析的 spec 渲染,不手写键名清单:手写清单会让新增的执行键
    # 从报告里静默消失。2026-10-15 复核实测:旧版只印了 5 个键,比引擎实际读的少 9 个。
    for k in sorted(summary["preset"]):
        parts.append(f'<tr><td>{e(k)}</td><td>'
                     f'{e(str(summary["preset"][k]))}</td></tr>')
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
        parts.append(candles_svg(win, t, sig_d, sym, uid=str(i)))
        parts.append('<div class="chead" style="margin-top:8px">'
                     f'<span class="tag">信号日 atr_pct {t["atr_pct"]*100:.2f}%</span>'
                     f'<span class="tag neg">止损宽度 {t["sl_width"]*100:.2f}%</span>'
                     f'<span class="tag pos">止盈宽度 {t["tp_width"]*100:.2f}%</span>'
                     f'<span class="tag">计划风险 ${t["risk_amount"]:,.0f}</span>'
                     f'<span class="tag">毛盈亏 ${t["gross_pnl"]:,.0f}</span>'
                     f'<span class="tag">费用 ${t["fees"]:,.0f}</span>'
                     # 分红必须单独列:净 = 毛 + 分红 − 费用(§3(c))。不列出来,持仓穿过
                     # 除息日的那几笔三个数就对不上(实测 VNOM 差 $1,662)。
                     + (f'<span class="tag pos">分红 ${t["dividends"]:,.0f}</span>'
                        if float(t.get("dividends") or 0) != 0 else '')
                     + f'<span class="tag">净盈亏 <b>${t["net_pnl"]:,.0f}</b></span>'
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

def run(lo: str, hi: str, preset_name: str = DEFAULT_PRESET,
        control_runs: int = 200) -> tuple:
    P = presets.get_preset(preset_name)
    if P.get("family"):
        # 族型 preset 的逐笔 HTML 报告尚未实现:报告里的「信号日条件清单」是 uptrend-pullback
        # 的 A~F 表,对族**没有意义**,照印会误导。宁可明确失败,不静默印一张无关的表。
        raise NotImplementedError(
            f"preset {preset_name!r} 走的是 families.py 的族({P['family']}),"
            f"逐笔 HTML 报告尚未支持;数字请用 research.run_baseline / research.family_search")
    p = panel_mod.load_panel_for_preset(P)
    ind = indicators.compute_all(p)
    # 卡片上的均线也要落在 **as-traded** 轴上(与蜡烛、止损止盈线、买卖标记同轴)——
    # 见 candles_svg 的注释:混轴会让图上出现从未发生的止损击穿。
    ind = ind.assign(
        raw_ma20=ind.groupby("symbol")["raw_close"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()),
        raw_ma60=ind.groupby("symbol")["raw_close"].transform(
            lambda s: s.rolling(60, min_periods=60).mean()),
    )
    # ⚠️ 必须把 preset 的 regime 参数传进去:此前这里用函数默认值做择时,而 CLI 用
    #    preset 值 —— preset 的 regime 一改,报告与 CLI 就分叉(审计实测差 13 笔成交)。
    reg = regime.compute_regime(ind, **(P.get("regime") or {}))

    # 信号:preset 里声明的 signal 字典直接透传(含可选过滤层)。
    # preset 是**单一事实来源** —— 不在这里重新拼参数,否则报告展示的配置
    # 会与 presets.py 里注册的那个悄悄分叉。
    sig = signals.select_entries(ind, start_date=lo, end_date=hi,
                                regime_df=reg if P.get("regime") else None,
                                **P["signal"])

    # 执行:**整包透传**。此前这里是一份 7 键白名单,对 broker 实际会读的键静默丢弃
    # (position_fraction / sl_pct / tp_pct / atr_slip_scale / trailing_sl),
    # 实测同一个 preset 在报告与 CLI 上跑出差 45.5pp 的两条曲线。
    # 现在唯一的展开入口是契约的 resolve(),CLI 与报告读到的是同一份参数。
    preset = resolve(P)
    res = broker.simulate(preset, sig, ind, start=lo, end=hi)

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
        "preset_name": preset_name, "preset": preset, "signal": P["signal"],
        "lo": lo, "hi": hi,
        "cagr": metrics.cagr(eq["equity"], eq["date"]),
        "max_dd": float(res["metrics"]["max_dd"]),
        "win": float(res["metrics"]["win_rate"]) if len(res["trades"]) else float("nan"),
        "e_r": float(ts["expectancy_r"]), "exposure": float(res["metrics"]["avg_exposure"]),
        "sig_dates": sig_dates, "n_trials": P.get("n_comparisons", 106),
    }

    # 运气对照:**真跑**,不硬编码。同止损止盈、同交易笔数的随机入场 200 次,
    # 再算本配置落在该分布的第几百分位。渲染一份报告多花 ~1 分钟,换的是
    # "报告里的每个数字都可复现"。
    from research import search_12m as _S
    # 对照配置:从**已解析的 broker preset** 反推,保证对照与策略同杠杆同止损止盈。
    # (search_12m.make_preset 读的是 sl_mult/tp_mult 这套键名,broker preset 用的是
    #  atr_sl_mult/atr_tp_mult —— 名字不同,值必须一致,否则对照就不成立。)
    cfg = {
        "sl_mult": preset["atr_sl_mult"], "tp_mult": preset["atr_tp_mult"],
        "max_hold": preset["max_hold"], "max_positions": preset["max_positions"],
        "risk_per_trade": preset["risk_per_trade"],
    }
    if control_runs > 0:
        # 必须把 reg 传进去:对照要在**策略自己的机会集**里抽(含 regime 门),
        # 否则比的是两个不同的池子(见 search_12m._control_pool)。
        ctl = _S.random_control(ind, cfg, len(res["trades"]),
                                n_runs=control_runs, lo=lo, hi=hi, seed=0, reg=reg)
        summary.update({
            "ctl_runs": ctl["n_runs"], "ctl_failed": ctl["n_failed"],
            "ctl_p90": ctl["p90"], "ctl_max": ctl["max"],
            "ctl_pct": float((ctl["cagrs"] < summary["cagr"]).mean() * 100.0),
        })
    else:
        # 跳过运气对照(测试与快速迭代用)。渲染层据此不出那条判词 ——
        # **不能**留着 NaN 让它渲染成 "nan 百分位"。
        summary.update({"ctl_runs": 0, "ctl_failed": 0, "ctl_p90": float("nan"),
                        "ctl_max": float("nan"), "ctl_pct": float("nan")})
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
    ap.add_argument("--preset", default=DEFAULT_PRESET)
    ap.add_argument("--out", default=str(OUT_HTML))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    res, eq, ind, bench, summary = run(a.lo, a.hi, a.preset)
    n = len(res["trades"])
    doc = render_html(res["trades"], eq, ind, summary["signal"], bench, summary)

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
