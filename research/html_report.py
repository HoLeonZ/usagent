"""HTML 回测报告的**公共能力** —— **变的只是策略，不变的是呈现**。

这一层只做一件事：把「一条策略在某个窗口上的逐笔明细」画成自包含的 HTML
（K 线卡片 + 入场/出场标记 + 止损止盈线 + 信号日的条件清单 + 权益曲线 + 口径头）。
它**不知道**自己画的是哪条策略：

    render_html(trades, eq, ind, cfg, bench, summary, explain=...)

唯一与策略有关的入口是 `explain` 钩子（信号日的条件清单由策略自己给）：

  * `condition_rows`      —— signal 型 preset 的 A~F 条件表（实际值 vs 阈值）；
  * `family_explain_rows` —— 族型策略：从**掩码源码**里读出它实际读了哪些列
    （AST 扫 `ind["X"]`），再把信号日那一行的值与参数一起列出来 —— 对任何新族都成立，
    不需要人手维护一张"这个族看哪些指标"的表。

自包含（不引外链、不执行 <script>）由 `selfcheck` 强制：元素个数、NaN 坐标、
外部引用三项 —— 这三项都是"纯文本检查看不见、只有数元素才暴露"的失败模式
（2026-10-14 两次实测教训，见 `selfcheck` 的 docstring）。
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 口径头的**唯一**来源：与 run_baseline / --json 用同一个函数 ⇒ 两个入口不可能各说一套
from core.preset_contract import describe_engine_spec

from research import signals          # condition_rows 要用 V21 的阈值做对照

PRE_SIG_BARS = 60          # 卡片左侧多看多少根 K 线
POST_EXIT_BARS = 8         # 卡片右侧多看多少根


class RenderError(RuntimeError):
    pass



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




def inline_md(text: str) -> str:
    """**先转义、再做最小行内 markdown** —— 只认 `**粗体**` 与 `` `代码` ``。

    为什么必须转义（2026-10-17 修）：这一行插的是**策略自己声明的 VERDICT** 或调用方给的
    `--note`，此前是裸插进 HTML；而本文件其余文案一律走 `html.escape`。一个 `<` 就能把
    文档弄坏（或更糟）。为什么不直接上 markdown 库：报告必须**自包含、无外部依赖**，
    而这里只需要两种行内标记。
    """
    out = html.escape(text or "")
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out, flags=re.S)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


# 出场原因的**中文图例**：引擎里是英文裸码（broker.EXIT_REASONS），报告此前只印码，
# 读者得去翻源码才知道 sl_gap 与 sl 的区别（审计实测「图例」在报告里出现 0 次）。
# 语义按 broker.py 的**赋值点**写，不是猜的：
#   sl/tp       —— 盘中触及止损/止盈价，按那个价成交（broker.py 的 close_position(..., sl_fill_at)）
#   sl_gap/tp_gap —— 开盘就跳空过线，按**开盘价**成交（close_position(pos, day, o, ...)）
#   max_hold    —— 持有到 spec["max_hold"] 上限，按当日收盘价平
#   eod         —— 窗口结束当天的收盘平仓
#   regime      —— regime 门翻掉，按开盘价平
EXIT_LEGEND = {
    "sl": "盘中触及止损价，按止损价成交",
    "sl_gap": "跳空跌破止损 —— 按**开盘价**成交（比止损价更差）",
    "tp": "盘中触及止盈价，按止盈价成交",
    "tp_gap": "跳空高过止盈 —— 按**开盘价**成交（比止盈价更好）",
    "max_hold": "持有到最长天数上限，按当日收盘平",
    "eod": "窗口结束，按最后一日收盘平",
    "regime": "regime 门翻转，按开盘价平",
}


def exit_reason_rows(trades: pd.DataFrame) -> list[dict]:
    """按出场原因分档 —— 读者要先知道"这一类出场整体如何"，才有必要往下翻几十张卡。

    为什么在这里算而不是复用 `broker.summarize` 的三个 count：那三个是**粗分组**
    （tp+tp_gap / sl+sl_gap / max_hold），回答不了"跳空止损这一小撮到底亏多少"。
    这里是同一列（exit_reason）的另一种分组，只做呈现，不改任何判据。
    """
    if trades is None or trades.empty or "exit_reason" not in trades.columns:
        return []
    rows = []
    for reason, g in trades.groupby("exit_reason", sort=False):
        rows.append({"reason": str(reason), "n": int(len(g)),
                     "share": float(len(g) / len(trades)),
                     "mean_net_return": float(g["net_return"].mean()),
                     "sum_net_pnl": float(g["net_pnl"].sum()),
                     "win_rate": float((g["net_pnl"] > 0).mean())})
    return sorted(rows, key=lambda d: -d["n"])


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
/* 逐笔明细的"只看某一类出场"（纯 CSS：报告不许带脚本标签 —— selfcheck 会直接扫这个字面
   量，所以这里连注释里都不能写它）。
   用 radio 而不是 checkbox：多选要表达"任一命中即显示"的并集，纯 CSS 表达不了；
   而单选的语义正好是读者要的那件事 —— "只看止损那批"。默认"全部"。
   用 :has() 是为了不把 radio 硬塞成卡片的兄弟节点（那会逼着改整个文档结构）。
   老浏览器不支持时**静默降级为不过滤** —— 报告照旧可读，只是少一个筛选。 */
.fbar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:10px 0 4px}
.fbar input{position:absolute;opacity:0;width:1px;height:1px}
.fbar label{font-size:12.5px;color:var(--fg);border:1px solid var(--line);border-radius:999px;
            padding:2px 10px;background:var(--card);cursor:pointer}
.fbar label:hover{border-color:var(--acc)}
.fbar input:focus-visible + label{outline:2px solid var(--acc);outline-offset:1px}
.fbar input:checked + label{border-color:var(--acc);color:var(--acc);background:#131a26}
body:has(#xf-sl:checked) .card:not([data-reason="sl"]),
body:has(#xf-sl_gap:checked) .card:not([data-reason="sl_gap"]),
body:has(#xf-tp:checked) .card:not([data-reason="tp"]),
body:has(#xf-tp_gap:checked) .card:not([data-reason="tp_gap"]),
body:has(#xf-max_hold:checked) .card:not([data-reason="max_hold"]),
body:has(#xf-eod:checked) .card:not([data-reason="eod"]),
body:has(#xf-regime:checked) .card:not([data-reason="regime"]){display:none}
.reasonbox{margin:12px 0 6px}
.rsum{font-size:12.5px}
.rsum td,.rsum th{padding:4px 8px}
.rleg{font-size:12px;color:var(--dim);margin:8px 0 0;line-height:1.7}
.rleg code{background:transparent;padding:0;color:var(--fg)}
.conds{font-size:12px;border-collapse:collapse;margin-top:10px;width:100%}
.conds td,.conds th{padding:3px 7px;border-bottom:1px solid #1e222b}
.ok{color:var(--up)}.no{color:var(--dn)}
.nodata{color:var(--dim);padding:16px}
code{background:#1b2029;padding:1px 5px;border-radius:4px;font-size:12px}
/* 锚点落点必须让开**自己的**那条粘性目录。实测（2026-10-18 审计，两名核验者独立复现）：
   .toc 是 position:sticky;top:0、渲染高 40.4px，而这里原先写 12px ⇒ 点任意一个目录项，
   章节标题都被目录条**整条盖住**（h2 的文字行落在目录条下面）。52px = 40.4 + 余量。 */
h2{scroll-margin-top:52px}
/* 逐笔卡的锚点（#t{i}）同样要让开目录条 —— 否则从总表跳到某一笔，卡片头也在目录条下。 */
.card[id^="t"]{scroll-margin-top:52px}
/* 粘性小节导航：这份报告实测 87k px 高（90+ 屏），目录放在最上面等于走到中段就够不着了。
   sticky 让它一直在顶上；打印时由 @media print 隐藏。 */
/* 屏幕端也要给链接上色：报告里 16 个「跳到该笔」的标的链接此前是 UA 默认蓝 #0000EE，
   在 --card #161a22 上只有约 1.9:1（打印块里那条 a{color:inherit} 只作用于打印）。 */
a{color:var(--acc)}
.toc{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0 4px;position:sticky;top:0;z-index:5;
     background:var(--bg);padding:8px 0 6px;border-bottom:1px solid var(--line)}
.toc a{font-size:12.5px;color:var(--fg);text-decoration:none;border:1px solid var(--line);
       border-radius:999px;padding:2px 10px;background:var(--card)}
.toc a:hover{border-color:var(--acc);color:var(--acc)}
.totop{font-size:12px;color:var(--dim);text-decoration:none;display:inline-block;margin:2px 0 0 8px}
.totop:hover{color:var(--acc)}
/* 逐笔明细很长（实测 91 张卡 / 108 屏）—— 走到底之后回得去 */
.totopcard{position:fixed;right:16px;bottom:16px;background:var(--card);border:1px solid var(--line);
           border-radius:999px;padding:7px 13px;font-size:12.5px;color:var(--fg);text-decoration:none}
.totopcard:hover{border-color:var(--acc);color:var(--acc)}
/* ---- 打印 / 另存 PDF：审计实测整份报告没有任何 @media，深色底上的浅色正文打到白纸上
   （勾了「背景图形」则整页铺黑）。这一段只改配色与分页，不动任何内容与结构。---- */
@media print{
  :root{--bg:#fff;--fg:#111;--dim:#555;--line:#bbb;--up:#0a7a4a;--dn:#b3261e;--acc:#1c4fa1;
        --sl:#b3261e;--tp:#0a7a4a;--warn:#8a5a00;--card:#fff}
  body{background:#fff;color:#111}
  .wrap{max-width:none;padding:0}
  .card{break-inside:avoid;border-color:#bbb}
  .kv div{break-inside:avoid;border-color:#bbb}
  .warn{background:#fff;border-color:#8a5a00}
  h2{break-after:avoid}
  a{color:inherit;text-decoration:none}
  .totopcard,.toc,.fbar{display:none}
  tr:hover td{background:transparent}
}
"""


def render_html(trades: pd.DataFrame, eq: pd.DataFrame, ind: pd.DataFrame,
                cfg: dict, bench: pd.Series, summary: dict,
                explain=None) -> str:
    """渲染**任意策略**的逐笔报告。

    `explain(ind, symbol, sig_date, cfg) -> [(code, 名称, 实际值, 阈值, 是否满足)]`
    是唯一的策略相关钩子：信号日的条件清单由**策略自己**给（preset 走 A~F 表，
    族走 `family_explain_rows` —— 从掩码源码里读出它实际读了哪些列）。
    其余一切（版式、卡片、口径头、自检）对任何策略完全相同。
    """
    explain = explain or condition_rows
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
    parts.append(f'<h1 id="top">回测明细 · {e(summary["preset_name"])}</h1>')
    _spec = summary["preset"]
    _slip = (f'含执行滑点(f(ATR,参与率),scale={_spec["atr_slip_scale"]:g})'
             if _spec["atr_slip_scale"] > 0 else '零滑点')
    parts.append(f'<div class="sub">窗口 {e(summary["lo"])} → {e(summary["hi"])} ｜ '
                 f'成交 {len(trades)} 笔 ｜ {_slip},成本双边 '
                 f'{_spec["cost_bps_roundtrip"]:g}bp</div>')
    # 口径自报（审计 E5 及其同类）：与 run_baseline / --json 用的是**同一个函数**，
    # 所以两个入口不可能再各说一套。
    # 口径标签里也有 `**强调**`（SPEC_LABELS 是为 Markdown 报告写的：**未设** / **不清算**），
    # 所以这里与判读那一行走**同一个** `inline_md`，免得同一份文档里两种写法。
    _items = " ・ ".join(f"{e(lab)} <b>{inline_md(val)}</b>"
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
        _ctl_line = ("运气对照本次未跑(control_runs=0)—— 这份报告因此<b>不含</b>"
                     "「位于随机分布第几百分位」那条判词。")
    # 试验次数有两个：声明值（这条策略是从多大的搜索空间里选出来的）与**实际用于 DSR 的 k**
    # （声明值 + 继承父策略的 + 这台机器上点过它的次数）。面板的「DSR 的 k」格与逐笔 CSV
    # 的表头写的都是后者 —— 报告里那句"按 §5，N 次试验的选择偏差折算后门槛还要更高"里的 N
    # 必须也是它，否则同一件事在报告与面板上是两个数（实测：报告写 1614、面板写 1616）。
    _k_decl = int(summary["n_trials"])
    _k_eff = summary.get("n_trials_effective")
    _k_eff = int(_k_eff) if _k_eff else _k_decl
    _k_note = ("" if _k_eff == _k_decl else
               f"（本机上的 k 比声明值大 {abs(_k_eff - _k_decl)}：含继承父策略的运行次数"
               f"与本机点过它的次数 —— 面板的「DSR 的 k」格写着它的构成）")
    parts.append('<div class="warn"><h3>读这份报告之前必须知道的五件事</h3><ul>'
                 f'<li>本策略在<b>本窗口</b> <b>CAGR {summary["cagr"]*100:+.2f}%</b>,'
                 f'同期无脑等权持有 CAGR {bench_cagr*100:+.2f}% —— '
                 f'<b>{_over} {abs(_diff)*100:.1f}pp</b>(同口径比较)。'
                 f'它是 {summary["n_trials"]} 次试验里选出来的。</li>'
                 f'<li><b>{inline_md(summary.get("verdict") or summary.get("note") or "（本策略没有声明负面证据—— §5 要求负面证据与参数同时记录）")}</b></li>'
                 f'<li><b>样本只有 {len(trades)} 笔</b>,敞口 {summary["exposure"]*100:.0f}%。'
                 f'{_ctl_line}'
                 f'按 §5,{_k_eff} 次试验的选择偏差折算后门槛还要更高{_k_note}。</li>'
                 '<li><b>卡片右侧显示出场之后的价格</b>(供复盘),那是展示,不参与信号。'
                 '信号日清单只用当日及之前的数据 —— 已由截断重算独立验证。</li>'
                 '</ul></div>')

    # ---- 目录：这份报告实测 91 张卡 / 108 屏，此前 5 个 h2 连 id 都没有、也没有一处
    # 「回到顶部」⇒ 看一眼逐笔就得靠浏览器后退（审计实测 backToTop=0、h2 id 全为 null）。
    # 纯 <a href="#...">，零 JS —— 报告必须保持自包含（selfcheck 禁 <script>）。
    parts.append('<div class="toc">'
                 '<a href="#s-summary">摘要</a><a href="#s-equity">权益曲线</a>'
                 '<a href="#s-config">配置</a><a href="#s-trades">全部交易</a>'
                 f'<a href="#s-detail">逐笔明细（{len(trades)} 笔）</a>'
                 '</div>')

    # ---- 摘要 ----
    parts.append('<h2 id="s-summary">摘要</h2><div class="kv">')
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

    parts.append('<h2 id="s-equity">权益曲线 vs 基准</h2><div class="card">')
    parts.append(equity_svg(eq, bench))
    parts.append('</div>')

    parts.append('<h2 id="s-config">配置</h2><div class="card"><table class="conds">')
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
    # ---- 出场原因分布（在 90 张卡之前先说清"钱是从哪一类出场来的"）----
    _rr = exit_reason_rows(trades)
    if _rr:
        parts.append('<h2 id="s-trades">全部交易</h2><div class="card">'
                     '<div class="reasonbox"><b>按出场原因（从多到少）</b>'
                     '<table class="rsum"><thead><tr>'
                     '<th>原因</th><th>笔数</th><th>占比</th><th>平均净收益</th>'
                     '<th>合计净盈亏</th><th>胜率</th></tr></thead><tbody>')
        for d in _rr:
            cls = "pos" if d["mean_net_return"] >= 0 else "neg"
            parts.append(
                f'<tr><td><code>{e(d["reason"])}</code></td><td>{d["n"]}</td>'
                f'<td>{d["share"]*100:.0f}%</td>'
                f'<td class="{cls}">{d["mean_net_return"]*100:+.2f}%</td>'
                f'<td class="{"pos" if d["sum_net_pnl"] >= 0 else "neg"}">'
                f'${d["sum_net_pnl"]:,.0f}</td>'
                f'<td>{d["win_rate"]*100:.0f}%</td></tr>')
        # 合计行必须与上面每一行**同一列同名**：这里是"逐笔平均净收益"（不是权益总收益 tot，
        # 那是另一回事；把它们并排放会让人以为逐笔平均 == 组合收益）。
        _mret = float(trades["net_return"].mean())
        _wpct = float((trades["net_pnl"] > 0).mean())
        parts.append(f'<tr><td><b>合计</b></td><td><b>{len(trades)}</b></td><td>100%</td>'
                     f'<td class="{"pos" if _mret >= 0 else "neg"}">{_mret*100:+.2f}%</td>'
                     f'<td class="{"pos" if trades["net_pnl"].sum() >= 0 else "neg"}">'
                     f'${trades["net_pnl"].sum():,.0f}</td>'
                     f'<td>{_wpct*100:.0f}%</td></tr>')
        parts.append('</tbody></table><div class="rleg">出场原因的含义：'
                     + " ・ ".join(f'<code>{e(k)}</code>={inline_md(v)}'
                                   for k, v in EXIT_LEGEND.items()
                                   if k in {d["reason"] for d in _rr})
                     + f'</div></div>')
        # 表格仍在本卡片里（同一个 .card），只是接在汇总下面
        parts.append('<table><thead><tr>')
    else:
        parts.append('<h2 id="s-trades">全部交易</h2><div class="card"><table><thead><tr>')
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
    parts.append('<h2 id="s-detail">逐笔明细</h2>')
    _reasons = exit_reason_rows(trades)
    if len(_reasons) >= 2 and len(trades) >= 10:
        parts.append('<div class="fbar"><span class="sub">只看：</span>'
                     '<input type="radio" name="xf" id="xf-all" checked>'
                     f'<label for="xf-all">全部 {len(trades)}</label>')
        for d_ in _reasons:
            parts.append(f'<input type="radio" name="xf" id="xf-{e(d_["reason"])}">'
                         f'<label for="xf-{e(d_["reason"])}">{e(d_["reason"])} {d_["n"]}</label>')
        parts.append('</div>')
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
        parts.append(f'<div class="card" id="t{i}" data-reason="{e(str(t["exit_reason"]))}">')
        # 每张卡自己带「回总表」：总表→逐笔有 #t{i} 锚点（本就有、实测 91 个全命中），
        # 但跳下去之后**没有回去的路**（审计实测 href 只有 #t{i}，无一处回目录）。
        parts.append('<a class="totop" href="#s-trades">↑ 回总表</a>'
                     '<a class="totop" href="#top">↑ 回顶部</a>')
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
        for code, name, val, thr, ok in explain(ind, sym, sig_d, cfg):
            # 第三态：`ok is None` ⇒ **不适用**（族不打勾：判定是多个条件的合取、阈值散在
            # FIXED/params 里，逐条重算等于把掩码实现写第二遍 —— 那是本仓库禁止的第二条路径）。
            # 少这一态就会出现"每一行都打 ✓"的假象（对抗核验式的自检抓到过）。
            _cls, _mark = ("ok", "✓") if ok else ("", "·") if ok is None else ("no", "✗")
            parts.append(f'<tr><td>{e(code)}</td><td>{e(name)}</td><td>{e(val)}</td>'
                         f'<td>{e(thr)}</td><td class="{_cls}">{_mark}</td></tr>')
        parts.append('</table></div>')

    parts.append('<a class="totopcard" href="#top">↑ 回到顶部</a>')
    parts.append('</div></body></html>')
    return "".join(parts)


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
    # 字面的 markdown 标记：`**粗体**` 与 `` `代码` `` 走 inline_md 转换过了，剩下的
    # `**` 就是**漏转**的 —— 读者会看到星号本身（实测：运气对照那句"**不含**"就这么印出来）。
    # 报告里凡是给人看的散文，标记都该是 HTML 标签。
    # ⚠️ 要**排除 <style> 里的 CSS 注释** —— 那些 `**` 是注释里的强调，人看不见
    # （我第一版没排除，selfcheck 立刻在真报告上抓出 6 个"漏转"的假警报）。
    _visible = re.sub(r"<style>.*?</style>", "", doc, flags=re.S)
    if "**" in _visible:
        problems.append(f"报告里还有 {_visible.count('**')} 个字面的 **（markdown 没转成标签）")
    return problems


# ---------------------------------------------------------------- 族的条件清单

def _mask_read_columns(mask) -> list[str]:
    """从**掩码源码**里读出它实际读了哪些面板列（`ind["X"]`）。

    为什么用 AST 而不是维护一张表：族的掩码是自由 Python（`family_defs/<id>.py`），
    手工维护"这个族看哪些指标"必然与代码分叉 —— 而分叉的报告会让读者以为自己在看
    依据。AST 扫源码是**唯一**一种"新增一个族不用改这一层"的做法。
    """
    import ast
    import inspect

    src = inspect.getsource(mask)
    cols: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "ind" and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            if node.slice.value not in cols:
                cols.append(node.slice.value)
    return cols


def family_explain_rows(ind: pd.DataFrame, symbol: str, sig_date: pd.Timestamp,
                        cfg: dict, *, mask, params: dict | None = None,
                        ) -> list[tuple[str, str, str, str, bool | None]]:
    """族型策略的信号日清单：**掩码实际读的列**的当日值 + 本次用的参数。

    诚实边界：这里不打"✓/✗"（族的判定是多个条件的合取，且阈值散在 FIXED/params 里，
    逐条重算等于把掩码实现写第二遍 —— 那正是本仓库禁止的两条路径）。所以第五项返回
    **None**（= 不适用），报告里显示为「·」而不是「✓」。读者要看的是**输入**：
    那些列的当日值，以及本次用了什么参数。
    """
    row = ind[(ind["symbol"] == symbol) & (ind["date"] == sig_date)]
    out: list[tuple[str, str, str, str, bool]] = []
    if row.empty:
        return [("—", f"信号日 {sig_date.date()} 没有 {symbol} 的行", "—", "—", False)]
    r = row.iloc[0]
    for col in _mask_read_columns(mask):
        v = r.get(col)
        try:
            txt = f"{float(v):.4g}" if v is not None and np.isfinite(float(v)) else "—"
        except (TypeError, ValueError):
            txt = str(v)
        out.append((col, f"掩码读了 {col}", txt, "—", None))
    for k, v in sorted((params or {}).items()):
        out.append((f"param.{k}", "本次用的参数", str(v), "—", None))
    if not out:
        out.append(("—", "这个掩码没有直接读面板列", "—", "—", False))
    return out
