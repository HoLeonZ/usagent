"""本地策略平台的服务端 —— **stdlib only**，只绑 127.0.0.1。

为什么不用 Flask/FastAPI：本仓库的依赖是钉死的（`requirements.txt`，升级=再验一次），
给一个单人本地工具加一个 Web 框架属于要单独决策的事；而 `http.server` 足够。
为什么**必须**绑 127.0.0.1：这个页面能一键跑回测、读本地库、写报告文件 ——
把它暴露到局域网上等于把"谁能读你的数据/占你的 CPU"交给别人。

路由（都很小，页面只用这几个）：
    GET  /                  页面（`static/index.html`，自包含、无 CDN）
    GET  /api/catalog       策略目录（preset 参数 / 族网格 / 负面证据）
    POST /api/run           提交回测 {kind,id,lo,hi,params?} → {job id}
    POST /api/cancel/<id>   取消一次运行（粗粒度：当前这一步跑完才停）
    GET  /api/job/<id>      任务状态（steps / metrics / report 链接）
    GET  /api/jobs          在途任务（queued/running）—— 刷新后能把面板与取消入口找回来
    GET  /api/runs          台账（最近 30 条 + 总条数）
    GET  /api/runs.csv      台账**全部**记录的 CSV（页面上的表有 30 条上限）
    GET  /api/reports       盘上已有的报告（结果可回访，不必为了看一眼而重跑）
    GET  /api/source/<kind>/<id>   只读地取定义体源码（开发时最常跳的地方）
    GET  /report/<id>       平台生成的回测报告（id = 任务 id）
    GET  /api/trades/<id>.csv   逐笔明细 CSV（机器可读；人看的那份仍是报告）
    GET  /report-file/<名>  盘上任何一份报告（含 CLI `--out` 生成的）—— 只认 results/ 下
                            的 `<名>.html`，不拼目录（越界必须 400/404）

用法：
    python3 -m webapp.server                 # 默认 http://127.0.0.1:8765
    python3 -m webapp.server --port 9000 --open
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.parse
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ⚠️ 关掉字节码写入（2026-10-18 实测踩到）：Python 用「源文件 mtime + 大小」判断 .pyc 是否
# 还有效，粒度是**秒**。于是"在**同一秒内**把 `preset_defs/x.py` 改动又改回来、且长度不变"
# 会让旧的 .pyc 继续被判为有效 —— 我自己的验证脚本正好踩出这一幕：文件里写着
# n_comparisons=1614，而进程读到的是 1615，并且**静默**用了三次。这是长驻开发进程最该防的
# 一类错（"我改了却没反应"的反面：**没改却生效了旧值**）。不写 .pyc 就没有这个类。
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import obs                                            # noqa: E402
from webapp import catalog, jobs                     # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
RESULTS = ROOT / "research" / "results"
_JOB_ID = re.compile(r"^[0-9a-f]{6,32}$")            # 报告文件名只接受任务 id 形态
# 命令行产物的命名规则（页面 cliCmd 给的）：web_report_cli_<策略id>_<lo>_<hi>[_p<指纹>].html
_CLI_REPORT = re.compile(r"^web_report_cli_.*_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})"
                         r"(?:_p[0-9a-f]{8})?\.html$")


def _cli_holdout_touches() -> int | None:
    """命令行产物里"碰了 holdout"的次数 —— **只认文件名里带着窗口的那种**。

    为什么不干脆扫所有产物：报告文件里确实有窗口（标题下的副标题），但从渲染好的 HTML 里
    反推事实是"一份事实两处"（本仓库为此立过规矩）。文件名是页面让用户写进去的、可机器读的
    那一份，认不出就明说认不出（返回 None 让界面别显示这一半）。
    """
    if not RESULTS.is_dir():
        return None
    n, saw_cli = 0, False
    for f in RESULTS.glob("web_report_cli_*.html"):
        m = _CLI_REPORT.match(f.name)
        if not m:
            continue
        saw_cli = True
        if catalog.touches_holdout((m.group(1), m.group(2))):
            n += 1
    return n if saw_cli else None


def _finite(obj):
    """把 NaN / ±Inf 换成 `null` —— **Python 的 json 默认会写出裸 `NaN`，而那不是合法 JSON**。

    2026-10-19 边界实测：一个 0 成交（或退化）的运行，metrics 里带 `t_alpha: NaN` / `E_R: NaN`
    ⇒ `/api/job/<id>` 的响应体里出现裸 `NaN` ⇒ **浏览器的 `JSON.parse` 直接抛错** ⇒ 页面永远
    渲染不出这次运行的结果（而台账里已经记下了），前端还会把它误报成"读不到任务状态" ——
    服务器明明好好的。所以在这条唯一的出口上统一净化：非有限浮点一律 null，前端对 null
    本来就渲染成「—」。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    server_version = "usagent-platform/1"

    # ---- 小工具 ----
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 本地工具同样不给浏览器任何"顺手嗅探"的空间
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(_finite(obj), ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _html(self, code: int, path: Path) -> None:
        if not path.is_file():
            self._json(404, {"error": f"找不到 {path.name}"})
            return
        self._send(code, path.read_bytes(), "text/html; charset=utf-8")

    def _report_missing(self, name: str) -> None:
        """报告文件不在了 —— 给人看的页面，而不是一坨 JSON。

        报告是由人**点链接打开**的（新标签页、甚至书签），而 research/results/ 里的产物会被
        手工清理 ⇒ 链接会变陈旧。实测：那种情况下浏览器里显示的是一行
        `{"error": "找不到 web_report_xxx.html"}` —— 对一个被点开的链接来说这是最没用的一种脸。
        API 路由照旧回 JSON（那边的读者是程序）。"""
        page = (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<title>报告不在盘上 · usagent</title></head>'
            '<body style="font:14px/1.7 -apple-system,PingFang SC,Arial,sans-serif;background:#0f1115;'
            'color:#e6e8ee;max-width:640px;margin:80px auto;padding:0 20px">'
            '<h1 style="font-size:18px">这份报告已经不在盘上了</h1>'
            f'<p style="color:#9aa3b2">文件名：<code style="color:#cbd5e1">{name}</code></p>'
            '<p>常见原因：<code>research/results/</code> 里的产物被手工清理过（报告不入库，'
            '也不该入库 —— §7）。</p>'
            '<p>这次运行本身仍在<strong>运行台账</strong>里（若它有 job id）：'
            '<a href="/#runs" style="color:#7aa7ff">回到平台 → 运行台账</a>。'
            '想重新生成就在对应策略卡片上再跑一次 —— 注意每跑一次都会记进台账、推高 k。</p>'
            '</body></html>')
        self._send(404, page.encode("utf-8"), "text/html; charset=utf-8")

    def _report(self, code: int, path: Path, job_id: str | None = None,
                ledger: bool = False) -> None:
        """托管一份报告 —— **在服务端**给它加一条"回到平台"的返回条。

        `target="_top"` 是必须的：平台把报告 iframe 内嵌在结果区里，不带 target 的话点这条
        返回条会把**整个平台**塞进那个 70vh 的小框里（审计实测）。

        为什么在服务端注入而不是写进报告文件：报告是**可下载、可归档**的自包含产物
        （`selfcheck` 明令不许出现 `https?://`），写死一个 `http://127.0.0.1:…` 会让
        自检失败、归档后链接也是死的。而注入的服务端那条只在**由本平台托管时**存在 ——
        下载下来的那一份仍然干净、仍然自包含。（审计当初给的也是这条建议。）
        """
        if not path.is_file():
            return self._json(404, {"error": f"找不到 {path.name}"})
        html = path.read_text(encoding="utf-8")
        bar = ('<style>@media print{.hostbar{display:none}}</style>'
               '<div class="hostbar" style="position:fixed;left:14px;bottom:14px;z-index:9;'
               'background:#161a22;border:1px solid #262b36;border-radius:999px;padding:6px 12px;'
               'font:13px/1.4 -apple-system,PingFang SC,Arial,sans-serif;color:#e6e8ee">'
               '← <a target="_top" href="/#runs" style="color:#8ab4f8;text-decoration:none">'
               + ("回到平台（这份运行在台账里）" if ledger else
                  "回到平台（这份是命令行 --out 产出的，不在台账里）")
               + '</a>'
               + (f'<span style="color:#9aa3b2"> · job {job_id}</span>' if job_id else "")
               + f'<span style="color:#9aa3b2"> · {path.name}</span></div>')
        if "</body>" in html:
            html = html.replace("</body>", bar + "</body>", 1)
        else:
            html += bar
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):               # 默认会往 stderr 刷每一条请求
        pass

    # ---- 路由 ----
    def do_GET(self) -> None:                        # noqa: N802
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            return self._html(200, STATIC / "index.html")
        if p == "/api/catalog":
            return self._json(200, catalog.build())
        if p == "/api/runs":
            # 表默认只显示最近 30 条 —— 但**共几条**要说出来（否则读者以为那就是全部），
            # 并且 `?limit=` 可以要更多（我此前给"看不见第 31 条"开的方子只是"去用 CSV 导出"，
            # 那是把用户赶出界面）。上限 500：台账是线性的，不该被一个查询拉爆。
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                limit = int((q.get("limit") or ["30"])[0])
            except ValueError:
                return self._json(400, {"error": "limit 必须是整数"})
            limit = max(1, min(limit, 500))
            runs, allruns = jobs.recent_runs(limit), jobs.all_runs()
            return self._json(200, {"runs": runs, "total": len(allruns),
                                    "shown": len(runs),
                                    # 纪律计数：碰过 holdout 的次数（卡片上那句"只准读一次"
                                    # 必须有个数字，否则它只是句口号）
                                    "holdout_touches": catalog.holdout_touches(allruns),
                                    # 命令行产物的**文件名**里带着窗口（页面给的命名规则），
                                    # 能认出来的也数上 —— 但必须与上面那个数分开报（§5 的账
                                    # 不能静默混两种来源）
                                    "cli_holdout_touches": _cli_holdout_touches(),
                                    # 当前代码版本：界面据此把"旧代码下跑的"那几次标出来
                                    # （sha 单独不够 —— 脏工作区同样返回一个 sha）
                                    "git_sha": obs.git_sha(), "git_dirty": obs.git_dirty(),
                                    # 卡片「已跑 N 次」与筛选下拉都按**全文**数 —— 它们与
                                    # DSR 的 k 是同一个计数，不能受"表只显示 30 条"影响
                                    "by_strategy": jobs.counts_by_strategy(),
                                    # 卡片「最近一次」也按全台账（与 by_strategy 同源）
                                    "last_by_strategy": jobs.last_by_strategy(),
                                    # 台账里读不出来的行数：>0 时页面要警告（少读一行 =
                                    # 少记一次试验 = k 偏小），不能让它静默
                                    "skipped_lines": jobs.skipped_lines(),
                                    # 权威 holdout 台账的状态（平台运行**不登记**进去）——
                                    # 不并排显示，读者会把平台那个数当成纪律账本上的次数
                                    "holdout_ledger": catalog.holdout_ledger_state(allruns)})
        if p.startswith("/api/compare/"):
            # 跑完两条候选**怎么比** —— 逐项差值 + 两条纪律提醒（k 不同 ⇒ DSR 不可比；
            # 窗口不同 ⇒ 其它指标也不是同口径）。这是上一轮我**手工**做的事，现在平台自己做。
            m = re.fullmatch(r"/api/compare/([0-9a-f]{6,32})/([0-9a-f]{6,32})", p)
            if not m:
                return self._json(400, {"error": "比较要两个任务 id：/api/compare/<a>/<b>"})
            by_id = {r.get("job_id"): r for r in jobs.all_runs() if r.get("job_id")}
            a, b = by_id.get(m.group(1)), by_id.get(m.group(2))
            if a is None or b is None:
                return self._json(404, {"error": "这两个任务 id 里有一个不在台账里"})
            fields = [("cagr", "CAGR", "pct"), ("excess", "曝光匹配超额", "pp"),
                      ("bench_cagr", "同窗等权池 CAGR", "pct"), ("exposure", "平均敞口", "pct"),
                      ("n_trades", "成交笔数", "int"), ("win_pct", "胜率", "pct"),
                      ("e_r", "E[R]", "num"), ("profit_factor", "盈亏比 PF", "num"),
                      ("maxdd", "最大回撤", "pct"), ("folds_positive", "4 折为正", "int"),
                      ("alpha_annual", "alpha 年化", "pct"), ("dsr", "DSR", "num"),
                      # 随机入场百分位：**整窗一个数**（判据④要的是逐折 ≥3/4 折在中位以上，
                      # 那是 family_search 的逐折输出）。只有跑过运气对照的那次才有。
                      ("ctl_pct", "随机入场百分位", "num")]
            rows = []
            for key, lab, kind in fields:
                va, vb = a.get(key), b.get(key)
                delta = (vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
                rows.append({"key": key, "label": lab, "unit": kind, "a": va, "b": vb, "delta": delta})
            ka, kb = a.get("n_trials_k"), b.get("n_trials_k")
            return self._json(200, {
                "a": {"job_id": a.get("job_id"), "strategy_id": a.get("strategy_id"),
                      "window": a.get("window"), "k": ka, "ts": a.get("ts"),
                      "params": a.get("params"), "derived_from": a.get("derived_from"),
                      "control_runs": a.get("control_runs"), "ctl_pct": a.get("ctl_pct")},
                "b": {"job_id": b.get("job_id"), "strategy_id": b.get("strategy_id"),
                      "window": b.get("window"), "k": kb, "ts": b.get("ts"),
                      "params": b.get("params"), "derived_from": b.get("derived_from"),
                      "control_runs": b.get("control_runs"), "ctl_pct": b.get("ctl_pct")},
                "rows": rows,
                # 两条纪律提醒（这是"对比"视图最该说的话）
                "k_differs": (ka is not None and kb is not None and ka != kb),
                "window_differs": (a.get("window") != b.get("window")),
                "params_differ": (json.dumps(a.get("params"), sort_keys=True)
                                  != json.dumps(b.get("params"), sort_keys=True)),
                # 运气对照是**勾选框**（默认不跑）⇒ 两次里常常只有一次有那个数。
                # 不标出来的话，那一行的"差值"是拿一个数减一个空格，读起来像"变好了"。
                "control_mismatch": (a.get("ctl_pct") is None) != (b.get("ctl_pct") is None),
            })
        if p == "/api/jobs":
            # 在途任务（页面刷新后据此重建面板/徽标/取消入口）。只给这个页面用得上的最小字段。
            jobs_now = jobs.inflight()
            for r in jobs_now:
                if r["seconds"] is None:
                    st = jobs.status(r["id"]) or {}
                    r["seconds"] = st.get("seconds")
            return self._json(200, {"jobs": jobs_now})
        if p == "/api/runs.csv":
            # 完整台账的 CSV（页面上的表有 30 条上限，导出不能跟着被截断）。
            import csv as _csv
            import io as _io
            rows = jobs.all_runs()
            cols = ["ts", "strategy_id", "window", "params", "n_trades", "excess", "dsr",
                    "n_trials_k", "prior_runs", "derived_from", "control_runs", "seconds", "note",
                    "job_id", "git_sha", "git_dirty", "kind", "name"]
            buf = _io.StringIO()
            w = _csv.writer(buf)
            w.writerow(cols + ["excess_pct"])
            for r in rows:
                w.writerow([json.dumps(r.get(c), ensure_ascii=False) if isinstance(r.get(c), (list, dict))
                            else r.get(c, "") for c in cols]
                           + ["" if r.get("excess") is None else round(r["excess"] * 100, 4)])
            body = buf.getvalue().encode("utf-8-sig")      # BOM：Excel 打开中文不乱码
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", 'attachment; filename="web_runs.csv"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            return self.wfile.write(body)
        if p == "/api/reports":
            # 盘上已有的报告（结果**可以回访**）。截取文件头部把策略名与窗口读回来 ——
            # 报告自己就带这两样（title 与副标题），不用另存一份元数据（一份事实一处）。
            out = []
            pj = jobs.params_by_job()          # job_id → 参数（报告里没有参数，台账里有）
            # 平台的产物 **和** CLI `--out` 的产物都要列出来：复制出来的那条命令如果跑完
            # 在页面上看不见，那"界面不是黑箱"就只是句口号（审计实测 --out 的命名对不上）。
            files = list(RESULTS.glob("web_report_*.html")) + list(RESULTS.glob("report_*.html"))
            total_reports = len(files)
            for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
                # ⚠️ 读头部不能只取前 N 个字符：CSS 一长，副标题就漂到切片外 —— 实测加了
                # 打印样式与目录之后，新报告全都解析成 window=None（旧的还能解析，看起来
                # 像"个别报告有问题"）。这里先把 <style> 整块去掉再取，头部大小与样式无关。
                txt = f.read_text(encoding="utf-8", errors="replace")
                txt = re.sub(r"<style>.*?</style>", "", txt, count=1, flags=re.S)
                head = txt[:4000]
                name = re.search(r"<title>回测明细 · ([^<]+)</title>", head)
                win = re.search(r"窗口 ([0-9-]+) → ([0-9-]+)", head)
                st = f.stat()
                # job_id **只**对平台自己的产物成立（web_report_<12位hex>.html）。CLI `--out`
                # 的文件名不是任务 id —— 塞一个假 job_id 进 API 会让调用方以为能按它查任务。
                hexid = re.fullmatch(r"web_report_([0-9a-f]{6,32})\.html", f.name)
                jid = hexid.group(1) if hexid else None
                out.append({"job_id": jid, "file": f.name,
                            "has_trades": bool(jid) and
                            (RESULTS / f"web_trades_{jid}.csv").is_file(),
                            "name": name.group(1) if name else "?",
                            "window": [win.group(1), win.group(2)] if win else None,
                            # 参数不是从报告里正则抓的，是按 job_id 回读台账（见 params_by_job）
                            "params": pj.get(jid),
                            "kb": round(st.st_size / 1024), "mtime": int(st.st_mtime)})
            # 截断必须说出来（台账那边是这么做的：静默截断读起来像"这就是全部"）
            return self._json(200, {"reports": out, "total": total_reports,
                                    "shown": len(out)})
        m = re.fullmatch(r"/api/source/(preset|family)/([A-Za-z0-9_]{1,64})", p)
        if m:
            # 只读地取**定义体源码**（开发时最常要跳的地方）。两条校验：策略必须存在，
            # 且文件名只能由 [A-Za-z0-9_] 组成（不拼用户输入、不做目录遍历）。
            kind, ident = m.group(1), m.group(2)
            if catalog.find(kind, ident) is None:
                return self._json(404, {"error": f"未知策略 {ident!r}"})
            sub = "preset_defs" if kind == "preset" else "family_defs"
            f = ROOT / "research" / sub / f"{ident}.py"
            if not f.is_file():
                return self._json(404, {"error": f"没有定义体文件 {f.name}"})
            text = f.read_text(encoding="utf-8")
            return self._json(200, {"path": f"research/{sub}/{ident}.py",
                                    "bytes": len(text), "text": text[:200_000]})
        m = re.fullmatch(r"/api/candidate_skeleton/([A-Za-z0-9_]{1,64})", p)
        if m:
            # "从这里开始做一条候选" 的骨架（catalog.candidate_skeleton 里有为什么）。
            # 只给 preset：族没有 NAME / n_comparisons / VERDICT 这几样要改的东西。
            ident = m.group(1)
            if catalog.find("preset", ident) is None:
                return self._json(404, {"error": f"未知 preset {ident!r}"})
            sk = catalog.candidate_skeleton(ident)
            if sk is None:
                return self._json(404, {"error": f"{ident} 没有可复制的定义体文件"})
            return self._json(200, {"path": sk["path"], "text": sk["text"],
                                    "parent": sk["parent"],
                                    "renamed_because": sk["renamed_because"]})
        m = re.fullmatch(r"/api/job/([0-9a-f]{6,32})", p)
        if m:
            st = jobs.status(m.group(1))
            return self._json(200 if st else 404, st or {"error": "未知任务"})
        m = re.fullmatch(r"/api/trades/([0-9a-f]{6,32})\.csv", p)
        if m:
            # 逐笔明细的机器可读件（报告是给人看的那些：K 线/指标/判词；这份给脚本与表格）
            f = RESULTS / f"web_trades_{m.group(1)}.csv"
            if not f.is_file():
                return self._json(404, {"error": "这次运行没有逐笔 CSV（可能早于该功能）"})
            return self._send(200, f.read_bytes(), "text/csv; charset=utf-8")
        m = re.fullmatch(r"/report-file/([A-Za-z0-9_.-]{1,120}\.html)", p)
        if m:
            # 按**文件名**打开一份报告：CLI `--out` 的产物不是任务 id，用不了 /report/<id>。
            # 两道校验 —— 名字只允许 [A-Za-z0-9_.-] 且必须以 .html 结尾；解析后父目录必须仍是
            # 我们自己的 results/（不拼用户输入、不做目录遍历）。
            f = (RESULTS / m.group(1)).resolve()
            if f.parent != RESULTS.resolve():
                return self._json(400, {"error": "报告路径越界"})
            if not f.is_file():
                return self._report_missing(f.name)
            # 与 /report/<id> 同判据：平台产物（web_report_<hex>.html）**且**台账里真有这条
            # 才敢说「这份运行在台账里」—— 否则同一份报告走两条路会给出互相矛盾的说法（审计实测）
            hexid = re.fullmatch(r"web_report_([0-9a-f]{6,32})\.html", f.name)
            in_ledger = bool(hexid) and any(r.get("job_id") == hexid.group(1)
                                            for r in jobs.all_runs())
            return self._report(200, f, job_id=hexid.group(1) if hexid else None,
                                ledger=in_ledger)
        m = re.fullmatch(r"/report/([0-9a-f]{6,32})", p)
        if m:
            jid = m.group(1)
            if not _JOB_ID.match(jid):
                return self._json(400, {"error": "任务 id 非法"})
            # 只接受由本次服务生成的报告文件（不拼用户输入、不做目录遍历）
            _f = RESULTS / f"web_report_{jid}.html"
            if not _f.is_file():
                return self._report_missing(_f.name)
            # 台账里有没有这一条：只有平台自己产出的才在（CLI `--out` 的从来不在）——
            # 返回条那句「这份运行在台账里」必须是真的（审计实测：CLI 产物上它是假的）
            return self._report(200, RESULTS / f"web_report_{jid}.html", job_id=jid,
                                ledger=any(r.get("job_id") == jid for r in jobs.all_runs()))
        self._json(404, {"error": f"未知路径 {p}"})

    def do_POST(self) -> None:                       # noqa: N802
        p = self.path.split("?", 1)[0]
        if p.startswith("/api/cancel/"):
            # 取消一次运行。粗粒度：当前这一步跑完才停（见 jobs.cancel 的说明）。
            # ⚠️ 只调用一次并记住结果：调两次的话，若任务恰好在两次之间跑完，
            # 第一次说"受理了"、第二次说"没受理"，返回给页面的是后者 —— 自相矛盾。
            m = re.fullmatch(r"/api/cancel/([0-9a-f]{6,32})", p)
            ok = bool(m) and jobs.cancel(m.group(1))
            return self._json(200 if ok else 404, {"cancelled": ok})
        if p != "/api/run":
            return self._json(404, {"error": "未知路径"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": f"请求体不是合法 JSON：{exc}"})
        for k in ("kind", "id", "lo", "hi"):
            if not req.get(k):
                return self._json(400, {"error": f"缺字段 {k}"})
        if req["kind"] not in ("preset", "family"):
            return self._json(400, {"error": "kind 只能是 preset 或 family"})
        if catalog.find(req["kind"], req["id"]) is None:
            return self._json(404, {"error": f"未知策略 {req['id']!r}"})
        return self._json(200, jobs.submit(req))


def serve(port: int = 8765, open_browser: bool = False) -> ThreadingHTTPServer:
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)     # 只监听本机
    except OSError as exc:
        # 端口被占时原先抛 15 行 Python traceback（审计实测），既不说是怎么回事也不说下一步。
        # 这是**运维**最常撞的一堵墙，值得一句人话。
        import errno
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            raise SystemExit(
                f"端口 {port} 用不了（{'已被占用' if exc.errno == errno.EADDRINUSE else '权限不足'}）——\n"
                f"  换一个端口：python3 -m webapp.server --port {port + 1}\n"
                f"  或先停掉原来那个：lsof -ti :{port} | xargs kill") from None
        raise
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"策略平台已启动：{url}   （只绑 127.0.0.1；Ctrl-C 退出）")
    print("  目录页读的是注册表本身：preset_defs/ 与 family_defs/ —— 新增策略后刷新即可")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本地策略平台（stdlib，只绑 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    a = ap.parse_args(argv)
    srv = serve(a.port, a.open)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
