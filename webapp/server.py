"""本地策略平台的服务端 —— **stdlib only**，只绑 127.0.0.1。

为什么不用 Flask/FastAPI：本仓库的依赖是钉死的（`requirements.txt`，升级=再验一次），
给一个单人本地工具加一个 Web 框架属于要单独决策的事；而 `http.server` 足够。
为什么**必须**绑 127.0.0.1：这个页面能一键跑回测、读本地库、写报告文件 ——
把它暴露到局域网上等于把"谁能读你的数据/占你的 CPU"交给别人。

路由（都很小，页面只用这几个）：
    GET  /                  页面（`static/index.html`，自包含、无 CDN）
    GET  /api/catalog       策略目录（preset 参数 / 族网格 / 负面证据）
    POST /api/run           提交回测 {kind,id,lo,hi,params?} → {job id}
    GET  /api/job/<id>      任务状态（steps / metrics / report 链接）
    GET  /api/runs          台账（最近 30 条 + 总条数）
    GET  /api/runs.csv      台账**全部**记录的 CSV（页面上的表有 30 条上限）
    GET  /api/reports       盘上已有的报告（结果可回访，不必为了看一眼而重跑）
    GET  /api/source/<kind>/<id>   只读地取定义体源码（开发时最常跳的地方）
    GET  /report/<id>       平台生成的回测报告（id = 任务 id）
    GET  /report-file/<名>  盘上任何一份报告（含 CLI `--out` 生成的）—— 只认 results/ 下
                            的 `<名>.html`，不拼目录（越界必须 400/404）

用法：
    python3 -m webapp.server                 # 默认 http://127.0.0.1:8765
    python3 -m webapp.server --port 9000 --open
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from webapp import catalog, jobs                     # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
RESULTS = ROOT / "research" / "results"
_JOB_ID = re.compile(r"^[0-9a-f]{6,32}$")            # 报告文件名只接受任务 id 形态


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
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _html(self, code: int, path: Path) -> None:
        if not path.is_file():
            self._json(404, {"error": f"找不到 {path.name}"})
            return
        self._send(code, path.read_bytes(), "text/html; charset=utf-8")

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
            # 表只显示最近 30 条 —— 但**共几条**要说出来（否则读者以为那就是全部）。
            runs = jobs.recent_runs(30)
            return self._json(200, {"runs": runs, "total": len(jobs.all_runs()),
                                    "shown": len(runs),
                                    # 卡片「已跑 N 次」与筛选下拉都按**全文**数 —— 它们与
                                    # DSR 的 k 是同一个计数，不能受"表只显示 30 条"影响
                                    "by_strategy": jobs.counts_by_strategy()})
        if p == "/api/runs.csv":
            # 完整台账的 CSV（页面上的表有 30 条上限，导出不能跟着被截断）。
            import csv as _csv
            import io as _io
            rows = jobs.all_runs()
            cols = ["ts", "strategy_id", "window", "params", "n_trades", "excess", "dsr",
                    "n_trials_k", "prior_runs", "control_runs", "seconds", "note",
                    "job_id", "git_sha", "kind", "name"]
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
                            "name": name.group(1) if name else "?",
                            "window": [win.group(1), win.group(2)] if win else None,
                            # 参数不是从报告里正则抓的，是按 job_id 回读台账（见 params_by_job）
                            "params": pj.get(jid),
                            "kb": round(st.st_size / 1024), "mtime": int(st.st_mtime)})
            return self._json(200, {"reports": out})
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
        m = re.fullmatch(r"/api/job/([0-9a-f]{6,32})", p)
        if m:
            st = jobs.status(m.group(1))
            return self._json(200 if st else 404, st or {"error": "未知任务"})
        m = re.fullmatch(r"/report-file/([A-Za-z0-9_.-]{1,120}\.html)", p)
        if m:
            # 按**文件名**打开一份报告：CLI `--out` 的产物不是任务 id，用不了 /report/<id>。
            # 两道校验 —— 名字只允许 [A-Za-z0-9_.-] 且必须以 .html 结尾；解析后父目录必须仍是
            # 我们自己的 results/（不拼用户输入、不做目录遍历）。
            f = (RESULTS / m.group(1)).resolve()
            if f.parent != RESULTS.resolve():
                return self._json(400, {"error": "报告路径越界"})
            return self._html(200, f)
        m = re.fullmatch(r"/report/([0-9a-f]{6,32})", p)
        if m:
            jid = m.group(1)
            if not _JOB_ID.match(jid):
                return self._json(400, {"error": "任务 id 非法"})
            # 只接受由本次服务生成的报告文件（不拼用户输入、不做目录遍历）
            return self._html(200, RESULTS / f"web_report_{jid}.html")
        self._json(404, {"error": f"未知路径 {p}"})

    def do_POST(self) -> None:                       # noqa: N802
        if self.path.split("?", 1)[0] != "/api/run":
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
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)     # 只监听本机
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
