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
    GET  /api/runs          台账（含"这条策略已被跑过几次"）
    GET  /report/<id>       回测报告（自包含 HTML）

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
            return self._json(200, {"runs": jobs.recent_runs(30)})
        m = re.fullmatch(r"/api/job/([0-9a-f]{6,32})", p)
        if m:
            st = jobs.status(m.group(1))
            return self._json(200 if st else 404, st or {"error": "未知任务"})
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
