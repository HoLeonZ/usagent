"""策略平台的测试 —— **离线**（确定性合成市场 + 真 HTTP 服务在随机端口）。

钉住四件事：
  1. 目录页的数据**来自注册表本身**（数得出来的 preset/族条数、每条都有参数与负面证据）；
  2. 一次任务能从头跑到尾：取链路 → 信号 → 回测 → 渲染（自检通过）→ 台账；
  3. **迭代记账**：同一策略跑第二次，DSR 的 k 会 +1（"让点击次数可见"不是口号）；
  4. 服务端路由与拒绝语义（未知路径 404 / 坏请求 400 / 未知策略 404），且页面自包含。
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db as dbmod                                        # noqa: E402
import obs                                                # noqa: E402
from research import families, indicators, panel as panel_mod, presets  # noqa: E402
from research import render_report as RR                   # noqa: E402
from research.tests import fixture_market as FX             # noqa: E402
from webapp import catalog, jobs, server as srv_mod         # noqa: E402

WIN = ("2024-01-01", "2025-01-01")          # 合成市场的可评估窗


# ---------------------------------------------------------------- 目录页

def test_catalog_lists_every_registered_strategy():
    cat = catalog.build()
    assert len(cat["presets"]) == len(presets.PRESETS)
    assert len(cat["families"]) == len(families.MASK)
    for e in cat["presets"]:
        assert e["engine_spec"], "preset 必须能展出引擎口径（与 CLI/报告同一个函数）"
        assert e["n_comparisons"] >= 1
        assert "window" in e
    for f in cat["families"]:
        assert f["mechanism"], "族要能展出机制叙述（掩码 docstring 首行）"
        assert f["grid_size"] >= 1 or f["reference"], "非参考族必须有预注册网格"


def test_family_default_params_are_the_grid_midpoint():
    """族的默认参数必须是**网格中点**（确定性、不看结果）—— 页面上也这么写。"""
    e = catalog.find("family", "str")
    assert e["default_params"] == {k: v[len(v) // 2] for k, v in e["grid"].items()}


def test_catalog_flags_windows_that_touch_the_holdout():
    ho = tuple(catalog.HOLDOUT)
    assert catalog._overlaps(("2025-01-01", "2026-01-01"), ho) is True
    assert catalog._overlaps(("2024-01-01", "2025-01-01"), ho) is False
    cat = catalog.build()
    assert any(e["touches_holdout"] for e in cat["presets"]), "应当至少有一条 preset 含 holdout 天数"


# ---------------------------------------------------------------- 任务（离线合成市场）

@pytest.fixture
def offline(monkeypatch, tmp_path):
    """把"链路"换成合成市场的（同一段真代码，只是换了库），并把产物/台账都落到 tmp。"""
    db_file = FX.build(tmp_path / "fixture.db")
    monkeypatch.setattr(dbmod, "DB_PATH", db_file)
    P = presets.get_preset("us_reversal_dip_v1")
    chain = RR._chain_from(P)                     # 真代码路径，只是库是合成的
    monkeypatch.setattr(jobs.RR, "_chain_from", lambda _P: chain)
    monkeypatch.setattr(jobs, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    monkeypatch.setattr(obs, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(obs, "ALERT_LOG", tmp_path / "alerts.jsonl")
    jobs._JOBS.clear()
    return tmp_path


def _wait(job_id: str, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = jobs.status(job_id)
        assert st is not None
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.2)
    raise AssertionError("任务超时")


def test_a_run_goes_all_the_way_and_lands_in_the_ledger(offline):
    job = jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                       "lo": WIN[0], "hi": WIN[1], "note": "测试"})
    st = _wait(job["id"])
    assert st["status"] == "done", st.get("error")
    assert [s["name"] for s in st["steps"]][:2] == ["取链路（命中缓存则近 0 秒）",
                                                   st["steps"][1]["name"]]
    assert st["metrics"]["n_trades"] > 0
    assert st["selfcheck"] == [], f"报告自检未通过：{st['selfcheck']}"
    rep = offline / f"web_report_{job['id']}.html"
    assert rep.is_file() and rep.stat().st_size > 10_000
    assert 'class="kchart"' in rep.read_text(encoding="utf-8")
    led = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()]
    assert led[-1]["strategy_id"] == "preset:us_reversal_dip_v1"
    assert led[-1]["note"] == "测试" and led[-1]["n_trades"] == st["metrics"]["n_trades"]


def test_second_run_raises_k_the_multiplicity_is_visible(offline):
    """**迭代记账**：同一策略跑第二次 ⇒ DSR 的 k 从 N+1 起算，页面上显示得出来。"""
    first = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                               "lo": WIN[0], "hi": WIN[1]})["id"])
    second = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                                "lo": WIN[0], "hi": WIN[1]})["id"])
    k1, k2 = first["metrics"]["dsr_k"], second["metrics"]["dsr_k"]
    assert k2 == k1 + 1, f"k 没随点击次数上涨：{k1} → {k2}"
    assert second["metrics"]["k_breakdown"]["platform_runs_before"] == 1


def test_unknown_strategy_and_bad_window_surface_as_errors(offline):
    st = _wait(jobs.submit({"kind": "preset", "id": "nope_v9",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "error" and "nope_v9" in st["error"]
    st2 = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                             "lo": WIN[1], "hi": WIN[0]})["id"])
    assert st2["status"] == "error" and "窗口非法" in st2["error"]


def test_family_path_uses_the_registry_template(offline):
    """族型请求走的是"已注册 preset + 换掩码"那条路（与 CLI 报告同一做法）。"""
    st = _wait(jobs.submit({"kind": "family", "id": "str",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    assert st["hint"].startswith("本族网格")
    assert st["metrics"]["n_trades"] > 0


# ---------------------------------------------------------------- 服务端

@pytest.fixture
def http(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), srv_mod.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_serves_the_page_and_the_catalog(http):
    code, body = _get(http + "/")
    assert code == 200 and b"usagent \xe7\xad\x96\xe7\x95\xa5\xe5\xb9\xb3\xe5\x8f\xb0" in body
    assert b"<script" not in body.lower().replace(b"<script>", b"")     # 页面只有一段内联脚本
    code, body = _get(http + "/api/catalog")
    assert code == 200
    cat = json.loads(body)
    assert len(cat["presets"]) == len(presets.PRESETS)


def test_server_rejects_unknown_paths_and_bad_requests(http):
    assert _get(http + "/nope")[0] == 404
    assert _get(http + "/report/deadbeef")[0] == 404          # 合法 id 但没有报告文件
    assert _get(http + "/report/../../etc/passwd")[0] == 404  # 不接受路径
    req = urllib.request.Request(http + "/api/run", data=b"{not json",
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("坏 JSON 应当 400")
    except urllib.error.HTTPError as e:
        assert e.code == 400
    req = urllib.request.Request(http + "/api/run", method="POST",
                                 data=json.dumps({"kind": "preset", "id": "nope",
                                                  "lo": WIN[0], "hi": WIN[1]}).encode())
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("未知策略应当 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_server_binds_localhost_only():
    """**只绑 127.0.0.1**：这个页面能一键跑回测、读本地库、写报告文件。"""
    src = (ROOT / "webapp" / "server.py").read_text(encoding="utf-8")
    assert '("127.0.0.1", port)' in src and "0.0.0.0" not in src
