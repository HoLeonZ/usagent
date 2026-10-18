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


def test_ledger_records_a_real_duration_and_the_params_that_ran(offline):
    """台账的两个"记了却用不上"的字段：seconds 曾恒为 null、params 有值却没渲染。

    seconds 的坑很典型：那一行在 `finally` **之前**读 `job["seconds"]`，而它正是 finally
    才赋的值 ⇒ 9 条台账全是 null（实测）。记了却永远用不上的字段比不记更坏 —— 它看起来
    像"有这件事"，让下一个人以为能拿它做统计。
    """
    st = _wait(jobs.submit({"kind": "family", "id": "str",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    rec = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()][-1]
    assert rec["seconds"] is not None and rec["seconds"] > 0, f"seconds 又变成 {rec['seconds']}"
    assert rec["params"], "族型运行必须记下**实际生效**的那组参数（网格中点）"


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


# ---------------------------------------------------------------- 按钮/导航清单（本循环的机器化）

def test_the_page_has_the_affordances_development_and_backtesting_need():
    """页面必须齐备「开发 / 回测」所需的按钮与导航 —— 少一个就红。

    为什么写成清单断言：这一层没有 JS 单测，"按钮还在不在"只能靠读页面源码；
    而**漏掉一个按钮**（例如"复制成可复现命令"）不会让任何别的东西变红 ——
    那正是本仓库最忌讳的"新东西默认不在护栏里"。清单本身就是这份护栏。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    need = {
        "搜索框（21 张卡片里找一条）": 'type="search"',
        "顶栏常驻导航": 'class="top"',
        "分段锚点": 'href="#presets"',
        "卡片锚点（可深链）": 'id="${a}"',
        "窗口预设（训练/全窗/声明/holdout）": "windowPresets(",
        "窗口非法即时提示": "窗口非法（起点必须早于终点）",
        "回测按钮": "onclick=\"run('${e.kind}','${e.id}')\"",
        "复制成可复现命令": "copyCmd(",
        "查看定义体源码": "showSrc(",
        "族参数可编辑（只列预注册网格取值）": 'select class="grid"',
        "台账可排序（对比两次运行）": "sortLedger(",
        "holdout 重叠标红": "与 holdout 重叠",
        "报告下载": "download>",
        "报告内嵌": "<iframe",
        # 2026-10-18 UX 审计补的四件（每条都有实测证据，见 webapp/jobs.py 的注释）
        "历史报告入口（结果可回访，不必重跑）": 'href="#reports"',
        "台账的报告列（回访某一次）": 'href="/report/${esc(r.job_id)}"',
        "在途守卫（双击不静默多花一次 k）": "btn.disabled = true; btn.textContent",
        "结果按 job 分块（并发不互相覆盖）": '"job-" + jobId',
        "已用秒数（冷启动不像假死）": "已用 ${el.toFixed(0)}s",
        "锚点不被常驻顶栏吃掉": "scroll-margin-top",
        "运气对照开关（标了耗时，默认关）": 'id="ctl"',
        # 2026-10-18 第二轮 UX 审计（5 视角并行取证）补的（每条都有实测数字）
        "长参数串不再撑破整页（实测 2329px→1200px）": "overflow-wrap:anywhere",
        "宽表只让自己横滚，不顶破整页": 'class="scrollx"',
        "提交失败不再静默（实测原来永远停在「正在回测…」）": "function fail(msg, btn)",
        "读目录失败说得出原因并给重试": "读不到目录：",
        "读台账失败说得出原因": "读台账失败：",
        "轮询连续失败要明说（不是无声重试）": "读不到任务状态（连续",
        "台账参数列（对比的前提）": '["params","参数"]',
        "台账的「超额」写清是曝光匹配口径": '["excess","曝光匹配超额"]',
        "等待文案可被读屏读到": 'role="status"',
        "运气对照真跑（共用 control_arm，不是第二份）": "control_runs: controlRuns",
    }
    missing = [k for k, v in need.items() if v not in html]
    assert not missing, f"页面缺这些按钮/导航：{missing}"


def test_source_route_serves_the_strategy_file_and_refuses_traversal(http):
    """「查看定义体」要有只读路由；且**不许**拼用户输入做路径（遍历必须 404）。"""
    if not (ROOT / "research" / "preset_defs" / "us_reversal_dip_v1.py").is_file():
        pytest.skip("定义体不在（不应发生）")
    code, body = _get(http + "/api/source/preset/us_reversal_dip_v1")
    assert code == 200
    d = json.loads(body)
    assert d["path"] == "research/preset_defs/us_reversal_dip_v1.py"
    assert "PRESET" in d["text"] and d["bytes"] == len(d["text"])
    assert _get(http + "/api/source/family/nope")[0] == 404
    assert _get(http + "/api/source/preset/..%2F..%2Fetc%2Fpasswd")[0] == 404
    assert _get(http + "/api/source/preset/" + "a" * 80)[0] == 404      # 超长名字也拒


def test_reports_route_lists_past_reports_and_parses_their_header(http, tmp_path, monkeypatch):
    """`/api/reports` 是"结果可回访"的服务端一半：文件名就是 id，名字与窗口**从报告自身读回**。

    为什么要从报告头部读：报告本来就有 title 与副标题（策略名/窗口），另存一份元数据
    就是"一份事实两处"（本仓库的老毛病）。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    (tmp_path / "web_report_abc123def456.html").write_text(
        "<!doctype html><title>回测明细 · family:xs_flow</title>"
        "<div class='sub'>窗口 2025-01-01 → 2025-10-13 ｜ 成交 28 笔</div>", encoding="utf-8")
    (tmp_path / "not_a_report.html").write_text("<title>别的</title>", encoding="utf-8")
    code, body = _get(http + "/api/reports")
    assert code == 200
    reps = json.loads(body)["reports"]
    assert [r["job_id"] for r in reps] == ["abc123def456"], "只该列 web_report_*"
    assert reps[0]["name"] == "family:xs_flow"
    assert reps[0]["window"] == ["2025-01-01", "2025-10-13"]


def test_reports_route_carries_the_params_that_run_used(http, tmp_path, monkeypatch):
    """报告列表要能回答"这份是哪组参数跑出来的" —— 参数**不在报告里**，在台账里。

    所以按 job_id 回读台账，而不是从渲染好的 HTML 里正则抓（那是从产物反推事实，
    一份事实两处）；实测同一族两次不同参数（{0.34,0.5,0.1} vs {0.2,0,0.05}）在列表上
    原本完全无法分辨 —— 而这个平台自己就宣传"两条策略的对比看这里"。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    (tmp_path / "web_report_deadbeef0001.html").write_text(
        "<!doctype html><title>回测明细 · family:xs_flow</title>"
        "<div class='sub'>窗口 2025-01-01 → 2025-10-13 ｜ 成交 33 笔</div>", encoding="utf-8")
    (tmp_path / "web_runs.jsonl").write_text(json.dumps(
        {"job_id": "deadbeef0001", "strategy_id": "family:xs_flow", "seconds": 34.3,
         "params": {"gap5_max": 0.34, "vol_z_min": 0.5}}) + "\n", encoding="utf-8")
    reps = json.loads(_get(http + "/api/reports")[1])["reports"]
    assert [r["job_id"] for r in reps] == ["deadbeef0001"]
    assert reps[0]["params"] == {"gap5_max": 0.34, "vol_z_min": 0.5}


def test_control_arm_can_be_switched_on_and_lands_in_the_report(offline):
    """运气对照必须能从平台开起来，并**写进报告** —— 那是把运气与本事分开的唯一判词。

    它此前在平台上结构性开不了（`_summary` 把 ctl_* 写死成 NaN），报告因此永远打印
    "本次未跑"。默认仍然是关（每次约 25 秒），但必须**有入口**。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1], "control_runs": 20})["id"])
    assert st["status"] == "done", st.get("error")
    rep = (offline / f"web_report_{st['id']}.html").read_text(encoding="utf-8")
    assert "位于随机分布的" in rep, "开了对照却没在报告里出判词"
    assert "运气对照本次未跑" not in rep
    assert any("运气对照" in s["name"] for s in st["steps"]), "步骤里没写这一步"
    led = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()]
    assert led[-1]["control_runs"] == 20, "台账要记下这次有没有跑对照"


def test_default_run_says_the_control_was_not_run(offline):
    """默认关时，报告必须**明说未跑**（不能留个 NaN 让它渲染成 "nan 百分位"）。"""
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    rep = (offline / f"web_report_{st['id']}.html").read_text(encoding="utf-8")
    assert "运气对照本次未跑" in rep and "nan 百分位" not in rep
