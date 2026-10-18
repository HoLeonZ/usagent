"""策略平台的测试 —— **离线**（确定性合成市场 + 真 HTTP 服务在随机端口）。

钉住四件事：
  1. 目录页的数据**来自注册表本身**（数得出来的 preset/族条数、每条都有参数与负面证据）；
  2. 一次任务能从头跑到尾：取链路 → 信号 → 回测 → 渲染（自检通过）→ 台账；
  3. **迭代记账**：同一策略跑第二次，DSR 的 k 会 +1（"让点击次数可见"不是口号）；
  4. 服务端路由与拒绝语义（未知路径 404 / 坏请求 400 / 未知策略 404），且页面自包含。
"""
from __future__ import annotations

import json
import re
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


def test_catalog_flags_definitions_changed_after_the_process_started(tmp_path, monkeypatch):
    """改了定义体、服务没重启 ⇒ 服务读的是旧值，界面必须**主动说**，不能只靠一句提示。

    这条路是静默的，而且实测有更坏的版本：同秒内改动又改回、文件长度不变时，Python 认为
    旧 .pyc 仍有效 —— 进程于是用了**仓库里不存在的值**（我自己的验证脚本踩出过这一幕）。
    """
    d = tmp_path / "research" / "preset_defs"
    d.mkdir(parents=True)
    f = d / "x.py"
    f.write_text("NAME = 'x'\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    # 比"启动时刻"更旧 ⇒ 不算陈旧
    assert catalog.stale_definitions(started_at=time.time() + 10) == []
    # 比"启动时刻"更新 ⇒ 陈旧，且路径是仓库相对形式
    assert catalog.stale_definitions(started_at=0.0) == ["research/preset_defs/x.py"]


def test_catalog_publishes_the_windows_from_their_real_sources():
    """页面上的窗口预设必须**取自出处**，不能在 JS 里抄一份。

    HOLDOUT 还担着"与 holdout 重叠"的警示判据 —— 抄错/漂移就会错报（把一次正常回测标红，
    或者更糟：放过一次真的碰了 holdout 的运行）。TRAIN 的出处是 WFV 的搜索期。
    """
    from research import wfv_search as W
    cat = catalog.build()
    assert cat["holdout"] == list(catalog.HOLDOUT), "holdout 必须就是 holdout 台账那个常量"
    assert cat["train"] == [W.SEARCH_LO, W.SEARCH_HI], "训练窗必须取自 wfv_search 的搜索期"

    # 而且页面真的**用了**它（不是发布了但没人读 —— 本仓库为"声明了却没人读"立过规矩）
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    assert "if (CAT.holdout) HOLDOUT = CAT.holdout" in html
    assert "if (CAT.train) TRAIN = CAT.train" in html


def test_holdout_touches_counts_what_the_page_warns_about():
    """卡片上写着「只准读一次」，那就得给出**被跨过几次** —— 否则它只是句口号。

    这条计数要覆盖**全台账**（不是显示的那 30 条）：它是纪律计数，不是表的一部分。
    """
    ho = tuple(catalog.HOLDOUT)
    rows = [{"window": [ho[0], ho[1]]},                 # 完全落在 holdout 里
            {"window": ["2024-01-01", "2025-01-01"]},   # 不碰
            {"window": ["2025-10-01", "2025-10-20"]},   # 边跨过去
            {"window": None},                           # 老记录可能没有窗口
            {}]
    assert catalog.holdout_touches(rows) == 2, catalog.holdout_touches(rows)
    assert catalog.holdout_touches([]) == 0


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
        if st["status"] in ("done", "error", "cancelled"):
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


def test_a_queued_run_can_be_cancelled_and_is_not_recorded(offline):
    """取消要真的生效，且**不进台账** —— k 只数"被看过的试验"，不数花掉的 CPU。

    为什么用"排队中"这条路径测：运行本身只要一两秒，抢时序必然 flaky；而排队是确定的 ——
    先握住重活锁，提交的任务必然停在 queued，取消它、再放锁，结果必然 cancelled。
    （粗粒度的诚实边界：正在跑的那一步跑完才停，这条断言只覆盖"还没开始跑"的那段。）
    """
    jobs._LOCK.acquire()
    try:
        jid = jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                           "lo": WIN[0], "hi": WIN[1]})["id"]
        assert jobs.status(jid)["status"] == "queued"
        assert jobs.cancel(jid) is True
        assert jobs.cancel(jid) is True, "重复请求取消应当是幂等的（任务还没结束）"
    finally:
        jobs._LOCK.release()
    st = _wait(jid)
    assert st["status"] == "cancelled", st
    assert "未记账" in st["error"], st["error"]
    # 台账里不能有它留下的任何一行
    led = offline / "web_runs.jsonl"
    assert not led.exists() or "cancelled" not in led.read_text(encoding="utf-8")
    assert jobs.prior_runs("preset:us_reversal_dip_v1") == 0, "取消的运行不该推高 k"


def test_cancel_refuses_unknown_or_finished_jobs(offline):
    assert jobs.cancel("deadbeefdead") is False, "未知任务不该被受理"
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done"
    assert jobs.cancel(st["id"]) is False, "已经跑完的任务不该被受理"


def test_the_hint_about_k_matches_the_k_that_was_actually_used(offline):
    """提示里那句「k 从这里起算」必须与 DSR 真正用的 k 是同一个数。

    验收实测漂移过一次：`us_reversal_dip_v1` 是**内部用族掩码的 preset**，提示按"它用了族"
    报了「本族网格 96 组 ⇒ k 从这里起算」，而 DSR 用的是它声明的 n_comparisons（1614）——
    同一份报告里 k 是 1622。根因是"用哪条信号路径"和"k 怎么算"两件事被写在同一个分支里。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    mb = st["metrics"]["k_breakdown"]
    declared = mb["declared_or_grid"]
    assert f"n_comparisons={declared}" in st["hint"], f"提示与 k 不是同一个数：{st['hint']}"
    assert st["metrics"]["dsr_k"] == declared + mb["platform_runs_before"]
    # 族型请求才该说"网格组数"
    st2 = _wait(jobs.submit({"kind": "family", "id": "str",
                             "lo": WIN[0], "hi": WIN[1]})["id"])
    assert "本族网格" in st2["hint"] and st2["metrics"]["dsr_k"] == \
        st2["metrics"]["k_breakdown"]["declared_or_grid"] + 0


def test_every_run_leaves_a_machine_readable_trades_csv(offline):
    """报告是给人看的；逐笔那 20 列必须还有一份**脚本能直接读**的。

    此前只有 1.9MB 的自包含 HTML —— 想把逐笔拿去做别的计算只能从 HTML 里抠。
    这里连"真的能读"都测了（`read_csv(comment="#")`），因为一份带注释行的 CSV 如果
    注释写得不规矩，等于没导出。
    """
    import pandas as pd
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    csv = offline / f"web_trades_{st['id']}.csv"
    assert csv.is_file(), "没有逐笔 CSV"
    text = csv.read_text(encoding="utf-8")
    head = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert len(head) >= 5 and any("as-traded" in h for h in head), head   # 口径必须随文件走
    assert any("read_csv" in h for h in head), "要告诉读者怎么读"
    df = pd.read_csv(csv, comment="#")
    assert len(df) == st["metrics"]["n_trades"], "CSV 行数与成交笔数不一致"
    for col in ("entry_date", "exit_date", "symbol", "exit_reason", "entry_price",
                "exit_price", "net_pnl", "net_return"):
        assert col in df.columns, f"少了 {col}"
    assert st["trades_csv"] == f"/api/trades/{st['id']}.csv"


def test_trades_csv_route_serves_only_known_jobs(http, tmp_path, monkeypatch):
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    (tmp_path / "web_trades_abc123def456.csv").write_text(
        "# usagent 逐笔明细\nentry_date,exit_date\n2025-01-02,2025-01-03\n", encoding="utf-8")
    code, body = _get(http + "/api/trades/abc123def456.csv")
    assert code == 200 and body.decode().startswith("# usagent")
    assert _get(http + "/api/trades/deadbeefdead.csv")[0] == 404, "不存在的任务要 404"
    assert _get(http + "/api/trades/..%2F..%2Fetc%2Fpasswd.csv")[0] == 404, "路径不许拼"


def test_unknown_strategy_and_bad_window_surface_as_errors(offline):
    st = _wait(jobs.submit({"kind": "preset", "id": "nope_v9",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "error" and "nope_v9" in st["error"]
    st2 = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                             "lo": WIN[1], "hi": WIN[0]})["id"])
    assert st2["status"] == "error" and "窗口非法" in st2["error"]


def test_a_window_with_no_trading_days_fails_fast_and_names_the_range(offline):
    """空窗口要在**信号之前**就被挡下，且报错必须说出合法范围（审计实测两样都没有）。

    原本的路径：链路面板载入 → 算信号（7–11 秒）→ broker 抛「窗口内没有任何交易日」，
    而这句话没告诉读者该填什么。交易日轴在载完面板之后几乎不值钱，就在那一步问。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": "1999-01-01", "hi": "1999-02-01"})["id"])
    assert st["status"] == "error", st
    assert "没有任何交易日" in st["error"] and "覆盖" in st["error"], st["error"]
    assert "1999" in st["error"]
    # 而且**没跑信号**（那一步的名字不该出现在步骤里）
    assert not any("计算信号" in s2["name"] for s2 in st["steps"]), st["steps"]


def test_a_run_reports_the_pool_range_so_the_page_can_warn_before_running(offline):
    """跑过一次之后，页面要知道"这个池子哪一段可评估" —— 据此在窗口旁显示范围并当场预警。"""
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    lo, hi = st["pool_range"]
    # 起止都必须是合法日期且起点早于终点（不写死具体年份：合成市场的跨度是 fixture 的事）
    import pandas as pd
    lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
    assert lo_ts < hi_ts, st["pool_range"]
    # 这条范围必须与刚跑过的窗口**重叠**（这正是页面拿它做的那个判断）—— 否则"可评估范围"
    # 与"刚跑出成交的窗口"互相矛盾。注意不能断言"含住窗口"：数据尾端早于窗口右端是常态
    # （合成市场到 2024-12-27，而评测窗到 2025-01-01），那种情况本来就只是"后面没数据"。
    assert lo_ts < pd.Timestamp(WIN[1]) and hi_ts > pd.Timestamp(WIN[0]), st["pool_range"]


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


def test_no_literal_markdown_leaks_into_visible_copy():
    """页面上的可见文案不许出现 `**` —— 那是 markdown 的强调，浏览器不认识，会原样打出来。

    同一个坑在这个仓库踩过三次（报告层靠 `inline_md` 先转义再做最小 markdown 解决，
    页面层一直没人管）。这条锁只看**标记之外**的部分：`<script>`（JS 注释里可以有）与
    `<style>`（CSS 注释里可以有）先剥掉。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    body = re.sub(r"<script>.*?</script>", "", html, flags=re.S)
    body = re.sub(r"<style>.*?</style>", "", body, flags=re.S)
    offenders = [ln.strip()[:70] for ln in body.splitlines() if "**" in ln]
    # ⚠️ 这个盲区我自己踩过第四次：**JS 生成的文案**（模板字符串拼出来的 HTML）不在静态标记里，
    # 上面的括号扫不到它。所以 script 里也要扫 —— 先剥掉注释（注释里可以有 markdown），
    # 再看剩下的还有没有 `**`。JS 里的 `**` 只可能是幂运算或文案，这个页面没有前者。
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    js = m.group(1) if m else ""
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    js = "\n".join(re.sub(r"(?<![:/])//.*$", "", ln) for ln in js.splitlines())
    offenders += [ln.strip()[:70] for ln in js.splitlines() if "**" in ln]
    assert not offenders, f"可见文案里有 markdown 的 **：{offenders}"


def test_run_counts_come_from_the_whole_ledger_not_the_shown_window(http, tmp_path, monkeypatch):
    """卡片「已跑 N 次」必须与 DSR 的 k 用**同一个计数**（都按全文台账数）。

    页面上的表只显示最近 30 条，而 k 用的是全文（`prior_runs` 逐行数）。此前卡片次数是从
    `/api/runs` 那 30 条里现数的 ⇒ 到第 31 次运行时界面会说"已跑 30"而 k 已经是 31：
    同一个量的两个数字（本仓库最忌讳的那种）。这里造 35 条，逼出这个分叉。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    lines = [json.dumps({"ts": f"2026-10-0{1+i%9}T00:00:0{i%9}", "strategy_id": "preset:x",
                         "n_trials_k": 10 + i, "excess": 0.01 * i, "window": ["2025-01-01", "2025-10-13"]})
             for i in range(35)]
    (tmp_path / "web_runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    d = json.loads(_get(http + "/api/runs")[1])
    assert d["total"] == 35 and d["shown"] == 30, d
    assert d["by_strategy"] == {"preset:x": 35}, d["by_strategy"]
    # 而且它与 k 的记账读的是同一份东西
    assert jobs.prior_runs("preset:x") == 35


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
        "备注输入框（台账那一列原本没人能填）": 'id="note"',
        "复制的命令带上运气对照开关（否则 CLI 默认跑 200 次）": "--control-runs ${ctl}",
        "复制的命令产物与平台同一命名空间（跑完页面上看得见）": "web_report_cli_",
        "窗口按钮自带「按下去会设成什么」": 'title="${ww[0]} .. ${ww[1]}"',
        "族那个空操作按钮已禁用并说明原因（声明窗是 preset 的属性）": "（族没有声明窗）",
        "两个日期输入可被读屏区分": 'aria-label="窗口起点"',
        "搜索 0 命中时分节标题说实话（原本还写着「6 条」）": "这一节没有命中",
        "窗口旁显示池子可评估范围（跑过一次后就有）": "池子可评估范围",
        "与池子范围不重叠时当场预警（不必白等一次）": "完全不重叠",
        "族卡的批量入口（整张网格一次跑完）": "copyGridCmd(",
        "族卡说清「网格才是一次搜索」（别把 96 个格点各点一次）": "一次搜索的完整空间",
        "台账按策略筛选（对比时要能只看两条）": 'id="lf"',
        "台账 Δ 列（同一配置重跑的对比）": '["d_excess","Δ超额（同配置）"]',
        "台账完整 CSV 导出（表只有 30 条上限）": '/api/runs.csv',
        "台账窗口可点回填到卡片（但不自动跑）": "fillWindow(",
        "台账明说只显示最近 N 条": "表只取最近",
        "卡片区排序（靠移动 DOM，不重渲 ⇒ 不吞用户输入）": 'id="csort"',
        "卡片区结构化过滤（只看跑过的 / 只看含 holdout）": 'id="cf-run"',
        "结构化过滤也算「在筛」（标题不能再说「6 条」）": "filtering = !!(q || onlyRun || onlyHeld)",
        "台账列头是真按钮（键盘可达）": 'class="thbtn"',
        "排序状态可被读屏读到": 'aria-sort=',
        # 2026-10-18 第三轮：滚走不失联 + 窄屏
        "顶栏「有任务在跑」徽标（滚到哪儿都看得见）": 'id="busy"',
        "徽标可点回结果": 'href="#job-',
        "徽标在轮询失败/任务丢失时也会摘掉（不留假在跑）": "trackInflight(jobId, false)",
        "超长机器串可断（实测 390px 下曾顶破 47px）": "color:#cbd5e1;overflow-wrap:anywhere",
        "窄屏压缩顶栏（sticky 的每一行都从正文永久扣走）": "@media (max-width:560px)",
        "运行中可取消（粗粒度，界面写明）": "function cancelJob(",
        "取消的终止态有专门分支（否则会掉进 metrics 渲染）": 'st.status === "cancelled"',
        "取消说清「不进台账」": "已取消的运行<b>不进台账</b>",
        "卡片显示最近一次读数（开发时第一个问题是「哪条最好」）": "function lastRunLine(",
        "读数写清是「最近一次」而不是策略水平": "不是这条策略的水平",
        "可按最近一次超额排序（没跑过的排最后）": 'data-last=',
        "顶栏「重新读目录」（页面旧 vs 服务旧要分清）": "reloadCatalog()",
        "改完定义体要说清「要重启服务」（注册表启动时读一次）": "srchint-",
        "holdout 的次数要显眼（一句没有计数的纪律等于没有纪律）": "只准读一次",
        "定义体在服务启动后被改过要主动警告（不只是提示）": "CAT.stale_defs",
        "逐笔 CSV 下载（报告是给人看的，这份给脚本）": "st.trades_csv",
        "历史报告列表里的逐笔 CSV 列": "r.has_trades",
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


def test_runs_route_counts_holdout_touches_over_the_whole_ledger(http, tmp_path, monkeypatch):
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    ho = list(catalog.HOLDOUT)
    lines = [json.dumps({"strategy_id": "preset:x", "window": ho}),
             json.dumps({"strategy_id": "preset:x", "window": ["2024-01-01", "2025-01-01"]}),
             json.dumps({"strategy_id": "preset:x", "window": ["2025-11-01", "2025-12-01"]})]
    (tmp_path / "web_runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    d = json.loads(_get(http + "/api/runs")[1])
    assert d["holdout_touches"] == 2, d          # 1 与 3 都落在 holdout 内
    assert d["total"] == 3


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


def test_a_report_can_be_opened_by_file_name_not_only_by_job_id(http, tmp_path, monkeypatch):
    """CLI `--out` 的产物不是任务 id ⇒ 必须能按**文件名**打开，且遍历必须被拒。

    「复制命令」与页面做的事已经逐项对齐（--control-runs/--note/同一个命名空间），
    但那条命令产出的文件名不是 12 位任务 id，用不了 `/report/<id>`。少了这条路由，
    复制出来的命令跑完，报告照样在页面上看不见。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    made = tmp_path / "web_report_cli_xs_flow_2025-01-01_2025-10-13.html"
    made.write_text("<!doctype html><title>回测明细 · family:xs_flow</title>"
                    "<div class='sub'>窗口 2025-01-01 → 2025-10-13</div>", encoding="utf-8")
    code, body = _get(http + "/report-file/" + made.name)
    assert code == 200 and "xs_flow" in body.decode()
    # 它也会出现在「历史报告」里，并且带 file 字段（表格用它拼链接）
    reps = json.loads(_get(http + "/api/reports")[1])["reports"]
    assert [r["file"] for r in reps] == [made.name]
    # 遍历与越界：一律拒（文件名里不许有 /，也不许跳出 results/）
    assert _get(http + "/report-file/../../etc/passwd")[0] == 404
    assert _get(http + "/report-file/sub/dir.html")[0] == 404
    assert _get(http + "/report-file/nope.html")[0] == 404


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
