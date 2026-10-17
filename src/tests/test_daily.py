"""每日驱动器（`src/daily.py`）的测试 —— 不联网、不碰库。

2026-10-17 机构就绪审计第 ②④ 项：每日任务的池子必须**就是引擎读的那一份**，
且每一步都要留痕（此前 `logs/` 是空目录、失败只 echo）。
这里钉住四件事：池子/源来自 config 单一来源；dry-run 不产生副作用；
**失败继续跑后续步骤但仍然非零退出并告警**（这条语义是有意为之，见 daily 的 docstring）；
机器可读结论的解析容忍垃圾输入。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import daily  # noqa: E402
from config import DAILY_RECENT_DAYS, DAILY_SOURCE, DAILY_UNIVERSE  # noqa: E402


@pytest.fixture
def logs(tmp_path, monkeypatch):
    import obs
    monkeypatch.setattr(obs, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(obs, "ALERT_LOG", tmp_path / "alerts.jsonl")
    return tmp_path


def test_plan_takes_universe_and_source_from_the_single_source_of_truth():
    first = " ".join(daily.plan("py")[0]["argv"])
    assert f"--universe {DAILY_UNIVERSE}" in first
    assert f"--source {DAILY_SOURCE}" in first
    assert f"--recent {DAILY_RECENT_DAYS}" in first
    # 这条断言是这次修复的**要害**：池子必须是引擎真正读的那份
    assert DAILY_UNIVERSE == "all", "每日任务必须覆盖全库（引擎读的是全库，不是 oi100）"
    assert DAILY_SOURCE == "stockanalysis", "默认源必须是在本机真的能用的那个"


def test_validate_step_writes_a_machine_readable_report():
    step = daily.plan("py")[1]
    assert step["stage"] == "数据校验"
    assert "--json" in step["argv"] and step["artifacts"], "校验步必须产出机器可读结论"


def test_dry_run_prints_the_plan_without_side_effects(logs, capsys):
    assert daily.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "download_ohlcv.py" in out and "validate.py" in out
    assert not (logs / "runs.jsonl").exists(), "dry-run 不许写台账"


class _CP:
    def __init__(self, rc, out=""):
        self.returncode, self.stdout, self.stderr = rc, out, ""


def test_failure_continues_but_exits_nonzero_and_alerts(logs, monkeypatch):
    """**保留的语义**：某步失败仍跑后续步骤（否则分不清"取数没跑成"与"库真的坏了"），
    但要记清楚、要告警、要非零退出。"""
    calls = []

    def _fake_run(argv, **kw):
        # ⚠️ 只拦**步骤**调用：`obs.git_sha()` 也走 subprocess.run，一并拦下会把
        # git 调用算成一次步骤（实测踩过：断言 2 次却数到 8 次）。
        name = " ".join(argv)
        if "download_ohlcv" not in name and "validate.py" not in name:
            return _CP(0, "")                       # git 等旁路调用
        calls.append(name)
        return _CP(1, "probe 不通") if "download_ohlcv" in name else _CP(0, "18/18 通过")

    monkeypatch.setattr(daily.subprocess, "run", _fake_run)
    rc = daily.main(["--quiet"])
    assert rc == 1, "有步骤失败必须非零退出"
    assert len(calls) == 2, "失败之后仍要跑后续步骤（这是有意保留的语义）"
    runs = [json.loads(x) for x in (logs / "runs.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in runs] == [False, True, False]      # 两步 + 汇总
    alerts = [json.loads(x) for x in (logs / "alerts.jsonl").read_text().splitlines()]
    assert alerts and alerts[0]["level"] == "error"
    assert any("OHLCV 增量" in a["title"] for a in alerts)


def test_all_green_logs_a_summary_and_returns_zero(logs, monkeypatch):
    monkeypatch.setattr(daily.subprocess, "run",
                        lambda argv, **kw: _CP(0, "ok" if "--short" not in argv else ""))
    assert daily.main(["--quiet"]) == 0
    runs = [json.loads(x) for x in (logs / "runs.jsonl").read_text().splitlines()]
    assert runs[-1]["stage"] == "每日任务" and runs[-1]["ok"] is True


def test_metrics_from_checks_parses_and_tolerates_garbage(tmp_path):
    good = tmp_path / "c.json"
    good.write_text(json.dumps({"passed": 17, "total": 18, "failed": ["覆盖目标池（池内）"],
                                "universe": "all"}), encoding="utf-8")
    m = daily._metrics_from_checks(good)
    assert m["checks_passed"] == 17 and m["checks_failed"] == ["覆盖目标池（池内）"]
    bad = tmp_path / "bad.json"
    bad.write_text("{ 不是 json", encoding="utf-8")
    assert daily._metrics_from_checks(bad) is None
    assert daily._metrics_from_checks(tmp_path / "missing.json") is None


def test_stale_artifact_is_not_attributed_to_this_run(tmp_path, monkeypatch):
    """**上一次**的校验结论不许被当成本次的 —— 失败步骤的台账里不该出现 checks_*。

    对抗核验实测：产物写失败时旧文件会留在磁盘上，于是"步骤失败(ok=false)"那一行
    却带着上一次的 passed=18/18 —— 自相矛盾，而且事后无法分辨。
    """
    import json
    import obs as obs_mod
    monkeypatch.setattr(obs_mod, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(obs_mod, "ALERT_LOG", tmp_path / "alerts.jsonl")
    art = tmp_path / "checks.json"
    art.write_text(json.dumps({"passed": 18, "total": 18, "failed": []}), encoding="utf-8")
    monkeypatch.setattr(daily, "CHECKS_JSON", art)

    def _fake_run(argv, **kw):
        name = " ".join(argv)
        if "validate.py" in name:                 # 校验步失败，且**没有**重写产物
            return _CP(1, "坏了")
        return _CP(0, "")

    monkeypatch.setattr(daily.subprocess, "run", _fake_run)
    assert daily.main(["--quiet"]) == 1
    runs = [json.loads(x) for x in (tmp_path / "runs.jsonl").read_text().splitlines()]
    step2 = [r for r in runs if "校验" in r["stage"]][0]
    assert step2["ok"] is False
    assert "checks_passed" not in step2, "把上一次的结论当成本次的了"
