"""观测层（`src/obs.py`）的测试 —— 它必须是**哑的**：观测失败不许把主流程弄挂。

2026-10-17 机构就绪审计第 ④ 项：`logs/` 曾是空目录、失败只 echo 到 stdout
⇒ "跑完了/跑失败了"事后无法区分。这一层补上台账与告警，但它的纪律是：
**写不进去就留一句 stderr，绝不冒泡**。所以这里的测试重点是"坏情况下不抛"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import obs  # noqa: E402


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(obs, "ALERT_LOG", tmp_path / "alerts.jsonl")
    return tmp_path


def _lines(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()]


def test_run_log_appends_one_json_line_with_hard_fields(logs):
    obs.log_run("取数", True, seconds=1.234, exit_code=0, rows=10)
    obs.log_run("校验", False, seconds=0.5, exit_code=1, detail="boom")
    rows = _lines(logs / "runs.jsonl")
    assert [r["stage"] for r in rows] == ["取数", "校验"]
    assert rows[0]["ok"] is True and rows[0]["seconds"] == 1.234 and rows[0]["rows"] == 10
    assert rows[1]["ok"] is False and rows[1]["exit_code"] == 1
    for r in rows:                                   # 每行都要能回答"哪份代码跑的、什么时候"
        assert r["kind"] == "run" and r["ts"] and r["git_sha"]


def test_alert_writes_a_line_and_a_stderr_note(logs, capsys):
    p = obs.alert("error", "取数失败", "probe 不通")
    assert p == logs / "alerts.jsonl"
    rec = _lines(logs / "alerts.jsonl")[0]
    assert rec["level"] == "error" and rec["title"] == "取数失败"
    assert "取数失败" in capsys.readouterr().err


def test_no_webhook_configured_means_no_network(logs, monkeypatch):
    """仓库里不放投递地址：没配环境变量就**只落盘**，不许偷偷发请求。"""
    def _boom(*a, **k):                              # pragma: no cover
        raise AssertionError("没配 webhook 时不该有任何网络调用")

    monkeypatch.delenv(obs.WEBHOOK_ENV, raising=False)
    monkeypatch.setattr(obs.urllib.request, "urlopen", _boom)
    obs.alert("warn", "只是一个警告")


def test_webhook_env_is_the_only_place_the_url_lives(logs, monkeypatch):
    seen = {}

    class _Resp:
        def close(self): ...

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setenv(obs.WEBHOOK_ENV, "https://example.invalid/hook")
    monkeypatch.setattr(obs.urllib.request, "urlopen", _fake_urlopen)
    obs.alert("error", "t", "d")
    assert seen["url"] == "https://example.invalid/hook"
    assert seen["body"]["title"] == "t" and seen["body"]["git_sha"]


def test_unwritable_log_does_not_raise(tmp_path, monkeypatch, capsys):
    """观测写失败 ⇒ 只留 stderr。**这条是这一层的存在理由的反面**：
    如果告警写不进去会把主流程弄挂，那运维更不敢接它。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("我是一个文件，不是目录", encoding="utf-8")
    monkeypatch.setattr(obs, "RUN_LOG", blocker / "sub" / "runs.jsonl")
    monkeypatch.setattr(obs, "ALERT_LOG", blocker / "sub" / "alerts.jsonl")
    obs.log_run("取数", True)                        # 不抛
    obs.alert("error", "t")                          # 不抛
    assert "观测写入失败" in capsys.readouterr().err


def test_manifest_carries_the_hard_fields():
    m = obs.manifest(data_vintage="rows=1;digest=abc")
    assert set(m) >= {"ts", "git_sha", "python", "platform", "data_vintage"}
    assert m["data_vintage"] == "rows=1;digest=abc"


def test_git_sha_never_raises_outside_a_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "REPO", tmp_path)
    assert obs.git_sha() == "unknown"
