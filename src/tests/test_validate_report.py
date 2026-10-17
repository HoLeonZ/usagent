"""`validate.py` 的两处改动（池子来源 + 机器可读出口）的测试 —— 不需要真库。

2026-10-17 机构就绪审计：闸门的"目标池"写死 `oi100`（596 只），而引擎读全库 3028 只
⇒ 覆盖率这个最该有牙的检查只盖住约 20% 的面板，且扩池后会因为**注定没有数据**的
退市/改名名字每天报红（告警疲劳）。这里钉住：默认池来自 config 单一来源、
覆盖率按池内判定、机器可读结论在任何返回路径上都落盘。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import validate  # noqa: E402
from config import DAILY_UNIVERSE  # noqa: E402


def test_default_universe_is_the_one_the_engine_reads():
    assert DAILY_UNIVERSE == validate._UNIVERSE_NAME == "all", \
        "校验闸门的目标池必须就是引擎真正读的那一份（否则闸门只是看起来在跑）"


def test_json_out_defaults_to_none():
    assert validate._JSON_OUT is None, "不在 main() 里跑就不该偷偷写文件"


def test_write_report_writes_counts_and_every_check(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "_JSON_OUT", tmp_path / "checks.json")
    monkeypatch.setattr(validate, "_UNIVERSE_NAME", "all")
    monkeypatch.setattr(validate, "CHECKS", [("甲", True, "ok"), ("乙", False, "坏了")])
    validate._write_report()
    d = json.loads((tmp_path / "checks.json").read_text(encoding="utf-8"))
    assert d["passed"] == 1 and d["total"] == 2 and d["failed"] == ["乙"]
    assert d["universe"] == "all" and d["ts"]
    assert [c["name"] for c in d["checks"]] == ["甲", "乙"]


def test_summary_exit_code_matches_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(validate, "_JSON_OUT", tmp_path / "c.json")
    monkeypatch.setattr(validate, "CHECKS", [("甲", True, ""), ("乙", True, "")])
    assert validate._summary() == 0
    monkeypatch.setattr(validate, "CHECKS", [("甲", True, ""), ("乙", False, "x")])
    assert validate._summary() == 1
    assert "1/2 通过" in capsys.readouterr().out


def test_report_write_failure_does_not_change_the_verdict(tmp_path, monkeypatch, capsys):
    """报告写不出来 ⇒ 只留 stderr。**不许**把"数据是好的"说成"校验失败"。"""
    blocker = tmp_path / "b"
    blocker.write_text("我是文件不是目录", encoding="utf-8")
    monkeypatch.setattr(validate, "_JSON_OUT", blocker / "sub" / "c.json")
    monkeypatch.setattr(validate, "CHECKS", [("甲", True, "")])
    assert validate._summary() == 0
    assert "写" in capsys.readouterr().err


def test_coverage_check_is_pool_aware():
    """池内缺名字才判失败；池外只进 detail（不静默、也不判死）。"""
    pool = {"A", "B"}
    missing = ["A", "C", "D"]
    in_pool, out_pool, empty = validate._pool_view(missing, pool)
    assert in_pool == ["A"] and out_pool == 2 and empty is False


def test_coverage_check_without_a_pool_judges_everything():
    """算不出流动性池时按**全部**标的判 —— 否则这类检查会静默通过。"""
    items = ["A", "B"]
    in_pool, out_pool, empty = validate._pool_view(items, set())
    assert in_pool == items and out_pool == 0 and empty is True
