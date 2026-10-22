"""preset 定义体的装载器 —— 「新增一条 preset = 新增一个文件」的机制。

这个重构把 preset 从「往 `presets.py` 的 dict 里加一条」改成「新增一个文件」。
装载器是它的守门人：**违规一律 fail-fast**，因为这些违规的共同后果是
"新加的东西默认不在护栏里"（这正是重构要消灭的失效方式）。

本文件钉住四条：文件名与 NAME 必须一致、`VERDICT`/`LOCK_WINDOW` 必填、
`PRESET` 里不许再写一份 verdict（一份事实只能有一处）、跨文件重名必须抛。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import preset_registry as R  # noqa: E402

GOOD = '''
NAME = "x1"
PRESET = {"max_positions": 1, "position_fraction": 1.0, "atr_slip_scale": 1.0,
          "intraday_tiebreak": "sl_first", "max_hold": 10, "cost_bps_roundtrip": 5.0,
          "sl_pct": 0.02, "tp_pct": 0.20}
VERDICT = "无可证实的正 alpha"
LOCK_WINDOW = ("2024-01-01", "2025-01-01")
LOCK_IS_CLI_DEFAULT = False
FIXTURE_WINDOW = ("2024-01-01", "2025-01-01")
'''


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / f"{name}.py"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_one_returns_the_declared_fields(tmp_path):
    got = R.load_one(_write(tmp_path, "x1", GOOD))
    assert got["name"] == "x1" and got["verdict"].startswith("无可证实的")
    assert got["window"] == ("2024-01-01", "2025-01-01")
    assert got["is_cli_default"] is False and got["preset"]["max_positions"] == 1


def test_file_name_must_match_name(tmp_path):
    with pytest.raises(ValueError, match="文件名"):
        R.load_one(_write(tmp_path, "other", GOOD))


def test_missing_verdict_fails_fast(tmp_path):
    body = GOOD.replace('VERDICT = "无可证实的正 alpha"', "")
    with pytest.raises(ValueError, match="VERDICT"):
        R.load_one(_write(tmp_path, "x1", body))


def test_missing_lock_window_fails_fast(tmp_path):
    body = GOOD.replace('LOCK_WINDOW = ("2024-01-01", "2025-01-01")', "")
    with pytest.raises(ValueError, match="LOCK_WINDOW"):
        R.load_one(_write(tmp_path, "x1", body))


def test_verdict_must_not_be_duplicated_inside_preset(tmp_path):
    """一份事实只能有一处：verdict 在模块级，PRESET 里不许再写一遍。"""
    body = GOOD.replace("PRESET = {", 'PRESET = {"verdict": "x",')
    with pytest.raises(ValueError, match="verdict"):
        R.load_one(_write(tmp_path, "x1", body))


def test_duplicate_names_across_files_fail_fast(tmp_path):
    _write(tmp_path, "x1", GOOD)
    _write(tmp_path, "x2", GOOD.replace('NAME = "x1"', 'NAME = "x2"'))
    models: dict = {}
    R.load_dir(models, tmp_path)
    assert set(models) == {"x1", "x2"}
    (tmp_path / "x1b.py").write_text(GOOD, encoding="utf-8")   # 文件名不符 ⇒ 抛
    with pytest.raises(ValueError):
        R.load_dir({}, tmp_path)


def test_load_dir_skips_private_files(tmp_path):
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "_helper.py").write_text("X = 1\n", encoding="utf-8")
    _write(tmp_path, "x1", GOOD)
    models: dict = {}
    R.load_dir(models, tmp_path)
    assert set(models) == {"x1"}


def test_fixture_window_and_exemption_are_mutually_recorded(tmp_path):
    got = R.load_one(_write(tmp_path, "x1", GOOD))
    assert got["fixture_window"] == ("2024-01-01", "2025-01-01")
    assert got["fixture_exempt"] is None
    body2 = GOOD.replace('FIXTURE_WINDOW = ("2024-01-01", "2025-01-01")',
                         'FIXTURE_EXEMPT_REASON = "合成市场上成交为 0，锁会变成空断言"')
    got2 = R.load_one(_write(tmp_path, "x1", body2))
    assert got2["fixture_window"] is None and "空断言" in got2["fixture_exempt"]


def test_offline_lock_membership_is_mandatory(tmp_path):
    """两者都不给 ⇒ 抛错：不给表态，等于这条 preset **默认不在离线护栏里**。"""
    body = GOOD.replace('FIXTURE_WINDOW = ("2024-01-01", "2025-01-01")', "")
    with pytest.raises(ValueError, match="离线锁"):
        R.load_one(_write(tmp_path, "x1", body))


def test_fixture_window_and_exemption_are_exclusive(tmp_path):
    body = GOOD + '\nFIXTURE_EXEMPT_REASON = "又想要锁又想要豁免"\n'
    with pytest.raises(ValueError, match="只能给一个"):
        R.load_one(_write(tmp_path, "x1", body))
