"""独立交易日历 —— 判据不得自指。

动机（2026-10-17 审计）：`validate.py` 此前用「库里出现过的所有日期」当交易日历，
再拿它判断个股缺档。这条判据**自指**：全市场都缺的那一天（下载中断、vendor 少返
一天）在检查里是隐形的 —— 判据不可能发现自己也被污染。机构的做法是持有一条
独立于被测数据的日历，且必须能表达**临时休市**（本窗口里有 2025-01-09 卡特国葬日）。
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import trading_calendar as TC  # noqa: E402


@pytest.mark.parametrize("d,why", [
    ("2025-01-09", "National Day of Mourning (Jimmy Carter)"),
    ("2024-07-04", "Independence Day"),
    ("2024-03-29", "Good Friday"),
    ("2024-11-28", "Thanksgiving"),
    ("2025-06-19", "Juneteenth"),
    ("2026-07-03", "Independence Day (observed; Jul 4 是周六)"),
])
def test_closures_are_closed(d, why):
    assert not TC.is_session(date.fromisoformat(d))
    assert TC.closure_name(date.fromisoformat(d)) == why


@pytest.mark.parametrize("d", [
    "2025-01-10",   # 卡特国葬次日（周五）
    "2024-07-05",   # 7/4 次日
    "2024-04-01",   # 耶稣受难日后的周一
    "2025-11-28",   # 感恩节次日（半日市，仍是交易日）
])
def test_normal_days_are_sessions(d):
    assert TC.is_session(date.fromisoformat(d))


def test_weekends_are_not_sessions():
    assert not TC.is_session(date(2026, 10, 17))   # 周六
    assert not TC.is_session(date(2026, 10, 18))   # 周日
    assert TC.is_session(date(2026, 10, 16))       # 周五


def test_expected_sessions_counts_around_a_closure():
    """2025-01-06(一) .. 2025-01-10(五) 只有 4 个交易日 —— 1/9 休市。"""
    got = TC.expected_sessions(date(2025, 1, 6), date(2025, 1, 10))
    assert [d.isoformat() for d in got] == [
        "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-10"]


def test_last_session_skips_closure_and_weekend():
    assert TC.last_session_on_or_before(date(2025, 1, 9)) == date(2025, 1, 8)
    assert TC.last_session_on_or_before(date(2026, 10, 17)) == date(2026, 10, 16)


def test_out_of_range_raises_instead_of_guessing():
    """超出表覆盖范围必须抛错，不许静默退回「按工作日推算」—— 那正是要消除的失败模式。"""
    with pytest.raises(TC.CalendarError):
        TC.is_session(date(2030, 1, 2))
    with pytest.raises(TC.CalendarError):
        TC.expected_sessions(date(2030, 1, 1), date(2030, 1, 5))


def test_table_is_sane():
    """表本身的结构性约束：无周末条目、无重复、有序、日期可解析。"""
    ds = [date.fromisoformat(d) for d, _ in TC.MARKET_CLOSURES]
    assert ds == sorted(ds), "休市表应按日期有序"
    assert len(ds) == len(set(ds)), "休市表有重复条目"
    assert all(d.weekday() < 5 for d in ds), "休市表里出现周末条目（周末本就不开市）"
    assert all(name.strip() for _, name in TC.MARKET_CLOSURES), "每条休市都要有名字"
