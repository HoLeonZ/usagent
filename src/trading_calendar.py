"""美国交易所日历（NYSE / Nasdaq 同休）—— **显式表，不靠规则推算**。

为什么必须有这一层（2026-10-17 审计）:

`validate.py` 此前用「库里出现过的日期」当交易日历
（`cal = {r[0] for r in con.execute("SELECT DISTINCT date FROM ohlcv")}`），
再拿它判断个股有没有缺档。这条判据是**自指的**：全市场都缺的那一天
（下载中断、Yahoo 少返一天、vendor 故障）在检查里是**隐形的** —— 判据不可能
发现自己也被污染。机构的做法是持有一条**独立于被测数据**的日历。

为什么不按规则推算（"周一至周五减掉第 N 个星期一"）：规则算不出**临时休市**，
而本窗口里就有一个真实的例子 —— 2025-01-09（卡特国葬日，NYSE/Nasdaq 全天休市）。
按规则算会把它当成"全市场缺档"，于是真正该报警的那天被淹没在噪声里。
所以这里用一张**显式的休市表**：可审计、可 diff、可被测试钉住。

范围：覆盖本项目的数据窗口（3 年历史 + 前向一年）。超出范围时 `expected_sessions`
抛错而不是静默按工作日推算 —— 静默推算正是本节要消除的东西。
"""
from __future__ import annotations

from datetime import date, timedelta

# 全天休市（半日市不在内：半日市有完整日线，不影响日线口径）。
# 每条都带名字，便于人工核对与 diff —— 这张表的价值就在于"看得见"。
MARKET_CLOSURES: tuple[tuple[str, str], ...] = (
    # 2023
    ("2023-11-23", "Thanksgiving"),
    ("2023-12-25", "Christmas"),
    # 2024
    ("2024-01-01", "New Year's Day"),
    ("2024-01-15", "MLK Day"),
    ("2024-02-19", "Washington's Birthday"),
    ("2024-03-29", "Good Friday"),
    ("2024-05-27", "Memorial Day"),
    ("2024-06-19", "Juneteenth"),
    ("2024-07-04", "Independence Day"),
    ("2024-09-02", "Labor Day"),
    ("2024-11-28", "Thanksgiving"),
    ("2024-12-25", "Christmas"),
    # 2025
    ("2025-01-01", "New Year's Day"),
    ("2025-01-09", "National Day of Mourning (Jimmy Carter)"),
    ("2025-01-20", "MLK Day"),
    ("2025-02-17", "Washington's Birthday"),
    ("2025-04-18", "Good Friday"),
    ("2025-05-26", "Memorial Day"),
    ("2025-06-19", "Juneteenth"),
    ("2025-07-04", "Independence Day"),
    ("2025-09-01", "Labor Day"),
    ("2025-11-27", "Thanksgiving"),
    ("2025-12-25", "Christmas"),
    # 2026
    ("2026-01-01", "New Year's Day"),
    ("2026-01-19", "MLK Day"),
    ("2026-02-16", "Washington's Birthday"),
    ("2026-04-03", "Good Friday"),
    ("2026-05-25", "Memorial Day"),
    ("2026-06-19", "Juneteenth"),
    ("2026-07-03", "Independence Day (observed; Jul 4 是周六)"),
    ("2026-09-07", "Labor Day"),
    ("2026-11-26", "Thanksgiving"),
    ("2026-12-25", "Christmas"),
    # 2027（前向一年，供新鲜度检查跨年使用）
    ("2027-01-01", "New Year's Day"),
)

COVERED_FROM = date(2023, 10, 1)
COVERED_TO = date(2027, 1, 1)
_CLOSURES = {date.fromisoformat(d): name for d, name in MARKET_CLOSURES}


class CalendarError(RuntimeError):
    """超出本表覆盖范围 —— 不静默按工作日推算（那正是要消除的失败模式）。"""


def is_session(d: date) -> bool:
    """这一天交易所是否开市（含范围校验）。"""
    if not (COVERED_FROM <= d <= COVERED_TO):
        raise CalendarError(
            f"{d} 超出显式日历覆盖范围 [{COVERED_FROM}, {COVERED_TO}]；"
            f"请扩充 trading_calendar.MARKET_CLOSURES（含临时休市），不要退回按工作日推算")
    return d.weekday() < 5 and d not in _CLOSURES


def closure_name(d: date) -> str | None:
    return _CLOSURES.get(d)


def expected_sessions(start: date, end: date) -> list[date]:
    """[start, end] 闭区间内的**应有**交易日（交易所开市日）。"""
    if end < start:
        return []
    out, cur = [], start
    while cur <= end:
        if is_session(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def last_session_on_or_before(d: date) -> date:
    while not is_session(d):
        d -= timedelta(days=1)
    return d
