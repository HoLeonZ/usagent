"""厂商重述 ⇒ 整段重下重写：增量刷新不得留下永久的价格基数断层。

缺陷本体（2026-10-17 审计 A1）：本库落盘的 Close **是拆股后价位**，而每日任务用
`--recent 10` 的增量刷新 + INSERT OR REPLACE 只重写最近 10 天。厂商在拆股后把
该日**之前**的整段历史重述 ⇒ 更早的行永久停在旧基数，在窗口左边界留下一个等于
累计拆股比的假单日跌幅（10:1 就是 −90%），污染该标的的全部下游指标与信号。

钉住三条不变量：
  1. 重叠日期的 close 变了 = 厂商重述过（阈值 5%，见 config.RESTATE_MIN_REL_DIFF）；
  2. 一旦判定重述，必须**整段重下重写**（局部修补会把断层挪到别处）；
  3. 修复之后，该标的的整条序列来自**单一 vintage**，且事件在 fetch_log 里可审计。
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import db  # noqa: E402
import download_ohlcv as D  # noqa: E402
from config import RESTATE_MIN_REL_DIFF, Universe  # noqa: E402


# ---------------------------------------------------------------- 纯函数判定

@pytest.mark.parametrize("old,new,hit", [
    (100.0, 100.0, False),          # 没变
    (100.0, 101.0, False),          # 1% 的正常修订,低于阈值
    (100.0, 104.9, False),          # 阈值下沿
    (100.0, 105.5, True),           # 阈值上沿
    (100.0, 10.0, True),            # 10:1 拆股的重述(−90%)
    (1.0, 15.0, True),              # 反向拆股(基数变小 ⇒ 历史价上调)
])
def test_threshold(old, new, hit):
    got = D.restated_dates({"2024-01-02": old}, {"2024-01-02": new})
    assert bool(got) is hit, f"{old}->{new} 应{'触发' if hit else '不触发'}"
    assert RESTATE_MIN_REL_DIFF == 0.05


def test_edges_are_skipped_not_flagged():
    """没有可比基准的行不参与判定（新行、旧值为空/非正）。"""
    assert D.restated_dates({}, {"2024-01-02": 10.0}) == []
    assert D.restated_dates({"2024-01-02": 0.0}, {"2024-01-02": 10.0}) == []
    assert D.restated_dates({"2024-01-02": 100.0}, {"2024-01-02": 0.0}) == []


def test_multiple_dates_sorted():
    got = D.restated_dates(
        {"2024-01-03": 100.0, "2024-01-02": 100.0, "2024-01-04": 100.0},
        {"2024-01-03": 10.0, "2024-01-02": 100.5, "2024-01-04": 10.0})
    assert [d for d, _, _ in got] == ["2024-01-03", "2024-01-04"]


# ---------------------------------------------------------------- 端到端

def _bars(sym: str, dates: list[str], closes: list[float]) -> pd.DataFrame:
    """构造 yf.download 形状的返回（MultiIndex 列，按 symbol 分层）。"""
    fields = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    cols = pd.MultiIndex.from_product([[sym], fields])
    data = [[c, c, c, c, 1_000_000.0, 0.0, 0.0] for c in closes]
    return pd.DataFrame(data, index=pd.to_datetime(dates), columns=cols)


def _dates_upto(n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-11-01", periods=n)]


def _stored(con, sym: str) -> list[tuple[str, float]]:
    return con.execute("SELECT date, close FROM ohlcv WHERE symbol=? ORDER BY date",
                       (sym,)).fetchall()


def test_restatement_triggers_full_rewrite(tmp_path, monkeypatch):
    """增量窗口里发现基数变化 ⇒ 必须整段重下，且落盘序列只剩单一 vintage。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()

    days = _dates_upto(60)
    old_basis = [100.0 + i for i in range(60)]      # 拆股前的旧基数
    new_basis = [c / 10.0 for c in old_basis]       # 厂商按 10:1 重述后的新基数

    # 播种：库里是旧基数的 3 年历史
    with db.session() as con:
        con.executemany("INSERT INTO ohlcv (symbol,date,open,high,low,close,volume,"
                        "dividends,splits,fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [("AAA", d, c, c, c, c, 1e6, 0.0, 0.0, "seed")
                         for d, c in zip(days, old_basis)])

    inc_start = "2024-01-22"                        # 增量窗口 = 最近 3 个交易日
    inc_days = days[-3:]
    calls: list[tuple] = []

    def fake_fetch_resilient(batch, start, end):
        calls.append(("batch", str(start), str(end)))
        return _bars("AAA", inc_days, new_basis[-3:]), []

    def fake_fetch_with_split(batch, start, end, depth=0):
        calls.append(("repair", str(start), str(end)))
        return _bars("AAA", days, new_basis), []     # 整段（新基数）

    monkeypatch.setattr(D, "_fetch_resilient", fake_fetch_resilient)
    monkeypatch.setattr(D, "fetch_with_split", fake_fetch_with_split)
    monkeypatch.setattr(D, "load_universe",
                        lambda name="oi100": Universe("fake", ("AAA",)))

    stats = D.download("oi100", None, False, recent_days=10, as_of=date(2024, 1, 31))

    assert stats["restated"] == 1, "重叠日期基数变了却没被判定为重述"
    assert stats["repaired"] == 1, "判定重述后没有触发整段重下"
    repair_calls = [c for c in calls if c[0] == "repair"]
    assert len(repair_calls) == 1
    # 修复用的是**整段**起点（历史起点），不是增量窗口
    assert repair_calls[0][1] < inc_start, f"修复窗口太窄：{repair_calls[0]}"

    with db.session() as con:
        got = _stored(con, "AAA")
        # 不变量：整条序列来自单一 vintage（全部新基数，无混排）
        assert all(abs(c - nb) < 1e-9 for (_d, c), nb in zip(got, new_basis)), \
            "落盘序列混排了新旧基数 —— 断层还在"
        events = {r[0] for r in con.execute(
            "SELECT status FROM fetch_log WHERE symbol='AAA'").fetchall()}
        assert {"ok", "restated", "repaired"} <= events, f"审计事件缺失：{events}"


def test_no_restatement_no_repair(tmp_path, monkeypatch):
    """价格没被厂商改过 ⇒ 不得触发任何整段重下（否则每天都会全量重下）。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    days = _dates_upto(30)
    closes = [100.0 + i * 0.1 for i in range(30)]
    with db.session() as con:
        con.executemany("INSERT INTO ohlcv (symbol,date,open,high,low,close,volume,"
                        "dividends,splits,fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [("AAA", d, c, c, c, c, 1e6, 0.0, 0.0, "seed")
                         for d, c in zip(days, closes)])

    calls: list[tuple] = []

    def fake_fetch_resilient(batch, start, end):
        calls.append(("batch", str(start)))
        return _bars("AAA", days[-3:], closes[-3:]), []

    def fake_fetch_with_split(batch, start, end, depth=0):
        calls.append(("repair", str(start)))
        return _bars("AAA", days, closes), []

    monkeypatch.setattr(D, "_fetch_resilient", fake_fetch_resilient)
    monkeypatch.setattr(D, "fetch_with_split", fake_fetch_with_split)
    monkeypatch.setattr(D, "load_universe", lambda name="oi100": Universe("fake", ("AAA",)))

    stats = D.download("oi100", None, False, recent_days=10, as_of=date(2024, 1, 31))
    assert stats["restated"] == 0 and stats["repaired"] == 0
    assert not [c for c in calls if c[0] == "repair"], "无重述却做了整段重下"
