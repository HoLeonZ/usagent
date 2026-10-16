"""数据质量的门：取数源、物理上不可能的价、说不清真伪的跳变。

2026-10-17 全量下载（3028 只 / 196.9 万行 / 3 年）之后，`validate.py` 一次就报出
三类问题，全部与"数据能不能直接用"有关：

  1. **470 行 `close ∉ [low, high]`**（物理上不可能），且全部集中在**最后一根 bar** ——
     vendor 当日快照未定型。处置：**落库前拒收**，而不是让不可能的价进面板。
  2. **池内 10 只中间缺口**（SHAZ 276/750 一类）、**池内 18 只整数倍跳变**
     （CORT/NVAX 一类）。处置：跳变按"标的×日期"排除出机会集（说不清就不交易）；
     缺口留给下一轮处置。
  3. **池外**大量稀疏交易与仙股噪声（实测有标的在 $0.00001 价位刷出 110 倍的假跳变、
     成交量几十股）—— 这些不该触发告警，否则运维层会训练出"反正每天都红"的疲劳。
     处置：判据改成**池内才判失败**，池外只报数。

另外补上审计 G9：`ohlcv` 加 `source` 列（含对已有库的幂等加列迁移 + 显式回填），
并加"同一标的不混源"的校验 —— 两个源的价格口径相同但**事件来源不同**，
混在一条序列里没有任何别的检查能发现。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import db  # noqa: E402
import download_ohlcv as D  # noqa: E402
import validate as validate_mod  # noqa: E402


# ---------------------------------------------------------------- 取数源

def test_migration_adds_source_column_idempotently(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    db.init()                                    # 再跑一次不得报错
    with db.session() as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(ohlcv)")]
    assert "source" in cols


def test_migration_preserves_existing_rows(tmp_path, monkeypatch):
    """加列不得动数据：老的库升级后行数、价格一字不变。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    with db.session() as con:
        con.execute("INSERT INTO ohlcv (symbol,date,close,fetched_at,source) "
                    "VALUES ('AAA','2024-01-02',100.0,'x','yahoo')")
        con.execute("ALTER TABLE ohlcv RENAME TO _old")     # 模拟"没有 source 列的老库"
        con.execute("CREATE TABLE ohlcv AS SELECT symbol,date,open,high,low,close,volume,"
                    "dividends,splits,fetched_at FROM _old")
        con.execute("DROP TABLE _old")
        con.execute("PRAGMA legacy_alter_table=OFF")
    db.init()                                    # 触发迁移
    with db.session() as con:
        assert [r[1] for r in con.execute("PRAGMA table_info(ohlcv)")][-1] == "source"
        n, c = con.execute("SELECT COUNT(*), SUM(close) FROM ohlcv").fetchone()
        assert (n, c) == (1, 100.0), "迁移动到了数据"
        assert con.execute("SELECT source FROM ohlcv").fetchone()[0] is None


def test_backfill_requires_an_explicit_source(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    with db.session() as con:
        con.execute("INSERT INTO ohlcv (symbol,date,close,fetched_at) "
                    "VALUES ('AAA','2024-01-02',100.0,'x')")
        with pytest.raises(ValueError, match="不许猜"):
            db.backfill_source(con, "")
        assert db.backfill_source(con, "stockanalysis") == 1
        assert con.execute("SELECT source FROM ohlcv").fetchone()[0] == "stockanalysis"


def test_rows_carry_the_source():
    df = pd.DataFrame({"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.0],
                       "Volume": [10.0], "Dividends": [0.0], "Stock Splits": [0.0]},
                      index=pd.DatetimeIndex(["2024-01-02"]))
    rows = D._to_rows("AAA", df, "now", "stockanalysis")
    assert rows and rows[0][-1] == "stockanalysis"
    assert len(rows[0]) == len(D.COLS)


# ---------------------------------------------------------------- 物理上不可能的价

def test_impossible_ohlc_is_rejected_not_stored():
    """`close` 落在 [low, high] 之外是物理上不可能的（实测 470 行，全在最后一根 bar）。"""
    before = D._IMPOSSIBLE["ohlc"]
    bad = pd.DataFrame({"Open": [348.28], "High": [350.66], "Low": [343.13],
                        "Close": [343.01], "Volume": [1e6], "Dividends": [0.0],
                        "Stock Splits": [0.0]}, index=pd.DatetimeIndex(["2026-10-16"]))
    assert D._to_rows("GOOGL", bad, "now", "stockanalysis") == []
    assert D._IMPOSSIBLE["ohlc"] == before + 1
    good = bad.assign(High=[352.0], Low=[342.0])   # 同时修 low（它原先 > close）
    assert len(D._to_rows("GOOGL", good, "now", "stockanalysis")) == 1


def test_boundary_tolerance_does_not_reject_valid_rows():
    """浮点误差级的越界（1e-5）不得被误拒 —— 拒收判据必须带容差。"""
    df = pd.DataFrame({"Open": [1.0], "High": [1.0 + 5e-5], "Low": [0.9],
                       "Close": [1.00001], "Volume": [10.0], "Dividends": [0.0],
                       "Stock Splits": [0.0]}, index=pd.DatetimeIndex(["2024-01-02"]))
    assert len(D._to_rows("AAA", df, "now", "yahoo")) == 1


# ---------------------------------------------------------------- 说不清真伪的跳变
from research import universe as U  # noqa: E402


def _panel(rows):
    return pd.DataFrame(rows, columns=["symbol", "date", "raw_close", "close", "splits"])


def test_integer_jump_without_a_split_is_untrustworthy():
    p = _panel([("A", "2024-01-02", 1.0, 1.0, 0.0), ("A", "2024-01-03", 3.0, 3.0, 0.0)])
    assert U.untrustworthy_jumps(p).tolist() == [False, True]


def test_a_recorded_split_explains_the_jump():
    p = _panel([("A", "2024-01-02", 1.0, 1.0, 0.0), ("A", "2024-01-03", 3.0, 3.0, 3.0)])
    assert U.untrustworthy_jumps(p).tolist() == [False, False]


def test_only_exact_integer_multiples_are_flagged():
    """**有意的取舍**：只抓恰好整数倍（容差 1%）。真实反拆股正是精确整数倍；
    而"带同期涨幅的反拆股"（3.15×）会漏掉 —— 宁可漏报，也不要把微盘的普通暴涨
    （2.857×）误判成数据问题。实测 209 处的倍数分布是 2.0×96 / 3.0×29 / 5.0×8，
    全部精确落在整数上，说明这个判据对**真实存在的那一类**是有效的。
    """
    p = _panel([("A", "2024-01-02", 1.05, 1.05, 0.0), ("A", "2024-01-03", 3.0, 3.0, 0.0)])
    assert not U.untrustworthy_jumps(p).any(), "2.857 倍不该被判为跳变（宁可漏报）"


def test_ordinary_moves_are_trustworthy():
    p = _panel([("A", "2024-01-02", 1.0, 1.0, 0.0), ("A", "2024-01-03", 1.09, 1.09, 0.0),
                ("A", "2024-01-04", 0.92, 0.92, 0.0)])
    assert not U.untrustworthy_jumps(p).any(), "9%/8% 的普通波动不该被判为跳变"


def test_jump_exclusion_applies_in_every_pool_mode():
    """数据可信度与"取池口径"正交：说不清的那一天，无论池子怎么定义都不该交易。"""
    p = _panel([("A", "2024-01-02", 1.0, 1.0, 0.0),
                ("A", "2024-01-03", 1.0, 1.0, 0.0),
                ("A", "2024-01-04", 3.0, 3.0, 0.0)])       # 第 2 行是**恰好** 3 倍
    p["amount"] = 1e9
    short = {"mode": "liquidity", "lookback": 2, "min_dollar_volume": 0.0}
    for spec in (None, short):
        ok = U.eligibility(p, U.parse(spec))
        assert not ok.iloc[2], f"模式 {spec} 下跳变日仍应被排除"
    assert U.eligibility(p, U.parse(None)).tolist() == [True, True, False]


def test_jumps_are_causal():
    """判据只看 t 与 t−1 —— 截断序列不得改变过去的值。"""
    p = _panel([("A", "2024-01-%02d" % d, float(d), float(d), 0.0) for d in range(2, 8)])
    full = U.untrustworthy_jumps(p)
    part = U.untrustworthy_jumps(p[p["date"] <= "2024-01-05"])
    assert full[:4].tolist() == part.tolist()


# ---------------------------------------------------------------- 池内外的分级

def test_pool_split_counts_in_and_out(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    with db.session() as con:
        rows = []
        for sym, vol in (("BIG", 1e7), ("TINY", 10.0)):
            for i in range(60):
                rows.append((sym, f"2024-0{1 + i // 30}-{i % 30 + 1:02d}", 10.0, 10.0,
                             10.1, 9.9, vol, 0.0, 0.0, "x", "yahoo"))
        con.executemany("INSERT OR REPLACE INTO ohlcv (symbol,date,open,high,low,close,"
                        "volume,dividends,splits,fetched_at,source) VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?)", rows)
        assert validate_mod._pool_symbols(con) == {"BIG"}
        assert validate_mod._split_pool(["BIG", "TINY", "TINY"], {"BIG"}, set()) == (1, 2, False)
        # 池为空时按全部判（第三项 True）—— 否则检查会静默通过
        assert validate_mod._split_pool(["BIG", "TINY"], set(), set()) == (2, 0, True)

