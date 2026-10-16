"""SQLite 落盘层：schema + 幂等写入。

设计要点
- 所有写入用 INSERT OR REPLACE / ON CONFLICT，所以任务重复跑是幂等的。
- WAL 模式，读（分析）写（抓取）不互相阻塞。
- 存原始值，不做复权。复权在读取时现算（见 derive.py）。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,          -- YYYY-MM-DD
    open        REAL, high REAL, low REAL, close REAL,
    volume      REAL,
    dividends   REAL DEFAULT 0,
    splits      REAL DEFAULT 0,
    fetched_at  TEXT NOT NULL,
    -- 取数源（yahoo / stockanalysis …）。2026-10-17 审计 G9：本库现在有**两个**
    -- 价格口径相同但事件来源不同的源；没有这一列时，用 A 源铺历史再被 B 源增量
    -- 刷新，同一标的的序列会混进两种口径而且**无从检测**。
    source      TEXT,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS ix_ohlcv_date ON ohlcv(date);

CREATE TABLE IF NOT EXISTS option_chain (
    snapshot_date       TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    expiry              TEXT NOT NULL,
    dte                 INTEGER NOT NULL,
    strike              REAL NOT NULL,
    right               TEXT NOT NULL,   -- C / P
    bid                 REAL, ask REAL, last REAL,
    volume              REAL, open_interest REAL,
    implied_volatility  REAL,
    underlying_price    REAL,
    fetched_at          TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, symbol, expiry, strike, right)
);
CREATE INDEX IF NOT EXISTS ix_chain_sym_date ON option_chain(symbol, snapshot_date);

CREATE TABLE IF NOT EXISTS iv_history (
    date            TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    iv30            REAL,               -- 30 天恒定到期 ATM IV（方差域线性插值）
    hv20            REAL,               -- 20 日已实现波动率（年化）
    vrp             REAL,               -- iv30 - hv20
    iv_rank         REAL,
    iv_percentile   REAL,
    n_used          INTEGER,            -- 参与插值的合约数
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    run_id   TEXT NOT NULL,
    ts       TEXT NOT NULL,
    kind     TEXT NOT NULL,             -- ohlcv / chain / iv
    symbol   TEXT,
    status   TEXT NOT NULL,             -- ok / empty / error / ratelimited
    detail   TEXT
);
CREATE INDEX IF NOT EXISTS ix_log_run ON fetch_log(run_id, kind, status);

CREATE TABLE IF NOT EXISTS coverage (
    kind          TEXT NOT NULL,        -- ohlcv / chain
    symbol        TEXT NOT NULL,
    rows          INTEGER DEFAULT 0,
    first_date    TEXT,
    last_date     TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (kind, symbol)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=120000")  # 抓取任务持写锁可达 40s，给足等待
    return con


# 需要向后兼容的列迁移：列名 → DDL 片段。**只加列、不改列、不删列。**
_MIGRATIONS: dict[str, str] = {
    "ohlcv": "ALTER TABLE ohlcv ADD COLUMN source TEXT",
}


def _migrate(con: sqlite3.Connection) -> None:
    """幂等的加列迁移（老的库没有 source 列）。

    为什么必须显式做这一层而不是靠 CREATE TABLE IF NOT EXISTS：后者对**已存在**的
    表什么都不做 —— 于是"新加的列"在最需要它（老库）的地方恰好不存在，
    而写入会因为列数不匹配而报错或静默丢列。
    """
    have = {r[1] for r in con.execute("PRAGMA table_info(ohlcv)").fetchall()}
    if not have:
        return
    if "source" not in have:
        con.execute(_MIGRATIONS["ohlcv"])


def backfill_source(con: sqlite3.Connection, source: str) -> int:
    """把 source 为空的历史行**显式**标注为某个源。

    不放进 init():默认值必须是"未知",而不是替操作者猜一个。谁跑这次回填、
    为什么是这个值，要由调用方说清（本项目唯一一次：2026-10-17 全量下载，
    3028 只全部来自 stockanalysis）。
    """
    if not source:
        raise ValueError("backfill_source 需要显式的源名 —— 不许猜")
    cur = con.execute("UPDATE ohlcv SET source=? WHERE source IS NULL", (source,))
    return int(cur.rowcount)


def init(path: Path | None = None) -> None:
    with connect(path) as con:
        con.executescript(SCHEMA)
        _migrate(con)
        con.commit()


@contextmanager
def session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    con = connect(path)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def upsert_many(con: sqlite3.Connection, table: str, cols: list[str],
                rows: Iterable[tuple]) -> int:
    """INSERT OR REPLACE 批量写入。返回写入行数。"""
    rows = list(rows)
    if not rows:
        return 0
    sql = (f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
           f"VALUES ({','.join('?' * len(cols))})")
    con.executemany(sql, rows)
    return len(rows)


def log(con: sqlite3.Connection, run_id: str, ts: str, kind: str,
        symbol: str | None, status: str, detail: str = "") -> None:
    con.execute("INSERT INTO fetch_log (run_id, ts, kind, symbol, status, detail) "
                "VALUES (?,?,?,?,?,?)", (run_id, ts, kind, symbol, status, detail))


def refresh_coverage(con: sqlite3.Connection, kind: str, table: str,
                     date_col: str, now: str, symbols: Iterable[str] | None = None) -> None:
    """重算某类数据的覆盖区间，用于断点续跑。"""
    where, params = "", []
    if symbols is not None:
        syms = list(symbols)
        if not syms:
            return
        where = f" WHERE symbol IN ({','.join('?' * len(syms))})"
        params = syms
    rows = con.execute(
        f"SELECT symbol, COUNT(*), MIN({date_col}), MAX({date_col}) "
        f"FROM {table}{where} GROUP BY symbol", params).fetchall()
    con.executemany(
        "INSERT OR REPLACE INTO coverage (kind, symbol, rows, first_date, last_date, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        [(kind, s, n, a, b, now) for s, n, a, b in rows])


def covered_symbols(con: sqlite3.Connection, kind: str, min_last: str) -> set[str]:
    """已覆盖到 min_last 之后的标的（用于跳过）。"""
    cur = con.execute("SELECT symbol FROM coverage WHERE kind=? AND last_date>=?",
                      (kind, min_last))
    return {r[0] for r in cur.fetchall()}
