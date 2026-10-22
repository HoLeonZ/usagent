"""面板层口径 —— 双价格体系(CLAUDE.md §3(b))与成交额代理量(§3(a)/§3(b))。

为什么 amount 必须用 as-traded 价:panel 用 `volume × close` 合成成交额代理量,而 close
是**分红回补**后的价(因子 = ∏_{事件晚于 t})。E 条件把这个量拿去比**绝对美元常数**
[3e7, 3e8],于是「T 日能不能进池」变成了 T 之后除息日的函数。凭证:2026-10-15 审计实测
全库 390,168 个就绪格中有 740 格的 E 判定被未来分红翻转(VISN 的因子低到 0.278,即
成交额被压低到真值的 28%)。把 amount 换成 as-traded 后,12m preset 的信号数 19→21、
CAGR +44.54%→+47.19%。

比例型指标(close>ma60、mom120、pullback)对同一个因子免疫 —— 分子分母同乘一个常数。
所以问题**只**出在「绝对美元阈值」这一处比对上,这也是为什么修的是 amount 而不是指标。
"""
from __future__ import annotations

import pandas as pd
import pytest

from research.panel import (
    MAX_LOOKBACK_BARS, RAW_COLS, PanelError, data_vintage, first_signal_date,
    price_digest, to_panel_frame,
)

DATES = pd.bdate_range("2024-01-01", periods=3)


def _series(close: list[float], dividends: list[float] | None = None,
            dates: pd.DatetimeIndex = DATES) -> pd.DataFrame:
    """模仿 src/series.daily 的输出:index=date,含 OHLCV + dividends/splits。"""
    n = len(close)
    df = pd.DataFrame({
        "date": dates,
        "open": close, "high": [c * 1.01 for c in close],
        "low": [c * 0.99 for c in close], "close": close,
        "volume": 1000.0,
        "dividends": dividends if dividends is not None else [0.0] * n,
        "splits": 0.0,
    })
    return df.set_index("date")


def test_amount_uses_as_traded_close():
    out = to_panel_frame("AAA", _series([50.0, 50.0, 50.0]), _series([100.0, 100.0, 100.0]))
    assert list(out["close"]) == [50.0] * 3, "指标口径必须仍是复权价"
    assert list(out["raw_close"]) == [100.0] * 3, "执行口径必须是 as-traded 价"
    assert list(out["amount"]) == [1000.0 * 100.0] * 3, "amount 必须用 as-traded 价算"


def test_frame_carries_both_price_series():
    out = to_panel_frame("AAA", _series([50.0] * 3), _series([100.0] * 3))
    assert set(RAW_COLS) <= set(out.columns)
    assert set(out["symbol"]) == {"AAA"}
    assert set(out["dividends"]) == {0.0}


def test_frame_fails_fast_when_the_two_series_do_not_align():
    """两条序列日期对不上 ⇒ fail-fast,不得静默留 NaN(§0 Fail-Fast)。"""
    short = pd.bdate_range("2024-01-01", periods=2)
    with pytest.raises(PanelError, match="as-traded"):
        to_panel_frame("AAA", _series([50.0] * 3), _series([100.0] * 2, dates=short))


# ---------------------------------------------------------------- 预热边界与 vintage

def test_first_signal_date_is_the_first_bar_past_the_warmup():
    """预热边界取「全表第 MAX_LOOKBACK_BARS 个交易日」= 第 121 根 K 线。"""
    dates = pd.bdate_range("2020-01-01", periods=200)
    panel = pd.DataFrame({"symbol": "AAA", "date": dates})
    assert first_signal_date(panel) == dates[MAX_LOOKBACK_BARS]


def test_first_signal_date_rejects_a_panel_shorter_than_the_warmup():
    dates = pd.bdate_range("2020-01-01", periods=MAX_LOOKBACK_BARS)
    with pytest.raises(PanelError, match="预热"):
        first_signal_date(pd.DataFrame({"symbol": "AAA", "date": dates}))


def test_data_vintage_is_a_stable_fingerprint():
    """计数部分逐字稳定；价格部分（A6 加的）是 32 位十六进制摘要。

    ⚠️ 这个 fixture 现在必须带 `raw_close`/`volume`：`data_vintage` 会在缺列时
    **抛错**（一个看不见价格的"vintage 指纹"正是 A6 要修的那个缺陷），
    所以最小 fixture 也得把这两个物理量给出来。
    """
    panel = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
        "raw_close": [1.0, 2.0, 3.0],
        "volume": [1e6, 1e6, 1e6],
    })
    v = data_vintage(panel)
    assert v.startswith("rows=3;symbols=2;from=2024-01-01;to=2024-01-02;prices=")
    digest = v.split("prices=")[1]
    assert len(digest) == 32 and all(c in "0123456789abcdef" for c in digest)


def test_load_full_panel_explains_a_missing_database(tmp_path, monkeypatch):
    """缺库/缺表时要报**可行动**的错,而不是抛一句 sqlite 原始异常(§0 Fail-Fast)。

    data/*.db 不入库(§7),所以 fresh clone 上第一条该跑的是取数。此前这里抛的是
    `sqlite3.OperationalError: no such table: ohlcv` —— 一个新读者拿到这句话不知道
    该干什么。实测于干净 worktree(只有 tracked 文件、没有 data/*.db)。
    """
    import sqlite3

    import db as dbmod
    from research import panel

    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    monkeypatch.setattr(dbmod, "connect", lambda: sqlite3.connect(str(empty)))
    with pytest.raises(panel.PanelError, match="download_ohlcv"):
        panel.load_full_panel()


# ---------------------------------------------------------------- 价格内容指纹（A6）

def _mini(close: float = 1.0, volume: float = 1e6) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
        "raw_close": [close, 2.0, 3.0], "volume": [volume, 1e6, 1e6],
    })


def test_vintage_detects_a_price_change_with_identical_counts():
    """**这就是 A6 的场景**：行数/标的数/日期全同，只有价格变了 —— 旧指纹看不出来，
    于是 golden 变红会被归因成"逻辑变了"。"""
    a, b = _mini(), _mini(close=1.0000001)
    assert (len(a), a["symbol"].nunique()) == (len(b), b["symbol"].nunique())
    old_style_a = f"rows={len(a)};symbols={a['symbol'].nunique()}"
    old_style_b = f"rows={len(b)};symbols={b['symbol'].nunique()}"
    assert old_style_a == old_style_b, "旧口径本来就分不开 —— 这条测试的前提"
    assert price_digest(a) != price_digest(b), "价格变了，指纹必须变"
    assert data_vintage(a) != data_vintage(b)


def test_vintage_detects_a_volume_change():
    assert price_digest(_mini()) != price_digest(_mini(volume=1.1e6))


def test_digest_is_independent_of_row_order():
    """分块哈希必须与行序无关 —— 否则同一份数据换个载入顺序就"变 vintage 了"。"""
    a = _mini()
    assert price_digest(a) == price_digest(a.iloc[::-1].reset_index(drop=True))


def test_digest_is_deterministic_and_short():
    a = _mini()
    d = price_digest(a)
    assert d == price_digest(a) and len(d) == 32 and all(c in "0123456789abcdef" for c in d)


def test_digest_fails_fast_on_missing_columns():
    with pytest.raises(PanelError, match="price_digest"):
        price_digest(pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2024-01-02")]}))


def test_vintage_fails_fast_without_prices():
    """缺价的面板造不出 vintage —— 不许退回"只有计数"的旧口径（那正是 A6）。"""
    with pytest.raises(PanelError):
        data_vintage(pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2024-01-02")]}))


def test_vintage_still_reports_the_counts():
    """计数那部分不能因为加了摘要就丢掉 —— 它回答的是另一个问题（有没有补到新数据）。"""
    v = data_vintage(_mini())
    assert "rows=3" in v and "symbols=2" in v and "from=2024-01-02" in v
    assert "prices=" in v
