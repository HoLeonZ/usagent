"""读取层:usagent SQLite → 研究用长表。

口径(设计文档 §1.1):
    价格用 series.daily(symbol, adjust="total") —— 拆股已由数据源复权,
    这里再叠分红 → 总收益口径。理由见 hiagent/uptrend_pullback/data.py 的
    模块 docstring:做多回测必须用复权价,否则除息日的裸价跳空会被误判为止损。
    **直接复用 src/series.py,不重写复权数学。**

    amount 列是代理量(设计文档 §1.2):usagent 的 ohlcv 没有成交额字段,
    用 volume × close 合成,以支撑 hiagent 的 E 条件。

一次性载入全历史而不是逐窗口载入,有两个好处:
  1. 指标只需算一遍(见 indicators 模块),而非 34 个窗口各算一遍;
  2. 预热更充分 —— 逐窗口载入时每个窗口的 MA120 都只拿到本窗口+400 自然日的
     数据,全历史载入则拿到全部既有历史。
这是等价变换:全体计算后再切片,与「带足够预热的逐窗口计算」在所有因果指标上
逐位相同(test_causality 钉住这条)。
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

import series

logger = logging.getLogger(__name__)

# 指标最长回看:ma120 与 mom120 都需 120 个交易日;above_ma60_ratio 需 60+60。
# 因此 3 年数据里最早的可用信号日约为数据起点之后第 120 个交易日。
MAX_LOOKBACK_BARS = 120

PANEL_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]


class PanelError(RuntimeError):
    """数据层不可用 / 结果为空 —— fail-fast,不静默返回空表。"""


def load_full_panel(
    symbols: Iterable[str] | None = None,
    *,
    adjust: str = "total",
    universe_path: str | None = None,
) -> pd.DataFrame:
    """载入全部标的的全部历史,返回长表。

    symbols 为 None 时,从 ohlcv 表里取全部 distinct symbol。
    adjust 默认 "total"(拆股+分红复权,总收益口径)。
    """
    import db  # 延迟导入:调用方可能用 -m 从仓库根运行

    con = db.connect()
    try:
        if symbols is None:
            rows = con.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
            symbols = [r[0] for r in rows]
        symbols = list(symbols)
        if not symbols:
            raise PanelError("ohlcv 表里没有任何 symbol —— 先跑 src/download_ohlcv.py")

        frames: list[pd.DataFrame] = []
        for sym in symbols:
            d = series.daily(sym, adjust=adjust, con=con)
            if d.empty:
                continue
            d = d.reset_index()
            d["symbol"] = sym
            frames.append(d)
    finally:
        con.close()

    if not frames:
        raise PanelError(
            f"{len(symbols)} 个标的全部无数据 —— 检查 DB 与 universe 是否对得上"
        )

    panel = pd.concat(frames, ignore_index=True)
    # amount 代理:成交额 ≈ 量 × 收盘(设计文档 §1.2)
    panel["amount"] = panel["volume"] * panel["close"]
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    logger.info(
        "panel: %d 行 / %d 标的 / %s → %s (adjust=%s)",
        len(panel), panel["symbol"].nunique(),
        panel["date"].min().date(), panel["date"].max().date(), adjust,
    )
    return panel[PANEL_COLS + ["dividends", "splits"]]


def first_signal_date(panel: pd.DataFrame) -> pd.Timestamp:
    """最早的「指标已就绪」日期。

    取全表第 MAX_LOOKBACK_BARS 个交易日(所有标的的交易日并集)。
    3 年数据里这意味着前约 120 个交易日无法产生信号 —— 评测期因此短于数据跨度,
    这一点必须在结论里如实写出来,不能把预热期算进 CAGR 的分母。
    """
    days = pd.Index(sorted(panel["date"].unique()))
    if len(days) <= MAX_LOOKBACK_BARS:
        raise PanelError(
            f"交易日只有 {len(days)} 天,不足以预热 {MAX_LOOKBACK_BARS} 根 K 线"
        )
    return pd.Timestamp(days[MAX_LOOKBACK_BARS])


def slice_window(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """按日期左闭右开切片。[start, end)。"""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    out = panel[(panel["date"] >= lo) & (panel["date"] < hi)]
    return out.reset_index(drop=True)
