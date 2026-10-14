"""指标层 —— hiagent/uptrend_pullback/signals.py::compute_indicators 的移植。

除了把 `thscode` 改名成 `symbol`,计算逐条保持原样。所有 rolling/shift/ewm 都按
symbol 分组,不存在跨股票泄露。

因果性(设计文档 §6,最重要的一条不变量):
    每个指标在 T 日的取值只依赖 T 及之前的行。整表计算后切片,与「截断到 T 再重算」
    必须逐位相同 —— test_causality 把这条钉死。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 60, 120)

INDICATOR_COLS = [
    "ma5", "ma10", "ma20", "ma60", "ma120",
    "above_ma20", "prev_close", "ret1",
    "above_ma60", "above_ma60_ratio",
    "down_day", "up_break", "down_id", "down_streak", "prev_down_streak",
    "high20", "pullback", "mom120", "amount60",
    "vol_ma20", "vol_ratio", "tr", "atr14", "atr_pct",
    "ema12", "ema26", "macd_dif", "macd_dea", "macd_bar", "macd_bar_prev",
]


def compute_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """追加均线 / 回调 / 连阴 / 动量 / 波动 / 流动性 / MACD 列。

    输入需含 [symbol, date, open, high, low, close, volume, amount]。
    """
    df = panel.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    # --- 均线 ---
    for w in MA_WINDOWS:
        df[f"ma{w}"] = g["close"].transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean()
        )
    df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)

    # --- 日收益(基于前一根 K 线收盘)---
    df["prev_close"] = g["close"].shift(1)
    df["ret1"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    # --- 趋势持续性:过去 60 日中 close > MA60 的占比 ---
    df["above_ma60"] = (df["close"] > df["ma60"]).astype(float)
    df["above_ma60_ratio"] = g["above_ma60"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()
    )

    # --- 连阴计数(按 symbol 分组,避免跨股票串联)---
    df["down_day"] = df["ret1"] < 0          # NaN 比较结果为 False
    df["up_break"] = (~df["down_day"]).astype(int)
    df["down_id"] = df.groupby("symbol")["up_break"].cumsum()
    df["down_streak"] = df.groupby(["symbol", "down_id"]).cumcount() + 1
    df.loc[~df["down_day"], "down_streak"] = 0
    df["prev_down_streak"] = g["down_streak"].shift(1)

    # --- 回调深度:距近 20 日最高价的跌幅(负值)---
    df["high20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    df["pullback"] = df["close"] / df["high20"] - 1.0

    # --- 中期动量(排序分):120 日涨幅 ---
    df["mom120"] = df["close"] / g["close"].shift(120) - 1.0

    # --- 流动性:60 日均成交额(代理量,见 panel.py)---
    df["amount60"] = g["amount"].transform(lambda s: s.rolling(60, min_periods=60).mean())

    # --- 量能:当日量 / 20 日均量 ---
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # --- ATR14 ---
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs(),
        ),
    )
    df["tr"] = tr
    df["atr14"] = g["tr"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["atr_pct"] = df["atr14"] / df["close"]

    # --- MACD(12/26/9 EMA)---
    df["ema12"] = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df["ema26"] = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd_dif"] = df["ema12"] - df["ema26"]
    df["macd_dea"] = g["macd_dif"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_bar"] = df["macd_dif"] - df["macd_dea"]
    df["macd_bar_prev"] = g["macd_bar"].shift(1)

    return df


def indicators_ready(df: pd.DataFrame) -> pd.Series:
    """所有信号依赖的列都非空的掩码 —— 预热完成的判据。"""
    need = ["ma120", "mom120", "amount60", "vol_ratio", "atr_pct",
            "macd_bar", "macd_bar_prev", "above_ma60_ratio", "prev_down_streak"]
    mask = pd.Series(True, index=df.index)
    for c in need:
        mask &= df[c].notna()
    return mask
