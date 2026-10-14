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


# ================================================================ 扩展指标
# 2026-10-14:为"把能用的指标都加上"补的这一组。全部按 symbol 分组、
# 只用 ≤T 的数据 —— 因果性由 test_indicators_ext 的截断重算测试钉住。

EXT_COLS = [
    # RSI(Wilder)
    "rsi14", "rsi7",
    # 布林带
    "bb_mid", "bb_sd", "bb_up", "bb_lo", "bb_pctb", "bb_width",
    # 随机指标 KDJ
    "stoch_k", "stoch_d",
    # 趋势强度 ADX/DMI
    "plus_di", "minus_di", "adx14",
    # 量价
    "obv", "obv_slope", "vol_z",
    # 位置
    "hh252", "ll252", "dist_high252", "dist_low252",
    # 波动
    "rvol20",
    # 横截面
    "mom60", "xs_mom120", "xs_mom60", "xs_rvol20",
]

RSI_PERIOD = 14
BB_PERIOD, BB_STD = 20, 2.0
STOCH_PERIOD, STOCH_SMOOTH = 14, 3
ADX_PERIOD = 14
HH252_BARS = 252


def _wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder 平滑 —— 等价于 alpha = 1/period 的 EWM。RSI/ADX 的标准口径。"""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def add_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """追加 RSI / 布林 / KDJ / ADX / OBV / 52 周位置 / 横截面相对强度。

    输入需已过 compute_indicators(用到 tr / ma20 / mom120 等)。
    """
    out = df.copy()
    g = out.groupby("symbol", group_keys=False)

    # ---- RSI(Wilder)------------------------------------------------
    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    for p in (RSI_PERIOD, 7):
        ag = gain.groupby(out["symbol"]).transform(lambda s, p=p: _wilder(s, p))
        al = loss.groupby(out["symbol"]).transform(lambda s, p=p: _wilder(s, p))
        rs = ag / al.replace(0.0, np.nan)
        rsi = (100.0 - 100.0 / (1.0 + rs)).where(al.notna())
        # 边界:窗口内**一次下跌都没有** → al == 0 → rs 为 inf → 上面会得 NaN。
        # 正确取值是 100(纯上涨)。这是经典 RSI 实现的坑:单调上涨的序列会
        # 整段变成 NaN,而 NaN 会静默退出所有下游筛选(看起来像"没信号")。
        rsi = rsi.mask((al == 0.0) & ag.notna() & (ag > 0.0), 100.0)
        # 完全无波动(ag 与 al 都是 0,价格横盘)→ 无方向,取中性 50
        rsi = rsi.mask((al == 0.0) & (ag == 0.0), 50.0)
        out[f"rsi{p}"] = rsi

    # ---- 布林带 -----------------------------------------------------
    out["bb_mid"] = g["close"].transform(
        lambda s: s.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean())
    out["bb_sd"] = g["close"].transform(
        lambda s: s.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0))
    out["bb_up"] = out["bb_mid"] + BB_STD * out["bb_sd"]
    out["bb_lo"] = out["bb_mid"] - BB_STD * out["bb_sd"]
    rng = (out["bb_up"] - out["bb_lo"]).replace(0.0, np.nan)
    out["bb_pctb"] = (out["close"] - out["bb_lo"]) / rng
    out["bb_width"] = rng / out["bb_mid"]

    # ---- 随机指标 KDJ -----------------------------------------------
    ll = g["low"].transform(
        lambda s: s.rolling(STOCH_PERIOD, min_periods=STOCH_PERIOD).min())
    hh = g["high"].transform(
        lambda s: s.rolling(STOCH_PERIOD, min_periods=STOCH_PERIOD).max())
    span = (hh - ll).replace(0.0, np.nan)
    out["stoch_k"] = 100.0 * (out["close"] - ll) / span
    out["stoch_d"] = out.groupby("symbol")["stoch_k"].transform(
        lambda s: s.rolling(STOCH_SMOOTH, min_periods=STOCH_SMOOTH).mean())

    # ---- ADX / DMI --------------------------------------------------
    up_move = g["high"].diff()
    down_move = -g["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    atr_w = out.groupby("symbol")["tr"].transform(lambda s: _wilder(s, ADX_PERIOD))
    pdi = 100.0 * plus_dm.groupby(out["symbol"]).transform(
        lambda s: _wilder(s, ADX_PERIOD)) / atr_w.replace(0.0, np.nan)
    mdi = 100.0 * minus_dm.groupby(out["symbol"]).transform(
        lambda s: _wilder(s, ADX_PERIOD)) / atr_w.replace(0.0, np.nan)
    out["plus_di"], out["minus_di"] = pdi, mdi
    denom = (pdi + mdi).replace(0.0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / denom
    out["adx14"] = dx.groupby(out["symbol"]).transform(lambda s: _wilder(s, ADX_PERIOD))

    # ---- OBV / 量能 z 分 --------------------------------------------
    sign = np.sign(g["close"].diff().fillna(0.0))
    out["obv"] = (sign * out["volume"]).groupby(out["symbol"]).cumsum()
    out["obv_slope"] = out.groupby("symbol")["obv"].transform(
        lambda s: s.diff(20) / s.abs().rolling(20, min_periods=20).mean().replace(0.0, np.nan))
    vmean = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    vsd = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).std(ddof=1))
    out["vol_z"] = (out["volume"] - vmean) / vsd.replace(0.0, np.nan)

    # ---- 52 周位置 ---------------------------------------------------
    out["hh252"] = g["high"].transform(
        lambda s: s.rolling(HH252_BARS, min_periods=HH252_BARS).max())
    out["ll252"] = g["low"].transform(
        lambda s: s.rolling(HH252_BARS, min_periods=HH252_BARS).min())
    out["dist_high252"] = out["close"] / out["hh252"] - 1.0
    out["dist_low252"] = out["close"] / out["ll252"] - 1.0

    # ---- 已实现波动 --------------------------------------------------
    out["rvol20"] = out.groupby("symbol")["ret1"].transform(
        lambda s: s.rolling(20, min_periods=20).std(ddof=1))
    out["mom60"] = out["close"] / g["close"].shift(60) - 1.0

    return out


def add_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """横截面相对强度:同一交易日、全池内的百分位排名。

    **因果性**:每个日期只用**当天**的横截面(各标的那一天的值),不涉及未来。
    这是"这只票相对同池其他票有多强",与"它自己涨了多少"不同 ——
    纯多头选股里,后者会被市场整体涨跌淹没。

    必须**在 add_extended_indicators 之后**调用(依赖 mom120 / mom60 / rvol20)。
    预热期的 NaN 会天然退出排名(pandas rank 默认 na_option='keep')。
    """
    out = df.copy()
    for src, dst in (("mom120", "xs_mom120"), ("mom60", "xs_mom60"),
                     ("rvol20", "xs_rvol20")):
        out[dst] = out.groupby("date")[src].rank(pct=True)
    return out


def compute_all(panel: pd.DataFrame) -> pd.DataFrame:
    """compute_indicators + add_extended_indicators + add_cross_sectional。"""
    return add_cross_sectional(add_extended_indicators(compute_indicators(panel)))
