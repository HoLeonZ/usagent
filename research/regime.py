"""大盘择时 —— hiagent/uptrend_pullback/regime.py 的移植。

用样本内个股自身合成等权指数,判断多头环境。为什么需要:目标年主板中位数个股
-8.1%、仅 38.8% 上涨(hiagent 的实测口径),在这种环境无差别做多必然亏损。

因果性:T 日的 regime 只用 ≤T 的数据。ew_index 是从面板首日起的 cumprod,
在 T 处只依赖 ≤T 的 ret1;ew_ma 是它的 20 日**滞后**均值。没有任何前视。

一处必须声明的偏差:这个「等权指数」本身建立在一个**幸存者偏差**的股票池上
(usagent 的 596 只是今日 OI≥100 的名单),所以它不是一个中性的市场基准,
而是一个偏乐观的基准。这一点在做 alpha 归因时**对我们有利**——分子分母同时被
抬高,做差时大部分对消。绝对 CAGR 仍应读作上界。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REGIME_COLS = ["date", "ew_index", "ew_ma", "breadth_sm", "regime_ok"]


def compute_regime(
    panel_ind: pd.DataFrame,
    *,
    ma_window: int = 20,
    breadth_window: int = 20,
    min_breadth: float = 0.45,
    require_rising: bool = False,
    rising_lookback: int = 5,
) -> pd.DataFrame:
    """返回 [date, ew_index, ew_ma, breadth_sm, regime_ok]。

    regime_ok = 等权指数 > 自身 MA(ma_window)
                且 市场宽度 ≥ min_breadth
                且(可选)MA 本身在上行

    require_rising 过滤「熊市反抽」——指数短暂上穿均线但均线仍在下行。
    ⚠️ hiagent 已实测 require_rising 更差(preset_search §5 M4),默认 False;
       保留该参数只为复现那次证伪,不要在 preset 里打开。
    """
    needed = {"date", "ret1", "above_ma20"}
    missing = needed - set(panel_ind.columns)
    if missing:
        raise ValueError(f"compute_regime 需要 {sorted(needed)},缺 {sorted(missing)}")

    daily = (
        panel_ind.groupby("date", sort=True)
        .agg(ret=("ret1", "mean"), breadth=("above_ma20", "mean"))
        .reset_index()
    )
    if daily.empty:
        raise ValueError("compute_regime 拿到空面板 —— fail-fast,不返回空 regime")

    daily["ret"] = daily["ret"].fillna(0.0)
    daily["ew_index"] = (1.0 + daily["ret"]).cumprod()
    daily["ew_ma"] = daily["ew_index"].rolling(ma_window, min_periods=ma_window).mean()
    daily["breadth_sm"] = daily["breadth"].rolling(breadth_window, min_periods=1).mean()

    ok = (daily["ew_index"] > daily["ew_ma"]) & (daily["breadth_sm"] >= min_breadth)
    if require_rising:
        ok &= daily["ew_ma"] > daily["ew_ma"].shift(rising_lookback)
    daily["regime_ok"] = ok.fillna(False)

    logger.info(
        "regime: %d/%d 天可做多 (%.1f%%)",
        int(daily["regime_ok"].sum()), len(daily),
        100.0 * daily["regime_ok"].mean(),
    )
    return daily[REGIME_COLS]


def regime_ok_dates(regime_df: pd.DataFrame) -> set[pd.Timestamp]:
    """regime 门为真的交易日集合,供信号层过滤用。"""
    return set(pd.to_datetime(regime_df.loc[regime_df["regime_ok"], "date"]).tolist())


def equal_weight_benchmark(regime_df: pd.DataFrame) -> pd.Series:
    """等权指数的日收益(基准),索引为 date。

    注意:这是**日度再平衡**的等权指数。作为 buy-and-hold 对照时,两者不完全可比
    (等权日再平衡隐含更高的换手);这一点在结论文档里要写明,而不是当成等价物。
    """
    return regime_df.set_index("date")["ew_index"].pct_change().fillna(0.0)
