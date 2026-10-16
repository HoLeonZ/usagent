"""读取层:usagent SQLite → 研究用长表。

口径(设计文档 §1.1):
    价格用 series.daily(symbol, adjust="total") —— 拆股已由数据源复权,
    这里再叠分红 → 总收益口径。理由见 hiagent/uptrend_pullback/data.py 的
    模块 docstring:做多回测必须用复权价,否则除息日的裸价跳空会被误判为止损。
    **直接复用 src/series.py,不重写复权数学。**

    amount 列是代理量(设计文档 §1.2):usagent 的 ohlcv 没有成交额字段,
    用 **volume × as-traded close** 合成,以支撑 hiagent 的 E 条件。
    ⚠️ 不能用复权 close:E 条件把它比**绝对美元阈值**,用复权价会让 T 日的判定
       依赖 T 之后的分红(见 to_panel_frame 的 docstring)。

一次性载入全历史而不是逐窗口载入,有两个好处:
  1. 指标只需算一遍(见 indicators 模块),而非 34 个窗口各算一遍;
  2. 预热更充分 —— 逐窗口载入时每个窗口的 MA120 都只拿到本窗口+400 自然日的
     数据,全历史载入则拿到全部既有历史。
这是等价变换:全体计算后再切片,与「带足够预热的逐窗口计算」在所有因果指标上
逐位相同(test_causality 钉住这条)。
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable

import pandas as pd

import series

logger = logging.getLogger(__name__)

# 指标最长回看:ma120 与 mom120 都需 120 个交易日;above_ma60_ratio 需 60+60。
# 因此 3 年数据里最早的可用信号日约为数据起点之后第 120 个交易日。
MAX_LOOKBACK_BARS = 120

PANEL_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]

# as-traded 价(= 未做**分红**回补)。数据源给的 Close 本身已是拆股后价位
# (src/series.py::source_is_split_adjusted 实测),所以「raw」在这里的含义就是
# 「没有把未来分红折进去」的那一套价 —— 止损伤、跳空判定、逐日市值与现金记账用它。
RAW_COLS = ["raw_open", "raw_high", "raw_low", "raw_close"]


def to_panel_frame(sym: str, adj: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """把一个标的的 (复权序列, as-traded 序列) 合成面板行。

    价格分两套(CLAUDE.md §3(b) Dual-Price System):
      * `open/high/low/close` —— 分红回补后的价,供数学指标(MA / MACD / ATR / 动量)用;
      * `raw_open/.../raw_close` —— as-traded 价,供止损止盈触发、跳空判定、逐日市值
        与现金记账用。全仓 grep 不到第二条价格序列,是 2026-10-15 审计的一处 HIGH。

    ⚠️ `amount` 必须用 **raw_close** 算。用复权 close 算会让 T 日的成交额成为
       T **之后**所有分红的函数,而 E 条件把它与绝对美元常数 [3e7, 3e8] 比较 ——
       于是「能不能进池」变成数据 vintage 的函数(实测 740 个就绪格的判定被翻转)。
       比例型指标对同一个因子免疫(分子分母同乘一个常数),所以只有绝对比对要改。
    """
    dates_adj = pd.DatetimeIndex(adj.index)
    dates_raw = pd.DatetimeIndex(raw.index)
    if not dates_adj.equals(dates_raw):
        raise PanelError(
            f"{sym}: 复权序列与 as-traded 序列的日期轴不一致"
            f"(复权 {len(dates_adj)} 行 / as-traded {len(dates_raw)} 行)"
        )

    out = adj.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    r = (raw.reset_index()[["date", "open", "high", "low", "close"]]
         .rename(columns={"open": "raw_open", "high": "raw_high",
                          "low": "raw_low", "close": "raw_close"}))
    out = out.merge(r, on="date", how="left", validate="one_to_one")
    out["symbol"] = sym
    if out[RAW_COLS].isna().any().any():
        raise PanelError(f"{sym}: as-traded 价有缺口 —— 两条序列日期对不上")
    out["amount"] = out["volume"] * out["raw_close"]
    return out


class PanelError(RuntimeError):
    """数据层不可用 / 结果为空 —— fail-fast,不静默返回空表。"""


def load_full_panel(
    symbols: Iterable[str] | None = None,
    *,
    adjust: str = "total",
    universe: dict | str | None = None,
) -> pd.DataFrame:
    """载入全部标的的全部历史,返回长表。

    symbols 为 None 时,从 ohlcv 表里取全部 distinct symbol。
    adjust 默认 "total"(拆股+分红复权,总收益口径)。

    `universe` 是**取池声明**（见 research/universe.py）:挂一列因果可判定的
    `eligible`,不删行。此前的 `universe_path` 形参是**死参数**（声明了却没人读,
    2026-10-17 审计 A2）,已删除并换成真正会被消费的 `universe`。
    """
    import db  # 延迟导入:调用方可能用 -m 从仓库根运行

    con = db.connect()
    try:
        if symbols is None:
            try:
                rows = con.execute(
                    "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
            except sqlite3.OperationalError as exc:
                # data/*.db 不入库(§7),fresh clone 上第一条该跑的是取数。裸抛
                # sqlite3.OperationalError 会让新读者不知道要干什么(§0 Fail-Fast 要求
                # 立刻抛,但也要抛得可行动)。
                raise PanelError(
                    f"读不到 ohlcv 表({exc})—— data/*.db 不随仓库分发,先跑取数:\n"
                    f"    python src/download_ohlcv.py --universe oi100 --as-of <YYYY-MM-DD>\n"
                    f"    (--as-of 钉住数据 vintage;窗口与命令见 research/README.md 的「复现」)"
                ) from exc
            symbols = [r[0] for r in rows]
        symbols = list(symbols)
        if not symbols:
            raise PanelError("ohlcv 表里没有任何 symbol —— 先跑 src/download_ohlcv.py")

        frames: list[pd.DataFrame] = []
        for sym in symbols:
            # 两套价:复权(指标) + as-traded(执行/记账)。见 to_panel_frame 的 docstring。
            adj = series.daily(sym, adjust=adjust, con=con)
            raw = series.daily(sym, adjust=None, con=con)
            if adj.empty or raw.empty:
                continue
            frames.append(to_panel_frame(sym, adj, raw))
    finally:
        con.close()

    if not frames:
        raise PanelError(
            f"{len(symbols)} 个标的全部无数据 —— 检查 DB 与 universe 是否对得上"
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    logger.info(
        "panel: %d 行 / %d 标的 / %s → %s (adjust=%s)",
        len(panel), panel["symbol"].nunique(),
        panel["date"].min().date(), panel["date"].max().date(), adjust,
    )
    panel = panel[PANEL_COLS + RAW_COLS + ["dividends", "splits"]]
    # 取池资格：挂列不删行（见 research/universe.attach 的 docstring）。
    from research import universe as mod_universe
    return mod_universe.attach(panel, universe)


def load_panel_for_preset(preset: dict, *, adjust: str = "total") -> pd.DataFrame:
    """按 preset 声明的 `universe` 载入面板 —— **取池的唯一入口**。

    存在的理由：preset 声明过 `universe`，而引擎此前根本不读它（core/preset_contract.py
    的 docstring 自己承认），回测池等于"DB 里恰好有什么"（2026-10-17 审计 A2）。
    把入口收成一个函数之后，「忘了带池子」在代码层就没有位置了 ——
    这也正是本仓库为根除"声明了却没人读"而立的规矩（§2(a)）。
    """
    if not isinstance(preset, dict):
        raise PanelError(f"load_panel_for_preset 需要 preset dict，得到 {type(preset).__name__}")
    return load_full_panel(adjust=adjust, universe=preset.get("universe"))


def data_vintage(panel: pd.DataFrame) -> str:
    """数据 vintage 指纹:行数 / 标的数 / 起止日。不含任何收益信息(§7)。

    被 run_baseline 的结果 JSON 与 golden 基线共用 —— 没有它,「当前代码 + 一条命令」
    复现不出任何历史数字,因为 data/*.db 本身不入库(§0/§7)。
    """
    return (f"rows={len(panel)};symbols={panel['symbol'].nunique()};"
            f"from={panel['date'].min():%Y-%m-%d};to={panel['date'].max():%Y-%m-%d}")


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
