"""读取层：从日线合成周线，并在读取时做复权。

为什么复权放在读取而不是落库（对应报告 §3.3 的硬规则）：
  对「前复权序列」做增量 append 会造出假单日跌幅。实测过 MO −15.52% 这种
  不存在的单日跌幅，之后所有 ATR / z-score / 状态过滤器全部失真。
  所以库里存 as-traded 原始价 + 公司行动事件，每次读取现算。

为什么必须复权：3 年窗口里 NVDA 有 10:1 拆股（2024-06-10）、WMT 有 3:1
（2024-02-26）。不复权的话周线上会出现一根假的 −90% 巨阴线，趋势判断直接废掉。

用法：
    from series import weekly, daily
    w = weekly("NVDA")                      # 默认（数据源已拆股复权，见下）
    w = weekly("NVDA", adjust="total")      # 再叠加分红 → 总收益口径
    w = panel_weekly(["NVDA","AAPL"])       # 多标的长表，做横截面用

命令行自检：
    python src/series.py NVDA
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd

import db

OHLC = ["open", "high", "low", "close"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "volume": "sum", "dividends": "sum", "splits": "prod"}


def source_is_split_adjusted(con: sqlite3.Connection | None = None) -> tuple[bool, str]:
    """检测数据源给的 Close 是否**已经**做了拆股复权。

    为什么要检测：这是本项目踩过的一个真坑。Yahoo 返回的 Close 事实上已经是
    拆股后价位（实测 NVDA 2024-06-10 标了 10:1 拆股，但前一天收盘 120.89、
    当天 121.79，同价位区间；真实未复权价应在 $1,200 量级）。如果读取层再乘一次
    1/split，就会在拆股次日造出一根 +892% 的假巨阳。

    判据：拆股当天的实际收益是否**接近拆股隐含的跳幅**，并取所有大拆股事件的**中位数**。
      - 未复权时，10:1 拆股当天会跌约 1/10−1 = −90%；1:15 反向拆股会涨约 +1400%
      - 已复权时，当天就是一个普通日收益（几个百分点）
    取中位而不是取最大，是为了不被单个噪音事件带偏（实测 ALM 一次 2:3 反向拆股
    当天真的涨了 35%，单看它会误判）。

    本机实测结论（2026-10-14）：596 只里 51 次 3:1 以上大拆股，吻合度中位 **0.01**、
    最大 0.10 → **Yahoo 的 Close 已经是拆股复权后的价位**。
    换数据源（marketdata.app / EODHD）时必须重跑这个检测。
    """
    own = con is None
    con = con or db.connect()
    try:
        rows = con.execute(
            "SELECT symbol, date, close, splits FROM ohlcv WHERE splits>0 "
            "ORDER BY symbol, date").fetchall()
        if not rows:
            return True, "窗口内无拆股事件，无法判定（按已复权处理）"
        ratios = []
        for sym, d, c, sp in rows:
            if not sp or sp <= 0:
                continue
            implied = abs(1.0 / sp - 1.0)
            if implied < 0.5:            # 只看 3:1 以上的大拆股，信号最强
                continue
            prev = con.execute(
                "SELECT close FROM ohlcv WHERE symbol=? AND date<? "
                "ORDER BY date DESC LIMIT 1", (sym, d)).fetchone()
            if not prev or not prev[0]:
                continue
            ratios.append(abs(c / prev[0] - 1) / implied)
        if len(ratios) < 3:
            return True, f"大拆股样本仅 {len(ratios)} 个，不足以判定（按已复权处理）"
        ratios.sort()
        med = ratios[len(ratios) // 2]
        if med < 0.5:
            return True, (f"已拆股复权（{len(ratios)} 次大拆股，吻合度中位 {med:.2f}，"
                          f"最大 {ratios[-1]:.2f}；越接近 1 越像未复权）")
        return False, (f"疑似未复权（{len(ratios)} 次大拆股，吻合度中位 {med:.2f}）")
    finally:
        if own:
            con.close()




def _back_factor(df: pd.DataFrame, adjust: str) -> pd.Series:
    """每个日期对应的「向后复权因子」，adj_price(t) = raw_price(t) × factor(t)。

    ⚠️ 这里**不做拆股复权**，因为数据源给的 Close 已经是拆股后价位
    （见 source_is_split_adjusted 的说明）。再做一次就是重复缩放。

    只做分红：Adj(t) = Close(t) × ∏_{事件晚于 t} (1 − D_i/前收盘_i)。
    这样得到的是总收益口径序列，用于「含股息的趋势/收益」分析。
    """
    n = len(df)
    f = np.ones(n, dtype=float)
    if adjust == "total":
        div = df["dividends"].fillna(0).to_numpy(float)
        prev_close = df["close"].shift(1).to_numpy(float)
        for i in range(n):
            if div[i] > 0 and prev_close[i] == prev_close[i] and prev_close[i] > 0:
                f[i] = max(1.0 - div[i] / prev_close[i], 1e-6)
    # 从今往回的累计乘积，整体右移一位把「当日」排除掉
    rev = np.cumprod(f[::-1])[::-1]          # rev[i] = ∏ f[i..n-1]
    out = np.ones(n, dtype=float)
    out[: n - 1] = rev[1:]                   # out[i] = ∏ f[i+1..n-1]
    return pd.Series(out, index=df.index)




def daily(symbol: str, adjust: str | None = None,
          con: sqlite3.Connection | None = None) -> pd.DataFrame:
    own = con is None
    con = con or db.connect()
    try:
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume, dividends, splits "
            "FROM ohlcv WHERE symbol=? ORDER BY date", con, params=(symbol,))
    finally:
        if own:
            con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    if adjust:
        f = _back_factor(df, adjust)
        for c in OHLC:
            df[c] = df[c] * f
    return df


def _resample(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg({**AGG, "dividends": "sum", "splits": "prod"})
    return w.dropna(subset=["close"])


def weekly(symbol: str, adjust: str | None = None,
           con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """周线（周五为周末端）。默认拆股复权。

    注意末端：如果今天不是周五，最后一根是「不完整周」，做形态判断时要么丢弃
    要么显式标记。`complete_only=True` 可只保留完整周。
    """
    d = daily(symbol, adjust=adjust, con=con)
    if d.empty:
        return d
    return _resample(d)


def weekly_complete(symbol: str, adjust: str | None = None,
                    con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """只保留完整周（丢掉当前未走完的那根）。做突破判断时更安全。"""
    w = weekly(symbol, adjust=adjust, con=con)
    if not w.empty and w.index[-1] > pd.Timestamp.now().normalize():
        w = w.iloc[:-1]
    return w


def panel_weekly(symbols: list[str], adjust: str | None = None) -> pd.DataFrame:
    """多标的周线长表：date / symbol / open / high / low / close / volume。

    做横截面（相对强度、排序）时用这个，比逐只 DataFrame 好处理。
    """
    con = db.connect()
    try:
        out = []
        for s in symbols:
            w = weekly(s, adjust=adjust, con=con)
            if w.empty:
                continue
            w = w.reset_index()
            w["symbol"] = s
            out.append(w[["date", "symbol", *OHLC, "volume"]])
    finally:
        con.close()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _selfcheck(symbols: list[str]) -> None:
    ok, msg = source_is_split_adjusted()
    print(f"数据源复权状态：{'已拆股复权' if ok else '疑似未复权'} — {msg}\n")
    con = db.connect()
    try:
        print(f"{'sym':<7}{'日线':>6}{'周线':>6}{'拆股事件':>22}{'最大单日(默认)':>16}{'最大单日(总收益)':>18}")
        for s in symbols:
            d = daily(s, adjust=None, con=con)
            t = daily(s, adjust="total", con=con)
            wk = weekly(s, adjust=None, con=con)
            if d.empty:
                print(f"{s:<7} 无数据")
                continue
            sp = ",".join(f"{x.date()}×{v:g}" for x, v in
                          d["splits"][d["splits"] > 0].items()) or "无"
            mr = d["close"].pct_change().abs().max() * 100
            mt = t["close"].pct_change().abs().max() * 100
            print(f"{s:<7}{len(d):>6}{len(wk):>6}{sp:>22}{mr:>15.2f}%{mt:>17.2f}%")
    finally:
        con.close()


if __name__ == "__main__":
    syms = sys.argv[1:] or ["NVDA", "WMT", "AAPL", "AGNC"]
    _selfcheck(syms)
    w = weekly(syms[0])
    print(f"\n=== {syms[0]} 周线最近 6 周 ===")
    print(w.tail(6)[OHLC + ["volume"]].round(2).to_string())
    print(f"\n共 {len(w)} 根周线；区间 {w.index[0].date()} → {w.index[-1].date()}")
    print(f"最后 3 根是否含未走完的当周: {w.index[-1].date()} > 今天 {pd.Timestamp.now().date()}"
          f" → {'是（用 weekly_complete 可丢掉）' if w.index[-1] > pd.Timestamp.now().normalize() else '否'}")

