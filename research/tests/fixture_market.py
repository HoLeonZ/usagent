"""确定性合成市场 —— 让 §6 的行为锁**在任何机器上都能跑**（审计 G1）。

为什么要这一层：真正的行为锁（`test_golden_trades`）需要一个 277MB 的
`data/usagent.db`，而它**不入库**。于是任何全新 clone、任何 CI 上，那把锁都会静默
退化成 `skip` —— "有护栏"与"护栏在跑"是两件事，而此前只有前者。

本模块用一个**纯确定性的合成市场**（无随机数）补上这一环：
  * 24 只 / 520 个交易日 / 12,480 行 ⇒ 整个锁跑完是秒级；
  * 两条策略路径都**真的出信号**（不是空转的 0 笔成交 —— 那样的锁几乎测不到东西）；
  * 形状是**被条件反推出来的**，不是随手写的（见下）。

形状与它满足的条件（每一条都是实测踩出来的）：

  * `trend` 型（三分之二）：长期上行 + 每 40 天一次三日 −3% 回调。
    ⇒ `uptrend-pullback` 的 A~F：`mom120 ≥ 18%`、`above_ma60_ratio ≥ 0.6693`、
    `down_streak ∈ [3,10]`、`ret1 ∈ [−8.8%, −2%]`、`amount60 ≥ $3e7`。
  * `stairs` 型（三分之一）：涨 30 日 → **五日 −5%/日深跌 40%** → 反弹 25 日（+1.5%/日）。
    ⇒ 族型 preset（`str` 短周期反转）要求**同时**满足三条几何，别处踩过两个坑：
      - 它要 `ret5 ≤ −12%`（**锐跌**）：慢跌不触发（−0.6%/日 × 60 日累计 −30%，
        但 `drop5` 只有 −3%）；
      - 它要 `dd120 ≤ −20%`（离 120 日高点够远）：锯齿形（−15% 后 +52%）不触发，
        因为净上行让 `dd120` 到不了 −20%；
      - 它还要 `ss_lo120 ≥ +5%`（**但没跌到 120 日低点附近**，避免退市边缘）：
        所以必须是"深跌钉出低点 → 反弹出低点 → 再跌但不破前低"的阶梯，而不是一路下行。

⚠️ 这是**测试资产**，不是研究数据：它不参与任何结论，也不该被当成市场样本。
它的唯一职责是让"交易列表漂移"这件事在**没有真库**的地方也能被抓住。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_SYMBOLS = 24
N_DAYS = 520
START = "2023-01-02"
VOLUME = 2_000_000.0


def _closes(kind: str, n_days: int, k: int) -> list[float]:
    px, out = 20.0 + 5.0 * k, []
    for i in range(n_days):
        if kind == "trend":
            cycle = i % (40 + k % 7)
            r = 0.0012 + 0.0004 * k + 0.004 * np.sin(i / 11.0 + k)
            if 0 < cycle <= 3:
                r = -0.030
        else:                                   # stairs：见模块 docstring 的三条几何
            ph = i % 65
            r = 0.005 if ph < 30 else (-0.050 if ph < 40 else 0.015)
        px *= (1.0 + r)
        out.append(px)
    return out


def rows() -> list[tuple]:
    """按 `db.ohlcv` 的列序产出全部行（与 `download_ohlcv.COLS` 同构）。

    `source='fixture'`：这一列不是装饰 —— 校验里有"同一标的不混源"，
    合成数据必须自报家门，免得将来有人把 fixture 库与真库混在一张表里。
    """
    dates = pd.bdate_range(START, periods=N_DAYS)
    out: list[tuple] = []
    for k in range(N_SYMBOLS):
        sym = f"SYN{k:02d}"
        c = pd.Series(_closes("trend" if k % 3 else "stairs", N_DAYS, k))
        o = c.shift(1).fillna(c.iloc[0])
        h = np.maximum(o.to_numpy(), c.to_numpy()) * 1.004
        lo = np.minimum(o.to_numpy(), c.to_numpy()) * 0.996
        for i, d in enumerate(dates):
            out.append((sym, d.strftime("%Y-%m-%d"), float(o.iloc[i]), float(h[i]),
                        float(lo[i]), float(c.iloc[i]), VOLUME, 0.0, 0.0,
                        "fixture", "fixture"))
    return out


def build(path) -> Path:
    """建一个合成库并返回路径。**幂等**（INSERT OR REPLACE）。"""
    import db as dbmod
    p = Path(path)
    dbmod.init(p)
    with dbmod.session(p) as con:
        con.executemany(
            "INSERT OR REPLACE INTO ohlcv (symbol,date,open,high,low,close,volume,"
            "dividends,splits,fetched_at,source) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows())
    return p
