"""第三步：从落盘的期权链反解 IV，算出 30 天恒定到期 ATM IV 与 IV Rank。

为什么不用厂商的 IV 字段：
  报告 §4.4 实测 yfinance 的 put IV 在 ITM/ATM 只有 18-24% 落在自算值 ±5% 内
  （call 在 OTM 是 97.7-98.1%）。所以统一**从 mid 反解 Black-Scholes**。

三条铁律（报告 §4.2，不做这三件事攒出来的 IV Rank 是错的）：
  1. 常数期限 30 日插值（在方差域做，不是直接对 IV 线性插值）
  2. 用 mid 反解，绝不用 bid 或 ask（ask 系统性 +3.4 vp、bid −1.0 vp）
  3. r 与 q 全程固定同一个值，不能逐日变（会往序列里注入台阶）

关于 q=0 的说明：IV Rank 是相对自身的度量，反解时固定的常数偏移会在
Rank 里对消；真正有害的是「逐日变化的 r/q」。所以固定值优于时变值。

用法：python src/derive_iv.py [--exclude-earnings]
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date

import numpy as np
import pandas as pd

import db
from config import load_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("iv")

RISK_FREE = 0.040      # 固定
DIV_YIELD = 0.0        # 固定（见模块 docstring）
TARGET_DAYS = 30
LOOKBACK = 252


def _npdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _npcf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, r, q, v, is_call):
    if T <= 0 or v <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    sq = v * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * v * v) * T) / sq
    d2 = d1 - sq
    fwd = S * math.exp(-q * T)
    dis = math.exp(-r * T)
    if is_call:
        return fwd * _npcf(d1) - K * dis * _npcf(d2)
    return K * dis * _npcf(-d2) - fwd * _npcf(-d1)


def implied_vol(mid, S, K, T, is_call, lo=1e-3, hi=5.0, tol=1e-6, iters=80):
    """二分反解。返回 None 表示无法反解（价格低于内在价值等）。"""
    if mid is None or S <= 0 or K <= 0 or T <= 0 or mid <= 0:
        return None
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if mid < intrinsic * 0.999:       # 低于内在价值：无解，属深实值无时间价值
        return None
    p_lo = bs_price(S, K, T, RISK_FREE, DIV_YIELD, lo, is_call)
    p_hi = bs_price(S, K, T, RISK_FREE, DIV_YIELD, hi, is_call)
    if not (p_lo <= mid <= p_hi):
        return None
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        p = bs_price(S, K, T, RISK_FREE, DIV_YIELD, m, is_call)
        if abs(p - mid) < tol:
            return m
        if p < mid:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def atm_iv_by_expiry(rows: pd.DataFrame) -> list[tuple[int, float]]:
    """每个到期日一个 (dte, ATM IV)。

    ATM = 最接近现货的行权价；该档的 call 与 put 都反解一次再取均值，
    两边都有解才算（单边有解时用单边）。

    注意：不要求行权价档数 —— 低价股行权价间距可达 $2.5，±10% 带内可能只有
    一两档，但算 ATM IV 需要的只是「最接近现货的那一档」。
    """
    out = []
    for dte, g in rows.groupby("dte"):
        spot = float(g["underlying_price"].iloc[0])
        if spot <= 0:
            continue
        k = float(g.iloc[(g["strike"] - spot).abs().argsort()].iloc[0]["strike"])
        ivs = []
        for _, r in g[g["strike"] == k].iterrows():
            mid = _mid(r)
            if mid is None:
                continue
            v = implied_vol(mid, spot, float(r["strike"]), float(dte) / 365.0,
                            str(r["right"]).upper().startswith("C"))
            if v is not None:
                ivs.append(v)
        if ivs:
            out.append((int(dte), float(np.mean(ivs))))
    return sorted(out)



def _mid(r):
    b, a = r.get("bid"), r.get("ask")
    try:
        b, a = float(b), float(a)
    except (TypeError, ValueError):
        return None
    if not (b == b and a == a) or b <= 0 or a <= 0 or a < b:
        return None
    return 0.5 * (a + b)


def interp_30d(pairs: list[tuple[int, float]]) -> float | None:
    """在方差域（σ²T 对 T 线性）插值到 30 天。"""
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0][1]
    T = np.array([d / 365.0 for d, _ in pairs])
    V = np.array([(v ** 2) * t for (_, v), t in zip(pairs, T)])
    tgt = TARGET_DAYS / 365.0
    if tgt <= T[0]:
        var = V[0] * tgt / T[0]
    elif tgt >= T[-1]:
        var = V[-1]
    else:
        order = np.argsort(T)
        var = float(np.interp(tgt, T[order], V[order]))
    return math.sqrt(max(var, 0.0) / tgt)


def realized_vol(closes: np.ndarray, window: int = 20) -> float | None:
    if len(closes) < window + 1:
        return None
    r = np.diff(np.log(closes[-(window + 1):]))
    return float(np.std(r, ddof=1) * math.sqrt(252))


def rank_pctile(series: pd.Series) -> tuple[float | None, float | None]:
    s = series.dropna()
    if len(s) < 20:
        return None, None
    cur, lo, hi = s.iloc[-1], s.min(), s.max()
    rank = 50.0 if hi == lo else (cur - lo) / (hi - lo) * 100
    pct = float((s < cur).mean() * 100)
    return float(rank), pct


def derive(symbols: list[str] | None) -> int:
    db.init()
    with db.session() as con:
        syms = symbols or [r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM option_chain ORDER BY symbol")]
        log.info("反解 %d 只", len(syms))
        written = 0
        for i, sym in enumerate(syms, 1):
            rows = pd.read_sql(
                "SELECT snapshot_date d, expiry, dte, strike, right, bid, ask, "
                "underlying_price FROM option_chain WHERE symbol=?", con, params=(sym,))
            if rows.empty:
                continue
            per_date = []
            for d, g in rows.groupby("d"):
                iv = interp_30d(atm_iv_by_expiry(g))
                if iv is not None:
                    per_date.append((d, iv))
            if not per_date:
                continue
            s = pd.DataFrame(per_date, columns=["date", "iv30"]).sort_values("date")
            s["iv_rank"] = s["iv30"].expanding(LOOKBACK).apply(
                lambda w: rank_pctile(w)[0], raw=False)
            s["iv_percentile"] = s["iv30"].expanding(LOOKBACK).apply(
                lambda w: rank_pctile(w)[1], raw=False)

            px = pd.read_sql(
                "SELECT date, close FROM ohlcv WHERE symbol=? ORDER BY date", con, params=(sym,))
            hv = {}
            if not px.empty:
                c = px["close"].to_numpy(float)
                for j, d in enumerate(px["date"]):
                    v = realized_vol(c[: j + 1])
                    if v is not None:
                        hv[d] = v

            out = [(d, sym, iv, hv.get(d), (iv - hv[d]) if d in hv else None,
                    rk, pc, None)
                   for (d, iv), rk, pc in zip(per_date, s["iv_rank"], s["iv_percentile"])]
            written += db.upsert_many(
                con, "iv_history",
                ["date", "symbol", "iv30", "hv20", "vrp", "iv_rank", "iv_percentile", "n_used"],
                out)
            if i % 50 == 0:
                con.commit()
                log.info("[%d/%d] 累计 %d 行", i, len(syms), written)
        con.commit()
    log.info("完成，写入 %d 行", written)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=None, choices=["all", "oi100", "tight"])
    ap.add_argument("--symbols", default=None)
    a = ap.parse_args()
    if a.symbols:
        syms = [s.strip().upper() for s in a.symbols.split(",")]
    elif a.universe:
        syms = list(load_universe(a.universe).symbols)
    else:
        syms = None
    derive(syms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
