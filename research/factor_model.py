"""归因：风格因子回归 + 跟踪误差 + 信息比率（审计 C4）。

缺陷本体：本仓库的"超额"一直**只有 CAGR 之差**。那回答不了机构里最常被问的那个问题：
**这份收益里有多少是风格 beta？** 一个长期拿高波动小盘股的策略，"跑赢基准"可能只是
承担了更多的市场/波动暴露，而不是选股能力；而 CAGR 之差对这种区别是瞎的。

本模块用**面板自己**构造风格因子代理（不需要外部因子库，也不需要我们没有的基本面数据）：

  * `market`  —— 可投资池等权指数（已有的 ew_index）；
  * `mom`     —— 横截面 120 日动量的多空（高三分位 − 低三分位，等权）；
  * `lowvol`  —— 横截面 20 日已实现波动的多空（**低**波 − **高**波）。

为什么是"多空价差"而不是"多头组合"：多空价差已经把市场 beta 减掉了，它衡量的是
**暴露于该风格维度本身的收益**，正好是回归里要的那一列。
（长空在**本模块里**只是测量工具，不经过 broker、不涉及借券假设 —— §4(g) 要求
任何**可交易**的空头声明借券模型，而这里不产生任何成交。）

⚠️ 已知缺口（不藏）：**没有价值因子**（我们没有基本面数据），**规模因子**只有一个
静态市值快照（`universe_nasdaq_all.json`，非 PIT ⇒ 用它会把未来信息带进来），
所以两者都不做。归因因此只覆盖 market / mom / lowvol —— 说清楚覆盖到哪里，
比给一个看起来完整的四因子表更诚实。

统计口径：t 值用 **Newey-West（HAC）**，因为日收益有自相关与波动聚集，
普通 OLS 的 t 会被系统性放大（本仓库实测日收益一阶自相关可达 +0.20）。
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from research import metrics as M

logger = logging.getLogger("factor_model")

TRADING_DAYS = 252
NW_LAGS = 5                 # Newey-West 滞后阶数（日频常用 5，覆盖一周的自相关）
N_QUANTILES = 3             # 三分位：多空两端
MIN_NAMES_PER_SIDE = 10     # 每一侧至少这么多名字才算得出因子收益


class FactorError(RuntimeError):
    """因子构造的前提不成立 —— fail-fast，不静默出一列假收益。"""


def style_factor_returns(ind: pd.DataFrame, *, lo: str, hi: str,
                         rebalance: str = "ME") -> pd.DataFrame:
    """构造 [market, mom, lowvol] 三列日频因子收益（索引=date）。

    因果性：每个再平衡日的分位划分**只用 ≤ 该日**的因子值（`xs_*` 列本身也是因果的），
    然后在**下一个**持有期内用实际收益计算价差 —— 没有用到划分日之后的信息去分组。

    `ind` 需含 [date, symbol, ret1, eligible, mom120, rvol20]。
    """
    need = {"date", "symbol", "ret1", "eligible", "mom120", "rvol20"}
    missing = need - set(ind.columns)
    if missing:
        raise FactorError(f"style_factor_returns 需要 {sorted(need)}，缺 {sorted(missing)}")

    df = ind.loc[ind["eligible"] & ind["ret1"].notna(),
                 ["date", "symbol", "ret1", "mom120", "rvol20"]].copy()
    if df.empty:
        raise FactorError("可投资行数为 0 —— 取池把整个面板筛掉了")
    dates = pd.Index(sorted(df["date"].unique()))
    lo_, hi_ = pd.Timestamp(lo), pd.Timestamp(hi)
    axis = dates[(dates >= lo_) & (dates < hi_)]
    if len(axis) < 40:
        raise FactorError(f"窗口内只有 {len(axis)} 个交易日，不足以做归因")

    piv = {c: df.pivot(index="date", columns="symbol", values=c)
           for c in ("ret1", "mom120", "rvol20")}
    reb = pd.Series(axis, index=axis).resample(rebalance).last().dropna().tolist()

    out = {c: pd.Series(0.0, index=axis) for c in ("mom", "lowvol")}
    for i, rd in enumerate(reb):
        # 持有期 = 本次再平衡日之后、下次再平衡日之前
        end = reb[i + 1] if i + 1 < len(reb) else axis[-1]
        hold = axis[(axis > rd) & (axis <= end)] if i + 1 < len(reb) else axis[axis > rd]
        if not len(hold):
            continue
        for col, name, sign in (("mom120", "mom", 1.0), ("rvol20", "lowvol", -1.0)):
            score = piv[col].loc[rd].dropna()
            if len(score) < 2 * MIN_NAMES_PER_SIDE:
                continue
            k = max(len(score) // N_QUANTILES, MIN_NAMES_PER_SIDE)
            hi_names = score.nlargest(k).index
            lo_names = score.nsmallest(k).index
            r = piv["ret1"].reindex(index=hold, columns=sorted(set(hi_names) | set(lo_names)))
            top = r[list(hi_names)].mean(axis=1)
            bot = r[list(lo_names)].mean(axis=1)
            # sign=+1：高值一端做多（动量）；sign=−1：低值一端做多（低波动）
            out[name].loc[hold] = (sign * (top - bot)).reindex(hold).fillna(0.0)
    res = pd.DataFrame(out)
    market = df.groupby("date")["ret1"].mean().reindex(axis).fillna(0.0)
    res["market"] = market
    return res[["market", "mom", "lowvol"]]


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS) -> dict:
    """带 HAC（Newey-West）标准误的 OLS。X **不含**截距列，函数自己加。

    为什么要 HAC：日收益有自相关与波动聚集，普通 OLS 假设独立同方差 ⇒ t 值被系统性
    放大（把噪声读成显著）。机构里报 alpha 的 t 值默认就该是 HAC 的。
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if n < X.shape[1] + 2:
        raise FactorError(f"样本 {n} 太少，无法估 {X.shape[1]} 个因子")
    A = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(A.T @ A)
    beta = XtX_inv @ A.T @ y
    resid = y - A @ beta
    # HAC（Newey-West, Bartlett 核）
    S = (A * resid[:, None]).T @ (A * resid[:, None])
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = (A[lag:] * resid[lag:, None]).T @ (A[:-lag] * resid[:-lag, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": t,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n": int(n), "lags": int(lags)}


def attribute(strategy_ret: pd.Series, factors: pd.DataFrame,
              bench_ret: pd.Series) -> dict:
    """策略日收益对 [market, mom, lowvol] 的归因 + 跟踪误差/信息比率。

    返回：
      alpha_annual / t_alpha   —— 年化截距与它的 HAC t 值（**这是"选股能力"的那一列**）
      beta_* / t_*             —— 各风格暴露
      r2                       —— 风格能解释多少
      te_annual / ir           —— 对基准的跟踪误差与信息比率（§5(e) 要求披露的量）
    """
    j = pd.concat([strategy_ret.rename("s"), factors, bench_ret.rename("b")],
                  axis=1).dropna()
    if len(j) < 60:
        raise FactorError(f"共同样本只有 {len(j)} 天，不足以归因")
    fit = newey_west_ols(j["s"].to_numpy(), j[["market", "mom", "lowvol"]].to_numpy())
    names = ["alpha", "market", "mom", "lowvol"]
    out = {"n_days": int(fit["n"]), "r2": fit["r2"], "lags": fit["lags"],
           "alpha_annual": float(fit["beta"][0]) * TRADING_DAYS,
           "t_alpha": float(fit["t"][0])}
    for i, nm in enumerate(names[1:], start=1):
        out[f"beta_{nm}"] = float(fit["beta"][i])
        out[f"t_{nm}"] = float(fit["t"][i])
    ex = (j["s"] - j["b"]).to_numpy()
    sd = float(np.std(ex, ddof=1))
    # 跟踪误差为 0 是**有意义的退化**（策略就是基准），报 0 而不是 nan；
    # 而 IR = 超额均值/跟踪误差 在 sd=0 时是 0/0 —— 那个才是未定义。
    out["te_annual"] = sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    out["ir"] = (float(np.mean(ex)) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    return out


def describe(d: dict) -> str:
    def f(v):
        return "—" if v is None or not np.isfinite(v) else f"{v:+.2f}"
    return (f"alpha 年化 {d['alpha_annual']*100:+.2f}% (HAC t={f(d['t_alpha'])}) ｜ "
            f"market β {d['beta_market']:+.2f}(t={f(d['t_market'])}) "
            f"mom β {d['beta_mom']:+.2f}(t={f(d['t_mom'])}) "
            f"lowvol β {d['beta_lowvol']:+.2f}(t={f(d['t_lowvol'])}) ｜ "
            f"R² {d['r2']:.2f} ｜ 跟踪误差 {d['te_annual']*100:.1f}% IR {f(d['ir'])}"
            f" ｜ {d['n_days']} 天，NW 滞后 {d['lags']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="风格因子归因（market/mom/lowvol）")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--preset", default="us_uptrend_pullback_v1")
    ap.add_argument("--lo", default=None)
    ap.add_argument("--hi", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if not a.run:
        ap.print_help()
        return 0

    from research import broker, indicators, panel as P, presets, regime, signals
    from research import families as F
    ps = presets.get_preset(a.preset)
    pan = P.load_panel_for_preset(ps)
    ind = indicators.compute_all(pan)
    lo = a.lo or str(ind["date"].iloc[125].date())
    hi = a.hi or str(ind["date"].max().date() + pd.Timedelta(days=1))
    reg = regime.compute_regime(ind, **(ps.get("regime") or {}))
    if ps.get("family"):
        ind = F.add_family_indicators(ind)
        sig = F.select(ind, ps["family"], dict(ps.get("family_params") or {}), lo, hi, reg=reg)
    else:
        sig = signals.select_entries(ind, start_date=lo, end_date=hi, **ps["signal"])
    res = broker.simulate(ps, sig, ind, start=lo, end=hi)
    eq = res["equity"].set_index("date")
    sret = eq["equity"].pct_change().dropna()
    fac = style_factor_returns(ind.loc[ind["date"].isin(sret.index)], lo=lo, hi=hi)
    bench = reg.set_index("date")["ew_index"].pct_change().reindex(sret.index).fillna(0.0)
    d = attribute(sret, fac, bench)
    print(f"{a.preset}  {lo}..{hi}")
    print("  " + describe(d))
    print(f"  因子口径：market=池等权；mom=横截面 120 日动量高/低三分位多空(等权)；"
          f"lowvol=20 日波动低/高三分位多空。月频再平衡，划分只用 ≤ 再平衡日的数据。")
    print("  ⚠️ 没有价值因子（无基本面数据）与规模因子（只有静态市值快照，非 PIT）——"
          "覆盖到哪里就说哪里。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
