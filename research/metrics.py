"""评估口径 —— 窗口生成 / E[R] / DSR / beta 归因。

三处移植,每处都带着 hiagent 踩过的坑:

1. `tail_stats` ← uptrend_pullback/tail_objective.py
   该族盈亏由**右尾**驱动,Sharpe 会用波动率惩罚创造边缘的那条尾巴,所以主判据是
   R-multiple 期望 E[R]。分母必须是**入场时固定的计划风险**(= 止损宽度 × 股数),
   不能是事后结果,也不能用 atr_pct 当分母(hiagent 原话:固定百分比止损路径下
   atr_pct 可能为空,会把整批 R 算成 NaN)。

2. `deflated_sharpe_ratio` ← dna_stats/deflated.py,**含那个单位一致性修复**。
   hiagent 曾拿**年化** Sharpe 去比**逐期**阈值,把中等 Sharpe 系统性折减到 0
   (实测年化 0.84 / k=19 / T=1071 被判 0.000,正确值约 0.44),而年化 3.0 这种高
   Sharpe 两法都接近 1,所以 bug 长期未被发现。test_metrics 钉住这条。

3. `monthly_windows` ← core/walkforward.py,2 个月窗 / 1 个月步,与 hiagent 记录口径一致。
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------- 窗口

def monthly_windows(start_month: str, end_month: str, window_months: int = 2,
                    step_months: int = 1) -> list[tuple[str, str]]:
    """月级滚动窗口,左闭右开。[start_month, end_month) 内生成 (start, end)。

    end_month 是**排他**上界:窗口的 end 可以等于 end_month,但不能超过。
    """
    if step_months < 1:
        raise ValueError(f"step_months 必须 >= 1,得到 {step_months}")
    sy, sm = map(int, start_month.split("-"))
    ey, em = map(int, end_month.split("-"))

    def add(y: int, m: int, n: int) -> tuple[int, int]:
        t = (y * 12 + (m - 1)) + n
        return t // 12, t % 12 + 1

    out: list[tuple[str, str]] = []
    cy, cm = sy, sm
    while True:
        ny, nm = add(cy, cm, window_months)
        if (ny, nm) > (ey, em):
            break
        out.append((f"{cy:04d}-{cm:02d}-01", f"{ny:04d}-{nm:02d}-01"))
        cy, cm = add(cy, cm, step_months)
    return out


# ---------------------------------------------------------------- CAGR

def cagr(equity: pd.Series, dates: pd.Series) -> float:
    """按**实际日历天数**年化。不要用 252 交易日近似,否则短窗会被放大。"""
    if len(equity) < 2:
        return float("nan")
    days = (pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days
    if days <= 0:
        return float("nan")
    total = float(equity.iloc[-1]) / float(equity.iloc[0])
    if total <= 0:
        return float("nan")
    return total ** (365.25 / days) - 1.0


# ---------------------------------------------------------------- R-multiple / 尾部

def r_multiples(trades: pd.DataFrame) -> pd.Series:
    """每笔的 R 倍数 = net_pnl / risk_amount。

    risk_amount 由 broker 逐笔记录(= 入场时的止损宽度 × 入场价 × 股数)。
    不用 sl_pct(ATR 归一化后逐笔不同),也不用 atr_pct。
    """
    if trades.empty:
        return pd.Series(dtype=float)
    risk = trades["risk_amount"].astype(float)
    if (risk <= 0).any():
        raise ValueError("存在非正的 risk_amount —— 止损宽度或股数为 0")
    return trades["net_pnl"].astype(float) / risk


def tail_stats(trades: pd.DataFrame) -> dict:
    """R 分布的核心统计量。`expectancy_r` 是主目标,其余是分布/尾部判据。

    `t_expectancy` 按**退出月**分组算均值再求 t —— 同月内交易相关(且
    max_positions>1 时可能重叠),按退出日分组无法消除这种相关。
    这一点与 hiagent 的字面实现(按 exit_date 分组)不同,是有意的:
    它的 docstring 说的是「窗口互不重叠」,而按日分组并不满足。
    """
    r = r_multiples(trades)
    if r.empty:
        return {"n": 0, "expectancy_r": float("nan"), "t_expectancy": float("nan")}

    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())

    monthly = (trades.assign(_r=r)
               .groupby(pd.to_datetime(trades["exit_date"]).dt.to_period("M"))["_r"].mean())
    n_m = len(monthly)
    if n_m > 2 and monthly.std(ddof=1) > 0:
        t_exp = float(monthly.mean() / (monthly.std(ddof=1) / math.sqrt(n_m)))
    else:
        t_exp = float("nan")

    top = r.nlargest(max(len(r) // 10, 1))
    return {
        "n": int(len(r)),
        "expectancy_r": float(r.mean()),
        "median_r": float(r.median()),
        "t_expectancy": t_exp,
        "n_periods": int(n_m),
        "win_pct": float((r > 0).mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss_r": float(losses.mean()) if len(losses) else float("nan"),
        "payoff": float(wins.mean() / abs(losses.mean()))
                  if len(wins) and len(losses) else float("nan"),
        "p_r_ge_5": float((r >= 5).mean()),
        "p_r_le_minus_1": float((r <= -1).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "top_decile_share": float(top.sum() / r.sum()) if r.sum() != 0 else float("nan"),
        "worst_r": float(r.min()), "best_r": float(r.max()),
    }


# ---------------------------------------------------------------- DSR

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """标准正态反函数(Beasley-Springer-Moro),不依赖 scipy。"""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 必须在 (0,1),得到 {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q, r = p - 0.5, (p - 0.5) ** 2
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def deflated_sharpe_ratio(observed_sharpe: float, n_trials: int, *,
                          skewness: float = 0.0, kurtosis: float = 3.0,
                          autocorr_lag1: float = 0.0, n_returns: int | None = None,
                          trading_days_per_year: int = TRADING_DAYS) -> dict:
    """Bailey & López de Prado (2014) 的通缩 Sharpe,含单位一致性修复。

    observed_sharpe 按本函数约定是**年化**值;内部换算成逐期再比逐期阈值。
    """
    if n_trials < 1:
        raise ValueError(f"n_trials 必须 >= 1,得到 {n_trials}")

    gamma = 0.5772156649015329
    if n_trials == 1:
        expected_max = 0.0
    else:
        try:
            z1 = _norm_ppf(1 - 1.0 / n_trials)
            z2 = _norm_ppf(1 - 1.0 / (n_trials * math.e))
            expected_max = (1 - gamma) * z1 + gamma * z2
        except (ValueError, ZeroDivisionError, OverflowError):
            expected_max = float("nan")

    # ← 单位一致性:年化 → 逐期。少了这一步会把中等 Sharpe 折减成 0。
    sr = observed_sharpe / math.sqrt(trading_days_per_year)

    if n_returns is None or n_returns < 2:
        se = 1.0
    else:
        nonnorm = (1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr ** 2) / (n_returns - 1)
        mult = (1.0 + 2.0 * autocorr_lag1 / (1.0 - autocorr_lag1)
                if abs(autocorr_lag1) < 1.0 else 1.0)
        se = math.sqrt(max(nonnorm * mult, 1e-12))

    if not math.isfinite(expected_max) or se <= 0:
        dsr = p = float("nan")
    else:
        dsr = _norm_cdf(sr / se - expected_max)
        p = 1.0 - dsr

    return {"observed_sharpe": float(observed_sharpe),
            "expected_max_sharpe": float(expected_max),
            "deflated_sharpe": dsr, "dsr_p_value": p, "n_trials": int(n_trials)}


def bonferroni_t_threshold(family_size: int, alpha: float = 0.05) -> float:
    """Bonferroni 校正后的**双侧** t 门槛(CLAUDE.md §5(c))。

    报告一次多参数扫描的结果时,族错误率要按比较次数折减:
        t_crit = Φ⁻¹(1 − α/(2k))
    单次(k=1, α=0.05)给出 1.96;4 层给 2.498;12 次给 2.865。
    没有这一步,「优于基准」这类判词会在未达族显著性的 t 上就印出来。
    """
    if family_size < 1:
        raise ValueError(f"family_size 必须 >= 1,得到 {family_size}")
    return _norm_ppf(1.0 - alpha / (2.0 * family_size))


def two_sided_p(t: float) -> float:
    """正态近似的双侧 p 值。

    ⚠️ 是**近似**：本项目的期数常在 6~36 之间，严格该用 t 分布。这里用正态是
    沿用本仓库既有的近似（`_norm_ppf` 那套），并在报告里把族规模与判据一起写出来 ——
    近似不是问题，"不校正族错误率就下判词"才是（审计 B4）。
    """
    if not math.isfinite(float(t)):
        return float("nan")
    return 2.0 * (1.0 - _norm_cdf(abs(float(t))))


def benjamini_hochberg(pvals, q: float = 0.05) -> dict:
    """BH 步进法控制**错误发现率**（FDR）—— 多因子扫描的正确工具。

    为什么不能只用 Bonferroni（审计 B4）:本仓库只有 screen.py 落了 Bonferroni，
    而因子探针一次要比较十几个因子 × 两侧 × 多个分位。Bonferroni 在最坏情况下
    控制 FWER、代价是功效极低（k=1614 时门槛 4.17，什么都过不了）；因子研究里
    通行做法是控制 FDR（Harvey-Liu-Zhu 那类工作用的就是"提高门槛 + 看整体"）。
    两者都给，读者才知道"点比较的显著"在族规模下还剩多少。

    NaN 的 p 值（不可检验，如期数不足）**不参与排序、也不判为拒绝** —— 当作缺失，
    不给它们"通过"的假象。
    """
    if not (0.0 < float(q) < 1.0):
        raise ValueError(f"q 必须在 (0,1)，得到 {q}")
    p = np.asarray(list(pvals), dtype=float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    out = {"q": float(q), "n": m, "n_rejected": 0, "crit_p": float("nan"),
           "rejected": [False] * int(p.size)}
    if m == 0:
        return out
    order = np.argsort(p[ok], kind="mergesort")
    ps = p[ok][order]
    thresh = float(q) * (np.arange(1, m + 1) / m)
    below = ps <= thresh
    if not below.any():
        return out
    k = int(np.max(np.nonzero(below)[0]) + 1)
    rej = np.zeros(int(p.size), dtype=bool)
    rej[np.nonzero(ok)[0][order[:k]]] = True
    out.update({"n_rejected": k, "crit_p": float(ps[k - 1]),
                "rejected": rej.tolist()})
    return out


def sharpe_significance_pvalue(observed_sharpe: float, n_returns: int,
                               trading_days_per_year: int = TRADING_DAYS) -> float:
    """年化 Sharpe 的显著性 p 值（**双侧**意义上的一次比较）。

    ⚠️ 单位一致性：传入的是**年化** Sharpe，而标准误是**逐期**的 1/√(T−1)，
    所以必须先把它换算成逐期 Sharpe 再比 —— 这正是 DSR 那次踩过的坑
    （见 deflated_sharpe_ratio 的注释）。先前的实现直接拿年化值去除逐期标准误，
    会把 p 值系统性压低（等于凭空放大显著性）。
    """
    if n_returns < 2:
        return float("nan")
    sr_period = float(observed_sharpe) / math.sqrt(trading_days_per_year)
    se = 1.0 / math.sqrt(n_returns - 1)
    return 1.0 - _norm_cdf(sr_period / se)


# ---------------------------------------------------------------- 从收益序列出发的口径

def annualized_sharpe(rets, periods_per_year: int = TRADING_DAYS) -> float:
    """一条收益序列的年化 Sharpe（逐期均值/标准差 × √ppy）。"""
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if not (sd > 0):
        return float("nan")
    return float(np.mean(r) / sd * math.sqrt(periods_per_year))


def pool_windows_rets(per_window: list[tuple[tuple[str, str], list[float]]]
                      ) -> list[float] | None:
    """把逐窗口的收益序列**拼接**成一条（用于在 pooled 口径上算 DSR）。

    只在窗口**互不重叠**时才允许拼接：重叠窗口会把同一段行情算两次，
    拼出来的"一条序列"比真实观测数长、自相关被人为抬高。不能拼就返回 None
    （调用方据此**不报** DSR，而不是拿一个拼不出来的数去报）。

    输入: [((lo, hi), [ret, ...]), ...]
    """
    ws = sorted(((str(a)[:10], str(b)[:10]), list(r)) for (a, b), r in per_window)
    for i in range(len(ws) - 1):
        if ws[i][0][1] > ws[i + 1][0][0]:
            logger.warning("窗口 %s 与 %s 重叠 ⇒ 不拼接收益序列（不报 DSR）",
                           ws[i][0], ws[i + 1][0])
            return None
    out: list[float] = []
    for _, r in ws:
        out.extend(float(x) for x in r)
    return out


def deflated_sharpe_from_returns(rets, n_trials: int, *,
                                 periods_per_year: int = TRADING_DAYS,
                                 label: str = "") -> dict:
    """**从收益序列出发的 DSR —— 本项目唯一被认可的口径**。

    为什么必须有这一层（2026-10-17 审计 B3）:`deflated_sharpe_ratio` 需要四个量
    ——观测数 T、偏度、峰度、一阶自相关——而四个调用点各传各的：`run_baseline` 传
    交易日数、`family_search`/`wfv_search` 传**成交笔数**、`factor_probe` 传**月数**，
    且**没有任何一处**传过偏度/峰度（非正态修正恒为默认值 ⇒ 惰性）。
    同一个函数在两个入口得到两种口径的数字，而这个数字是用来做判据的。

    这里把所有派生量都从**同一条收益序列**算出来，调用方只能提供序列与试验次数，
    于是"T 是什么"在类型层面就不再是一个可以各说各话的问题。

    `periods_per_year` 与序列频率必须一致（日线 252、月线 12）——
    传错的后果是 Sharpe 被换算错一次，而 DSR 恰好对此很敏感。
    """
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3:
        return {"observed_sharpe": float("nan"), "expected_max_sharpe": float("nan"),
                "deflated_sharpe": float("nan"), "dsr_p_value": float("nan"),
                "n_returns": n, "n_trials": int(n_trials), "label": label,
                "reason": "样本不足（<3 个收益观测）"}
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if not (sd > 0):
        return {"observed_sharpe": float("nan"), "expected_max_sharpe": float("nan"),
                "deflated_sharpe": float("nan"), "dsr_p_value": float("nan"),
                "n_returns": n, "n_trials": int(n_trials), "label": label,
                "reason": "收益序列方差为 0"}
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))                     # 非超额峰度：正态 = 3
    if n > 2:
        x, y = r[:-1], r[1:]
        sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
        ac = (float(np.corrcoef(x, y)[0, 1]) if sx > 0 and sy > 0 else 0.0)
    else:
        ac = 0.0
    out = deflated_sharpe_ratio(annualized_sharpe(r, periods_per_year),
                               n_trials=n_trials, skewness=skew, kurtosis=kurt,
                               autocorr_lag1=ac, n_returns=n,
                               trading_days_per_year=periods_per_year)
    out.update({"n_returns": n, "skewness": skew, "kurtosis": kurt,
                "autocorr_lag1": ac, "periods_per_year": int(periods_per_year),
                "label": label})
    return out


# ---------------------------------------------------------------- beta 归因

def beta_attribution(strategy_ret: pd.Series, bench_ret: pd.Series,
                     exposure: pd.Series) -> dict:
    """策略日收益 vs 基准日收益的归因。

    返回的关键量:
      beta          —— 全样本 OLS 斜率(空仓日的零收益自然压低它)
      alpha_annual  —— 截距 × 252,及其 t 值
      bench_on_exposed_days —— **只**在策略有敞口的那些天持有基准的累计收益。
        这一项把「择时」与「敞口」分开:若它 ≈ 策略收益,说明收益全来自
        「在市场里的那些天市场涨了」,而这只是 beta + 择时运气,不是选股 alpha。
    """
    df = pd.concat([strategy_ret.rename("s"), bench_ret.rename("b"),
                    exposure.rename("e")], axis=1).dropna()
    if len(df) < 30:
        return {"beta": float("nan"), "alpha_annual": float("nan"),
                "t_alpha": float("nan"), "n_days": int(len(df)),
                "bench_on_exposed_days": float("nan"), "pct_days_exposed": float("nan")}

    s, b = df["s"].to_numpy(float), df["b"].to_numpy(float)
    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(s, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    alpha = float(np.mean(s) - beta * np.mean(b))
    resid = s - (alpha + beta * b)
    se_alpha = (float(np.std(resid, ddof=1)) / math.sqrt(len(s))) if len(s) > 2 else float("nan")
    t_alpha = alpha / se_alpha if se_alpha and se_alpha > 0 else float("nan")

    expo = df["e"].to_numpy(float) > 0
    bench_exposed = float(np.prod(1.0 + b[expo]) - 1.0) if expo.any() else float("nan")

    return {
        "beta": beta,
        "alpha_annual": alpha * 252,
        "t_alpha": t_alpha,
        "n_days": int(len(df)),
        "corr": float(np.corrcoef(s, b)[0, 1]) if var_b > 0 else float("nan"),
        "bench_on_exposed_days": bench_exposed,
        "pct_days_exposed": float(expo.mean()),
    }
