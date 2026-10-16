"""买卖价差估计（Corwin & Schultz 2012）—— 日线数据下唯一站得住的价差代理。

为什么必须有这一层（2026-10-17 审计 E1）:

本仓库的成本模型此前只有两块：**佣金/规费**（core/costs.py 的双边 5bp）与
**冲击滑点**（ATR × √参与率）。两者都不是"买卖价差" —— 而价差是**尺寸无关**的
那一半成本：你买 100 股和买 10 万股，都要先跨过同一个价差。对一个允许 $1 级别、
盘后价差上到 30% 的池子（见 data/universe_oi100.csv 的 spread_pct_afterhours 列），
不建模价差等于把最贵的那一项当成零。

为什么用 Corwin-Schultz 而不是"拍一个 bp 数"：日线数据里没有 NBBO，
但**两日最高最低价**里含有价差信息（若两天的价格区间只由真实波动决定，则两日合并区间
的方差应等于两日各自方差之和；超出的那部分归因于价差造成的"反弹"）。
这是文献里标准的日线价差估计量，**因果**（只用 ≤t 的两根 bar），且对每只票逐日不同 ——
正好对上"价差是尺寸无关但**标的/时间相关**"的性质。

公式（两日窗口 t−1, t）:
    β = ln(H_{t−1}/L_{t−1})² + ln(H_t/L_t)²
    γ = ln(max(H_{t−1},H_t) / min(L_{t−1},L_t))²
    α = (√(2β) − √β)/(3 − 2√2) − √(γ/(3 − 2√2))
    S = 2(e^α − 1)/(1 + e^α)

已知偏差（**实测过**，不藏起来）:
  * 系统性**低估**，且越是窄价差越明显。用"确定的中间价 + 注入已知价差"做对照：
    注入 100bp → 估回 72bp；注入 300bp → 估回 247bp；注入 ≤20bp → 常被压到 0。
    所以这个估计量给出的价差应读作**下限**，真实成本只会更高 —— 方向是对的
    （它把"价差=0"这个假设换成了一个偏低但不为零的估计），但它**没有**高估的风险。
  * 本库实测中位数：AAPL ~10bp、MSFT ~5bp、WMT ~16bp、PTEN ~39bp、GRAB ~46bp
  * 估计量有噪声，且单日估计**可为负**（此时按 0 处理，即"不能证明有价差"）；
  * 未做原文的隔夜跳空调整（该调整引入额外假设，而本库的价格是 as-traded 口径、
    跳空本身是真实信息）；
  * 低价股的绝对最小跳价（1 分钱）会让它更明显低估 —— 下限常数因此取 0 而不是某个正数。
"""
from __future__ import annotations

import math

import numpy as np

# 合法上界：超过它的一律截断。日线估计量在极端行情/停牌前后会爆出几百 bp 的假值，
# 而真实价差极少超过 5%（宽价差小盘的上限量级）。截断是**保守方向**（少算成本是
# 乐观，但 500bp 的假值会让个别交易的结论完全失真 —— 两害相权取截断）。
MAX_SPREAD_BPS: float = 500.0

# 估计下限：低于它的按 0 处理（见模块 docstring：单日估计可为负）。
MIN_SPREAD_BPS: float = 0.0

_K = 3.0 - 2.0 * math.sqrt(2.0)


def corwin_schultz(high: np.ndarray, low: np.ndarray, *,
                   cap_bps: float = MAX_SPREAD_BPS) -> np.ndarray:
    """逐日的价差估计（**bp**，即 1e-4）。第 i 个值用 bar i−1 与 i，第 0 个是 NaN。

    因果：第 i 个估计只依赖 ≤ i 的两根 bar ⇒ 在 i 日**收盘后**即可知，
    正好够用于"T 日收盘出信号、T+1 开盘成交"的决策时序。

    输入为 as-traded 价（执行口径）。非有限值/非正价的位置返回 NaN（不可判定），
    由调用方决定如何处理（本项目按"不建模"处理并在报告里标注）。
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    n = len(h)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out

    ok = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0) & (h >= l)
    # 用两日窗口：只有 (i-1, i) 都合法才估第 i 日
    both = ok[1:] & ok[:-1]
    idx = np.nonzero(both)[0] + 1
    if idx.size == 0:
        return out

    h0, l0 = h[idx - 1], l[idx - 1]
    h1, l1 = h[idx], l[idx]
    beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
    hh = np.maximum(h0, h1)
    ll = np.minimum(l0, l1)
    gamma = np.log(hh / ll) ** 2

    alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K
             - np.sqrt(np.maximum(gamma / _K, 0.0)))
    # e^α 在 α 较大时会溢出；用 expm1 并钳住指数上界（对应 cap_bps 量级）。
    alpha = np.clip(alpha, -50.0, math.log(cap_bps / 1e4 / 2.0 + 1.0))
    spread = 2.0 * np.expm1(alpha) / (2.0 + np.expm1(alpha))
    bps = spread * 1e4

    # 负估计 ⇒ "不能证明存在价差" ⇒ 0（不是取绝对值：那会把噪声变成成本）。
    bps = np.where(np.isfinite(bps), bps, np.nan)
    bps = np.clip(bps, MIN_SPREAD_BPS, cap_bps)
    out[idx] = bps
    # 两日窗口里任一根不合法 ⇒ 该日不可判定（保持 NaN），不参与建模
    out[idx[~np.isfinite(bps)]] = np.nan
    return out


# 平滑窗口与最小有效观测数。见 rolling_median 的 docstring：原始估计量有**厚右尾**
# （实测 AAPL 中位 9bp / 均值 43bp），直接用逐日原值会把平均成本算高好几倍。
SMOOTH_LOOKBACK: int = 20
SMOOTH_MIN_PERIODS: int = 5


def rolling_median(x: np.ndarray, lookback: int = SMOOTH_LOOKBACK,
                   min_periods: int = SMOOTH_MIN_PERIODS) -> np.ndarray:
    """**因果**的滚动中位（窗口以 t 结束，右端就是 t）。

    为什么必须平滑：Corwin-Schultz 的逐日估计在放量/高波动日会爆出几百 bp 的尖峰
    （实测 AAPL：中位 9bp、均值 43bp）。用原值等于把尖峰当成每天的常态，
    把价差成本算高 3~5 倍；用全样本中位又会**看未来**。
    滚动中位是这两者之间唯一站得住的选择：逐日变化、只依赖 ≤t、
    且对尖峰稳健。无效观测不足 min_periods ⇒ NaN（不可判定，不建模）。
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n == 0 or lookback < 1:
        return out
    if n < lookback:
        return out                     # 不足一个完整窗口 ⇒ 全部不可判定
    win = np.lib.stride_tricks.sliding_window_view(x, lookback)   # (n-lookback+1, lookback)
    valid = np.isfinite(win)
    cnt = valid.sum(axis=1)
    med = np.full(win.shape[0], np.nan)
    any_valid = cnt > 0
    if any_valid.any():
        # 只在有有效值的行上算中位 —— 否则会触发 "All-NaN slice" 运行时告警，
        # 而那条告警正是"某只票整段没有有效价差估计"的信号，不该被淹没在噪声里。
        med[any_valid] = np.nanmedian(
            np.where(valid[any_valid], win[any_valid], np.nan), axis=1)
    # 头部 lookback-1 根没有完整窗口（保守：那几天不建模价差，代价是少算成本，仅占窗口 2~3%）
    out[lookback - 1:] = np.where(cnt >= min_periods, med, np.nan)
    return out


def half_spread_rate(spread_bps: float) -> float:
    """bp → **单边**费率（买付一半、卖付一半；跨价差的总额是 spread_bps）。"""
    if not math.isfinite(spread_bps) or spread_bps <= 0:
        return 0.0
    return float(spread_bps) / 2.0 / 1e4


def describe() -> str:
    return (f"Corwin-Schultz (2012) 日线价差估计,逐日逐标的,截断于 {MAX_SPREAD_BPS:.0f}bp,"
            f"负估计按 0(不证明有价差)")
