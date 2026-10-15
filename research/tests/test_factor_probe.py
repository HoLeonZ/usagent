"""横截面因子探针的测试。

这个文件最重要的不是「代码跑通」,而是**证明探针能抓到因子**:
如果 IC 代码是坏的,「0 个因子通过」与「因子存在但探针看不见」看起来一模一样。
所以这里用**合成数据植入一个已知因子**,要求探针必须把它抓出来 ——
再把植入的因子**滞后一期**(仍然是个完全相同的列,只是时间错了),要求探针必须抓不出来。
两者一起才说明这套检验是有效的,而不是恒真的。

另外钉住两类这个仓库反复踩过的坑:
  * 因果性:前 k 期的收益必须**不随窗口延长而改变**(没有用到窗口外的信息);
  * 对齐:逐月配对的超额必须**按日期对齐**,不能按位置 —— 位置对齐在缺期时会静默错位。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import factor_probe as FP

GATE_OK = 1.0e8          # 高于门槛
GATE_LOW = 1.0e6         # 低于门槛(必须被排除)


def _daily_frame(n_sym: int = 60, n_months: int = 24, *, beta: float = 0.06,
                 sigma: float = 0.08, seed: int = 7) -> pd.DataFrame:
    """合成一份**日频**横截面 panel(月末是再平衡日),并植入一个已知的预测关系。

    构造:t 月末的当期特征 alpha[t,i] 决定 t → t+1 的整月收益
              r[t, i] = beta * alpha[t, i] + sigma * eps[t, i]
    月内价格按**对数线性**从上一个月的水平爬到本月水平。这一点很关键:探针的持有期是
    「再平衡日的**下一个**交易日 → 下一个再平衡日」,所以量到的是整月收益除以一天;
    对数线性爬坡让这个变换对 r 严格单调 ⇒ **排名不变 ⇒ Spearman 逐位不变**,
    植入的 IC 被精确保留(不像插值那样被随意衰减)。

    ⇒ 列 `f_true`(值 = alpha[t])应当有强 IC;
      列 `f_stale`(值 = alpha[t-1])是个同样的列,只是**时间错了一期**,
       按构造与 r[t] 无关 ⇒ IC 应当约等于 0。
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.0, 1.0, size=(n_months, n_sym))
    eps = rng.normal(0.0, 1.0, size=(n_months, n_sym))
    ret = beta * alpha + sigma * eps                      # r[t] = 从 t 到 t+1

    level = np.empty((n_months, n_sym))
    level[0] = 100.0
    for t in range(n_months - 1):
        level[t + 1] = level[t] * (1.0 + ret[t])

    bdays = pd.bdate_range("2024-01-01", periods=1)       # 占位,下面按月份展开
    months: list[np.ndarray] = []
    cur, last = [], None
    for d in pd.bdate_range("2024-01-01", "2026-06-30"):
        key = (d.year, d.month)
        if last is not None and key != last:
            months.append(np.array(cur))
            cur = []
        cur.append(d)
        last = key
        if len(months) == n_months:
            break
    months = months[:n_months]
    assert len(months) == n_months
    del bdays

    rows = []
    half = n_months // 2
    for t, days in enumerate(months):
        prior = level[t - 1] if t > 0 else level[0]
        k = np.arange(1, len(days) + 1, dtype=float) / len(days)
        ramp = prior[None, :] * (level[t] / prior)[None, :] ** k[:, None]   # (K, n_sym)
        # 第 5 个月整个池子流动性都不达标 ⇒ 该期无候选,被判据跳过。
        # 这一条是为了让「按位置对齐」这类 bug 露出马脚(缺期时位置与日期不再一致)。
        dry = (t == 5)
        for j, d in enumerate(days):
            for i in range(n_sym):
                rows.append({
                    "symbol": f"S{i:03d}", "date": d, "close": ramp[j, i],
                    # 前半流动性达标、后半不达标 —— 门必须真的起作用
                    "amount60": GATE_LOW if dry else (GATE_OK if i < n_sym // 2 else GATE_LOW),
                    "f_true": alpha[t, i],
                    # 滞后一期:同一个分布、同一个列名语义,只是时间错了
                    "f_stale": alpha[t - 1, i] if t > 0 else np.nan,
                    # 前半段同号、后半段**符号翻转**(模拟「半段行情」冒充因子)
                    "f_flip": alpha[t, i] * (1.0 if t < half else -1.0),
                    # 与任何东西都无关的一列(纯噪声,月内固定以便复现)
                    "f_noise": rng.normal(0.0, 1.0),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    return _daily_frame()


@pytest.fixture
def n_months() -> int:
    return 24


# ---------------------------------------------------------------- 探针有效性

def test_ic_finds_a_planted_factor(frame):
    """植入的当期特征必须被抓出来 —— 这是「探针没坏」的证据。"""
    ic = FP.rank_ic(frame, "f_true", lo="2024-01-01", hi="2026-02-01")
    assert ic["n"] >= 12
    assert ic["mean"] > 0.20, f"植入因子没被抓到:IC={ic['mean']:.3f}"
    assert ic["t"] > 3.0, f"植入因子的 t 值太低:{ic['t']:.2f}"


def test_ic_rejects_the_same_factor_shifted_by_one_period(frame):
    """同一列**滞后一期**后必须抓不出来 —— 否则这套检验是恒真的。"""
    good = FP.rank_ic(frame, "f_true", lo="2024-01-01", hi="2026-02-01")
    stale = FP.rank_ic(frame, "f_stale", lo="2024-01-01", hi="2026-02-01")
    assert abs(stale["mean"]) < 0.15, f"滞后因子不该有 IC:{stale['mean']:.3f}"
    assert abs(stale["t"]) < 2.0
    assert good["mean"] > stale["mean"] + 0.15


def test_ic_of_pure_noise_is_zero(frame):
    ic = FP.rank_ic(frame, "f_noise", lo="2024-01-01", hi="2026-02-01")
    assert abs(ic["mean"]) < 0.15 and abs(ic["t"]) < 2.0


def test_stability_flags_a_planted_factor_as_consistent(frame):
    """合成数据里植入的因子是**全程**有效的,所以两半应当同号(判据②不误伤真因子)。"""
    st = FP.stability(frame, ["f_true", "f_stale"], lo="2024-01-01", mid="2025-01-01",
                      hi="2026-02-01")
    assert st["f_true"]["same_sign"] is True
    assert st["f_true"]["kind"] == "一致"


def test_stability_catches_a_sign_flipping_factor(frame):
    """符号会翻转的因子必须被判据② 拦下 —— 这条判据就是用来杀「半段行情」的。"""
    st = FP.stability(frame, ["f_flip"], lo="2024-01-01", mid="2025-01-01", hi="2026-02-01")
    s = st["f_flip"]
    assert s["first"]["mean"] > 0.2 and s["second"]["mean"] < -0.2
    assert s["same_sign"] is False
    assert s["kind"] == "翻转"


def test_periods_are_skipped_not_misaligned_when_a_month_has_no_candidates(frame):
    """第 5 个月流动性全不达标 ⇒ 该期无候选。此时日期与收益**仍须一一对应**。"""
    st = FP.portfolio(frame, "f_true", "high", 0.3, lo="2024-01-01", hi="2026-02-01",
                      cost_bps=5.0)
    dates = pd.DatetimeIndex(frame["date"].unique())
    ends = set(FP.month_ends(dates))
    assert len(st["idx"]) == len(st["net_rets"]) == st["months"]
    assert all(d in ends for d in st["idx"])
    # 24 个月 ⇒ 23 个持有期(最后一个再平衡日没有下一个月可持有),再跳过 1 个空期 = 22
    assert len(st["idx"]) == 22


# ---------------------------------------------------------------- 因果性

def test_first_periods_are_invariant_to_extending_the_window(frame):
    """延长窗口不能改变前几期的收益 —— 探针没有用到窗口外的信息。"""
    short = FP.portfolio(frame, "f_true", "high", 0.3, lo="2024-01-01", hi="2024-08-31",
                         cost_bps=5.0)
    long = FP.portfolio(frame, "f_true", "high", 0.3, lo="2024-01-01", hi="2026-02-01",
                        cost_bps=5.0)
    assert len(long["idx"]) > len(short["idx"])
    k = len(short["idx"])
    assert short["idx"] == long["idx"][:k]
    np.testing.assert_allclose(short["net_rets"], long["net_rets"][:k], rtol=0, atol=1e-12)


def test_rebalance_dates_are_each_months_last_row(frame, n_months):
    dates = pd.DatetimeIndex(frame["date"].unique())
    ends = FP.month_ends(dates)
    assert len(ends) == n_months
    # 每个再平衡日都必须是它那个月**在数据里**的最后一个交易日(否则用了当月之后的 bar)
    for d in ends:
        same_month = dates[(dates.year == d.year) & (dates.month == d.month)]
        assert d == same_month.max()
    assert ends == sorted(ends)


# ---------------------------------------------------------------- 门槛与对齐

def test_gate_excludes_illiquid_names_from_pool_and_portfolio(frame):
    """流动性门必须真的过滤掉一半标的(否则「同门全池」这个基准是假的)。"""
    base = FP.equal_weight_pool(frame, lo="2024-01-01", hi="2026-02-01", cost_bps=5.0)
    assert base["n_holdings"] == frame["symbol"].nunique() // 2
    st = FP.portfolio(frame, "f_true", "high", 0.3, lo="2024-01-01", hi="2026-02-01",
                      cost_bps=5.0)
    assert st["n_holdings"] == max(2, int(frame["symbol"].nunique() // 2 * 0.3))


def test_excess_stats_align_by_date_not_by_position():
    """把日期与收益一起倒序,结果必须逐位相同 —— 位置对齐会在这里出错。

    关键是两个序列**长度相同但日期错开一格**:按日期对齐得到 7 期,
    按位置对齐会得到 8 期(而且配错)。
    """
    d = pd.date_range("2024-01-31", periods=9, freq="ME")
    port = 0.01 * np.arange(8, dtype=float)        # 日期 d[0..7]
    pool = np.zeros(8)                             # 日期 d[1..8]
    a = {"idx": list(d[:8]), "net_rets": port}
    b = {"idx": list(d[1:]), "net_rets": pool}
    by_date = FP.excess_stats(a, b)
    assert by_date["n"] == 7, "两段日期只重叠 7 期"
    # 逐月超额 = port[1..7] − 0(按日期对齐);按位置对齐会拿 port[0..6] 去比
    exp = port[1:8]
    sd = exp.std(ddof=1)
    assert by_date["t"] == pytest.approx(exp.mean() / (sd / np.sqrt(7)), rel=1e-12)
    assert by_date["win"] == pytest.approx(1.0)

    # 同一份数据、日期与收益一起倒序 ⇒ 结果必须逐位相同
    back = FP.excess_stats({"idx": list(d[:8])[::-1], "net_rets": port[::-1]},
                           {"idx": list(d[1:])[::-1], "net_rets": pool[::-1]})
    assert back["n"] == by_date["n"]
    assert back["t"] == pytest.approx(by_date["t"], rel=1e-12)
    assert back["win"] == by_date["win"]


def test_the_held_period_is_t_plus_one_to_the_next_rebalance(frame):
    """持有期的定义:再平衡日的**下一个交易日** → 下一个再平衡日。

    这里从**长表**独立复算第一期的收益(不复用探针的 pivot 路径),所以它同时钉住:
      * idx 的第一个元素必须是**第二个**再平衡日(不是第三个、也不是被跳过的那个);
      * 收益必须从 r 之后的第一个交易日算起,而不是从 r 当天算起;
      * 首次建仓的换手是 100%。
    """
    lo, hi = "2024-01-01", "2026-02-01"
    f = frame[frame["symbol"] < "S030"]                      # 只有这 30 只过得了流动性门
    st = FP.portfolio(frame, "f_true", "high", 0.3, lo=lo, hi=hi, cost_bps=5.0)
    ends = FP.month_ends(pd.DatetimeIndex(frame["date"].unique()))
    r0, nxt0 = ends[0], ends[1]
    assert st["idx"][0] == nxt0, "第一期必须结束在第二个再平衡日"

    # 手工选股:r0 当天按 f_true 取最高的 30%×30 = 9 只
    at_r0 = f[(f["date"] == r0) & (f["amount60"] >= 3e7)][["symbol", "f_true"]]
    pick = set(at_r0.nlargest(max(2, int(len(at_r0) * 0.3)), "f_true")["symbol"])
    # 手工复算:每只票在 (r0, nxt0] 内的第一次与最后一次观测价之比,再等权
    held = f[(f["symbol"].isin(pick)) & (f["date"] > r0) & (f["date"] <= nxt0)]
    assert held["date"].nunique() > 5, "持有期内应当有多个交易日(不是单点)"
    per_sym = held.sort_values("date").groupby("symbol")["close"]
    rets = (per_sym.last() / per_sym.first() - 1.0)
    expected = rets.mean() - 1.0 * 5.0 / 1e4                  # 首次建仓换手 100%
    assert st["net_rets"][0] == pytest.approx(expected, rel=1e-12)


def test_portfolio_with_a_negative_excess_has_negative_paired_t(frame):
    """一个纯噪声因子不该拿到正的超额 t 值。"""
    base = FP.equal_weight_pool(frame, lo="2024-01-01", hi="2026-02-01", cost_bps=5.0)
    st = FP.portfolio(frame, "f_noise", "high", 0.3, lo="2024-01-01", hi="2026-02-01",
                      cost_bps=5.0)
    es = FP.excess_stats(st, base)
    assert abs(es["t"]) < 3.0


def test_gate_scan_runs_and_reports_all_three_windows(frame, capsys):
    """梯度检查是个报告函数 —— 至少钉住它不崩、且三档窗口都算了(格式串里含百分号)。"""
    FP._gate_scan(frame, lo="2024-01-01", mid="2025-01-01", hi="2026-02-01", cost_bps=5.0)
    out = capsys.readouterr().out
    assert "标的资格梯度" in out
    assert out.count("CAGR") >= 3 and "maxDD" in out
    # 门越低,全窗持仓只数越多(这一列必须随门变化,否则说明 gate 没传进去)
    counts = [int(l.split()[-1]) for l in out.splitlines() if l.strip()[:1].isdigit()]
    assert len(counts) == len(FP.GATE_SCAN)
    assert counts == sorted(counts, reverse=True), f"只数应随门单调递减:{counts}"
