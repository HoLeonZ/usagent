"""搜索层必须带基准（审计 C3）。

缺陷本体：`search_12m.evaluate()` 的返回值里**没有基准** —— 于是整个搜索的目标是对
**绝对 CAGR** 求 argmax。在牛市里选出来的"最好配置"是"拿得最多的那个"，
而不是"相对池子做得最好的那个"；而报告里也没有任何一列能让人看出这一点。

本文件钉住：
  1. 基准的定义**只有一处**（`regime.window_benchmark_cagr`），family_search 的私有
     函数只是委托 —— 否则又会出现"两个入口两把尺"；
  2. 基准取自**同一窗口**（左闭右开），且按**日历天数**年化（与 `metrics.cagr` 同口径）；
  3. `evaluate` 的每次返回都带 `bench_cagr` 与 `excess`；
  4. 打印表里出现超额列（"看不见"等于没有）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import indicators  # noqa: E402
from research import regime as R  # noqa: E402
from research import search_12m as S  # noqa: E402
from research.tests.conftest import make_synth_panel  # noqa: E402


def _regime(n: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "date": dates,
        "ew_index": np.cumprod(1 + np.full(n, 0.001)),      # 每天 +0.1%
        "ew_ma": np.ones(n), "breadth_sm": np.ones(n), "regime_ok": True,
    })


# ---------------------------------------------------------------- 基准定义

def test_window_benchmark_is_annualized_by_calendar_days():
    reg = _regime(400)
    lo, hi = "2020-06-01", "2021-01-01"
    got = R.window_benchmark_cagr(reg, lo, hi)
    idx = reg.set_index("date")["ew_index"]
    b = idx[(idx.index >= pd.Timestamp(lo)) & (idx.index < pd.Timestamp(hi))]
    days = (b.index[-1] - b.index[0]).days
    want = (b.iloc[-1] / b.iloc[0]) ** (365.25 / days) - 1.0
    assert got == pytest.approx(want)


def test_window_is_left_closed_right_open():
    reg = _regime(100)
    lo, hi = "2020-01-01", "2020-02-01"
    full = R.window_benchmark_cagr(reg, lo, hi)
    assert full == pytest.approx(R.window_benchmark_cagr(reg, lo, "2020-02-01"))
    # 右端排他：把 hi 提前到窗口内，结果必须变
    assert R.window_benchmark_cagr(reg, lo, "2020-01-15" ) != pytest.approx(full)


def test_too_short_window_is_nan():
    assert np.isnan(R.window_benchmark_cagr(_regime(10), "2020-01-02", "2020-01-03"))


def test_family_search_delegates_to_the_single_definition():
    """family_search 的私有函数必须是委托 —— 否则就是两把尺。"""
    from research import family_search as FS
    reg = _regime(200)
    assert (FS._bench_cagr(reg, "2020-03-02", "2020-08-03")
            == pytest.approx(R.window_benchmark_cagr(reg, "2020-03-02", "2020-08-03")))


# ---------------------------------------------------------------- evaluate 带基准

def test_evaluate_returns_benchmark_and_excess():
    panel = make_synth_panel(("AAA", "BBB"), bars=360)
    ind = indicators.compute_all(panel)
    reg = R.compute_regime(ind)
    cfg = dict(S.wf_grid()[0])
    lo = str(ind["date"].iloc[200].date())
    hi = str(ind["date"].iloc[-1].date())
    r = S.evaluate(ind, reg, cfg, lo, hi)
    assert "bench_cagr" in r and "excess" in r, "搜索层的每次评估都必须带基准与超额"
    assert np.isfinite(r["bench_cagr"]), "这个夹具的池指数是单调上行的，基准必须能算出来"
    if r["cagr"] is not None:
        assert r["excess"] == pytest.approx(r["cagr"] - r["bench_cagr"])


def test_evaluate_without_signals_still_carries_the_benchmark():
    """空信号窗返回 cagr=None（三态可区分），但基准照样要有 —— 否则那一行无从比较。"""
    ind = indicators.compute_all(make_synth_panel(("AAA",), bars=360))
    reg = R.compute_regime(ind)
    with_sig = S.evaluate(ind, reg, dict(S.wf_grid()[0]),
                          str(ind["date"].iloc[200].date()),
                          str(ind["date"].iloc[-1].date()))
    assert "bench_cagr" in with_sig


def test_stage_table_prints_excess():
    """打印表里必须有超额列 —— 数字算出来了但没人看见，等于没算。"""
    src = (ROOT / "research" / "search_12m.py").read_text()
    assert '"excess"' in src and "d[\"excess\"]" in src
    assert 'd[["axis", "value", "trades", "cagr", "excess"' in src
