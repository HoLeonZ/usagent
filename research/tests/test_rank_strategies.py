"""回测排名工具（`research/rank_strategies.py`）的输出格式测试 —— 不读真库、不联网。

它守一个我踩过的坑：**比率型字段存的是分数，打印时必须 ×100**。
第一版 `_row()` 用 `'{:+8.2f}%'.format(0.2931)` 打出 `+0.29%`（而真实是 `+29.31%`），
于是整张表的 `%`/`pp` 列都小 100 倍 —— 偏偏笔数/敞口/beta/t/DSR 那些**分数列**照旧正确，
所以表"看起来合理"却全错。这类错误的危险在于它不会被"看一眼"发现。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import rank_strategies as R  # noqa: E402


def _row(**kw) -> str:
    base = dict(name="x", n_trades=0, own_ret=None, cagr=None, exposure=0.0,
                excess=None, excess_per_exposure=None, beta=0.0, alpha_annual=None,
                t_alpha=None, folds_positive=0, maxdd=None, r_median=None, dsr=None,
                dsr_k=1)
    base.update(kw)
    return R._row(base)


def test_ratio_columns_are_scaled_by_100():
    r = _row(n_trades=114, own_ret=0.2931, cagr=0.11, exposure=0.2216, excess=-0.0597,
             excess_per_exposure=-0.27, beta=0.33, alpha_annual=0.0078, t_alpha=0.10,
             folds_positive=3, maxdd=-0.1, r_median=-1.01, dsr=0.019, dsr_k=1614)
    assert "+29.31%" in r, r
    assert "-5.97pp" in r, r
    assert "22.16%" in r, r
    assert "-10.0%" in r, r             # maxDD 也是比率（一位小数）
    assert "+0.78%" in r and "+0.10" in r
    assert "0.019@1614" in r            # DSR 与它的 k 都要打出来（k 逐行不同）


def test_missing_values_print_a_dash_not_a_zero():
    """缺读数要打「—」而不是 0：0 是一个具体的读数，会把"没测到"伪装成"测到 0"。"""
    r = _row(own_ret=None, excess=float("nan"), maxdd=None)
    assert r.count("—") >= 3, r


def test_windows_and_controls_are_labelled():
    """共同窗必须写明它不是样本外；对照（什么都不做）要在表里。"""
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "含 holdout 天数，非样本外" in src
    assert "DO_NOTHING" in src
