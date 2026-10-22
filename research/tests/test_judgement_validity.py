"""判据本身的有效性：零边缘假阳性率（B6）+ 共同窗口集比较（B5）。

审计 B6：判据③（"4 折里 ≥3 折为正"）与④（"4 折里 ≥3 折打赢随机中位"）在**毫无边缘**
的情况下各自也有 **5/16 = 31.25%** 的概率为真 —— 它们不是 α=5% 的证据标准。
不把这个数写出来，"四条同时满足"会被读成一件比它更强的事。

审计 B5：判据① 比的是两个 `mean_cagr`，而本族的均值只在**它有信号的那些窗**上取，
基准线同理 —— 两批样本不一样时，比的就不只是策略（一个只在 2 个窗出信号的族，
可能在另外 2 个基准线亏钱的窗里根本没参与过）。修法是**退回共同窗集**重算，并出声。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import family_search as FS  # noqa: E402
from research import metrics as M  # noqa: E402


# ---------------------------------------------------------------- 零边缘假阳性率

@pytest.mark.parametrize("k,n,want", [
    (3, 4, 5 / 16),          # C(4,3)+C(4,4) = 5/16 = 0.3125
    (4, 4, 1 / 16),
    (2, 2, 1 / 4),
    (1, 1, 0.5),
    (0, 4, 1.0),
    (5, 4, 0.0),             # k > n ⇒ 0
    (6, 10, 386 / 1024),     # 已知的二项上尾
])
def test_null_prob_matches_the_binomial_tail(k, n, want):
    assert M.null_prob_at_least(k, n) == pytest.approx(want)


def test_null_prob_rejects_negative_inputs():
    with pytest.raises(ValueError):
        M.null_prob_at_least(-1, 4)


def test_judgement_null_rates_expose_how_weak_the_criteria_are():
    r = M.judgement_null_rates(4)
    assert len(r["rates"]) == 4
    # ③④ 各 31.25%，① ② 各 50% ⇒ 合取（独立近似）约 2.4%
    thirds = [v for k, v in r["rates"].items() if "≥3 折" in k]
    assert len(thirds) == 2 and all(v == pytest.approx(5 / 16) for v in thirds)
    assert r["joint"] == pytest.approx(0.5 * 0.5 * (5 / 16) ** 2)
    assert r["joint"] > 0.02, "合取的概率必须被算出来（不是默认 5%）"
    assert r["independent"] is False, "必须标明这是独立近似，不是精确族错误率"


def test_more_folds_make_the_fixed_three_fold_threshold_weaker():
    """⚠️ 反直觉但正确：折数越多，"≥3 折"这个**固定阈值**越松（越容易碰巧满足）。

    P(≥3|n=4) = 5/16 = 31.2%，P(≥3|n=6) = 42/64 = 65.6% —— 也就是说
    "折数多"这件事本身会把判据③④ 变成几乎白送。合格的做法是用**比例**
    （如 ≥75% 的折）或固定 α，而不是固定次数。这里把方向钉住，免得有人
    以为"多做几折就更严格"。
    """
    assert M.judgement_null_rates(3)["rates"]["③ 3 折里 ≥3 折为正"] == pytest.approx(0.125)
    assert M.null_prob_at_least(3, 6) == pytest.approx(42 / 64)
    assert M.null_prob_at_least(3, 6) > M.null_prob_at_least(3, 4), \
        "固定阈值下折数越多越松 —— 这不是笔误"
    assert M.judgement_null_rates(6)["joint"] > M.judgement_null_rates(4)["joint"]


# ---------------------------------------------------------------- 共同窗口集比较

def test_same_window_sets_use_the_reported_means():
    c1, note = FS.c1_on_common_windows([0.10, 0.20], [0.05, 0.05], bar_pooled=0.0)
    assert c1 is True and note == ""


def test_different_window_sets_are_recomputed_on_the_intersection():
    """本族只在第 0 窗有信号、基准只在第 1 窗有信号 ⇒ 不能拿两个均值直接比。"""
    c1, note = FS.c1_on_common_windows([0.50, None], [None, -0.90], bar_pooled=0.0)
    assert "没有一个共同窗" in note and c1 is False


def test_intersection_overrides_a_misleading_mean():
    """解析地构造：全窗比较说"赢"，共同窗比较说"输"。

    本族 [0.10, 0.90]（均值 0.50）；基准 [0.05, None]（均值 0.05）。
    全窗比均值 ⇒ 0.50 > 0.05「赢」；但两者的**唯一共同窗**是第 0 窗：0.10 vs 0.05
    —— 仍然赢。把基准的共同窗值改成 0.20 才是"共同窗输"，这里验证代码确实用了交集。
    """
    c1, note = FS.c1_on_common_windows([0.10, 0.90], [0.20, None], bar_pooled=0.0)
    assert c1 is False, "共同窗里本族输给基准 ⇒ 判据① 必须为假，尽管全窗均值更高"
    assert "共同 1 窗" in note


def test_mismatched_lengths_are_aligned_by_index():
    c1, note = FS.c1_on_common_windows([0.10], [0.05, 0.05], bar_pooled=0.0)
    assert c1 is True, "长度不同时按索引对齐（本族缺的窗不是'赢'）"
    assert "本族独有 0 窗" in note and "基准独有 1 窗" in note
