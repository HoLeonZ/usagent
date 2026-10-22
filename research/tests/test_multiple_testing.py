"""族错误率校正（审计 B4）—— 一次扫描十几个因子，判词不能靠单点比较。

缺陷本体：本仓库只有 `screen.py` 落了 Bonferroni，而 `factor_probe` 的判据①/③
是在**逐因子点比较** |t| ≥ 2 上直接下判词的 —— 扫 8 个因子"总有一个过 2"这件事
对读者完全不可见。判据④（随机对照）甚至被合成复核指出在零边缘下约 31% 为真。
按 CLAUDE.md §5(c)，报告多参数扫描的结果时必须给族校正后的判据。

本文件钉住：
  1. `two_sided_p` 与 `bonferroni_t_threshold` 互相自洽（同一 α 下门槛与 p 对得上）；
  2. BH 步进法算对（经典例子、全拒绝、全不拒绝、单调于 q）；
  3. **NaN 不参与排序、也绝不判为拒绝**（不可检验的比较不许"混过去"）；
  4. 族规模下点比较的显著会被收回（12 次比较时 |t|=2.2 过不了 Bonferroni）；
  5. 源码级护栏：factor_probe 的两族都真的接了校正报告。
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import metrics as M  # noqa: E402


# ---------------------------------------------------------------- 两个门槛自洽

def test_bonferroni_threshold_and_p_value_are_consistent():
    for k in (1, 4, 12, 100):
        tc = M.bonferroni_t_threshold(k, 0.05)
        assert M.two_sided_p(tc) == pytest.approx(0.05 / k, rel=1e-6)


def test_two_sided_p_known_values():
    assert M.two_sided_p(0.0) == pytest.approx(1.0)
    assert M.two_sided_p(1.96) == pytest.approx(0.05, abs=1e-3)
    assert M.two_sided_p(2.5758) == pytest.approx(0.01, abs=1e-3)
    assert M.two_sided_p(-1.96) == pytest.approx(M.two_sided_p(1.96))
    assert math.isnan(M.two_sided_p(float("nan")))


# ---------------------------------------------------------------- BH 步进法

def test_bh_classic_step_up_example():
    """m=15、q=0.05 的经典例子：只有 p=0.001 过线（阈值 0.05·i/15）。"""
    ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212,
          0.216, 0.222, 0.251, 0.269, 0.275, 0.34]
    out = M.benjamini_hochberg(ps, 0.05)
    assert out["n"] == 15 and out["n_rejected"] == 1
    assert out["crit_p"] == pytest.approx(0.001)
    assert out["rejected"][0] is True and sum(out["rejected"]) == 1


def test_bh_rejects_everything_when_all_p_are_tiny():
    out = M.benjamini_hochberg([1e-8] * 12, 0.05)
    assert out["n_rejected"] == 12


def test_bh_rejects_nothing_when_all_p_are_large():
    out = M.benjamini_hochberg([0.6, 0.7, 0.8, 0.9], 0.05)
    assert out["n_rejected"] == 0 and math.isnan(out["crit_p"])


def test_bh_is_monotone_in_q():
    ps = [0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.2]
    prev = -1
    for q in (0.01, 0.05, 0.1, 0.25):
        n = M.benjamini_hochberg(ps, q)["n_rejected"]
        assert n >= prev, f"q 变大不该让拒绝数下降（q={q}）"
        prev = n


def test_bh_ignores_nan_but_does_not_reject_them():
    """不可检验的比较（期数不足 ⇒ t 是 NaN）不参与排序，也绝不给它们"通过"的假象。"""
    out = M.benjamini_hochberg([0.001, float("nan"), 0.002], 0.05)
    assert out["n"] == 2, "NaN 不计入可检验数"
    assert out["rejected"][1] is False
    assert out["n_rejected"] == 2


def test_bh_all_nan():
    out = M.benjamini_hochberg([float("nan")] * 5, 0.05)
    assert out["n"] == 0 and out["n_rejected"] == 0


def test_bh_validates_q():
    with pytest.raises(ValueError):
        M.benjamini_hochberg([0.01], 0.0)
    with pytest.raises(ValueError):
        M.benjamini_hochberg([0.01], 1.5)


# ---------------------------------------------------------------- 族规模下判词被收回

def test_pointwise_significance_does_not_survive_a_realistic_factor_family():
    """12 个因子里有一个 |t|=2.2（点比较"显著"）—— Bonferroni 门槛 2.87，它过不了。"""
    ts = [2.2] + [0.4, -0.3, 0.9, -1.2, 0.1, 1.5] * 2       # 13 个，一个点显著
    m = len(ts)
    t_bonf = M.bonferroni_t_threshold(m, 0.05)
    assert abs(ts[0]) < t_bonf, "该例子必须过不了 Bonferroni，否则测试无意义"
    p = [M.two_sided_p(t) for t in ts]
    bh = M.benjamini_hochberg(p, 0.05)
    assert bh["n_rejected"] == 0, "FDR 也不该在这个族里放进任何一个"


def test_a_strong_effect_survives_both_criteria():
    ts = [4.0] + [0.3] * 11
    p = [M.two_sided_p(t) for t in ts]
    assert sum(1 for t in ts if abs(t) >= M.bonferroni_t_threshold(len(ts))) == 1
    assert M.benjamini_hochberg(p, 0.05)["n_rejected"] >= 1


def test_pre_registered_criterion_is_not_silently_relaxed():
    """预注册的 |t| ≥ 2 是**预注册**的：校正的产物是"还剩几个"，不是把判据改成别的数。

    （事后把 2 改成 2.87 就成了一次新的搜索 —— §5 要求它计入 n_trials。）
    """
    from research import factor_probe as FP
    assert FP.PRE_REG_T == 2.0


# ---------------------------------------------------------------- 源码级护栏

def test_factor_probe_reports_family_correction_for_both_criteria():
    """两族判据都必须真的接上校正报告 —— 否则"已控制族错误率"只是注释里的话。"""
    src = (ROOT / "research" / "factor_probe.py").read_text()
    calls = re.findall(r"_family_report\(", src)
    assert len(calls) >= 3, f"_family_report 只被调用 {len(calls)-1} 次（含定义）"
    assert "判据① 秩相关 IC" in src and "判据③ 分位组合" in src
