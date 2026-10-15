"""Walk-Forward Validation —— CLAUDE.md §5(b)。

铁律原文:Optimization logic MUST enforce Walk-Forward Validation (WFV) with strict
out-of-sample (OOS) testing windows.

2026-10-15 审计实测:仓库里**没有 WFV**。search_12m 的优化逻辑只有一个窗口
(WIN_LO..WIN_HI),而 `--windows` 是把冻结的 preset 逐窗重跑 —— 那是稳定性展示,
不是训练/测试分离。于是「搜到的配置」与「在这 12 个月上拟合出来的配置」在代码层不可
区分,交付 preset 的五个参数也就**全部落在网格边界**上(边界最优 = 最优由网格取景范围
决定,不由数据决定)。

这里测的是 WFV 的核心不变量:**选择只看训练窗**。用一个「作弊配置」判定 —— 它在测试窗
上极好、在训练窗上平庸。若选择逻辑偷看了测试窗,它会被选中。
"""
from __future__ import annotations

import numpy as np
import pytest

from research import search_12m as S

GOOD = {"max_rsi14": 60.0}      # 训练窗上更好
CHEAT = {"max_rsi14": 40.0}     # 只在**测试窗**上更好
TEST_ENDS = ("2025-01-02", "2025-04-01")

FOLDS = [
    ("2024-01-02", "2024-07-01", "2024-07-01", "2025-01-02"),
    ("2024-04-01", "2024-10-01", "2024-10-01", "2025-04-01"),
]


def _fake_evaluate(record):
    """训练窗:GOOD 0.2 / CHEAT 0.1;测试窗:CHEAT 0.9 / GOOD 0.1。"""
    def f(ind, reg, cfg, lo, hi, **kw):
        record.append((dict(cfg), lo, hi))
        cheat = cfg.get("max_rsi14") == CHEAT["max_rsi14"]
        on_test = hi in TEST_ENDS
        cagr = (0.9 if cheat else 0.1) if on_test else (0.1 if cheat else 0.2)
        return {"cagr": cagr, "trades": 5, "sharpe": 0.1, "total": cagr,
                "max_dd": -0.1, "win": 0.5, "e_r": 0.1, "exposure": 0.2,
                "reason": "ok", "cfg": dict(cfg)}
    return f


# ---------------------------------------------------------------- 折窗切分

def test_make_folds_are_train_then_test_with_no_overlap():
    folds = S.make_folds("2024-01-01", "2025-07-01",
                         train_months=6, test_months=3, step_months=3)
    assert folds
    for tr_lo, tr_hi, te_lo, te_hi in folds:
        assert tr_lo < tr_hi <= te_lo < te_hi, "训练窗必须在测试窗之前且不重叠"
        assert te_hi <= "2025-07-01"


def test_make_folds_advances_by_the_step():
    folds = S.make_folds("2024-01-01", "2025-01-01",
                         train_months=3, test_months=3, step_months=3)
    assert folds[0][0] == "2024-01-01"
    assert folds[0][2] == "2024-04-01"
    assert folds[1][0] == "2024-04-01"


def test_make_folds_rejects_a_window_without_room_for_one_fold():
    with pytest.raises(ValueError, match="fold"):
        S.make_folds("2024-01-01", "2024-04-01", train_months=6, test_months=3,
                     step_months=3)


# ---------------------------------------------------------------- 选择只看训练窗

def test_walk_forward_selects_only_on_the_train_window():
    """核心不变量:测试窗的结果只能**事后**被读,不得参与选择。"""
    record: list = []
    df = S.walk_forward(None, None, grid=[GOOD, CHEAT], folds=FOLDS,
                        evaluate_fn=_fake_evaluate(record))

    assert len(df) == len(FOLDS)
    # 训练窗上 GOOD(0.2) > CHEAT(0.1) ⇒ 每折都必须选 GOOD,
    # 于是测试窗上读到的是 GOOD 的 0.1,而不是 CHEAT 的 0.9。
    for _, row in df.iterrows():
        assert row["cfg"] == S.cfg_key(GOOD), f"选择偷看了测试窗:{row['cfg']}"
        assert row["test_cagr"] == pytest.approx(0.1)

    # 每个测试窗只被评估一次(选完之后才看)
    assert len([1 for cfg, lo, hi in record if hi in TEST_ENDS]) == len(FOLDS)


def test_walk_forward_defaults_to_the_module_evaluate(monkeypatch):
    """默认路径必须走模块级 evaluate —— 免得将来有人在别处另接一条选择逻辑。"""
    seen: list = []
    monkeypatch.setattr(S, "evaluate", _fake_evaluate(seen))
    S.walk_forward(None, None, grid=[GOOD], folds=FOLDS[:1])
    assert seen, "walk_forward 默认没有调用模块级 evaluate"


def test_walk_forward_reports_per_fold_train_and_test():
    df = S.walk_forward(None, None, grid=[GOOD, CHEAT], folds=FOLDS,
                        evaluate_fn=_fake_evaluate([]))
    assert {"train_lo", "train_hi", "test_lo", "test_hi",
            "train_cagr", "test_cagr", "cfg"} <= set(df.columns)
    assert np.isfinite(df["test_cagr"]).all()
    assert np.isfinite(df["train_cagr"]).all()


def test_walk_forward_requires_folds():
    with pytest.raises(ValueError, match="fold"):
        S.walk_forward(None, None, grid=[GOOD], folds=[])


def test_walk_forward_requires_a_non_empty_grid():
    with pytest.raises(ValueError, match="grid"):
        S.walk_forward(None, None, grid=[], folds=FOLDS)
