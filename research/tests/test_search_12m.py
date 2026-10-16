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

from pathlib import Path

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


# ------------------------------- 搜索结果的三态必须可区分(argmax(CAGR) 的正确性前提)
# 2026-10-15 审计:stage1 的 37 行里有 3 行是「过滤层杀光信号 → 空转 → 记成 +0.0%,
# reason=ok」。在一个 argmax(CAGR) 的搜索里,+0.0% 会**赢过所有负收益配置** ——
# 搜索会系统性朝「过度过滤」的方向跑。这正是挖高 CAGR 前必须先修掉的东西。

def _one_signal():
    import pandas as pd
    from research import signals
    row = {c: float("nan") for c in signals.ENTRY_COLS}
    row.update({"date": pd.Timestamp("2024-01-02"), "symbol": "AAA",
                "score": 1.0, "atr_pct": 0.02})
    return pd.DataFrame([row])[signals.ENTRY_COLS]


def _empty_signals():
    import pandas as pd
    from research import signals
    return pd.DataFrame(columns=signals.ENTRY_COLS)


def test_evaluate_does_not_report_zero_cagr_for_an_empty_signal_set(monkeypatch):
    from research import signals
    monkeypatch.setattr(signals, "select_entries", lambda *a, **k: _empty_signals())
    monkeypatch.setattr(S.broker, "simulate",
                        lambda *a, **k: pytest.fail("空信号不该去跑回测(会得到平直曲线 → +0.0%)"))
    r = S.evaluate(None, None, {})
    assert r["cagr"] is None, "空信号被记成了数字 —— argmax(CAGR) 会把空转配置选成最优"
    assert r["reason"] == "no_signals"
    assert r["trades"] == 0


def test_evaluate_propagates_broker_errors(monkeypatch):
    """broker 的 fail-fast 不得被 evaluate 吞成 cagr=None(§0)。

    审计实测:evaluate 把 BrokerError 降级成一行 cagr=None,于是「记账断言被触发」
    这类真错误在 106 次试验里静默消失,而 n_trials 照加。
    """
    from research import signals
    monkeypatch.setattr(signals, "select_entries", lambda *a, **k: _one_signal())

    def boom(*a, **k):
        raise S.broker.BrokerError("权益曲线含非有限值 —— 记账错了")

    monkeypatch.setattr(S.broker, "simulate", boom)
    with pytest.raises(S.broker.BrokerError, match="记账错了"):
        S.evaluate(None, None, {})


def test_random_control_counts_failed_draws(monkeypatch):
    """随机对照必须报出失败次数 —— 「200 次」不能只是名义值(不许有 silent cap)。"""
    from research import broker as B
    from research.indicators import compute_indicators
    from research.tests.conftest import make_synth_panel

    ind = compute_indicators(make_synth_panel(("AAA", "BBB"), bars=140))
    days = sorted(ind["date"].unique())
    real = B.simulate
    calls = {"n": 0}

    def flaky(preset, sig, panel, **kw):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise B.BrokerError("模拟一次失败")
        return real(preset, sig, panel, **kw)

    monkeypatch.setattr(B, "simulate", flaky)
    out = S.random_control(ind, {"sl_mult": 1.5, "tp_mult": 4.0}, n_trades=2, n_runs=6,
                           lo=str(days[125])[:10], hi=str(days[-1])[:10], seed=0)
    assert out["n_failed"] > 0, "失败的重抽被静默丢弃了"
    assert out["n_runs"] + out["n_failed"] == 6, (
        f"成功 {out['n_runs']} + 失败 {out['n_failed']} 应等于请求的 6 次")


def test_random_control_draws_from_the_strategys_opportunity_set():
    """随机对照必须从**策略自己够得着的池子**里抽,不是全池。

    2026-10-15 实测:旧实现从全池 596 只里抽(含小盘股),200 次里 33 次在出场时撞上
    §4(a) 的量能上限而 fail-fast —— 因为 E 条件的流动性门(amount60 ≥ $3e7)只在策略
    一侧生效。拿一个「永远不会碰小盘股」的策略去和「会碰小盘股」的随机对照比,
    对照本身就不成立;顺带还让重抽失败率 16.5%,分布只由成功子集给出。
    """
    import pandas as pd
    from research.indicators import compute_indicators
    from research.tests.conftest import make_synth_panel

    ind = compute_indicators(make_synth_panel(("AAA", "BBB"), bars=140))
    # 人为把其中一只标的的流动性打到门槛之下;对照池不应再抽到它
    ind.loc[ind["symbol"] == "BBB", "amount60"] = 1.0
    days = sorted(ind["date"].unique())
    lo, hi = str(days[125])[:10], str(days[-1])[:10]
    seen = set()
    real = S.broker.simulate

    def spy(preset, sig, panel, **kw):
        seen.update(sig["symbol"].tolist())
        return real(preset, sig, panel, **kw)

    S.broker.simulate = spy
    try:
        S.random_control(ind, {"sl_mult": 1.5, "tp_mult": 4.0}, n_trades=2,
                         n_runs=8, lo=lo, hi=hi, seed=0)
    finally:
        S.broker.simulate = real
    pool = set(S._control_pool(ind, lo, hi)["symbol"])
    assert "BBB" not in pool, "流动性不达标的标的仍在对照池里"
    assert seen and seen <= {"AAA"}, f"抽到了池子外的标的:{seen - {'AAA'}}"


# ---------------------------------------------------------------- 机会集对齐（C2）

def test_explicit_none_removes_the_liquidity_upper_bound():
    """`{"max_amount": None}` = **显式移除**流动性上界；缺键 = 保留 v21 的上界。

    两种语义必须分开，否则没办法表达"移除"（审计 C2：基准线带着 A 股的
    `amount60 ≤ $3e8` 上界，而族型策略没有这一条 —— 不显式去掉，"族 vs 基准线"
    就是两个机会集的比较）。
    """
    from research import signals
    base = S.signal_kwargs({})
    assert base["max_amount"] == signals.V21_SIGNAL_PARAMS["max_amount"]
    removed = S.signal_kwargs({"max_amount": None})
    assert removed["max_amount"] is None, "显式的 None 应当把上界去掉"
    assert S.signal_kwargs({"max_amount": 5e8})["max_amount"] == 5e8


def test_none_on_other_axes_still_means_do_not_touch():
    """其它轴上 None 仍表示"这一轴不动" —— 与 max_amount 的语义**刻意不同**。"""
    base = S.signal_kwargs({})
    assert S.signal_kwargs({"min_mom120": None})["min_mom120"] == base["min_mom120"]


def test_family_search_judges_against_the_aligned_baseline():
    """判据① 必须用**去掉上界**的那条基准线（同机会集才可比），且把两条都打印出来。"""
    src = (Path(__file__).resolve().parents[2] / "research" / "family_search.py").read_text()
    assert 'bar_pooled, bar_cont = upb_te_noband["mean_cagr"]' in src, \
        "判据① 又用回带 A 股上界的那条线了"
    assert "机会集对齐（C2）" in src, "两条线都要打印出来，读者才能看到上界的影响"
