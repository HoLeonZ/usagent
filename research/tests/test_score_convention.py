"""抢仓位槽的**符号约定**（审计 F2）。

缺陷本体：排序一律**降序**（score 越大越先抢到仓位槽），而 `idio*` 的机制是
"残差越负（跌得越'自己'）越优先"、`cap_inv` 的机制是"收盘趴低（卖压持续）越优先"——
两者却直接把**裸列**当 score。于是降序排在最前的是**最不极端**的那一批，与声明的
机制**相反**，而那几个族的水平数字（以及"低波动承重"那条线索）正是从这些排序里读出来的。
（讽刺的是：文件里那句注释写的就是"残差越负越优先"，而代码做的是相反的事。）

另外 `volshock_score` / `lhr_score` 在**指标层**已经取过负，所以它们**不能**再进取负集合
—— 两处都取负会抵消。这条也一并钉住。
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

from research import families as F  # noqa: E402
from research import indicators  # noqa: E402
from research.tests.conftest import make_synth_panel  # noqa: E402

# 每族**声明的**优先方向（越小越优先 ⇒ score 必须取负）。
# 这张表是判据本身：改代码而不改这张表、或反过来，测试都会红。
INTENDED_DIRECTION = {
    "idio": "越小越优先",            # 残差越负（跌得越"自己"）
    "idio_mktdown": "越小越优先",
    "idio_highvol": "越小越优先",
    "idio_highrvol": "越小越优先",
    "cap_inv": "越小越优先",          # 收盘趴低 = 卖压持续
    "cap": "越大越优先",              # 收盘强 = 有人接盘
    "volshock": "越大越优先",         # volshock_score 在指标层已取负
    "lhr": "越大越优先",              # lhr_score 在指标层已取负
}


def test_negated_set_matches_the_declared_directions():
    want = {f for f, d in INTENDED_DIRECTION.items() if d == "越小越优先"}
    assert F.SCORE_NEGATED == want, (
        f"取负集合与声明方向不一致：代码 {sorted(F.SCORE_NEGATED)} vs 判据 {sorted(want)}")


def test_indicators_already_negated_are_not_negated_twice():
    """`volshock_score`/`lhr_score` 是指标层取过负的列 —— 再取负会抵消。"""
    ind = F.add_family_indicators(indicators.compute_all(make_synth_panel(("AAA",), bars=300)))
    # volshock_score = -shock_z_S（指标层），lhr_score = -(dd120/atr_pct_ma120)
    s = ind[["shock_z_S", "volshock_score"]].dropna()
    if len(s):
        assert np.allclose(s["volshock_score"], -s["shock_z_S"]), \
            "volshock_score 的符号变了 —— 那 SCORE_NEGATED 的名单要跟着改"
    assert "volshock" not in F.SCORE_NEGATED and "lhr" not in F.SCORE_NEGATED


@pytest.fixture
def ind():
    return F.add_family_indicators(indicators.compute_all(
        make_synth_panel(("AAA", "BBB"), bars=300)))


def _hits_with_raw(monkeypatch, ind, family: str, params: dict | None = None
                   ) -> pd.DataFrame:
    """把掩码换成"全放行"，取出排序后的命中，并**从输入面板接回**裸列。

    ⚠️ `select()` 的输出只有 `signals.ENTRY_COLS` —— 裸列（clv/__resid/mom120）不在里面，
    所以要按 (symbol, date) 接回来才能验证 score 的符号（实测踩过：直接取列会 KeyError）。
    """
    monkeypatch.setitem(F.MASK, family, lambda df, p: pd.Series(True, index=df.index))
    days = sorted(ind["date"].unique())
    lo, hi = str(days[130])[:10], str(days[-1])[:10]
    hits = F.select(ind, family, params or {}, lo, hi)
    if hits.empty:
        return hits
    cols = [c for c in ("clv", "_resid", "mom120") if c in ind.columns]
    raw = ind[["symbol", "date", *cols]].drop_duplicates(["symbol", "date"])
    return hits.merge(raw, on=["symbol", "date"], how="left")


def test_cap_inv_prefers_the_lowest_clv(monkeypatch, ind):
    """`cap_inv`：收盘越趴低越优先 ⇒ score 必须是 **−clv**。"""
    hits = _hits_with_raw(monkeypatch, ind, "cap_inv")
    assert not hits.empty
    for _, day in hits.groupby("date"):
        if len(day) >= 2:
            assert day.iloc[0]["clv"] <= day.iloc[-1]["clv"], \
                "同一天里，收盘趴低的那只必须先抢到仓位槽"
            break
    else:
        pytest.skip("合成面板里没有同一天的多候选 —— 由下一条的列级断言覆盖")


def test_score_is_the_negation_of_the_raw_column(monkeypatch, ind):
    """直接对列：取负集合里的族，score 必须等于裸列取负。"""
    for fam, col in (("cap_inv", "clv"), ("idio", "_resid")):
        hits = _hits_with_raw(monkeypatch, ind, fam)
        assert not hits.empty and hits[col].notna().any(), fam
        m = hits[col].notna()
        assert np.allclose(hits.loc[m, "score"].to_numpy(dtype=float),
                           -hits.loc[m, col].to_numpy(dtype=float)), \
            f"{fam} 的 score 不是 -{col} 的逐元素取负"


def test_cap_prefers_the_highest_clv(monkeypatch, ind):
    """反向对照：`cap`（收盘强=有人接）**不**取负 —— 否则两族会被改成同一件事。"""
    hits = _hits_with_raw(monkeypatch, ind, "cap")
    m = hits["clv"].notna()
    assert m.any()
    assert np.allclose(hits.loc[m, "score"].to_numpy(dtype=float),
                       hits.loc[m, "clv"].to_numpy(dtype=float))


def test_family_without_a_score_column_defaults_to_mom120(monkeypatch, ind):
    """没有登记 score 的族用 mom120（与 uptrend-pullback 可比），且**不**取负。"""
    hits = _hits_with_raw(monkeypatch, ind, "xs_flow")
    m = hits["mom120"].notna()
    assert m.any()
    assert np.allclose(hits.loc[m, "score"].to_numpy(dtype=float),
                       hits.loc[m, "mom120"].to_numpy(dtype=float))
