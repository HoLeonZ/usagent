"""因果性 / 未来函数防线(设计文档 §6 第一条,最重要)。

做法:整表算一遍指标,再把输入**截断到 T** 重算一遍,断言 T 那一行的每个指标逐位相同。
只要有任何 rolling(center=True) / shift(-x) / bfill / 全表归一化,这条测试就会红。

第二条:把某一只在 T 之后的数据整体缩放,断言另一只在 T 处的指标不受影响 ——
钉住「所有 rolling/shift 都按 symbol 分组」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.indicators import INDICATOR_COLS, compute_indicators

# 相对容差 0(逐位)在这里是安全的:输入相同、算子相同、顺序相同 —— 浮点结果必然一致。
# 若某天有人换成非确定性算子(如并行 reduce),这条会先红,这正是我们要的。
TOL = 0.0

CHECK_COLS = [c for c in INDICATOR_COLS if c != "down_id"]


def _values_at(df: pd.DataFrame, symbol: str, ts: pd.Timestamp) -> dict:
    row = df[(df["symbol"] == symbol) & (df["date"] == ts)]
    assert len(row) == 1, f"{symbol} @ {ts} 应有且仅有一行"
    return {c: row.iloc[0][c] for c in CHECK_COLS}


@pytest.mark.parametrize("symbol", ["AAA", "BBB"])
def test_truncating_history_does_not_change_past_values(synth_panel, symbol):
    """截断到 T 后重算,T 处指标必须与整表一致。"""
    full = compute_indicators(synth_panel)
    dates = sorted(synth_panel["date"].unique())

    # 取若干 T:预热之后、末端之前,跨越不同形态
    probes = [dates[150], dates[200], dates[260], dates[320]]
    for ts in probes:
        ts = pd.Timestamp(ts)
        trunc = compute_indicators(synth_panel[synth_panel["date"] <= ts])
        a = _values_at(full, symbol, ts)
        b = _values_at(trunc, symbol, ts)
        for col in CHECK_COLS:
            av, bv = a[col], b[col]
            if isinstance(av, float) and (np.isnan(av) or np.isnan(bv)):
                assert (isinstance(av, float) and isinstance(bv, float)
                        and np.isnan(av) and np.isnan(bv)), \
                    f"{symbol} {ts.date()} {col}: NaN 不一致 ({av!r} vs {bv!r})"
                continue
            assert float(av) == pytest.approx(float(bv), abs=TOL), \
                f"{symbol} {ts.date()} {col}: 截断后变了 ({av!r} -> {bv!r}) — 疑似未来函数"


def test_no_cross_symbol_leak(synth_panel):
    """把 BBB 在第 200 根之后的数据整体 ×3,AAA 的指标必须一字不变。"""
    dates = sorted(synth_panel["date"].unique())
    cut = pd.Timestamp(dates[200])

    base = compute_indicators(synth_panel)

    mutated = synth_panel.copy()
    m = (mutated["symbol"] == "BBB") & (mutated["date"] > cut)
    for c in ("open", "high", "low", "close"):
        mutated.loc[m, c] = mutated.loc[m, c] * 3.0
    mutated.loc[m, "amount"] = mutated.loc[m, "volume"] * mutated.loc[m, "close"]
    after = compute_indicators(mutated)

    probe = pd.Timestamp(dates[200])          # 在改动点之前
    for sym in ("AAA", "BBB"):
        a = _values_at(base, sym, probe)
        b = _values_at(after, sym, probe)
        for col in CHECK_COLS:
            av, bv = a[col], b[col]
            if isinstance(av, float) and np.isnan(av):
                assert np.isnan(bv)
                continue
            assert float(av) == pytest.approx(float(bv), abs=TOL), \
                f"{sym} {col} 受另一只标的的未来改动影响 — 分组错了"


def test_indicators_are_nan_until_warmup(synth_panel):
    """预热期必须诚实为 NaN,不能靠 bfill 之类的补值蒙混。"""
    ind = compute_indicators(synth_panel)
    a = ind[ind["symbol"] == "AAA"].reset_index(drop=True)

    # 实测的预热边界(2026-10-14),留作回归锚点:
    #   atr14            14      vol_ma20        19
    #   ma60 / amount60 / above_ma60_ratio   59
    #   ma120           119      mom120         120  ← 最紧的一条,决定最早可用信号日
    assert a["atr14"].iloc[:14].isna().all() and not np.isnan(a["atr14"].iloc[14])
    assert a["vol_ma20"].iloc[:19].isna().all() and not np.isnan(a["vol_ma20"].iloc[19])
    assert a["ma60"].iloc[:59].isna().all() and not np.isnan(a["ma60"].iloc[59])
    assert a["ma120"].iloc[:119].isna().all() and not np.isnan(a["ma120"].iloc[119])
    assert a["mom120"].iloc[:120].isna().all() and not np.isnan(a["mom120"].iloc[120])

    # panel.MAX_LOOKBACK_BARS 必须覆盖最紧的一条(mom120 需 121 根)
    from research.panel import MAX_LOOKBACK_BARS
    assert MAX_LOOKBACK_BARS >= 120


def test_above_ma60_ratio_uses_fabricated_zeros_during_warmup(synth_panel):
    """记录一处**继承自 hiagent 原码**的细节,不是 bug,但必须知情。

    `above_ma60 = (close > ma60).astype(float)` —— 在 ma60 还是 NaN 的预热期,
    `close > NaN` 求值为 False,所以 above_ma60 是 **0.0 而不是 NaN**。
    于是它的 60 日滚动均值从索引 59 起就有值,把「当时其实不知道」读成了「当时在均线下方」。
    索引 59-63 那几行因此是被人造 0 拉低的伪值。

    为什么无害:信号只在 mom120 就绪(索引 120)之后才评估,而那时窗口已完全落在
    真实 ma60 区间内。本测试把这个边界钉住 —— 若将来有人把信号评估提前到预热期,
    这里会提醒他 above_ma60_ratio 在 59-118 段不可信。
    """
    a = compute_indicators(synth_panel)
    a = a[a["symbol"] == "AAA"].reset_index(drop=True)

    # 预热期 above_ma60 是人造 0.0,不是 NaN
    assert (a["above_ma60"].iloc[:59] == 0.0).all()
    assert a["above_ma60"].iloc[:59].notna().all()

    # 因此 ratio 从 59 起就有值,且开头被这些人造 0 压低
    assert not np.isnan(a["above_ma60_ratio"].iloc[59])
    assert a["above_ma60_ratio"].iloc[59] == pytest.approx(0.0)

    # 到索引 120(信号可评估的最早日)窗口已完全落在真实 ma60 区间内
    assert a.index[120] == 120
    window = a["above_ma60"].iloc[61:121]
    assert a["above_ma60_ratio"].iloc[120] == pytest.approx(window.mean())


def test_indicators_pass_amount_through_without_recomputing(synth_panel):
    """compute_indicators 必须**原样透传** amount,不许用 close 自己重算。

    amount 的口径由 panel.to_panel_frame 定死(volume × **as-traded** close);
    指标层若"顺手"用分红回补 close 重算一遍,就把未来分红又注入了 E 条件 ——
    那正是 2026-10-15 审计抓到的前视(740 个就绪格的 E 判定被未来分红翻转)。

    这条测试用「与 close 完全无关的 amount」来钉:只要指标层重算,立刻红。
    (旧版本写的是 `amount == volume × close`,在新口径下它恒真 —— 因为它拿的是同一个
     被复权的 close,区分不出两套价。)
    """
    p = synth_panel.copy()
    p["amount"] = p["volume"] * 7.5            # 与 close 无关的常数
    ind = compute_indicators(p)
    assert np.allclose(ind["amount"].to_numpy(), p["amount"].to_numpy(), rtol=0, atol=0)

