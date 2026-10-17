"""归因脚本的**窗口纪律** —— 护栏要能在没有真库的机器上跑。

这条测试守的是协议 §5 的一句话：归因阶段不许看 holdout。它值得写成测试，是因为
它的失效方式是**无声的**：某轮 cron（或某个人）把默认窗口顺手改成 holdout，
"一次性 holdout" 就悄悄没了 —— 而那正是本仓库反复吃亏的那类缺陷。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import event_attribution as EA  # noqa: E402


def test_default_window_is_the_train_window():
    assert EA.TRAIN == ("2024-04-09", "2025-10-13")
    assert EA.HOLDOUT[0] > EA.TRAIN[1], "训练窗与 holdout 不许重叠"


def test_train_window_is_allowed():
    EA._assert_window_is_allowed(*EA.TRAIN, allow_holdout=False)       # 不抛


def test_sub_window_inside_train_is_allowed():
    EA._assert_window_is_allowed("2024-06-01", "2025-01-31", allow_holdout=False)


@pytest.mark.parametrize("lo,hi", [
    ("2025-10-14", "2026-10-13"),          # 整段 holdout
    ("2025-09-01", "2025-12-01"),          # 跨过训练窗右端的部分重叠
    ("2024-01-01", "2024-06-01"),          # 左端越界（预热期）
])
def test_out_of_train_windows_are_refused(lo, hi):
    with pytest.raises(SystemExit, match="训练窗"):
        EA._assert_window_is_allowed(lo, hi, allow_holdout=False)


def test_explicit_flag_allows_the_holdout():
    EA._assert_window_is_allowed(*EA.HOLDOUT, allow_holdout=True)       # 不抛


def test_pool_returns_come_from_the_single_index_definition():
    """归因里的"池子"必须走 `overlay.pool_daily_returns` —— 不许自己再算一条池子日收益。

    注意断言要精确：`pct_change` 用**策略自身**的权益曲线上是合法的（那是策略收益），
    被禁止的是对 `ew_index` 就地求日收益 —— 那会造出第二份池子口径。
    """
    src = Path(EA.__file__).read_text(encoding="utf-8")
    assert '["ew_index"]' in src, "没有用 regime 的等权指数"
    assert '"ew_index"].pct_change' not in src, \
        "归因自己又算了一遍池子日收益 —— 池子只有一处定义（overlay.pool_daily_returns）"
    assert "overlay.pool_daily_returns" in src, "池子日收益应当取自唯一定义"


# ---------------------------------------------------------------- 分组语义（无需真库）

def _trade(sym, sig, ent, ext, r=0.5):
    return {"symbol": sym, "signal_date": pd.Timestamp(sig), "entry_date": pd.Timestamp(ent),
            "exit_date": pd.Timestamp(ext), "net_pnl": r * 100.0, "risk_amount": 100.0,
            "entry_price": 10.0, "size": 100.0, "sl_penetration": 0.01,
            "hold_days": 5, "exit_reason": "tp"}


def test_recency_groups_bound_the_trading_day_window():
    """`rec{k}` 分组按**信号日**与上一份财报的交易日数切 —— 边界要可判。

    夹具：同一只票在轴上的第 0/10/20 天各有一次财报；四笔信号分别距最近一份财报
    0 / 4 / 7 / 30 个交易日 ⇒ 依次落进 rec3、rec5、rec10、都不进。
    """
    axis = pd.bdate_range("2024-01-01", periods=40)
    by_symbol = {"AAA": [axis[0], axis[10], axis[20]]}
    # 声明的覆盖比合成轴更宽：前瞻窗（信号日 +26 自然日）会越过轴的末尾，
    # 而 A5 的覆盖校验**必须**拦下越界查询（那条护栏另由 earnings 的测试钉住）。
    coverage = (axis[0], axis[-1] + pd.Timedelta(days=45))
    trades = pd.DataFrame([
        _trade("AAA", axis[0], axis[1], axis[6]),      # gap 0
        _trade("AAA", axis[14], axis[15], axis[20]),   # gap 4
        _trade("AAA", axis[27], axis[28], axis[32]),   # gap 7
        _trade("AAA", axis[39], axis[39], axis[39]),   # gap 19（轴上第 20 天之后）
    ])
    g = EA._trade_groups(trades, by_symbol, coverage, forward_days=26, trading_days=axis)
    groups = g["groups"]
    assert groups["rec3_yes"] == [0]
    assert groups["rec5_yes"] == [0, 1]
    assert groups["rec10_yes"] == [0, 1, 2]
    assert groups["rec10_no"] == [3], "距上一份财报 19 个交易日 ⇒ 不算「刚过」"
    assert sorted(groups["rec3_yes"] + groups["rec3_no"]) == [0, 1, 2, 3], "分组必须覆盖全部成交"
    assert g["gap_unknown"] == 0, "这四笔的上一份财报都在日历里"


def test_unknown_gap_counts_as_not_recent():
    """查不到上一份财报 ⇒ 归入「事件未刚过」，**不许**当成「刚出过」。"""
    axis = pd.bdate_range("2024-01-01", periods=10)
    trades = pd.DataFrame([_trade("ZZZ", axis[5], axis[6], axis[8])])
    g = EA._trade_groups(trades, {"AAA": [axis[0]]},
                         (axis[0], axis[-1] + pd.Timedelta(days=45)),
                         forward_days=5, trading_days=axis)
    assert g["gap_unknown"] == 1
    assert g["groups"]["rec10_yes"] == [] and g["groups"]["rec10_no"] == [0]
