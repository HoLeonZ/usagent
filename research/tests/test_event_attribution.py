"""归因脚本的**窗口纪律** —— 护栏要能在没有真库的机器上跑。

这条测试守的是协议 §5 的一句话：归因阶段不许看 holdout。它值得写成测试，是因为
它的失效方式是**无声的**：某轮 cron（或某个人）把默认窗口顺手改成 holdout，
"一次性 holdout" 就悄悄没了 —— 而那正是本仓库反复吃亏的那类缺陷。
"""
from __future__ import annotations

import sys
from pathlib import Path

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
    """归因里的"池子"必须用 regime 的等权指数 —— 不许自己再算一条日收益序列。

    同窗池收益直接用 `ew_index` 的两个点位相除（比值），所以这个文件里不该出现
    `pct_change`：出现就说明有人又造了一份池子口径。
    """
    src = Path(EA.__file__).read_text(encoding="utf-8")
    assert '["ew_index"]' in src, "没有用 regime 的等权指数"
    assert "pct_change" not in src, "归因自己又算了一遍池子收益 —— 池子只有一处定义"
