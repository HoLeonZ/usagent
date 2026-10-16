"""执行/风险口径必须**自报**（审计 E5 及其同类）。

缺陷本体：报告只报数字，不报"这些数字是在什么成本与风险假设下产生的"。读者要么去读
preset 源码，要么默认"没有价差、没有滑点、组合层没有上限"—— 而后一种默认恰好可能是
最乐观的那一种。E5 的抱怨是**静默**（`atr_slip_scale=0` 可以合法地把两侧滑点关掉，
报告里看不出来），不是能力。

修法：`core/preset_contract.describe_engine_spec()` 把生效口径翻成 [(标签, 值)]，
**run_baseline 的 CLI 与 HTML 报告用同一个函数**（于是不可能再各说一套）。

钉住：
  1. 标签表**覆盖全部 ENGINE_KEYS** —— 将来加一个键却忘了登记，测试就红
     （否则又会造出一个"报告里看不见的旋钮"）；
  2. "关闭/未设"这类**否定事实**必须显示出来，不能因为值为 0/None 就省略；
  3. 两个入口共用同一个函数。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.preset_contract import (ENGINE_KEYS, SPEC_LABELS, describe_engine_spec,  # noqa: E402
                                  spec_header)
from research import presets  # noqa: E402


def _spec_of(**over) -> dict:
    p = presets.get_preset("us_uptrend_pullback_12m_target")
    return {**p, **over}


def _find(items, label: str) -> str:
    hit = [v for lab, v in items if lab == label]
    assert hit, f"口径头里没有 {label!r} —— 它会是报告里看不见的那一项"
    return hit[0]


# ---------------------------------------------------------------- 覆盖性（最重要）

def test_labels_cover_every_engine_key():
    missing = [k for k in ENGINE_KEYS if k not in SPEC_LABELS]
    assert not missing, (
        f"这些引擎键没有在 SPEC_LABELS 里登记：{missing} —— "
        f"新键会让报告里多出一个看不见的旋钮，正是 E5 那类缺陷")


def test_no_dead_labels():
    extra = [k for k in SPEC_LABELS if k not in ENGINE_KEYS]
    assert not extra, f"SPEC_LABELS 里有不属于引擎的键：{extra}"


def test_every_label_renders_for_every_preset():
    for name in presets.PRESETS:
        items = describe_engine_spec(presets.get_preset(name))
        assert len(items) >= len(ENGINE_KEYS) - 3, name     # 互斥的那两套会各少一项
        for lab, val in items:
            assert lab.strip() and isinstance(val, str) and val.strip(), (name, lab)


# ---------------------------------------------------------------- 否定事实必须可见

def test_disabled_slippage_is_shown_not_silent():
    v = _find(describe_engine_spec(_spec_of(atr_slip_scale=0.0)), "冲击滑点")
    assert "关闭" in v, "关掉滑点这件事必须在报告里看得见"


def test_absent_risk_limits_are_shown_as_absent():
    items = describe_engine_spec(_spec_of(max_weight=None, max_heat=None,
                                          max_drawdown_halt=None))
    assert "未设" in _find(items, "单名上限")
    assert "未设" in _find(items, "组合热风险上限")
    assert "未设" in _find(items, "回撤停机")


def test_missing_spread_model_is_called_out():
    v = _find(describe_engine_spec(_spec_of(spread_model="none")), "价差")
    assert "未建模" in v


def test_no_end_liquidation_is_called_out():
    v = _find(describe_engine_spec(_spec_of(liquidate_at_end=False)), "窗口末尾")
    assert "不清算" in v and "不同源" in v


def test_zero_overshoot_is_shown_as_zero_not_hidden():
    v = _find(describe_engine_spec(_spec_of(sl_overshoot=0.0)), "止损穿价")
    assert "θ=0" in v and "穿价仅记录" in v


# ---------------------------------------------------------------- 互斥的退出宽度

def test_only_the_active_exit_scheme_is_listed():
    atr_items = describe_engine_spec(_spec_of(atr_sl_mult=2.0, atr_tp_mult=6.0,
                                              sl_pct=None, tp_pct=None))
    pct_items = describe_engine_spec(_spec_of(atr_sl_mult=None, atr_tp_mult=None,
                                              sl_pct=0.021, tp_pct=0.26))
    assert "2×ATR" in _find(atr_items, "止损宽度")
    assert "2.10%" in _find(pct_items, "止损宽度")
    # 未启用的那一套不应出现两次
    assert sum(1 for lab, _ in pct_items if lab == "止损宽度") == 1


def test_header_is_a_single_line():
    h = spec_header(_spec_of())
    assert "\n" not in h and len(h) > 80
    assert h.count("|") >= 10


# ---------------------------------------------------------------- 单一口径来源

def test_both_entry_points_use_the_same_function():
    """CLI 与 HTML 报告必须共用 describe_engine_spec / spec_header ——
    否则又是一次"两个入口两套口径"。"""
    for name, needle in (("run_baseline.py", "spec_header"),
                         ("render_report.py", "describe_engine_spec")):
        src = (ROOT / "research" / name).read_text()
        assert needle in src, f"{name} 没有用共用的口径函数（{needle}）"
        # 导入可能是多行（PEP8 换行），所以用 DOTALL
        assert re.search(r"from core\.preset_contract import.*" + needle, src, re.S), name


def test_unknown_engine_key_still_fails_fast():
    """契约的既有保证不得因为加了口径头而放松。"""
    with pytest.raises(ValueError, match="未知键"):
        describe_engine_spec({**_spec_of(), "atr_slip_scal": 1.0})
