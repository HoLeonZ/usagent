"""契约允许、但下游没接上的两处（G5/G6）。

G5：契约**允许**两套退出几何（ATR 归一化 / 固定百分比，`validate` 强制恰好一套），
    而 `render_report` 的"运气对照"直接读 `preset["atr_sl_mult"]` ⇒ 一个合法的
    pct preset 会让它 KeyError，"运气对照"跑不起来（或更糟：跑一个几何不同的假对照）。
G6：`_BAR_INDEX_CACHE` 的缓存项是 `(弱引用, key=(), 值)`，而那个 key **恒为空元组、
    从不参与比较** —— 它看起来像"按 key 校验"，实际只按对象身份命中。留着它比删掉更坏：
    下一个读者会以为缓存有（按内容失效之类的）逻辑，并在别处依赖一个不存在的保证。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.preset_contract import exit_scheme, validate  # noqa: E402
from research import broker, presets, search_12m as S  # noqa: E402
from research.tests.test_broker import BASE, flat_days, mk_panel, mk_signal  # noqa: E402


def _atr_preset() -> dict:
    """一份 ATR 归一化退出的 preset。

    ⚠️ 注意 `BASE`（test_broker 的基准 preset）本身就是**百分比**退出
    （`sl_pct=0.021 / tp_pct=0.2628`）—— 我第一版测试把它当成 ATR 那套，
    被自己的断言当场纠正。
    """
    return {**BASE, "sl_pct": None, "tp_pct": None,
            "atr_sl_mult": 1.5, "atr_tp_mult": 4.0}


# ---------------------------------------------------------------- G5

def test_exit_scheme_names_the_scheme_and_refuses_neither():
    assert exit_scheme(BASE) == "pct", "BASE 用的是固定百分比（sl_pct/tp_pct）"
    assert exit_scheme(_atr_preset()) == "atr"
    bad = {k: v for k, v in BASE.items()
           if k not in ("atr_sl_mult", "atr_tp_mult", "sl_pct", "tp_pct")}
    with pytest.raises(ValueError):
        exit_scheme(bad)          # 两套都不给 ⇒ 契约校验先拒（失败得早是对的）
    # ⚠️ BASE 自带 sl_pct/tp_pct ⇒ 要构造"两套同时给"必须**加上** atr_* 那套
    both = {**BASE, "atr_sl_mult": 1.0, "atr_tp_mult": 3.0}
    with pytest.raises(ValueError):
        exit_scheme(both)         # 两套同时给 ⇒ 也拒


def test_a_pct_preset_is_legal_and_the_engine_takes_it():
    p = BASE
    validate(p)                                    # 契约：合法
    r = broker.simulate(p, mk_signal("2024-01-01"), mk_panel(flat_days(25)),
                        start="2024-01-01", end="2024-02-10", cost_bps_roundtrip=0.0)
    assert "trades" in r


def test_make_preset_supports_both_schemes():
    a = S.make_preset({"sl_mult": 2.0, "tp_mult": 6.0, "max_hold": 10})
    assert a["atr_sl_mult"] == 2.0 and "sl_pct" not in a
    b = S.make_preset({"sl_pct": 0.05, "tp_pct": 0.20, "max_hold": 10})
    assert b["sl_pct"] == 0.05 and "atr_sl_mult" not in b
    validate({**BASE, **b})                        # 产出的 preset 必须合法


def test_make_preset_refuses_two_schemes_at_once():
    with pytest.raises(S.SearchError):
        S.make_preset({"sl_mult": 2.0, "sl_pct": 0.05})


def test_render_report_builds_the_control_cfg_for_either_scheme():
    """运气对照的 cfg 必须按 preset **实际用的那套**建 —— 否则合法 preset 会 KeyError。

    这里把 render_report 的构建规则**照抄一遍**（它内联在函数里），然后喂给
    `make_preset` + 引擎：只要这两步不抛，对照就能成立。
    """
    for p in (_atr_preset(), BASE):
        scheme = exit_scheme(p)
        cfg = {"max_hold": p["max_hold"], "max_positions": p["max_positions"],
               # BASE 没声明 risk_per_trade（它是 all-in）；用 make_preset 的默认
               "risk_per_trade": p.get("risk_per_trade", 0.01)}
        if scheme == "atr":
            cfg.update({"sl_mult": p["atr_sl_mult"], "tp_mult": p["atr_tp_mult"]})
        else:
            cfg.update({"sl_pct": p["sl_pct"], "tp_pct": p["tp_pct"]})
        made = S.make_preset(cfg)
        # 合并时要**清掉 BASE 里另一套的键**，否则两套同时在场（契约拒绝），
        # 而那不是被测代码的问题 —— 是我这个拼装写错了（实测踩过）。
        merged = {**BASE, **made}
        for k in ("atr_sl_mult", "atr_tp_mult", "sl_pct", "tp_pct"):
            if k in made:
                merged[k] = made[k]
            elif k in ("sl_pct", "tp_pct") or scheme == "pct":
                merged[k] = None
        if scheme == "pct":
            merged["atr_sl_mult"] = merged["atr_tp_mult"] = None
        else:
            merged["sl_pct"] = merged["tp_pct"] = None
        assert exit_scheme(merged) == scheme, "对照的几何与策略不一致"


def test_render_report_no_longer_hardcodes_the_atr_keys():
    """直接读 `preset["atr_sl_mult"]` 只允许出现在**按几何分支**的那一支里。

    判据：先去掉注释行（注释里也会提到这个键名），再看剩下的代码里有没有
    **不带条件**的读取 —— 具体做法是要求 `exit_scheme(` 出现，且
    `atr_sl_mult` 的每一次直接读取都落在 `if _scheme == "atr"` 之后。
    """
    lines = [ln for ln in (ROOT / "research" / "render_report.py").read_text(
        encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    code = "\n".join(lines)
    assert "exit_scheme(" in code, "render_report 没有按几何分支"
    guard = code.index('if _scheme == "atr"')
    for m in ("preset[\"atr_sl_mult\"]", "preset[\"atr_tp_mult\"]"):
        assert m not in code[:guard], f"{m} 出现在分支之前 —— 又变成唯一口径了"


# ---------------------------------------------------------------- G6

def test_bar_index_cache_has_no_dead_key_slot():
    """缓存项不得再带那个恒为空、从不比较的 key 槽。"""
    for item in broker._BAR_INDEX_CACHE:
        assert len(item) == 2, f"缓存项还是多元组（含死 key 槽）：{len(item)}"
    src = (ROOT / "research" / "broker.py").read_text(encoding="utf-8")
    assert "for ref, key, val in _BAR_INDEX_CACHE" not in src
    assert "tuple[weakref.ref, tuple, tuple]" not in src, "三元组注解还在"


def test_bar_index_cache_hits_by_identity_only():
    a = mk_panel(flat_days(25))
    b = mk_panel(flat_days(25))
    ra1 = broker._bar_index_cached(a)
    ra2 = broker._bar_index_cached(a)
    rb = broker._bar_index_cached(b)
    assert ra1 is ra2, "同一面板必须命中缓存"
    assert rb is not ra1, "不同面板不得互相命中（身份比较）"
