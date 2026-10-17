"""行为锁的**覆盖判据** —— 每个 preset 要么在锁里，要么在显式豁免表里（2026-10-17 注册表重构）。

这条测试要防的失效方式与本仓库反复吃亏的那类同源：`CASES` 此前是**手写清单**，
新增 preset 时漏掉它**什么都不会发生**（没有报错、没有红），新 preset 只是从此不在
任何护栏里。现在用例从注册表派生，于是"在不在锁里"变成一个可以被断言的事实。
"""
from __future__ import annotations

import pytest

from research import golden_baseline as G
from research import presets


def test_every_preset_is_locked_or_explicitly_exempt():
    locked = {c["preset"] for c in G.CASES}
    for name in presets.PRESETS:
        assert name in locked or name in G.LOCK_EXEMPT, \
            f"{name} 既不在行为锁里也没有豁免理由 —— 它默认不被任何护栏覆盖"


def test_every_exemption_has_a_reason():
    for name, why in G.LOCK_EXEMPT.items():
        assert why and why.strip(), f"{name} 的豁免没写理由"


def test_exempt_table_cannot_grow_silently():
    """豁免表一旦能无声扩大，这条护栏就等于没有 —— 所以把它钉成一份显式清单。"""
    assert set(G.LOCK_EXEMPT) == set(), \
        "豁免表变了：要么把新 preset 加进锁里，要么在这里显式写明为什么它不能锁"


def test_cases_carry_the_declared_windows():
    """用例的窗口必须**就是**定义体声明的那个 —— 不是另抄一份。"""
    by = {c["preset"]: c for c in G.CASES}
    for name, rec in presets.REGISTRY.items():
        if name in by:
            assert (by[name]["start"], by[name]["end"]) == rec["window"], \
                f"{name}: 锁的窗口与定义体声明的不一致"
    assert sum(1 for c in G.CASES if c.get("is_cli_default")) == 1, \
        "必须正好有一条用例标注为 CLI 默认窗口（既有测试按这个标注校验窗口推导）"


def test_lock_exemption_and_the_registry_agree():
    """豁免表里的名字必须真实存在 —— 打错一个字就等于豁免了一个不存在的 preset。"""
    for name in G.LOCK_EXEMPT:
        assert name in presets.PRESETS, f"豁免表里的 {name} 不是一条 preset"


def test_no_preset_is_locked_twice():
    names = [c["preset"] for c in G.CASES]
    assert len(names) == len(set(names)), f"同一条 preset 被锁了两次：{names}"


@pytest.mark.parametrize("name,rec", list(presets.REGISTRY.items()),
                         ids=list(presets.REGISTRY))
def test_offline_lock_membership_is_a_conscious_decision(name, rec):
    """每条 preset 都必须就**离线锁**表态：入锁（给窗口）或写明为什么不入（给理由）。

    这条与装载器的 fail-fast 是同一件事的两面：装载器让"没说"这件事在 import 期就
    报错，这里让"说了什么"在测试期可见。
    """
    assert bool(rec["fixture_window"]) ^ bool(rec["fixture_exempt"]), \
        f"{name}: 必须**恰好**给一个 —— FIXTURE_WINDOW（入离线锁）或 FIXTURE_EXEMPT_REASON"
