"""执行契约与费用常数 —— CLAUDE.md §2(a)/§2(b)/§0 Fail-Fast。

§2(b) 原文:commission / stamp-duty / min-fee constants are defined **EXACTLY ONCE**
(in core/). Every engine MUST import them; redefining fee constants inside an engine or
preset is a fail-fast violation.

§2(a) 原文:Every preset MUST explicitly declare `max_positions` and `position_fraction`
— a preset that leaves them implicit is a contract violation.

2026-10-15 审计实测(这两条都曾是 HIGH):
  * 仓库里没有 core/,双边 5.0 被独立写出 7 处,而且 preset 里的 `cost_bps_roundtrip`
    在 CLI 跑数路径上**从未被读取**(`--cost-bps` 的默认值说了算,只有 render_report
    读 preset 字段)—— 同一份 preset 在两处得到不同的成本假设,且没有任何机制会发现。
    反讽的是测试读的正是 preset 字段,于是 89 个用例全绿掩盖了它。
  * 两个 preset 都没有 position_fraction;`position_sizing` 这个声明字段无人读取,
    改它的值没有任何效果、改 risk_per_trade 的键才有效。
  * 同一条链路的两个入口给 broker 喂不同的键:render_report 的 7 键白名单对 broker
    实际读的 13 个键静默丢 5 个,实测同 preset 跑出差 45.5pp 的两条曲线。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from core import costs
from core.preset_contract import (
    ENGINE_KEYS,
    derived_position_sizing,
    resolve,
    validate,
)
from research import (broker, metrics, presets, render_report, run_baseline, screen,
                      search_12m)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- §2(b) 单一来源

# (键名, 不允许再出现的字面量)。这些常数只允许在 core/costs.py 里出现一次。
def _production_modules() -> list[Path]:
    """**全部**生产模块：`research/` + `src/` + `tools/` 的 .py（自动枚举）。

    为什么必须自动枚举（审计 G4）：此前两条护栏各自手写一个 **7 元组**，而仓库有 18 个
    research 模块、`src/` 与 `tools/` 完全在外 —— 于是"新加的模块"默认落在护栏之外，
    而**只覆盖一部分的护栏比没有护栏更坏**：它给出虚假的安全感（测试全绿，但保护的是
    另一批文件）。排除测试与 `__init__`。
    """
    roots = [REPO_ROOT / "research", REPO_ROOT / "src", REPO_ROOT / "tools"]
    out: list[Path] = []
    for r in roots:
        for f in sorted(r.glob("*.py")):
            if f.name.startswith("_") or "test" in f.name or f.name == "conftest.py":
                continue
            out.append(f)
    return out


# (键名, 不允许再出现的字面量)。这些常数只允许在 core/costs.py 里出现一次。
CONST_LITERALS = [
    ("cost_bps_roundtrip", "5.0"),
    ("max_volume_participation", "0.10"),
    ("atr_slip_scale", "1.0"),
    ("initial_cash", "100_000.0"),
]


def test_execution_constants_have_a_single_definition():
    """费用/流动性/滑点常数不得在 core/ 之外再定义一遍(§2(b))。

    两层判据:
      * 这一条按**绑定形状**匹配(键名 + 冒号/等号 + 该字面量)。不能用「同一行同时出现」
        —— 那会把 `1.0 - atr_slip_scale * …` 这种算术误报成重定义(实测踩过)。
      * 间接写法(`_FEE_BPS_ = 5.0` + 键名另起一行)由下面的 AST 用例负责。
      * CLI 默认值写死(`--cost-bps … default=5.0`)由下面那条专用用例负责 ——
        审计抓到的那处缺陷正是它。

    只扫生产模块:测试 fixture 里的字面量是为隔离而故意写的,不算「引擎或 preset 里
    重定义常数」。
    """
    for path in _production_modules():
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "=" not in line and '":' not in line:
                continue
            for key, lit in CONST_LITERALS:
                # 只看**绑定形状**:键名后面紧跟冒号/等号再跟那个字面量。
                # 不能用「同一行同时出现」——那会把 `1.0 - atr_slip_scale * …`
                # 这种算术误报成重定义(实测踩过)。间接变量写法由下面的 AST 用例负责。
                shapes = (f'"{key}": {lit}', f"{key}={lit}", f"{key} = {lit}",
                          f"{key}: float = {lit}", f'"{key}": {lit},')
                assert not any(sh in line for sh in shapes), (
                    f"{path.relative_to(REPO_ROOT)}:{i} 又定义了一遍 {key} = {lit} —— "
                    f"常数只能 import core.costs(§2(b) 要求恰好定义一次)"
                )


def test_cli_cost_override_has_no_numeric_default():
    """`--cost-bps` 的默认必须是 None(= 用 preset 口径),不能写死一个数。

    这正是审计实测的那处缺陷:CLI 读自己的 `default=5.0` 而不读 preset,于是改 preset
    的成本假设不会反映到任何 CLI 数字上,只有 HTML 报告会变。
    """
    hits = [ln for ln in Path(run_baseline.__file__).read_text(
        encoding="utf-8").splitlines() if "--cost-bps" in ln]
    assert hits, "找不到 --cost-bps 的定义"
    for ln in hits:
        assert "default=None" in ln, f"--cost-bps 又写死了默认值:{ln.strip()}"


@pytest.mark.parametrize("name", sorted(presets.PRESETS))
def test_preset_constants_are_the_core_ones(name):
    """preset 里声明的这几个值必须**就是** core 的常数(而不是各自抄的数字)。"""
    p = presets.get_preset(name)
    assert p["cost_bps_roundtrip"] == costs.COST_BPS_ROUNDTRIP
    assert p["atr_slip_scale"] == costs.ATR_SLIP_SCALE
    assert p["max_volume_participation"] == costs.MAX_VOLUME_PARTICIPATION


def test_core_costs_carries_the_declared_values():
    assert costs.COST_BPS_ROUNDTRIP == 5.0
    assert costs.MAX_VOLUME_PARTICIPATION == 0.10
    assert costs.ATR_SLIP_SCALE == 1.0
    assert costs.INITIAL_CASH == 100_000.0
    assert costs.half_rate(costs.COST_BPS_ROUNDTRIP) == pytest.approx(0.00025)


# ---------------------------------------------------------------- §2(a) 契约

@pytest.mark.parametrize("name", sorted(presets.PRESETS))
def test_every_preset_satisfies_the_execution_contract(name):
    validate(presets.get_preset(name))


@pytest.mark.parametrize("name", sorted(presets.PRESETS))
def test_every_preset_declares_position_fraction_and_max_positions(name):
    p = presets.get_preset(name)
    assert p["max_positions"] >= 1, "§2(a) 要求显式声明 max_positions"
    assert 0 < p["position_fraction"] <= 1.0, "§2(a) 要求显式声明 position_fraction"


@pytest.mark.parametrize("key", ["position_fraction", "max_positions", "atr_slip_scale",
                                 "cost_bps_roundtrip", "intraday_tiebreak", "max_hold"])
def test_missing_required_key_is_a_contract_violation(key):
    p = presets.get_preset("us_uptrend_pullback_v1")
    del p[key]
    with pytest.raises(ValueError, match=key):
        validate(p)


def test_unknown_key_is_rejected():
    """拼错的执行键必须报错,不能被当成元数据静默忽略。

    这正是 render_report 白名单那类缺陷的根因:没有任何一层知道
    「哪些键引擎会读」,于是丢键、错拼都无声通过。
    """
    p = presets.get_preset("us_uptrend_pullback_v1")
    p["atr_slip_scal"] = 1.0                     # 少一个 e
    with pytest.raises(ValueError, match="atr_slip_scal"):
        validate(p)


def test_declared_position_sizing_must_agree_with_the_keys():
    """`position_sizing` 是派生量,声明值必须与实际生效的键一致。

    审计前的矛盾:preset 写 position_sizing="risk_budget" 而设计文档写
    position_fraction=1.0(all-in),两份说法都无人校验。
    """
    p = presets.get_preset("us_uptrend_pullback_v1")
    assert derived_position_sizing(p) == "risk_budget"
    p["position_sizing"] = "all_in"
    with pytest.raises(ValueError, match="position_sizing"):
        validate(p)

    q = {k: v for k, v in p.items() if k != "risk_per_trade"}
    q["position_sizing"] = "all_in"
    validate(q)                                   # 去掉风险预算 ⇒ 派生为 all_in,一致
    assert derived_position_sizing(q) == "all_in"


def test_resolve_returns_exactly_the_engine_keys():
    """resolve() 的输出就是引擎读的全部 —— 声明了却没人读的键在结构上不可能存在。"""
    spec = resolve(presets.get_preset("us_uptrend_pullback_v1"))
    assert set(spec) == set(ENGINE_KEYS)


def test_resolve_fills_defaults_from_core():
    spec = resolve(presets.get_preset("us_uptrend_pullback_v1"))
    assert spec["max_volume_participation"] == costs.MAX_VOLUME_PARTICIPATION
    assert spec["initial_cash"] == costs.INITIAL_CASH
    assert spec["atr_slip_scale"] == costs.ATR_SLIP_SCALE


def test_price_adjust_is_not_a_preset_knob():
    """指标价轴是引擎约定(§3(b)),不是 preset 键。

    第三层复核实测:把它做成 preset 键时,5 个 panel 载入点里只有 1 个读它 ——
    「同一 preset 两条曲线」又换了个载体回来。删掉这个键,带着它即被契约拒绝。
    """
    from core.preset_contract import INDICATOR_PRICE_ADJUST
    assert INDICATOR_PRICE_ADJUST == "total"
    p = presets.get_preset("us_uptrend_pullback_v1")
    p["price_adjust"] = "none"
    with pytest.raises(ValueError, match="price_adjust"):
        validate(p)


# 名字里出现这些片段 ⇒ 该名字承载的就是这个常数
_NAME_FRAGMENTS = {
    "cost_bps_roundtrip": ("cost", "bps", "fee", "commission"),
    "max_volume_participation": ("participation",),
    "atr_slip_scale": ("slip",),
    "initial_cash": ("initial_cash", "cash"),
}
_LITERALS = {k: float(v) for k, v in CONST_LITERALS}


def _literal_number(node):
    return (float(node.value) if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
            else None)


def _binds_a_constant(name: str, value_node) -> str | None:
    """这个名字 + 右值,是不是又把某个常数写了一遍?是则返回那个键名。"""
    num = _literal_number(value_node)
    if num is None:
        return None
    for key, frags in _NAME_FRAGMENTS.items():
        if any(f in name.lower() for f in frags) and num == _LITERALS[key]:
            return f"{key}(={num})"
    return None


def test_cost_constants_cannot_be_rebound_through_an_intermediate_name():
    """把常数挪到相邻一行 / 中间变量,也必须被抓住。

    第三层复核实测:行级判据能被两行写法绕过 ——
        _FEE_BPS_: float = 5.0
        "cost_bps_roundtrip": _FEE_BPS_,
    两行各自只满足一半条件,160 个用例全绿。这里按 AST 走一遍:任何被赋值的名字、
    dict 键、关键字参数,只要名字里含有该常数的语义片段,其右值就不许是那个字面量。
    """
    for path in _production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            pairs: list[tuple[str, object]] = []
            if isinstance(node, ast.Assign):
                pairs = [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                pairs = [(node.target.id, node.value)]
            elif isinstance(node, ast.Dict):
                pairs = [(k.value, v) for k, v in zip(node.keys, node.values)
                         if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            elif isinstance(node, ast.Call):
                pairs = [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
            for name, value in pairs:
                hit = _binds_a_constant(name, value)
                assert hit is None, (
                    f"{path.relative_to(REPO_ROOT)}:{getattr(node, 'lineno', '?')} "
                    f"用中间名字 {name!r} "
                    f"又定义了一遍常数 {hit} —— 常数只能 import core.costs(§2(b))"
                )


# ---------------------------------------------------------------- 护栏的覆盖范围（G4）

def test_the_guard_covers_every_production_module():
    """护栏必须覆盖**全部**生产模块，而不是一个手写的子集（审计 G4）。

    此前两条护栏各写一个 7 元组，而 research/ 有 18 个模块、`src/` 与 `tools/` 完全在外。
    只覆盖一部分的护栏**比没有护栏更坏**：它给出虚假的安全感。
    """
    mods = _production_modules()
    names = {p.name for p in mods}
    assert len(names) >= 18, f"只发现 {len(names)} 个生产模块：{sorted(names)}"
    # 此前明确落在覆盖之外的几个，现在必须在册
    for must in ("families.py", "family_search.py", "wfv_search.py", "factor_probe.py",
                 "holdout.py", "universe.py", "factor_model.py", "download_ohlcv.py",
                 "snapshot_options.py", "parse_family_logs.py"):
        assert must in names, f"{must} 不在护栏的覆盖范围内"
    src = Path(__file__).read_text(encoding="utf-8")
    assert "_production_modules()" in src
    # ⚠️ 判据本身要拼出来：直接写字面量的话，这一行**会把自己扫到**（自指）。
    # 这个坑在本仓库已经踩过三次（护栏扫到自己），一律用拼接规避。
    hand_written = "for mod in " + "(broker"
    assert hand_written not in src, "护栏又被写回手写模块元组了"


def test_the_guard_would_actually_catch_a_violation():
    """护栏要有牙：直接喂一个"用中间名字重定义常数"的结点，必须被认出来。"""
    hit = _binds_a_constant("_FEE_BPS_", ast.parse("5.0").body[0].value)
    assert hit is not None and "cost_bps_roundtrip" in hit
    assert _binds_a_constant("_SLIP_", ast.parse("1.0").body[0].value) is not None
    assert _binds_a_constant("_CASH_", ast.parse("100_000.0").body[0].value) is not None
    # 反向：与常数无关的名字不该误报
    assert _binds_a_constant("max_positions", ast.parse("5.0").body[0].value) is None
    assert _binds_a_constant("_FEE_BPS_", ast.parse("'5.0'").body[0].value) is None
