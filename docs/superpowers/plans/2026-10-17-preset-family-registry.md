# preset / 族注册表重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让「新增一条 preset / 一个族」= 新增一个文件，并把"每个 preset 必须在行为锁里或显式豁免"变成测试强制的纪律。

**Architecture:** 两条注册表。定义体搬进 `research/preset_defs/<name>.py` 与 `research/family_defs/<id>.py`，每个文件自报 `NAME`/`ID` 与全部配置；`presets.py` 与 `families.py` 退化为**注册表 + 兼容门面**（`pkgutil` 扫描 + fail-fast 装载校验），继续导出全部既有名字，因此 20 多处调用点一行不改。

**Tech Stack:** Python 3.14、pandas、pytest；SQLite（数据层，本重构不涉及）。

## Global Constraints

* 设计依据：`docs/superpowers/specs/2026-10-17-preset-family-registry-design.md`。
* **行为必须逐字节不变**：重构前后 7 个 `trades_sha256` 必须相等。若不等 ⇒ 查根因，**不许直接重生成基线**（仓库已吃过一次：两条入口读两个面板，只改数字会把分歧固化成基线）。
* 只允许改**引用了被移动符号名**的测试；不许为了让重构通过而放宽任何断言。
* 常数/口径仍只在 `core/` 定义一次（§2(b)）；本重构不新增任何引擎键。
* 装载违规一律 **fail-fast**，不静默跳过（§0）。

### 重构前的基线哈希（Task 1 记下，Task 5 复验必须逐字节相等）

| 用例 | trades_sha256 |
|---|---|
| `us_uptrend_pullback_v1` | `c6ef92039a072ae22a3e6ef8afa67a01d6c235dd9965e154c0a52b5a12c0659b` |
| `us_uptrend_pullback_12m_target` | `6ab77c97ec930f314e5ed0782a2cc4110bcf45e681c24a26dadf32f432c543a3` |
| `us_uptrend_pullback_wfv` | `cdca9be636ef6b02db4b814003a7b4b1e70e01780cf3b2d093bb2f5d55ac8c70` |
| `us_reversal_dip_v1` | `9c08fb334c1357dd7456ae31525a0f2ae2448ccbfe28d14dea55449f4e76fb9a` |
| `fixture_uptrend_pullback_v1` | `9e5e1921201ecebe4db0ff49b81cf7578f83d911ff39e6d1d8ef83d9b26d3fbd` |
| `fixture_12m_target` | `4ba62d2da62403b9cd77bfefd25d3709ba073abaa3270e352886544803a0debb` |
| `fixture_reversal_dip_v1` | `9c85e16375e5b720144a334fef4a40355f79f614f06c5082058e337856a18ae9` |

### 现状坐标（迁移时按此切）

* `research/presets.py`：`RISK_LIMITS`(:55) `DEFAULT_RISK_PER_TRADE`(:98)、
  `PRESETS[...]` 四条(:100 v1、:287 12m、:409 wfv、:551 reversal)、
  `WFV_SEARCH_NOTE`(:349)、`FAMILY_SEARCH_NOTE`(:463)、`get_preset`(:606)。
* `research/families.py`：15 个 `_mask_*`(:146..456)、`MASK`(:457)、`SCORE_COL`(:486)、
  `SCORE_NEGATED`(:492)、`NEEDS_REGIME`(:496)、`FIXED`(:500)、`REFERENCE_FAMILIES`(:518)、
  `GRIDS`(:536)、`CONTROL_ARM`(:636)、`GRID_CAP`(:639)、`grid_size`(:642)、`select`(:649)。
* 现有 real 锁窗口：v1 `2024-04-10..2026-10-17`(is_cli_default)、12m `2025-10-14..2026-10-14`、
  wfv `2024-04-09..2026-10-14`、reversal `2024-04-09..2026-10-14`。

---

## Task 1: 基线对照（动代码之前）

**Files:** 无改动。

- [ ] **Step 1: 确认两道锁现在是绿的**

```bash
cd /Users/zhl/code/usagent
python3 -m pytest research/tests/test_golden_trades.py research/tests/test_golden_fixture_lock.py -q
```
Expected: `19 passed`（14 real + 5 fixture/uniqueness）。若不为绿 ⇒ 停，先搞清为什么不绿。

- [ ] **Step 2: 抄下哈希**

```bash
for f in research/tests/baselines/*.golden; do
  printf "%-42s %s\n" "$(basename $f)" "$(grep -h trades_sha256 $f | cut -d= -f2)"
done
```
Expected: 与文件头那张表**逐字节相同**。不同 ⇒ 停。

---

## Task 2: preset 注册表骨架 + 装载校验（TDD）

**Files:**
- Create: `research/preset_defs/__init__.py`（空文件，仅包标记）
- Create: `research/preset_registry.py`
- Test: `research/tests/test_preset_registry.py`

**Interfaces:**
- Produces: `preset_registry.load_one(path: Path) -> dict`（返回 `{name, preset, verdict, window, is_cli_default, fixture_window}`）、
  `preset_registry.load_dir(models: dict, dirpath: Path) -> None`（把每个 def 注册进 `models`，重名抛错）。

- [ ] **Step 1: 写失败的测试**

```python
# research/tests/test_preset_registry.py
from pathlib import Path
import pytest
from research import preset_registry as R

GOOD = '''
NAME = "x1"
PRESET = {"max_positions": 1, "position_fraction": 1.0, "atr_slip_scale": 1.0,
          "intraday_tiebreak": "sl_first", "max_hold": 10, "cost_bps_roundtrip": 5.0,
          "sl_pct": 0.02, "tp_pct": 0.20}
VERDICT = "无可证实的正 alpha"
LOCK_WINDOW = ("2024-01-01", "2025-01-01")
LOCK_IS_CLI_DEFAULT = False
'''

def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / f"{name}.py"
    p.write_text(body, encoding="utf-8")
    return p

def test_load_one_returns_the_declared_fields(tmp_path):
    got = R.load_one(_write(tmp_path, "x1", GOOD))
    assert got["name"] == "x1" and got["verdict"].startswith("无可证实的")
    assert got["window"] == ("2024-01-01", "2025-01-01")
    assert got["is_cli_default"] is False and got["preset"]["max_positions"] == 1

def test_file_name_must_match_name(tmp_path):
    with pytest.raises(ValueError, match="文件名"):
        R.load_one(_write(tmp_path, "other", GOOD))

def test_missing_verdict_fails_fast(tmp_path):
    body = GOOD.replace('VERDICT = "无可证实的正 alpha"', "")
    with pytest.raises(ValueError, match="VERDICT"):
        R.load_one(_write(tmp_path, "x1", body))

def test_missing_lock_window_fails_fast(tmp_path):
    body = GOOD.replace('LOCK_WINDOW = ("2024-01-01", "2025-01-01")', "")
    with pytest.raises(ValueError, match="LOCK_WINDOW"):
        R.load_one(_write(tmp_path, "x1", body))

def test_verdict_must_not_be_duplicated_inside_preset(tmp_path):
    body = GOOD.replace('PRESET = {', 'PRESET = {"verdict": "x",')
    with pytest.raises(ValueError, match="verdict"):
        R.load_one(_write(tmp_path, "x1", body))

def test_duplicate_names_across_files_fail_fast(tmp_path):
    _write(tmp_path, "x1", GOOD)
    _write(tmp_path, "x2", GOOD.replace('NAME = "x1"', 'NAME = "x2"').replace("x1", "x2"))
    models: dict = {}
    R.load_dir(models, tmp_path)
    assert set(models) == {"x1", "x2"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest research/tests/test_preset_registry.py -q`
Expected: `ModuleNotFoundError: No module named 'research.preset_registry'`

- [ ] **Step 3: 实现**

```python
# research/preset_registry.py
"""preset 定义体的装载器 —— 「新增一条 preset = 新增一个文件」的机制（2026-10-17 重构）。

设计见 docs/superpowers/specs/2026-10-17-preset-family-registry-design.md。
违规一律 fail-fast：文件名与 NAME 不符、缺 VERDICT / LOCK_WINDOW、名字重复、
以及在 PRESET 里再写一份 verdict（一份事实只能有一处）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REQUIRED_ATTRS = ("NAME", "PRESET", "VERDICT", "LOCK_WINDOW")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_preset_def_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法装载 preset 定义 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_one(path: Path) -> dict:
    mod = _load_module(path)
    missing = [a for a in REQUIRED_ATTRS if not hasattr(mod, a)]
    if missing:
        raise ValueError(f"{path.name} 缺 {missing} —— §5 要求负面证据与参数同时记录")
    if mod.NAME != path.stem:
        raise ValueError(f"{path.name}: NAME={mod.NAME!r} 与文件名不符 —— "
                         f"名字与文件对不上这件事必须立刻发现")
    preset = dict(mod.PRESET)
    if "verdict" in preset:
        raise ValueError(f"{path.name}: PRESET 里不要再写 verdict —— 用模块级 VERDICT "
                         f"（一份事实只能有一处）")
    win = tuple(mod.LOCK_WINDOW)
    if len(win) != 2:
        raise ValueError(f"{path.name}: LOCK_WINDOW 必须是 (start, end)")
    return {"name": mod.NAME, "preset": preset, "verdict": mod.VERDICT,
            "window": win,
            "is_cli_default": bool(getattr(mod, "LOCK_IS_CLI_DEFAULT", False)),
            "fixture_window": tuple(getattr(mod, "FIXTURE_WINDOW", ())) or None,
            "fixture_exempt": getattr(mod, "FIXTURE_EXEMPT_REASON", None)}


def load_dir(models: dict, dirpath: Path) -> None:
    for path in sorted(Path(dirpath).glob("*.py")):
        if path.name.startswith("_"):
            continue
        rec = load_one(path)
        if rec["name"] in models:
            raise ValueError(f"preset 名字重复：{rec['name']}")
        models[rec["name"]] = rec
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest research/tests/test_preset_registry.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add research/preset_registry.py research/preset_defs/__init__.py research/tests/test_preset_registry.py
git commit -m "feat(presets): preset 定义体的装载器（新增 preset = 新增文件，第 1 步）"
```

---

## Task 3: 迁移 4 条 preset 到文件 + 门面

**Files:**
- Create: `research/preset_defs/{us_uptrend_pullback_v1,us_uptrend_pullback_12m_target,us_uptrend_pullback_wfv,us_reversal_dip_v1}.py`
- Modify: `research/presets.py`（删掉被搬走的块，改成扫描 + 门面）

**Interfaces:**
- Consumes: `preset_registry.load_dir`（Task 2）。
- Produces: `presets.PRESETS`（同今天：`name -> dict`，且 `preset["verdict"]` 由装载器注入）、`presets.CASES`（Task 5 用）。

- [ ] **Step 1: 逐条搬文件**

对每条 preset：新建 `research/preset_defs/<name>.py`，内容 = 文件头（`NAME`/`LOCK_WINDOW`/`LOCK_IS_CLI_DEFAULT`/`FIXTURE_WINDOW` 或 `FIXTURE_EXEMPT_REASON`）+ **把 `research/presets.py` 里那个 dict 字面量原样粘贴**住 `PRESET = {...}` + 把原 dict 里的 `"verdict"` 值原样搬到 `VERDICT = """…"""` 并**从 PRESET 里删掉那一行**。

窗口取值（来自现状坐标，逐字照抄）：

| preset | LOCK_WINDOW | LOCK_IS_CLI_DEFAULT | FIXTURE_WINDOW |
|---|---|---|---|
| `us_uptrend_pullback_v1` | `("2024-04-10", "2026-10-17")` | `True` | `("2024-01-01", "2025-01-01")` |
| `us_uptrend_pullback_12m_target` | `("2025-10-14", "2026-10-14")` | `False` | `("2024-01-01", "2025-01-01")` |
| `us_uptrend_pullback_wfv` | `("2024-04-09", "2026-10-14")` | `False` | 无 ⇒ `FIXTURE_EXEMPT_REASON = "合成市场上该配置成交为 0，锁会变成空断言"` |
| `us_reversal_dip_v1` | `("2024-04-09", "2026-10-14")` | `False` | `("2024-01-01", "2025-01-01")` |

- [ ] **Step 2: 改 `presets.py` 为扫描 + 门面**

在 `presets.py` 顶部加：

```python
from pathlib import Path as _Path

from research.preset_registry import load_dir as _load_dir

_DEFS_DIR = _Path(__file__).resolve().parent / "preset_defs"
_REGISTRY: dict[str, dict] = {}
_load_dir(_REGISTRY, _DEFS_DIR)          # fail-fast：违规在 import 时就抛

PRESETS: dict[str, dict] = {}
for _name, _rec in _REGISTRY.items():
    PRESETS[_name] = {**_rec["preset"], "verdict": _rec["verdict"]}
```

并删掉被搬走的四个 dict 字面量。**保留**：`RISK_LIMITS`、`DEFAULT_RISK_PER_TRADE`、
`V21_*`、`WFV_SEARCH_NOTE`、`FAMILY_SEARCH_NOTE`、`get_preset`（原样不动）。
`get_preset` 里的 `VALIDATED`/`n_comparisons` 检查照旧。

- [ ] **Step 3: 跑全量测试**

```bash
python3 -m pytest research/tests src/tests -q
```
Expected: 全绿（若 `test_presets.py` 的 `p["verdict"]` 断言失败 ⇒ 装载器没注入 verdict，回 Step 2 查）。

- [ ] **Step 4: 复验哈希（本段最硬的一条）**

```bash
python3 -m pytest research/tests/test_golden_trades.py -q
```
Expected: `14 passed`。红 ⇒ **停**：说明搬迁改了行为，逐一比对 dict 内容（最可能是漏了一个键或 verdict 注入改变了某个键）。**不许** `--regen`。

- [ ] **Step 5: Commit**

```bash
git add -A research/
git commit -m "refactor(presets): 4 条 preset 搬进 preset_defs/，presets.py 退化为注册表+门面"
```

---

## Task 4: family 注册表骨架 + 装载校验（TDD）

**Files:**
- Create: `research/family_defs/__init__.py`（空）
- Create: `research/family_registry.py`
- Test: `research/tests/test_family_registry.py`

**Interfaces:**
- Produces: `family_registry.load_one(path) -> dict`（`{id, mask, fixed, grid, score_col, score_negated, needs_regime, control, desc, legacy_names}`）、`family_registry.load_dir(models, dirpath)`。

- [ ] **Step 1: 写失败的测试**

```python
# research/tests/test_family_registry.py
import pytest
from research import family_registry as R

GOOD = '''
ID = "zz"
DOC = """机制叙述"""
def mask(ind, p):
    return ind["close"] > 0
FIXED = {}
GRID = {"a": [1, 2]}
'''

def _w(tmp_path, name, body):
    p = tmp_path / f"{name}.py"; p.write_text(body, encoding="utf-8"); return p

def test_load_one_returns_the_declared_fields(tmp_path):
    got = R.load_one(_w(tmp_path, "zz", GOOD))
    assert got["id"] == "zz" and callable(got["mask"])
    assert got["grid"] == {"a": [1, 2]} and got["needs_regime"] is False
    assert got["control"] is None and got["score_negated"] is False

def test_id_must_match_file_name(tmp_path):
    with pytest.raises(ValueError, match="文件名"):
        R.load_one(_w(tmp_path, "other", GOOD))

def test_mask_is_required_and_callable(tmp_path):
    with pytest.raises(ValueError, match="mask"):
        R.load_one(_w(tmp_path, "zz", GOOD.replace("def mask", "def nope")))

def test_empty_grid_fails_fast(tmp_path):
    with pytest.raises(ValueError, match="GRID"):
        R.load_one(_w(tmp_path, "zz", GOOD.replace('GRID = {"a": [1, 2]}', "GRID = {}")))
```

- [ ] **Step 2: 跑测试确认失败** → `ModuleNotFoundError`

- [ ] **Step 3: 实现**

```python
# research/family_registry.py
"""族定义体的装载器 —— 「新增一个族 = 新增一个文件」的机制（2026-10-17 重构）。

违规 fail-fast：ID 与文件名不符、缺 mask、GRID 为空。
GRID 为空必须抛：预注册网格是**纪律**（§5 迭代记账），不是可选项。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_family_def_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法装载族定义 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_one(path: Path) -> dict:
    mod = _load_module(path)
    if not hasattr(mod, "ID"):
        raise ValueError(f"{path.name} 缺 ID")
    if mod.ID != path.stem:
        raise ValueError(f"{path.name}: ID={mod.ID!r} 与文件名不符")
    fn = getattr(mod, "mask", None)
    if not callable(fn):
        raise ValueError(f"{path.name} 必须定义可调用的 mask(ind, p)")
    grid = dict(getattr(mod, "GRID", {}) or {})
    if not grid:
        raise ValueError(f"{path.name}: GRID 不能为空 —— 预注册网格是纪律（§5）")
    return {"id": mod.ID, "mask": fn, "fixed": dict(getattr(mod, "FIXED", {}) or {}),
            "grid": grid,
            "score_col": getattr(mod, "SCORE_COL", None),
            "score_negated": bool(getattr(mod, "SCORE_NEGATED", False)),
            "needs_regime": bool(getattr(mod, "NEEDS_REGIME", False)),
            "control": getattr(mod, "CONTROL", None),
            "desc": (getattr(mod, "DOC", "") or "").strip(),
            "legacy_names": tuple(getattr(mod, "LEGACY_NAMES", ()) or ()),
            "reference": bool(getattr(mod, "REFERENCE", False))}


def load_dir(models: dict, dirpath: Path) -> None:
    for path in sorted(Path(dirpath).glob("*.py")):
        if path.name.startswith("_"):
            continue
        rec = load_one(path)
        if rec["id"] in models:
            raise ValueError(f"族 ID 重复：{rec['id']}")
        models[rec["id"]] = rec
```

- [ ] **Step 4: 跑测试确认通过** → `4 passed`

- [ ] **Step 5: Commit**（`feat(families): 族定义体的装载器`）

---

## Task 5: 迁移 15 个族 + 门面（含 `_mask_*` 别名）

**Files:**
- Create: `research/family_defs/<id>.py` ×15
- Modify: `research/families.py`（保留 `MIN_PRICE`/`above_price_floor`/`add_family_indicators`/`select`/`grid_size`，其余改成扫描 + 门面）

**Interfaces:**
- Produces: `families.MASK / GRIDS / FIXED / SCORE_COL / SCORE_NEGATED / NEEDS_REGIME / NEW_FAMILIES / REFERENCE_FAMILIES / CONTROL_ARM / GRID_CAP`，以及**原符号名** `_mask_short_reversal` … `_mask_sf_calm`（别名）。

- [ ] **Step 1: 逐族搬文件**：`ID` = 族 id；`mask` = 原 `_mask_<...>` 函数体（docstring 保留为函数 docstring）；`DOC` = 机制叙述；`FIXED`/`GRID` 从字典里取该 id 那一项；`SCORE_COL`/`SCORE_NEGATED`/`NEEDS_REGIME`/`CONTROL` 照现状表；`LEGACY_NAMES = ("_mask_sf_stress",)` 之类（仅测试引用到的两个需要）。
  ⚠️ 两个族的 docstring 里有**审计 F3 的撤回说明**——搬的时候**逐字保留**，不要重写。

- [ ] **Step 2: 改 `families.py` 为扫描 + 门面**

```python
from pathlib import Path as _Path

from research.family_registry import load_dir as _load_dir

_DEFS_DIR = _Path(__file__).resolve().parent / "family_defs"
_REGISTRY: dict[str, dict] = {}
_load_dir(_REGISTRY, _DEFS_DIR)          # fail-fast：违规在 import 时就抛

MASK: dict[str, callable] = {k: r["mask"] for k, r in _REGISTRY.items()}
FIXED: dict[str, dict] = {k: r["fixed"] for k, r in _REGISTRY.items()}
GRIDS: dict[str, dict] = {k: r["grid"] for k, r in _REGISTRY.items()}
SCORE_COL: dict[str, str] = {k: r["score_col"] for k, r in _REGISTRY.items()
                             if r["score_col"]}
SCORE_NEGATED = frozenset(k for k, r in _REGISTRY.items() if r["score_negated"])
NEEDS_REGIME = frozenset(k for k, r in _REGISTRY.items() if r["needs_regime"])
CONTROL_ARM: dict[str, str] = {k: r["control"] for k, r in _REGISTRY.items()
                               if r["control"]}
REFERENCE_FAMILIES = frozenset(k for k, r in _REGISTRY.items() if r["reference"])
NEW_FAMILIES = [k for k in MASK if k not in REFERENCE_FAMILIES]
GRID_CAP = 96                            # 见 spec：声明等于事实，由测试钉住

# 原符号名再导出（`tests/test_families.py` 的 F3 护栏引用它；不改那些测试）
for _k, _r in _REGISTRY.items():
    for _alias in _r["legacy_names"]:
        globals()[_alias] = _r["mask"]
```

- [ ] **Step 3: 跑全量测试** → 全绿（`test_families.py`/`test_score_convention.py` **不该需要改**；若报 `AttributeError: _mask_sf_stress` ⇒ 别名没做上，回 Step 2）。

- [ ] **Step 4: 复验哈希** → `test_golden_trades` 14 passed + `test_golden_fixture_lock` 5 passed；红则停并查根因。

- [ ] **Step 5: Commit**（`refactor(families): 15 个族搬进 family_defs/，families.py 退化为注册表+门面`）

---

## Task 6: 行为锁覆盖（`CASES` 派生 + 豁免表 + 三条断言）

**Files:**
- Modify: `research/golden_baseline.py`（`CASES` 改成从注册表派生）
- Modify: `research/tests/test_golden_fixture_lock.py`（fixture 用例也派生）
- Test: `research/tests/test_lock_coverage.py`

- [ ] **Step 1: 写失败的测试**

```python
# research/tests/test_lock_coverage.py
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
    assert set(G.LOCK_EXEMPT) == set(), \
        "豁免表变了 —— 一旦它能无声扩大，这条护栏就等于没有"

def test_cases_carry_the_declared_windows():
    by = {c["preset"]: c for c in G.CASES}
    for name, rec in presets.PRESETS.items():
        if name in by:
            assert (by[name]["start"], by[name]["end"]) == rec["window"]
    assert sum(1 for c in G.CASES if c.get("is_cli_default")) == 1
```

- [ ] **Step 2: 跑测试确认失败** → `AttributeError: LOCK_EXEMPT`

- [ ] **Step 3: 实现**：`golden_baseline.py` 里 `CASES` 改为

```python
LOCK_EXEMPT: dict[str, str] = {}      # preset 名 -> 为什么不锁（空 = 全都锁）

def _cases_from_registry() -> list[dict]:
    out = []
    for name, rec in presets.PRESETS.items():
        if name in LOCK_EXEMPT:
            continue
        out.append({"name": name, "preset": name,
                    "start": rec["window"][0], "end": rec["window"][1],
                    "is_cli_default": rec["is_cli_default"]})
    return out

CASES: list[dict] = _cases_from_registry()
```

  并把 `test_golden_fixture_lock.py` 的 `CASES` 同样改为从 `presets.PRESETS` 派生（有 `fixture_window` 的入锁，有 `fixture_exempt` 理由的不入锁，两者都没有 ⇒ **抛错**，逼新增时做决定）。

- [ ] **Step 4: 跑测试确认通过**

```bash
python3 -m pytest research/tests/test_lock_coverage.py research/tests/test_golden_trades.py research/tests/test_golden_fixture_lock.py -q
```
Expected: 全绿，且**哈希仍未变**（派生出的窗口与原来逐字相同才可能）。

- [ ] **Step 5: Commit**（`test(golden): 行为锁用例从注册表派生 + 每个 preset 必须锁或显式豁免`）

---

## Task 7: 实操验收（判据 4）

**Files:** 临时新增两个文件，验完删除。

- [ ] **Step 1: 加一条 preset，只碰一个新文件**

按 Task 3 的格式新建 `research/preset_defs/zz_probe_v1.py`（键全用现有引擎键）。

- [ ] **Step 2: 验证它被自动接管**

```bash
python3 -c "
import sys;sys.path.insert(0,'.')
from research import presets, golden_baseline as G
p = presets.get_preset('zz_probe_v1')
print('可装载 ✓', p['n_comparisons'] >= 1)
print('在锁里 ✓', any(c['preset'] == 'zz_probe_v1' for c in G.CASES))"
python3 -m pytest research/tests/test_lock_coverage.py -q      # 必须绿
```
Expected: 三行都对。

- [ ] **Step 3: 加一个族，同样只碰一个新文件**：新建 `research/family_defs/zz.py`，跑
  `python3 -m pytest research/tests/test_families.py research/tests/test_score_convention.py -q`，
  并用 `python3 -c "from research import families as F; print('zz' in F.NEW_FAMILIES, F.grid_size('zz'))"` 确认自动进入横向比较。

- [ ] **Step 4: 删除两个探针文件并复验干净**

```bash
rm research/preset_defs/zz_probe_v1.py research/family_defs/zz.py
python3 -m pytest research/tests src/tests -q
```
Expected: 全绿（探针曾经的用例随文件消失）。

- [ ] **Step 5: Commit**（`chore: 实操验收记录 —— 新增 preset/族只需新增一个文件`，只提交台账/说明，不提交探针）

---

## Task 8: 收尾

- [ ] **Step 1: 全量 + 两道锁 + validate**

```bash
python3 -m pytest research/tests src/tests -q
python3 src/validate.py | tail -3
```
Expected: 全绿；`validate` 仍 18/18。

- [ ] **Step 2: 更新台账** `docs/2026-10-17-institutional-fix-plan.md`：加一节
  「注册表重构」，写明三段各改了什么、7 个哈希未变的证据、以及**已知边界**
  （`test_presets.py` 仍只测默认 preset；fixture 锁需要 `FIXTURE_WINDOW` 或豁免理由）。

- [ ] **Step 3: Commit** 台账。
