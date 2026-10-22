"""族定义体的装载器 —— 「新增一个族 = 新增一个文件」的机制（2026-10-17 重构）。

设计见 `docs/superpowers/specs/2026-10-17-preset-family-registry-design.md`。

为什么要有这一层：族的定义此前散在 `families.py` 的四处（`MASK` 函数、`FIXED`、
`GRIDS`、`SCORE_COL`/`NEEDS_REGIME`）加 `CONTROL_ARM`，新增一个族要同时改这些地方。
漏掉 `GRIDS` 会 `KeyError`，但漏掉别处**什么都不会发生** —— 新族就这样悄悄不在
护栏里，或者网格能被事后改动。搬成一个文件后，"这个族是什么"只有一处。

违规一律 fail-fast（§0），其中 `GRID` 为空必须抛：预注册网格是**纪律**（§5 迭代记账），
不是可选项 —— 允许空网格，等于允许"先看结果再决定搜什么"。唯一的例外要**明写理由**
（`GRID_EXEMPT_REASON`）：参考族 `upb_ref` 不参与搜索，它没有网格不是疏忽而是事实。
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
    """装载一个族定义文件，返回规范化记录。任何违规都抛 ValueError。"""
    mod = _load_module(path)
    if not hasattr(mod, "ID"):
        raise ValueError(f"{path.name} 缺 ID")
    if mod.ID != path.stem:
        raise ValueError(f"{path.name}: ID={mod.ID!r} 与文件名不符 —— "
                         f"名字与文件对不上这件事必须立刻发现")
    fn = getattr(mod, "mask", None)
    if not callable(fn):
        raise ValueError(f"{path.name} 必须定义可调用的 mask(ind, p) —— "
                         f"族的核心就是这个判定函数")
    grid = dict(getattr(mod, "GRID", {}) or {})
    exempt = (getattr(mod, "GRID_EXEMPT_REASON", "") or "").strip()
    if not grid and not exempt:
        raise ValueError(f"{path.name}: GRID 不能为空 —— 预注册网格是纪律（§5）："
                         f"允许空网格等于允许事后决定搜什么")
    if grid and exempt:
        raise ValueError(f"{path.name}: 既有 GRID 又声明 GRID_EXEMPT_REASON —— "
                         f"读了不知道该信哪个")
    return {
        "id": mod.ID,
        "mask": fn,
        "fixed": dict(getattr(mod, "FIXED", {}) or {}),
        "grid": grid,
        "grid_exempt": exempt or None,
        "score_col": getattr(mod, "SCORE_COL", None),
        "score_negated": bool(getattr(mod, "SCORE_NEGATED", False)),
        "needs_regime": bool(getattr(mod, "NEEDS_REGIME", False)),
        "control": getattr(mod, "CONTROL", None),
        "desc": (getattr(mod, "DOC", "") or "").strip(),
        # 原符号名（例如 `_mask_sf_stress`）：门面按这些名字再导出一次，
        # 好让守着 F3 结论的那两条测试**一个字都不用改**。
        "legacy_names": tuple(getattr(mod, "LEGACY_NAMES", ()) or ()),
        "reference": bool(getattr(mod, "REFERENCE", False)),
    }


def load_dir(models: dict, dirpath: Path) -> None:
    """扫一个目录里的全部族定义（跳过 `_` 开头的私有文件），把记录塞进 `models`。"""
    path = Path(dirpath)
    if not path.is_dir():
        raise ValueError(f"族定义目录不存在：{path}")
    for f in sorted(path.glob("*.py")):
        if f.name.startswith("_"):
            continue
        rec = load_one(f)
        if rec["id"] in models:
            raise ValueError(f"族 ID 重复：{rec['id']}（{f.name}）")
        models[rec["id"]] = rec
