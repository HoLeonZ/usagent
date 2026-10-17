"""preset 定义体的装载器 —— 「新增一条 preset = 新增一个文件」的机制（2026-10-17 重构）。

设计见 `docs/superpowers/specs/2026-10-17-preset-family-registry-design.md`。

为什么要有这一层：preset 此前定义在 `presets.py` 的 `PRESETS` dict 里，新增一条要改
**现有文件**；而族型 preset 还要再改 `families.py` 的 4 处。后果不是"麻烦"，而是
**新东西默认不在护栏里** —— `golden_baseline.CASES` 是手写清单，漏了就什么都不会发生。

装载违规一律 **fail-fast**（§0），因为它们的共同后果正是上面那种失效方式：
  * 文件名与 `NAME` 不符 ⇒ "名字与文件对不上"没人发现；
  * 缺 `VERDICT` ⇒ §5 要求的负面证据被省掉；
  * 缺 `LOCK_WINDOW` ⇒ 行为锁只能去用**会随数据漂移**的默认窗口，
    而锁的全部意义就是它不会自己动；
  * `PRESET` 里再写一份 `verdict` ⇒ 一份事实两处（迟早不一致）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# 定义体必须声明的字段。LOCK_WINDOW 也在内：见模块 docstring 最后两条。
REQUIRED_ATTRS = ("NAME", "PRESET", "VERDICT", "LOCK_WINDOW")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_preset_def_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法装载 preset 定义 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_one(path: Path) -> dict:
    """装载一个 preset 定义文件，返回规范化记录。任何违规都抛 ValueError。"""
    mod = _load_module(path)
    missing = [a for a in REQUIRED_ATTRS if not hasattr(mod, a)]
    if missing:
        raise ValueError(f"{path.name} 缺 {missing} —— §5 要求负面证据与参数同时记录，"
                         f"且行为锁需要显式的绝对窗口")
    if mod.NAME != path.stem:
        raise ValueError(f"{path.name}: NAME={mod.NAME!r} 与文件名不符 —— "
                         f"名字与文件对不上这件事必须立刻发现")
    if not isinstance(mod.PRESET, dict):
        raise ValueError(f"{path.name}: PRESET 必须是 dict")
    preset = dict(mod.PRESET)
    if "verdict" in preset:
        raise ValueError(f"{path.name}: PRESET 里不要再写 verdict —— 用模块级 VERDICT"
                         f"（一份事实只能有一处）")
    win = tuple(mod.LOCK_WINDOW)
    if len(win) != 2 or not all(isinstance(x, str) and x for x in win):
        raise ValueError(f"{path.name}: LOCK_WINDOW 必须是 (start, end) 两个日期串，"
                         f"得到 {mod.LOCK_WINDOW!r}")
    fx = tuple(getattr(mod, "FIXTURE_WINDOW", ()) or ())
    if fx and len(fx) != 2:
        raise ValueError(f"{path.name}: FIXTURE_WINDOW 必须是 (start, end)")
    exempt = getattr(mod, "FIXTURE_EXEMPT_REASON", None)
    if fx and exempt:
        raise ValueError(f"{path.name}: FIXTURE_WINDOW 与 FIXTURE_EXEMPT_REASON 只能给一个 —— "
                         f"既入离线锁又声明豁免，读了不知道该信哪个")
    if not fx and not exempt:
        raise ValueError(f"{path.name}: 必须就**离线锁**表态 —— 给 FIXTURE_WINDOW（入离线锁，"
                         f"任何机器上都能跑）或给 FIXTURE_EXEMPT_REASON（为什么不入）。"
                         f"两者都不给，这条 preset 就是**默认不在离线护栏里**，"
                         f"而那正是这次重构要消灭的失效方式")
    return {
        "name": mod.NAME,
        "preset": preset,
        "verdict": mod.VERDICT,
        "window": win,
        "is_cli_default": bool(getattr(mod, "LOCK_IS_CLI_DEFAULT", False)),
        "fixture_window": fx or None,
        "fixture_exempt": exempt,
        # 定义体要**再导出**的名字（例：长篇负面证据 NOTES）。门面会把它们挂到
        # presets.py 的模块级 —— 这样"证据跟着 preset 走"与"外部仍能 import"同时成立。
        "exports": dict(getattr(mod, "EXPORTS", {}) or {}),
        "source": path.name,
    }


def load_dir(models: dict, dirpath: Path) -> None:
    """扫一个目录里的全部定义（跳过 `_` 开头的私有文件），把记录塞进 `models`。"""
    path = Path(dirpath)
    if not path.is_dir():
        raise ValueError(f"定义目录不存在：{path}")
    for f in sorted(path.glob("*.py")):
        if f.name.startswith("_"):
            continue
        rec = load_one(f)
        if rec["name"] in models:
            raise ValueError(f"preset 名字重复：{rec['name']}（{f.name}）")
        models[rec["name"]] = rec
