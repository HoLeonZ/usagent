"""仓库卫生：机器相关的绝对路径、写死的解释器、未钉的依赖。

动机（2026-10-17 审计 G2）：本仓库的核心主张是"同输入 ⇒ 同输出"（§0），但它此前
  * 没有任何依赖锁定（无 requirements/pyproject/lock）；
  * 把**某一台机器**的绝对解释器路径当环境契约写进文档与脚本
    （`PY=/opt/homebrew/.../python`、`cd /Users/<某人>/code/usagent`）。
实测这些路径在**当前这台机器上就已经失效** —— 也就是新读者照着文档跑，第一步就报
"command not found"，而这与策略本身毫无关系。

钉住的规则（只禁**本仓库自己的**机器耦合写法，不禁"在散文里提到别的项目在哪"）：
  1. 不得出现本仓库的绝对路径 `/Users/*/code/usagent`；
  2. 不得写死某个包管理器下的解释器（`PY=/opt/homebrew...`、`PY=/usr/local/...`）；
  3. `requirements.txt` 必须存在，且每个依赖都带 `==` 版本。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEXT_SUFFIXES = (".md", ".py", ".sh", ".txt", ".yaml", ".yml", ".toml", ".cfg")

# 唯一豁免：本文件。它**必须**写出这些模式才能定义它们（正则字面量与 docstring 里
# 各一份）。豁免名单只允许这一个，且由 test_the_allowlist_is_exactly_one_file 钉住 ——
# 豁免名单一旦能悄悄变长，这条护栏就失效了。
_ALLOWLIST = {Path(__file__).resolve()}
_SELF = Path(__file__).resolve()

# (正则, 为什么禁止)
FORBIDDEN = [
    (re.compile(r"/Users/[^/\s`]+/code/usagent"),
     "本仓库的绝对路径 —— 换台机器就失效（用相对路径或 <仓库根>）"),
    (re.compile(r"PY=/opt/homebrew|PY=/usr/local|PY=/Users/"),
     "写死的解释器路径 —— 用 ${PY:-python3} 让调用方覆盖"),
]


def _tracked_text_files():
    import subprocess
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [ROOT / p for p in out.stdout.splitlines() if p.endswith(TEXT_SUFFIXES)]


def test_no_machine_specific_paths_in_tracked_files():
    files = _tracked_text_files()
    if not files:
        import pytest
        pytest.skip("不是 git 工作区，无法枚举被跟踪文件")
    hits = []
    for f in files:
        if f.resolve() in _ALLOWLIST:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for pat, why in FORBIDDEN:
            for m in pat.finditer(text):
                line = text[:m.start()].count("\n") + 1
                hits.append(f"{f.relative_to(ROOT)}:{line}: {m.group(0)!r} —— {why}")
    assert not hits, "仓库里出现机器相关的绝对路径：\n" + "\n".join(hits)


def test_dependencies_are_pinned():
    req = ROOT / "requirements.txt"
    assert req.exists(), ("没有 requirements.txt —— 不钉版本时，同一份代码在另一台机器上"
                          "跑出的数字可能不同，而'不同'会被误读成策略的问题")
    lines = [l.strip() for l in req.read_text().splitlines()]
    deps = [l for l in lines if l and not l.startswith("#")]
    assert deps, "requirements.txt 里没有任何依赖"
    unpinned = [d for d in deps if "==" not in d]
    assert not unpinned, f"这些依赖没钉版本：{unpinned}"


def test_the_allowlist_is_exactly_one_file():
    """豁免只能有一个（本文件自己）—— 它一旦能悄悄变长，这条护栏就等于没有。"""
    assert _ALLOWLIST == {_SELF}, f"护栏豁免名单被扩大了：{_ALLOWLIST - {_SELF}}"


def test_run_daily_does_not_hardcode_an_interpreter():
    src = (ROOT / "src" / "run_daily.sh").read_text()
    assert 'PY="${PY:-python3}"' in src, "run_daily.sh 必须让调用方可以覆盖解释器"
    assert not re.search(r"^PY=/", src, re.M), "仍然写死了 PY 的绝对路径"
