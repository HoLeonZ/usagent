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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 扫描面（2026-10-17 对抗核验：此前只有 8 种后缀 ⇒ 坏路径写进 .json/.html/无扩展名文件
# 一律看不见）。宁可多扫几个文本后缀，也别让"换个扩展名就能溜过去"。
TEXT_SUFFIXES = (".md", ".py", ".sh", ".txt", ".yaml", ".yml", ".toml", ".cfg",
                 ".json", ".csv", ".html", ".ini", ".env", ".sql", ".ipynb")

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
    # 2026-10-17 对抗核验：上面那条只认 `PY=<绝对路径>` 这一种拼法，
    # `exec /opt/homebrew/bin/python3` 这类**裸的绝对解释器路径**照样溜过去。
    (re.compile(r"(?<![\w./])/(?:opt/homebrew|usr/local|Users/[^/\s]+)/[^\s]*?/?(?:python[0-9.]*|python3)"),
     "写死的解释器绝对路径（不限于 PY= 形式）—— 用 ${PY:-python3}"),
]


def _clean_git_env() -> dict:
    """去掉继承来的 GIT_* —— 让 git 只按 cwd 发现仓库。

    为什么必须有：`GIT_DIR` / `GIT_WORK_TREE` / `GIT_INDEX_FILE` / `GIT_COMMON_DIR` 一旦被
    导出（linked worktree 里的钩子就是这种环境），在别处跑的 `git init` / `git add` 会作用到
    **那个**仓库上 —— 实测能把共享 .git/config 的 core.bare 写成 true，让主工作区不可用。
    """
    import os
    return {k: v for k, v in os.environ.items()
            if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                         "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES")}


def _tracked_text_files():
    """被跟踪的文本文件 + **已暂存但还没提交**的新文件。

    为什么要并上"已暂存"（2026-10-17 实测教训）：`git ls-files` 看不到刚写好、还没
    `git add` 的新文件。于是存在这条缝 —— 先跑测试（绿）→ 再 `git add` 一个新文件 →
    提交。提交后的状态**从未被这道护栏看过**，而它就是被提交的那个状态。
    （实际发生过一次：`research/preset_ranking.py` 写死了本机绝对路径，测试跑在它
    入索引之前，于是"596 passed"放行了一个红着的提交。）
    """
    import subprocess

    def _git(*args):
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.splitlines() if out.returncode == 0 else []

    paths = set(_git("ls-files"))
    # --diff-filter=ACMR：新增/复制/改动/改名，排除删除（删掉的文件读不到）
    paths |= set(_git("diff", "--cached", "--name-only", "--diff-filter=ACMR"))
    return [ROOT / p for p in sorted(paths) if p.endswith(TEXT_SUFFIXES)]


def _index_text(rel: str) -> str | None:
    """取**索引里**那份内容（= 将要被提交的那份）。看不见索引就返回 None。

    为什么必须读索引（2026-10-17 对抗核验实测）：此前只从索引取**文件名**，内容却读工作区
    ⇒「`git add` 一个带违规的文件、再把工作区改回干净」能让护栏放行，而**提交进历史的是
    违规那一份**（实测复现）。要守住"被提交的状态"，就必须读索引的内容。
    """
    try:
        out = subprocess.run(["git", "show", f":{rel}"], cwd=ROOT,
                             capture_output=True, text=True, env=_clean_git_env())
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def test_no_machine_specific_paths_in_tracked_files():
    files = _tracked_text_files()
    if not files:
        import pytest
        pytest.skip("不是 git 工作区，无法枚举被跟踪文件")
    hits = []
    for f in files:
        if f.resolve() in _ALLOWLIST:
            continue
        rel = str(f.relative_to(ROOT))
        copies: list[tuple[str, str]] = []
        try:
            copies.append(("工作区", f.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, FileNotFoundError):
            pass
        idx = _index_text(rel)
        if idx is not None:
            copies.append(("索引", idx))
        for where, text in copies:
            for pat, why in FORBIDDEN:
                for m in pat.finditer(text):
                    line = text[:m.start()].count("\n") + 1
                    hits.append(f"{rel}:{line}[{where}]: {m.group(0)!r} —— {why}")
    assert not hits, "仓库里出现机器相关的绝对路径：\n" + "\n".join(hits)


def test_the_holdout_boundary_is_defined_in_exactly_one_place():
    """预注册的那条分界线（搜索期 / holdout）只能有**一处**定义。

    为什么值得一条锁（2026-10-22 实测）：它此前至少有三份：
      * `research/wfv_search.py` 的 `SEARCH_LO/HI`、`HOLDOUT_LO/HI`（预注册正文，正主）；
      * `webapp/catalog.py` 又抄了一份**字面量**（注释还写着"与 holdout 台账、重叠判据
        同一个常量"——那是副本），平台据此给"与 holdout 重叠"标红；
      * `research/event_attribution.py` 抄了一份 `TRAIN`，外加一个**没人用**的 `HOLDOUT`。
    改了预注册却漏改副本 ⇒ 平台的警示与归因阶段的闸门会对着**旧的边界**工作，
    而且不会报错。这类副本越多，纪律越像口号。

    只认"赋值里的字面量"，不认散文里的提及（注释与文档里写日期是正常的、也是必要的）。
    """
    pat = re.compile(r'^\s*(?:HOLDOUT|HOLDOUT_LO|HOLDOUT_HI|SEARCH_LO|SEARCH_HI)'
                     r'(?:\s*,\s*(?:HOLDOUT|HOLDOUT_LO|HOLDOUT_HI|SEARCH_LO|SEARCH_HI))*'
                     r'\s*=\s*[\(\[]?\s*["\'](\d{4}-\d\d-\d\d)["\']', re.M)
    found: list[str] = []
    where: set[str] = set()
    for d in ("research", "webapp", "src", "tools"):
        for f in sorted((ROOT / d).rglob("*.py")):
            if "tests" in f.parts or f.name == "conftest.py":
                continue
            for m in pat.finditer(f.read_text(encoding="utf-8")):
                where.add(str(f.relative_to(ROOT)))
                found.append(f"{f.relative_to(ROOT)} → {m.group(1)}")
    # 一个文件里可以同时定义搜索期与 holdout（那正是预注册正文所在的地方）；
    # **别处**不许再写一份字面量。
    assert where == {"research/wfv_search.py"}, \
        f"这条分界线只能在预注册那一个文件里定义，别处有副本：{sorted(where)}"
    assert "2025-10-14" in " ".join(found), found


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


def test_preflight_finds_an_interpreter_instead_of_assuming_the_first_one():
    """preflight 必须在 PATH 上**找**有依赖的解释器，而不是假定第一个 `python3` 就是。

    为什么这条是行为锁而不是洁癖（2026-10-18 实测）：pre-commit 钩子不继承你 shell 里的
    `PY=...`（那是命令前缀，不是导出）。本机 `which -a python3` 第一个是 /usr/bin/python3
    （无 numpy）⇒ 钩子把一个**环境问题**报成红，于是每次提交都被挡下；而被一个永远挡人的
    钩子挡住，正确反应会退化成 `--no-verify` —— 那等于把全部护栏一起关掉，比没有钩子更糟。
    这条断言钉的就是"去找、并说出选中的那个"。
    """
    src = (ROOT / "tools" / "preflight.sh").read_text()
    assert "type -a -p python3" in src, "preflight 没有沿着 PATH 去找能用的解释器"
    assert 'echo "[preflight] 解释器：' in src, "换解释器这件事不能静默（要打印选中的那个）"


def test_staged_but_uncommitted_files_are_also_scanned(tmp_path, monkeypatch):
    """提交前就要抓：**已暂存、未提交**的新文件也必须被扫。

    这正是漏过一次的那道缝：先跑测试（绿）→ 再 `git add` 一个新文件 → 提交，
    于是"被提交的那个状态"从未被这道护栏看过。实测发生过（`research/preset_ranking.py`
    写死了本机绝对路径，而 596 passed 的测试跑在它入索引之前）。
    """
    import subprocess

    import src.tests.test_repo_hygiene as H          # noqa: N813

    monkeypatch.setattr(H, "ROOT", tmp_path)         # _git() 在调用时读模块全局
    # ⚠️ **必须清掉继承来的 GIT_\* 环境变量**（2026-10-17 对抗核验抓到的**破坏性**回归）：
    # linked worktree 里 git 会向钩子导出 GIT_DIR=<repo>/.git/worktrees/<name>；测试若带着
    # 这个变量在 tmp 目录里 `git init`，git 会顺着 commondir 写回**共享**的 .git/config
    # （core.bare = true）⇒ 主工作区随后 git status/add/checkout 全部 fatal，要手工修配置。
    # 实测在三个独立副本上复现过。凡是在临时目录里跑 git 的测试都必须清环境。
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True,
                   env=_clean_git_env())
    bad = tmp_path / "bad.py"
    bad.write_text('ROOT = "/Users/someone/code/usagent"\n', encoding="utf-8")
    subprocess.run(["git", "add", "bad.py"], cwd=tmp_path, check=True)   # 只暂存，不提交

    assert bad in H._tracked_text_files(), "已暂存未提交的新文件必须被扫到"
