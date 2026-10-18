"""策略目录 —— 把「当前所有策略与参数」编译成一份可以给页面看的结构。

**只读注册表，不重写任何策略逻辑**：preset 的权威定义在 `research/preset_defs/`，
族的在 `research/family_defs/`（见各自的 registry）。这一层做的是"取出来、翻成人话、
标出可运行的那几种形态"，任何参数都不在这里二次加工 —— 一旦两处定义，就会分叉。

两个来源：
  * **preset**：6 条交付物。给出引擎口径（`describe_engine_spec` 的**同一份**人话标签）、
    `LOCK_WINDOW`、`n_comparisons`、`validated`，以及它自己的 `VERDICT`
    （§5：负面证据与参数同时记录 ⇒ 页面上也必须同时显示，不能只展示参数）。
  * **族**：15 个候选机制。给出掩码机制叙述（docstring 首行）、预注册网格与组数、
    固定轴、对照臂、排序口径；并标出"哪条已交付的 preset 用的就是它"。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 本模块被加载的时刻 ≈ 服务启动时刻（服务一启动就 import 它）——"定义体在此之后被改过"
# 就是对它比。用一个模块级常量而不是到处传时间戳：它只有一个来源。
_PROCESS_START = time.time()
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.preset_contract import describe_engine_spec        # noqa: E402
from research import families, presets                       # noqa: E402

# 交付物的窗口若与这段重叠，页面上必须标红：holdout 只准读一次，且台账里已有 claim。
HOLDOUT = ("2025-10-14", "2026-10-13")
# 训练窗的定义处（WFV 的搜索期）—— 页面从这里取，别在自己那份里再写一遍
from research import wfv_search as W                       # noqa: E402


def _overlaps(a: tuple[str, str], b: tuple[str, str]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def preset_entry(name: str) -> dict:
    """一条 preset 的可展示形态。"""
    rec = presets.REGISTRY[name]
    P = presets.get_preset(name)
    win = tuple(rec["window"])
    verdict = (rec.get("verdict") or "").strip()
    return {
        "kind": "preset",
        "id": name,
        "name": name,
        "window": list(win),
        "is_cli_default": bool(rec["is_cli_default"]),
        "touches_holdout": _overlaps(win, HOLDOUT),
        # 引擎口径：与 CLI / HTML 报告**同一个函数**，所以页面上的口径不可能与实跑不一致
        "engine_spec": [{"label": lab, "value": val} for lab, val in describe_engine_spec(P)],
        "strategy": {
            "type": "family" if P.get("family") else "signal",
            "family": P.get("family"),
            "family_params": P.get("family_params") or {},
            "signal_params": P.get("signal") or {},
            "regime": P.get("regime") or None,
        },
        "n_comparisons": int(P.get("n_comparisons") or 1),
        "validated": bool(P.get("validated", False)),
        "verdict": verdict,          # ← 负面证据必须与参数同屏
    }


def family_entry(fid: str) -> dict:
    """一个族的可展示形态（含"怎么把它跑起来"）。"""
    rec = families._REGISTRY.get(fid, {})
    grid = families.GRIDS.get(fid) or {}
    mask = families.MASK.get(fid)
    doc = (getattr(mask, "__doc__", "") or "").strip().splitlines()
    delivered = sorted(n for n, r in presets.REGISTRY.items()
                       if (r["preset"].get("family") == fid))
    # 族的可运行模板：与 rank_strategies / render_report --family 同一条路
    # （退出/仓位口径取自那条已注册的 preset，只有入场掩码换成本族）
    return {
        "kind": "family",
        "id": fid,
        "name": fid,
        "mechanism": doc[0] if doc else "",
        "grid": {k: list(v) for k, v in grid.items()},
        "grid_size": int(families.grid_size(fid)) if grid else 0,
        "fixed": dict(families.FIXED.get(fid) or {}),
        "score_col": families.SCORE_COL.get(fid, "mom120（默认）"),
        "score_negated": fid in families.SCORE_NEGATED,
        "needs_regime": fid in families.NEEDS_REGIME,
        "control_arm": families.CONTROL_ARM.get(fid),
        "reference": fid in families.REFERENCE_FAMILIES,
        "delivered_as": delivered,   # 它是不是已经是某条 preset 的入场逻辑
        # 网格**中点**那一组（确定性、不看结果）—— 手点回测时的默认参数
        "default_params": {k: v[len(v) // 2] for k, v in grid.items()},
    }


def build() -> dict:
    """整份目录。页面与 API 都读它 —— 单一来源。"""
    presets_list = [preset_entry(n) for n in sorted(presets.REGISTRY)]
    fams = [family_entry(f) for f in sorted(families.MASK)]
    holdout_claims = sum(1 for p in presets_list if p["touches_holdout"])
    return {
        "schema": "usagent.platform.catalog/v1",
        # 三个窗口预设都从**它们的真实出处**发布，不要让页面再抄一份（抄一份就会各自漂移）：
        #   holdout —— 本文件的 HOLDOUT（与 holdout 台账、重叠判据同一个常量）；
        #   train   —— WFV 的搜索期（`wfv_search.SEARCH_LO/HI`，那是"训练窗"的定义处）。
        # 页面里剩下的 FULL（训练窗起点 .. 数据末端）是**视图默认值**，不是台账里的事实，
        # 所以不上行 —— 但它的按钮 title 始终印出"按下去会设成哪两天"，不会撒谎。
        "holdout": list(HOLDOUT),
        # 启动之后被动过的定义体：服务读的是旧值 —— 摆到界面上，别让人在"改了没反应"里打转
        "stale_defs": stale_definitions(),
        "train": [W.SEARCH_LO, W.SEARCH_HI],
        "presets": presets_list,
        "families": fams,
        "notes": {
            "holdout_presets": holdout_claims,
            "family_default_params": "网格中点那一组（不是搜索最优）—— 点'运行'不会因此变成搜索",
            "multiplicity": "每点一次运行都会进台账；DSR 的 k 随之为涨（§5 迭代记账）",
        },
    }


def stale_definitions(started_at: float | None = None) -> list[str]:
    """比"服务启动时刻"更新的定义体文件 —— 改了它们但还没重启，服务读的是旧值。

    为什么值得单独一个函数：这条路**静默**（改了没反应）且现在页面上只写了句提示。
    实测过的更坏版本：同秒内改动又改回、长度不变时，Python 连 .pyc 都认为有效，
    于是进程用的是**不存在于仓库里的值**（见 webapp/server.py 顶部那段注记）。
    这里只做一件事：把"哪些文件在启动之后动过"摆到界面上，让人自己决定要不要重启。
    """
    t0 = started_at if started_at is not None else _PROCESS_START
    out = []
    for sub in ("preset_defs", "family_defs"):
        d = ROOT / "research" / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            if f.stat().st_mtime > t0:
                out.append(f"research/{sub}/{f.name}")
    return out


def holdout_touches(runs: list[dict]) -> int:
    """台账里有多少次运行的窗口碰了 holdout。

    为什么要有这个计数：卡片上写着「与 holdout 重叠（只准读一次）」，但页面上**从没有数过
    这条规矩被跨过几次** —— 一句没有计数的纪律等于没有纪律（§5 的整个做法就是"让次数可见"）。
    数据早就在台账里（每条运行的 window），这里只是把它数出来。
    """
    n = 0
    for r in runs or []:
        w = r.get("window")
        if isinstance(w, (list, tuple)) and len(w) == 2 and _overlaps((w[0], w[1]), HOLDOUT):
            n += 1
    return n


def find(kind: str, ident: str) -> dict | None:
    for e in build()[("presets" if kind == "preset" else "families")]:
        if e["id"] == ident:
            return e
    return None


if __name__ == "__main__":       # 便于命令行看一眼目录
    import json
    print(json.dumps(build(), ensure_ascii=False, indent=1))
