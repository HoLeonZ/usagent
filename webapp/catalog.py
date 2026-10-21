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


def touches_holdout(window: tuple[str, str], holdout: tuple[str, str] = HOLDOUT) -> bool:
    """这次运行**真的读了** holdout 里的天数吗？

    ⚠️ 两个区间的语义**不一样** —— 这正是一开始判错的地方（2026-10-18 审计实测，两名核验者独立复现）：
      * 运行窗口是**左闭右开** `[lo, hi)`：引擎按 `axis = [d for d in days if lo <= d < hi]` 取舍
        （`research/broker.py:373`）⇒ **终点那一天不会被读**；
      * holdout 是**按日点名**的一段，两端都算。
    所以"读了"的充要条件是 `hi > holdout_start and lo <= holdout_end`。

    原先用的是闭区间判据 `not (hi < h0 or h1 < lo)`，把"终点正好落在 holdout 第一天"的
    **合法**窗口误报成越界 —— 既会把用户从能用的输入上吓走，也会让纪律计数多算
    （实测：`2024-09-01..2025-10-14` 引擎一天都没读，旧判据却说"碰了"）。
    """
    return window[1] > holdout[0] and window[0] <= holdout[1]


_BASELINE_DIR = ROOT / "research" / "tests" / "baselines"


def _golden_state(name: str) -> dict:
    """这条 preset 的行为锁基线在不在、什么时候生成的。"""
    f = _BASELINE_DIR / f"{name}.golden"
    if not f.is_file():
        return {"exists": False, "mtime": None}
    import datetime as _dt
    return {"exists": True,
            "mtime": _dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}


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
        "touches_holdout": touches_holdout(win),
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
        # 这条 preset 的**行为锁基线**在不在（§6）。交付一条新候选时必须生成它，否则
        # test_golden_trades 会对它红 —— 而那件事此前只有跑一次测试才知道（2026-10-20 实测：
        # 建了兄弟候选，测试直接红，我得自己反推是 --regen 的事）。平台该一眼说出来。
        "golden_baseline": _golden_state(name),
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
        # 机制那一行是给**人读**的（卡片直接印它），所以在这里就清成纯文本：
        # ① 掩码 docstring 首行自带「族:」/「族 1:」前缀，而卡片模板前面又加了「族 · 」
        #    ⇒ 实测 15 张族卡里 11 张印成「族 · 族:**…**」；
        # ② 首行里有 markdown 的 **强调**，而卡片是 esc() 后原样显示的 ⇒ 星号直接印在脸上
        #    （我的 markdown 护栏扫的是 index.html 的静态文案，看不见**数据**里的这类标记）。
        "mechanism": _plain_mechanism(doc[0] if doc else ""),
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
        # 数据覆盖到哪一天（拿多新的数据跑的结论）—— 只读装饰，取不到就不显示
        "data_through": data_through(),
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


def data_through() -> dict | None:
    """价格数据**覆盖到哪一天** —— 让人一眼看出"这份结论是拿多新的数据跑的"。

    用 coverage 表的 MAX(last_date)（实测 9 毫秒；直接对 ohlcv 做 MIN/MAX 要 ~1 秒，
    每次打开页面都付不起）。与真跑一次得到的池子末端逐字一致（实测都是 2026-10-16）。

    ⚠️ 两条必须守住的：
      1) 只在**文件存在**时查 —— `sqlite3.connect` 会为不存在的路径凭空建一个空库文件，
         CI 上就会留下一个假的 data/usagent.db；
      2) `days_ago` 只是自然日差，**不判"是不是交易日"**（周末差 2 天完全正常）——
         所以界面只在差得离谱时才提示，而不是把每个周末都报成陈旧。
    """
    try:
        import db as dbmod
    except Exception:                                  # noqa: BLE001 —— 没有 db 模块就静默跳过
        return None
    try:
        import sqlite3
        import datetime
        p = Path(dbmod.DB_PATH)
        if not p.is_file():
            return None
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as con:
            row = con.execute("SELECT MAX(last_date), COUNT(*) FROM coverage "
                              "WHERE kind='ohlcv'").fetchone()
        last = row[0] if row else None
        if not last:
            return None
        d = datetime.date.fromisoformat(str(last)[:10])
        return {"last_date": d.isoformat(), "symbols": int(row[1] or 0),
                "days_ago": (datetime.date.today() - d).days}
    except Exception:                                  # noqa: BLE001 —— 只读装饰，出错就别显示
        return None


def _plain_mechanism(line: str) -> str:
    """把掩码 docstring 的首行清成"能直接印在页面上"的纯文本（去前缀、去强调标记）。

    为什么在**数据层**做而不是在页面渲染时做：这一行是"机制说明"，它的读者是页面；
    而页面拿到的 catalog 本来就是"已编译好的展示形态"（这一层的职责）。放在页面里做意味着
    每个渲染点都要记得清一次 —— 漏一处就是又一次"星号印在脸上"。
    """
    import re as _re
    t = (line or "").strip()
    t = _re.sub(r"^族\s*\d*\s*[:：]\s*", "", t)          # 「族:」/「族 1:」前缀
    t = _re.sub(r"\*\*(.+?)\*\*", r"\1", t)             # **强调** → 强调
    t = _re.sub(r"`([^`]+)`", r"\1", t)                    # `code` → code
    return t.strip()


def holdout_ledger_state(runs: list[dict] | None = None) -> dict:
    """仓库**权威** holdout 台账的状态（`research/results/holdout_ledger.jsonl`）。

    为什么页面上必须有它：卡片写着「与 holdout 重叠（只准读一次）」，而页面上此前只有**平台自己
    数的**那个数（`web_runs.jsonl` 里数窗口）。可"只准读一次"这条规矩真正被登记的地方是这本
    台账（`wfv_search` / `family_search` 通过 `holdout.consume` 写它），而**平台运行从不登记
    进去**（`webapp/jobs.py` 不调 `consume`）。两本账不并排显示，读者就会把平台那个数误当成
    "纪律账本上的次数"。

    三类事实缺一不可：① 权威账本里有几条 claim（真读过 holdout 的）、几条 lock（只登记不烧窗口）；
    ② **规范窗口自己被 claim 过没有**；③ 平台的越界运行里有几条以 claim 身份登记进去了。
    """
    try:
        from research import holdout as H
        hist = H.history()
    except Exception as exc:                     # noqa: BLE001
        # 台账坏了要说出来（H 自己就是抛 LedgerCorrupt 的态度："坏了必须修，不能跳过"）
        return {"error": f"{type(exc).__name__}: {exc}"}
    claims = [r for r in hist if r.get("kind") == H.KIND_CLAIM]
    out = {
        "claims": len(claims),
        "locks": len(hist) - len(claims),
        "windows": H.n_windows(),
        "claimed_windows": H.claimed_windows(),
        "current_window_claimed": f"{HOLDOUT[0]}..{HOLDOUT[1]}" in set(H.claimed_windows()),
        "last_claim": ({k: claims[-1].get(k) for k in ("ts", "script", "tag", "window")}
                       if claims else None),
    }
    if runs is not None:
        claimed = set(out["claimed_windows"])
        crossers, registered = 0, 0
        for r in runs or []:
            w = r.get("window")
            if not (isinstance(w, (list, tuple)) and len(w) == 2):
                continue
            if touches_holdout((w[0], w[1])):
                crossers += 1
                if f"{w[0]}..{w[1]}" in claimed:
                    registered += 1
        out["platform_crossers"] = crossers
        out["platform_registered"] = registered
    return out


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
        if isinstance(w, (list, tuple)) and len(w) == 2 and touches_holdout((w[0], w[1])):
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
