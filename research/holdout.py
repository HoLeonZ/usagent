"""一次性 holdout 的**消费台账** —— 把「这段窗口没参与过选择」从承诺变成机器可验证的事实。

为什么需要这一层（2026-10-17 审计 B1，CRITICAL 级）:

hiagent CLAUDE.md §5 原话: *Final claims MUST be validated on a holdout window that
participated in no iteration.* 但本仓库里 holdout 只是 `wfv_search.report()` 中的一段
**无条件代码** —— `python -m research.wfv_search --run` 跑一次就再消费一次，
不留任何痕迹：

  * `EVAL_COUNT`（wfv_search.py）是 **import 归零的进程内计数器**，不落盘、
    不区分 holdout 与搜索期，`report()` 的返回值在 `main()` 里被丢弃；
  * `factor_probe --mid 2025-10-14` 又把**同一条分界线**当作判据②的设计参数；
  * 于是「该窗口是样本外」只能靠文档和注释维持 —— `family_search.py` 里就写着
    「holdout 已在上一轮用过一次」，但那是**人写的注释**，不是机器能验证的状态。

本模块把消费做成**追加写、按窗口记账、默认拒绝二次消费**的台账：

  * 台账落在 `research/results/`（§7:run artifact，不入库）；
  * 同一个 `(lo, hi)` 窗口第二次被当作**claim**消费 ⇒ 抛 `HoldoutBurned`；
    要继续必须给出显式理由（`force_reason`），而理由本身也会被记账 ——
    于是「我又看了一次 holdout」从不可能发生的事变成一条可审计的记录；
  * `kind="lock"`（行为锁、golden 基线、单元测试）只记账、不烧窗口：
    重跑一次 golden 不是为了拿 holdout 出结论，把它们算成消费会让台账失去意义；
  * 台账同时是 `n_trials` 的一个事实来源:`n_windows()` 给出「一共动过几个评估窗口」，
    §5 明确要求**改动评估窗口本身**也计入迭代次数。

台账损坏一律 fail-fast：静默跳过坏行会让计数偏小，那正是这套台账要消灭的失败模式。
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LEDGER_PATH = RESULTS_DIR / "holdout_ledger.jsonl"

# 消费类型。只有 claim 会「烧掉」窗口；lock 只登记。
KIND_CLAIM = "claim"     # 拿这个窗口的数字去支持一个结论
KIND_LOCK = "lock"       # 行为锁 / golden 基线 / 单元测试：登记但不烧窗口
KINDS = (KIND_CLAIM, KIND_LOCK)


class HoldoutBurned(RuntimeError):
    """这段窗口已经被当作 claim 消费过 —— 它不再是样本外。"""


class LedgerCorrupt(RuntimeError):
    """台账里有读不懂的行。不静默跳过：静默跳过会让计数偏小。"""


def window_key(lo: str, hi: str) -> str:
    """窗口的规范键。用日期字符串而不是 Timestamp，好让台账可读、可 diff。"""
    a, b = str(lo)[:10], str(hi)[:10]
    if a >= b:
        raise ValueError(f"窗口必须 lo < hi，得到 {a}..{b}")
    return f"{a}..{b}"


def _git_rev() -> str | None:
    """当前代码版本（短 sha）。拿不到就返回 None —— 台账不该因为缺 git 而写不进去。"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 —— 版本信息是尽力而为
        return None


def history(lo: str | None = None, hi: str | None = None,
            path: Path | None = None) -> list[dict]:
    """台账里的全部条目（可按窗口过滤），按写入顺序。"""
    p = path or LEDGER_PATH
    if not p.exists():
        return []
    key = window_key(lo, hi) if (lo and hi) else None
    out: list[dict] = []
    for i, line in enumerate(p.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerCorrupt(f"{p}:{i} 不是合法 JSON —— 台账坏了必须修，不能跳过") from exc
        if not isinstance(rec, dict) or "window" not in rec or "kind" not in rec:
            raise LedgerCorrupt(f"{p}:{i} 缺 window/kind 字段：{line[:80]!r}")
        if key is None or rec["window"] == key:
            out.append(rec)
    return out


def n_evaluations(path: Path | None = None) -> int:
    """台账里登记过的评估总次数（claim + lock）。"""
    return len(history(path=path))


def n_windows(path: Path | None = None) -> int:
    """一共动过几个**不同的**评估窗口 —— §5 要求窗口改动计入 n_trials。"""
    return len({r["window"] for r in history(path=path)})


def claimed_windows(path: Path | None = None) -> list[str]:
    """被当作 claim 消费过的窗口（= 已经不再是样本外的那些）。"""
    return sorted({r["window"] for r in history(path=path) if r["kind"] == KIND_CLAIM})


def consume(lo: str, hi: str, *, script: str, tag: str,
            kind: str = KIND_CLAIM, force_reason: str | None = None,
            metrics: dict | None = None, now: str | None = None,
            path: Path | None = None) -> int:
    """登记一次 holdout 消费。返回**本次之前**该窗口被 claim 消费的次数。

    行为：
      * `kind="claim"` 且该窗口此前已被 claim 消费过：
          - 没给 `force_reason` ⇒ 抛 `HoldoutBurned`（默认拒绝，而不是打条日志继续跑）；
          - 给了 `force_reason` ⇒ 放行，并把理由写进台账。
      * `kind="lock"` ⇒ 只登记，永不抛错（重跑 golden 不是拿 holdout 出结论）。
    """
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一，得到 {kind!r}")
    key = window_key(lo, hi)
    prior = history(lo, hi, path=path)
    n_prior_claims = sum(1 for r in prior if r["kind"] == KIND_CLAIM)

    if kind == KIND_CLAIM and n_prior_claims > 0 and not force_reason:
        last = [r for r in prior if r["kind"] == KIND_CLAIM][-1]
        raise HoldoutBurned(
            f"窗口 {key} 已被 claim 消费 {n_prior_claims} 次"
            f"（最近一次：{last.get('ts', '?')} / {last.get('script', '?')} / "
            f"{last.get('tag', '?')}）—— 它**不再是样本外**。\n"
            f"§5 要求最终结论落在未参与任何迭代的窗口上。要继续只有两条正路：\n"
            f"  1) 换一段从未消费过的窗口；或\n"
            f"  2) 显式声明「我知道它已被消费」并给出理由："
            f"--force-holdout \"<理由>\"（理由会写进台账，可审计）。"
        )

    rec = {
        "ts": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": key,
        "kind": kind,
        "script": script,
        "tag": tag,
        "git": _git_rev(),
        "force_reason": force_reason,
        "metrics": metrics or {},
    }
    p = path or LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:          # 追加写：台账只增不改
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("holdout 台账：登记 %s %s（此前 claim %d 次）", kind, key, n_prior_claims)
    return n_prior_claims
