"""每日任务驱动器 —— 把「刷数据」从 shell 字符串挪进可测的 Python。

为什么要换掉 shell（2026-10-17 机构就绪审计，逐条都有证据）：
  1. `run_daily.sh` 只刷 `--universe oi100`（596 只，且那是台账自己因幸存者/前视弃用的
     "今日期权 OI"静态名单），而**引擎与面板读全库 3028 只** ⇒ 第二天起多数可交易名字
     是陈价；更糟的是"数据新鲜"判的是全库 MAX(date)（由永远在更新的 596 只决定）
     ⇒ **判据看不见这件事**。
  2. 失败只 echo 到 stdout，"跑完了/跑失败了"事后无法区分（`logs/` 是空目录）。
  3. 池子与数据源散在 shell 里，没有单一来源，也没有任何机器可读的结论。

现在：**池子/源/增量天数在 `src/config.py` 声明（单一来源）**；每一步进 `logs/runs.jsonl`；
失败与越界进 `logs/alerts.jsonl`（可选投递 webhook，地址只从环境变量读）；
退出码仍是"任一步失败即非零"，并额外留一行带失败项名字的告警。

成本（实测 2026-10-17，不是估）：全池 3,030 只、10 天窗口，单只中位 **1.25s**
（含 0.3s 节流）⇒ 串行约 **63 分钟**。延迟由每请求开销/节流主导，**与窗口大小几乎无关**
（3 年全历史单只 1.30s）。cron 定在北京 10:00，而链路的入场价钉在美东 T+1 开盘
（北京 21:30）⇒ **余量约 11 小时**，串行可接受。
下一步优化（单独一轮，因为它要动数据层的取数路径）：给 `download_ohlcv` 加 `--workers N`
（6 并发实测 ≈ 11 分钟）。`--no-skip` 在增量模式下是**语义上的空操作**（代码里写着
"增量模式必须刷所有标的" —— 重述检测依赖重叠窗口），保留它是为了把意图写在命令行上。

保留的语义（有意为之，不是漏改）：**某一步失败后继续跑后续步骤**。理由：取数失败时
仍要跑校验，才能知道"是取数没跑成，还是库真的坏了"—— 那正是运维要区分的第一件事。
`set -e` 会把这个信息抹掉。所以这里"继续 + 记清楚 + 最后非零退出 + 告警"。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import obs
from config import DAILY_RECENT_DAYS, DAILY_SOURCE, DAILY_UNIVERSE

REPO = Path(__file__).resolve().parents[1]
CHECKS_JSON = REPO / "logs" / "checks.json"


def plan(python: str) -> list[dict]:
    """今日要跑什么 —— 计划是**数据**（可打印、可测试），不是一串 shell 拼接。

    每一步带 `artifacts`：它产出的机器可读文件（供台账取指标，缺了不算失败）。
    """
    return [
        {"stage": "OHLCV 增量",
         "argv": [python, "-u", "src/download_ohlcv.py",
                  "--universe", DAILY_UNIVERSE, "--source", DAILY_SOURCE,
                  "--recent", str(DAILY_RECENT_DAYS), "--no-skip"],
         "artifacts": []},
        {"stage": "数据校验",
         "argv": [python, "-u", "src/validate.py",
                  "--universe", DAILY_UNIVERSE, "--json", str(CHECKS_JSON)],
         "artifacts": [CHECKS_JSON]},
    ]


def _tail(text: str, n: int = 12) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _metrics_from_checks(path: Path) -> dict | None:
    """校验步的机器可读结论 → 台账指标。读不到就返回 None（不猜、不算失败）。"""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict) or "passed" not in d:
        return None
    return {"checks_passed": d.get("passed"), "checks_total": d.get("total"),
            "checks_failed": d.get("failed", [])[:8], "universe": d.get("universe")}


def run_step(step: dict, *, python: str, verbose: bool) -> bool:
    """跑一步，记台账；失败则告警。返回是否成功。"""
    name, argv = step["stage"], step["argv"]
    print(f"--- {name}")
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    except OSError as exc:                       # 解释器/脚本不存在
        dt = time.time() - t0
        obs.log_run(name, False, seconds=dt, detail=f"{type(exc).__name__}: {exc}",
                    argv=argv)
        obs.alert("error", f"{name} 无法启动", f"{type(exc).__name__}: {exc}")
        print(f"!!! {name} 无法启动：{exc}", file=sys.stderr)
        return False
    dt = time.time() - t0
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if verbose or p.returncode != 0:
        print(out.rstrip())
    ok = p.returncode == 0
    fields = {"exit_code": p.returncode, "argv": argv}
    for art in step.get("artifacts", []):
        m = _metrics_from_checks(art)
        if m:
            fields.update(m)
    obs.log_run(name, ok, seconds=dt, detail=_tail(out), **fields)
    if ok:
        print(f"--- {name} 完成（{dt:.1f}s）")
    else:
        obs.alert("error", f"{name} 失败（exit {p.returncode}）", _tail(out),
                  stage=name, argv=argv)
        print(f"!!! {name} 失败 (exit {p.returncode}) —— 继续后续步骤（见本文件 docstring）",
              file=sys.stderr)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="每日数据任务（池子/源/观测一处定义）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行、不写台账")
    ap.add_argument("--python", default=sys.executable,
                    help="跑子步骤用的解释器（默认=当前解释器，避免挑到另一个环境）")
    ap.add_argument("--quiet", action="store_true", help="成功步骤不打印其输出")
    a = ap.parse_args(argv)

    steps = plan(a.python)
    print(f"===== {obs.now_iso()} 每日任务：池 {DAILY_UNIVERSE} / 源 {DAILY_SOURCE} "
          f"/ 增量 {DAILY_RECENT_DAYS} 天 / git {obs.git_sha()} =====")
    if a.dry_run:
        for s in steps:
            print("  " + " ".join(s["argv"]))
        return 0

    t0 = time.time()
    failed: list[str] = []
    for s in steps:
        if not run_step(s, python=a.python, verbose=not a.quiet):
            failed.append(s["stage"])
    dt = time.time() - t0

    ok = not failed
    obs.log_run("每日任务", ok, seconds=dt, stages=len(steps), failed=failed)
    if ok:
        print(f"===== {obs.now_iso()} 结束（全部通过，{dt:.1f}s）=====")
        return 0
    obs.alert("error", "每日任务有失败步骤", f"失败：{failed}；耗时 {dt:.1f}s", failed=failed)
    print(f"===== {obs.now_iso()} 结束（有失败：{failed}）=====", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
