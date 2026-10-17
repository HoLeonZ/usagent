"""运行观测、告警出口与**决策凭证** —— 最小可用的一层。

为什么需要它（2026-10-17 机构就绪审计的第 ④ 项，实测证据）：
`logs/` 是空目录；`run_daily.sh` 失败只 echo 到 stdout；**没有任何步骤级的运行台账**，
也**没有任何投递目标**（默认部署下告警等于不告警）。于是"跑完了"与"跑失败了"在一周后
无法区分。
⚠️ 更正一处我写错的动机（2026-10-17 对抗核验）：这里原先写「`grep -rniE 监控|告警|值班`
全仓 0 命中」——**是假的**，改动前的父提交上命中 13 行 / 7 个文件（`README.md`、
`validate.py`、`download_ohlcv.py` 里都在谈"接告警"）。真实的缺口不是"没人提过告警"，
而是"**有告警的想法、没有告警的机制**"。把它写成"零命中"是拿一个不成立的事实支撑结论 ——
本仓库正是为根除这种写法才建的台账。

这一层只做三件事，三件都是线上化的前置：
  1. **运行台账**：每步一行 JSONL（时间 / git sha / 阶段 / 成败 / 耗时 / 关键指标）；
  2. **告警**：失败或指标越界 ⇒ 另写一行 `alerts.jsonl`，并**可选**投递到一个 webhook
     （地址从环境变量读，**不落仓库**：密钥与地址都不进版本库）；
  3. **凭证**：任何决策产物都能挂同一份 `manifest()`（ts + git sha + 数据 vintage），
     于是"这份数字是哪份代码 + 哪份数据算的"可回答（§0/§7）。

边界（不夸大）：**这不是监控系统**。没有指标聚合、没有值班轮转、没有自动降级/退役 ——
那些属于生产层（审计第 6–8 步）。这里只保证"发生过什么有据可查"。
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
RUN_LOG = LOGS / "runs.jsonl"
ALERT_LOG = LOGS / "alerts.jsonl"

# 告警投递地址从环境变量读：仓库里不放任何地址或密钥（§7 不留外部耦合）。
WEBHOOK_ENV = "USAGENT_ALERT_WEBHOOK"
WEBHOOK_TIMEOUT_S = 5.0


def now_iso() -> str:
    """本地时间 + 时区偏移（ISO8601）。带偏移是因为"什么时候跑的"必须无歧义：
    这条链的入场价钉在美东 T+1 开盘（审计：时钟/时区是承重不变量），
    时间戳少了偏移，日后连"这次运行是否赶上开盘前"都判不了。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _clean_git_env() -> dict:
    """去掉继承来的 GIT_*：git 只按 cwd（= 本模块所在仓库）发现仓库。

    为什么必须有（2026-10-17 对抗核验抓到的**破坏性**回归）：linked worktree 里 git 会向
    子进程导出 `GIT_DIR=<主仓>/.git/worktrees/<name>`。带着它跑 `git init`（测试里会）能
    顺着 commondir 写回**共享**的 `.git/config`（core.bare=true），让主工作区不可用 ——
    三个独立副本上都复现过。观测层问的永远是"本模块所在仓库"，所以环境必须清干净。
    """
    return {k: v for k, v in os.environ.items()
            if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                         "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES")}


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                             timeout=10, env=_clean_git_env())
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def git_sha(short: bool = True) -> str:
    """当前 HEAD 的 sha；不在 git 工作区或没有 git 时返回 "unknown"（不抛）。"""
    sha = _git("rev-parse", "--short" if short else "--verify", "HEAD")
    return sha or "unknown"


def git_dirty() -> bool | None:
    """工作区是否有未提交改动。None = 判不出来（不是 git 仓库）。

    为什么需要它：只有 sha 答不出"跑的是不是这份代码" —— 脏工作区上 `rev-parse HEAD`
    照样返回同一个 sha（对抗核验实测：往被测文件里追一行注释，sha 不变）。
    "哪份代码算的"这句话在脏树上不成立，所以要把脏这件事**记下来**。
    """
    if not _git("rev-parse", "--git-dir"):
        return None                       # 不是 git 仓库 ⇒ 答不出，就不说"干净"
    return bool(_git("status", "--porcelain"))


def manifest(**extra) -> dict:
    """一份决策产物的凭证头。字段刻意少而硬：时间、代码版本、解释器、解释器版本。"""
    m = {"ts": now_iso(), "git_sha": git_sha(), "git_dirty": git_dirty(),
         "python": sys.version.split()[0], "platform": platform.system()}
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def _append(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def log_run(stage: str, ok: bool, *, seconds: float | None = None,
            exit_code: int | None = None, detail: str = "", **fields) -> Path:
    """记一行运行台账。**任何调用者都不要吞这里的异常**？不 —— 这里是观测层，
    观测不能把主流程弄挂：写失败只在 stderr 留一句（见 `_safe`）。"""
    rec = {"kind": "run", "stage": stage, "ok": bool(ok), "ts": now_iso(),
           "git_sha": git_sha(), "git_dirty": git_dirty()}
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 3)
    if exit_code is not None:
        rec["exit_code"] = int(exit_code)
    if detail:
        rec["detail"] = detail[:2000]
    rec.update({k: v for k, v in fields.items() if v is not None})
    return _safe(_append, RUN_LOG, rec)


def alert(level: str, title: str, detail: str = "", **fields) -> Path | None:
    """写一行告警，并尝试投递（未配置 webhook 就只落盘 + stderr）。

    `level` ∈ {"warn","error"}。返回落盘路径（写失败返回 None）—— 观测层不抛。
    """
    rec = {"kind": "alert", "level": level, "title": title, "detail": detail[:4000],
           "ts": now_iso(), "git_sha": git_sha(), "git_dirty": git_dirty()}
    rec.update({k: v for k, v in fields.items() if v is not None})
    path = _safe(_append, ALERT_LOG, rec)
    _safe(_stderr, f"[ALERT/{level}] {title} {detail}")
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    if url:
        # 投递失败必须与"落盘失败"区分开（否则台账答不出"这条告警有没有送到人"）
        if _safe(_post, url, rec) is None:
            _safe(_append, ALERT_LOG, {**rec, "kind": "alert_delivery_failed",
                                       "ts": now_iso()})
    return path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _post(url: str, rec: dict) -> None:
    body = json.dumps(rec, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_S).close()


def _safe(fn, *args):
    """观测层的所有副作用都从这里过：失败只留痕，不冒泡。"""
    try:
        return fn(*args)
    except Exception as exc:                                        # noqa: BLE001
        # 连 stderr 也可能写不出去（磁盘满 / 管道断）—— 那一层再兜一次，
        # 因为"观测把主流程弄挂"是这一层最不能犯的错（对抗核验点名要查这条）。
        try:
            print(f"[obs] 观测写入失败（不影响主流程）: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        except Exception:                                           # noqa: BLE001
            pass
        return None


def main() -> int:
    """`python3 src/obs.py` —— 打印最近若干行台账/告警，便于人工扫一眼。"""
    import argparse
    ap = argparse.ArgumentParser(description="运行台账 / 告警查看")
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--alerts", action="store_true", help="只看告警")
    a = ap.parse_args()
    path = ALERT_LOG if a.alerts else RUN_LOG
    if not path.exists():
        print(f"（还没有 {path.relative_to(REPO)}）")
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()[-a.tail:]
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
