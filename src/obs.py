"""运行观测、告警出口与**决策凭证** —— 最小可用的一层。

为什么需要它（2026-10-17 机构就绪审计的第 ④ 项，实测证据）：
`logs/` 是空目录；`run_daily.sh` 失败只 echo 到 stdout；`grep -rniE 监控|告警|值班` 全仓
0 命中。于是"跑完了"与"跑失败了"在一周后无法区分，而**任何一个"今天的候选清单"都没有
时间戳 / git sha / 数据 vintage**（`research/screen.py` 只 `print()`）⇒ 从信号到下单这一段
没有任何可对账的凭证。

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


def git_sha(short: bool = True) -> str:
    """当前 HEAD 的 sha；不在 git 工作区或没有 git 时返回 "unknown"（不抛）。"""
    args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    if not short:
        args = ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def manifest(**extra) -> dict:
    """一份决策产物的凭证头。字段刻意少而硬：时间、代码版本、解释器、解释器版本。"""
    m = {"ts": now_iso(), "git_sha": git_sha(),
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
           "git_sha": git_sha()}
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
           "ts": now_iso(), "git_sha": git_sha()}
    rec.update({k: v for k, v in fields.items() if v is not None})
    path = _safe(_append, ALERT_LOG, rec)
    print(f"[ALERT/{level}] {title} {detail}", file=sys.stderr)
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    if url:
        _safe(_post, url, rec)
    return path


def _post(url: str, rec: dict) -> None:
    body = json.dumps(rec, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_S).close()


def _safe(fn, *args):
    """观测层的所有副作用都从这里过：失败只留痕，不冒泡。"""
    try:
        return fn(*args)
    except (OSError, ValueError, urllib.error.URLError) as exc:      # noqa: BLE001
        print(f"[obs] 观测写入失败（不影响主流程）: {type(exc).__name__}: {exc}",
              file=sys.stderr)
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
