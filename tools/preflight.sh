#!/usr/bin/env bash
# 提交前预检（preflight）—— **快的那一组**，目的只有一个：让"红着提交"这件事不发生。
#
# 为什么需要它（2026-10-17 机构就绪审计）：CI 早已存在，但**没有任何东西强制它**——
# 没有 pre-commit、没有分支保护，本轮 90 个提交未推未评审。实测付出过代价一次：
# `research/preset_ranking.py` 写死了本机绝对路径，而"596 passed"是在它 `git add`
# **之前**跑的 ⇒ 提交后的状态从没被扫过，仓库红着进了历史。
#
# 它跑什么（目标 < 5 秒，否则没人会留它）：
#   1. 仓库卫生（机器相关绝对路径 / 写死解释器 / 依赖钉版）—— 上面那次事故的守卫；
#   2. 装载契约（preset / family 注册表）与行为锁覆盖判据；
#   3. **离线行为锁**（合成市场，1 秒左右，任何机器可跑）。
# 真数据行为锁（`test_golden_trades`）不进这里：它要 277MB 的库、跑 4-5 分钟，
# 属于 CI 与显式回归的那一档（`--full`）。
#
# 用法：
#   bash tools/preflight.sh          # 快检（pre-commit 用的就是它）
#   bash tools/preflight.sh --full   # 再跑全量（research/tests + src/tests）
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python3}"
FAST=(-q
  src/tests/test_repo_hygiene.py
  src/tests/test_daily.py
  src/tests/test_obs.py
  src/tests/test_validate_report.py
  research/tests/test_preset_registry.py
  research/tests/test_family_registry.py
  research/tests/test_lock_coverage.py
  research/tests/test_golden_fixture_lock.py)

echo "[preflight] 快检（卫生 / 契约 / 覆盖判据 / 离线行为锁）"
"$PY" -m pytest "${FAST[@]}" || { echo "[preflight] ✗ 快检未过 —— 不要提交"; exit 1; }

if [ "${1:-}" = "--full" ]; then
  echo "[preflight] 全量（research/tests + src/tests）"
  "$PY" -m pytest research/tests src/tests -q || { echo "[preflight] ✗ 全量未过"; exit 1; }
fi
echo "[preflight] ✓ 通过"
