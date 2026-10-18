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
FAST=(-q
  src/tests/test_repo_hygiene.py
  src/tests/test_daily.py
  src/tests/test_obs.py
  src/tests/test_validate_report.py
  research/tests/test_preset_registry.py
  research/tests/test_family_registry.py
  research/tests/test_lock_coverage.py
  research/tests/test_golden_fixture_lock.py)

# 解释器解析（2026-10-17 教训 + 2026-10-18 补）：仓库把依赖钉在 requirements.txt，但
# **没有任何东西**保证 shell 里的 `python3` 就是装过它们的那个 —— PATH 一变它可能落到
# 系统 python，于是报错是 "No module named 'numpy'"，而真正该说的是"你用错了解释器"。
#
# 为什么不能只在 `$PY` 不对时报错（2026-10-18 实测）：pre-commit 钩子**不继承**你 shell 里
# 的 `PY=...`（那是一次性的命令前缀），于是本机每次提交都被挡下 —— 而"被一个永远挡人的
# 钩子挡住"的正确反应会退化成 `--no-verify`，那等于把全部护栏一起关掉，比没有钩子更糟。
# 本机实测：`which -a python3` 第一个是 /usr/bin/python3（无 numpy），第二个才有。
# 所以按 $PY → PATH 上每一个 python3 的顺序找**能 import numpy/pandas** 的那个，
# 并把选中的解释器**打印出来** —— 换解释器这件事不静默。
_PICKED=""
for _cand in ${PY:-} $(type -a -p python3 2>/dev/null) python3; do
  [ -n "$_cand" ] || continue
  if "$_cand" -c "import numpy, pandas" >/dev/null 2>&1; then _PICKED="$_cand"; break; fi
done
if [ -z "$_PICKED" ]; then
  echo "[preflight] ✗ PATH 上没有任何 python3 能 import numpy/pandas —— 依赖钉在 requirements.txt"
  echo "           试过：$(type -a -p python3 2>/dev/null | tr '\n' ' ')"
  echo "           用 PY=<venv>/bin/python3 bash tools/preflight.sh 显式指定"
  exit 1
fi
PY="$_PICKED"
echo "[preflight] 解释器：$("$PY" -c 'import sys;print(sys.executable)')"

echo "[preflight] 快检（卫生 / 契约 / 覆盖判据 / 离线行为锁）"
"$PY" -m pytest "${FAST[@]}" || { echo "[preflight] ✗ 快检未过 —— 不要提交"; exit 1; }

if [ "${1:-}" = "--full" ]; then
  echo "[preflight] 全量（research/tests + src/tests + webapp/tests）"
  "$PY" -m pytest research/tests src/tests webapp/tests -q \
    || { echo "[preflight] ✗ 全量未过"; exit 1; }
fi
echo "[preflight] ✓ 通过"
