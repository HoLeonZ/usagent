#!/usr/bin/env bash
# 安装本地 git 钩子 —— 让"提交前跑一次快检"这件事不依赖记性。
#
# 为什么用安装器而不是直接放 `.git/hooks/pre-commit`：`.git/` 不入库，钩子必须**可复制**，
# 所以真正的逻辑在 `tools/preflight.sh`（入库、可评审），钩子只是调用它的一行。
#
# 用法：
#   bash tools/install_git_hooks.sh          # 安装/覆盖 pre-commit
#   bash tools/install_git_hooks.sh --remove # 卸载
#
# 注意（诚实边界）：本地钩子可以被 `--no-verify` 绕过，也管不到别人克隆的仓库。
# 真正的强制在远端：**分支保护 + CI 必过**（这是仓库设置，不是代码 —— 见 README「提交前」一节）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOOK="${GITDIR}/hooks/pre-commit"

# ⚠️ 不能判「.git 是不是目录」：linked worktree 里 .git 是**文件**（对抗核验实测
# 在此处 exit 1）。用 git 自己回答"这是不是一个工作区"。
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "不是 git 工作区，无法安装钩子"; exit 1
fi
GITDIR=$(git rev-parse --git-dir)

if [ "${1:-}" = "--remove" ]; then
  rm -f "$HOOK" && echo "已移除 $HOOK"
  exit 0
fi

mkdir -p "${GITDIR}/hooks"
cat > "$HOOK" <<'HOOK_BODY'
#!/usr/bin/env bash
# 由 tools/install_git_hooks.sh 生成 —— 真正的逻辑在 tools/preflight.sh（可评审、入库）。
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
exec bash tools/preflight.sh
HOOK_BODY
chmod +x "$HOOK"
echo "已安装 ${HOOK}（调用 tools/preflight.sh；用 git commit --no-verify 可临时绕过）"
