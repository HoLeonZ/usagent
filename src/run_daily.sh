#!/usr/bin/env bash
# usagent 每日数据任务。
#
# ⚠️ 运行时刻是这个数据层唯一真正会「做错」的地方，先读完这段。
#
# Yahoo 在美东深夜会把期权报价**整体清零**。实测（2026-10-14）：
#   美东 23:48  AAPL/NVDA 平值档 bid/ask 正常
#   美东 00:51  全变成 0.0 / 0.0，IV 变 0.0000，一直持续到次日开盘前
# 在这个窗口抓，会得到 2 万行 bid=0 的废数据 —— 表面"抓成功"，实际反解不出任何 IV。
# （本项目第一次跑快照就踩了这个坑，整个 10-13 那天的期权链报废且无法回补。）
#
# 有效窗口 = 美股收盘(16:15 ET) → 约 23:45 ET：
#     北京 04:15–11:45（夏令时） / 05:15–12:45（冬令时）
#   两者交集 = 北京 05:15–11:45，全年可用。
#
# 所以下面默认排在北京 10:00 —— 它同时在两个 DST 制式下都落在窗口内，
# 而且是白天，不需要机器通宵开机。（原先把默认设成 21:10 是错的：那对应
# 美东 08:10/09:10 开盘前，正落在清零窗口里。）
#
# snapshot_options.py 里已加飞行前检查：探针发现报价清零会直接拒绝写入并返回 3，
# 所以就算 cron 时刻设错，也是响亮报错而不是静默落垃圾。
#
# 用法：
#   bash src/run_daily.sh
#   crontab（机器时区需为北京）:  0 10 * * 1-5  cd /Users/holeon/code/usagent && bash src/run_daily.sh >> logs/daily.log 2>&1

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python
mkdir -p logs
STAMP=$(date "+%Y-%m-%d %H:%M:%S%z")

echo "===== $STAMP 开始 ====="
echo "美东时间: $(TZ=America/New_York date '+%F %H:%M')"

run() {                       # run <名称> <命令...>
  local name="$1"; shift
  echo "--- $name"
  if "$@"; then
    echo "--- $name 完成"
  else
    local rc=$?
    echo "!!! $name 失败 (exit $rc) —— 继续后续步骤"
    FAILED="${FAILED:-} $name"
  fi
}

# 1) 价格：增量，只拉最近 10 天（约 1 分钟；全量回补第一次手动跑 --universe oi100）
run "OHLCV 增量" $PY -u src/download_ohlcv.py --universe oi100 --recent 10 --no-skip

# 2) 期权链快照：唯一"不落盘就永久丢失"的一步，放第二位保证它一定被执行
run "期权链快照" $PY -u src/snapshot_options.py --universe oi100

# 3) 衍生：IV30 / HV20 / IV Rank
run "IV 反解" $PY -u src/derive_iv.py

# 4) 校验：任何结构性断言失败都会非零退出，便于接告警
run "数据校验" $PY -u src/validate.py

if [ -n "${FAILED:-}" ]; then
  echo "===== $(date '+%F %T') 结束（有失败:$FAILED）====="
  exit 1
fi
echo "===== $(date '+%F %T') 结束（全部通过）====="
