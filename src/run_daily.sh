#!/usr/bin/env bash
# usagent 每日数据任务 —— **只采集标的数据**。
#
# 2026-10-14:本脚本原含「期权链快照」与「IV 反解」两步,已移除。
# 原因:只需要标的数据做分析、手动下单,不需要期权链。
#
# 移除的连带效果(三条都是好事):
#   1. **运行时刻不再有约束。** 原先整个脚本是为期权链设的 —— Yahoo 在美东
#      深夜会把期权报价清零,所以只能在 ET 16:15-23:45(北京 05:15-11:45)跑。
#      OHLCV 没有这个问题:某一天的收盘价一旦形成就不再变,补抓随时可以。
#   2. **cron 的星期字段不再是"每周丢一天"。** 原先 `1-5` 覆盖不到美东周五
#      那一场,而期权链错过就永久消失;现在 OHLCV 用 `--recent 10`,漏掉的
#      那天下次运行会自动补回来。
#   3. 单轮耗时从 ~110 分钟降到 ~7 分钟。
#
# 若将来要重新采集期权链:把 config.COLLECT_OPTIONS 置 True,并在此加回
#   run "期权链快照" $PY -u src/snapshot_options.py --universe oi100
#   run "IV 反解"    $PY -u src/derive_iv.py
# 注意届时运行时刻约束会重新生效。
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

# 1) 价格:增量,只拉最近 10 天。
#    `--recent 10` 兼作补抓:漏跑一天不会在库里留下空洞(不会覆盖更早的行,
#    因为 --no-skip 只影响"是否跳过已覆盖标的",不影响写入区间)。
#    ⚠️ 但"只重写最近 10 天"本身是个陷阱:厂商(拆股后)会把更早的历史**整段重述**,
#    于是更早的行会永久停在旧基数、在窗口左边界留下假单日跌幅。所以下载器内置了
#    重述检测:增量窗口里发现同一 (symbol,date) 的 close 变了 ⇒ 该标的**整段重下重写**
#    (download_ohlcv.write_symbol / repair_symbol,阈值见 config.RESTATE_MIN_REL_DIFF)。
run "OHLCV 增量" $PY -u src/download_ohlcv.py --universe oi100 --recent 10 --no-skip

# 2) 校验:任何结构性断言失败都会非零退出,便于接告警
run "数据校验" $PY -u src/validate.py

if [ -n "${FAILED:-}" ]; then
  echo "===== $(date '+%F %T') 结束（有失败:$FAILED）====="
  exit 1
fi
echo "===== $(date '+%F %T') 结束（全部通过）====="
