"""执行成本与流动性常数 —— 全仓**唯一**定义(CLAUDE.md §2(b))。

不许在任何引擎、preset、脚本里再写一遍这些字面量;research/tests/test_contract.py
用源码 grep 把这条钉住(它会让「又写了一遍」的改动红掉)。
"""
from __future__ import annotations

# 双边总成本(bp)。美股无印花税:IBKR Lite 佣金 $0 + SEC §31(卖出,≈0.278bp)
# + FINRA TAF($0.000166/股,上限 $8.30)。回测按双边 5bp 折算。
COST_BPS_ROUNDTRIP: float = 5.0

# 单根 bar 的成交量参与率上限(CLAUDE.md §4(a)):
#     Max_Fill_Qty = MIN(Target_Qty, Bar_Volume × 0.10)
MAX_VOLUME_PARTICIPATION: float = 0.10

# 执行滑点系数(CLAUDE.md §4(c) 要求滑点必须是 f(ATR, 参与率)):
#     slip = scale × 信号日 atr_pct × sqrt(该单股数 / 本 bar 成交量)
# 平方根冲击律。scale=1.0 取 Almgren 类经验区间 [0.5, 1] 的**保守端**
# (即「吃掉整根 bar 的量」付一个 ATR)。系数先在机制上定死、再看结果 —— 不是调参。
ATR_SLIP_SCALE: float = 1.0

# 回测初始名义权益($)。只用于把收益率归一化,不影响结构性结论。
INITIAL_CASH: float = 100_000.0


def half_rate(cost_bps_roundtrip: float) -> float:
    """双边 bp → **单边**费率(买入、卖出各收一次)。"""
    if cost_bps_roundtrip < 0:
        raise ValueError(f"cost_bps_roundtrip 必须 >= 0,得到 {cost_bps_roundtrip}")
    return cost_bps_roundtrip / 2.0 / 10_000.0
