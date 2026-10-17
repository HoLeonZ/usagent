"""preset 定义体：us_uptrend_pullback_wfv（见 research/preset_registry.py 的装载契约）。
"""
from __future__ import annotations

from core import costs
from research import universe as mod_universe
from research.preset_common import (DEFAULT_RISK_PER_TRADE, RISK_LIMITS,
                                    V21_PAYOFF_GEOMETRY, V21_SIGNAL_PARAMS)

NAME = "us_uptrend_pullback_wfv"

NOTES = """
【怎么搜的】协议、网格、折窗与判据都写在 research/wfv_search.py 的模块 docstring 里,
**运行之前定稿**。要点:搜索期 2024-04-09..2025-10-13 分 4 折(train 6m / test 3m / step 3m),
选参只看训练窗(池化),头条数字只来自测试窗;n_trials = 138(A 单轴 18 + B 组合 64 +
C 退出 48 + D 杠杆 8)。复现:
    python -m research.wfv_search --run

【结果:过滤层相对基线有改善,但两个口径给出的幅度差 5 倍 —— 按这个顺序读】
    ① 池化测试窗(WFV 协议统计量):
       本 preset CAGR +14.19%  平均 maxDD -1.2%  E[R] +0.398  22 笔
       v21 基线   CAGR  +3.69%  平均 maxDD -1.7%  E[R] -0.118  10 笔
       ⇒ +10.5pp。⚠️ 但这是把 4 个**独立 3 个月窗**的 CAGR 取平均(每折重置资金、年化放大),
         而这四个数里第一折 +49.2% 远大于其余三折(-2.6%/+7.0%/+3.2%)—— **均值被单折主导**。
    ② 连续净值(一个账户从 2024-04-09 跑到 2025-10-13,这才是「真赚到多少」):
       本 preset CAGR +8.15%  平均敞口 4.9%  maxDD -1.9%
       v21 基线 CAGR +6.21%  平均敞口 8.1%  maxDD -8.2%
       ⇒ 优势只有 **+1.94pp**
    逐折测试窗 CAGR: +49.2% / -2.6% / +7.0% / +3.2%
    DSR:测试窗平均 Sharpe 2.10,k=138 ⇒ **0.021**(p=0.979)
      —— 按 §5 的选择偏差折算后,这个样本外 Sharpe 与噪声不可区分(22 笔、k=138)。

【零对照:同止损止盈、同笔数、同机会集的随机入场 200 次/窗(seed=0)】
    逐窗百分位 100.0 / 45.5 / 76.5 / 58.0
      —— 四窗里三窗落在随机分布之内。第一折(2024-10)打到第 100 百分位,
         但那一折正是把均值抬起来的那一折 ⇒ **别把 +14.19% 当成稳定的边缘**。

【一次性 holdout(2025-10-14..2026-10-13,全程未参与任何选择;只评估了这一次)】
    本 preset(risk=1%):  CAGR  -2.19%   平均 maxDD -4.9%   E[R] -0.120  18 笔
    v21 基线(同窗同 sizing): CAGR -7.78%   平均 maxDD -8.9%   E[R] -0.531  15 笔
    ⇒ 仍然跑赢基线 **+5.6pp**,但**绝对水平是负的、且大幅跑输基准**(-18.79pp):
      这段窗口里它没赚钱。注意它与搜索期的连续口径(+1.94pp)也不同 —— 两个窗口一个 +1.94pp、
      一个 +5.6pp,合起来只能说「相对改善在 +2~6pp 量级,方向一致」,不能说「有 +10.5pp 的边缘」。

【风险-收益前沿(搜索期测试窗池化;**只用于看形状,不是可引用结论**)】
    risk=1%  -> CAGR +14.19%  平均 maxDD -1.2%
    risk=2%  -> CAGR +33.82%  平均 maxDD -2.4%
    risk=3%  -> CAGR +55.96%  平均 maxDD -3.5%
    risk=5%  -> CAGR +86.35%  平均 maxDD -4.2%
    risk=8%  -> CAGR +98.24%  平均 maxDD -5.8%
    risk=12% -> CAGR +70.68%  平均 maxDD -9.3%
    risk=20% -> CAGR +70.02%  平均 maxDD -9.4%
    risk=25% -> CAGR +69.50%  平均 maxDD -9.4%
    ⚠️ 三点必须一起读:
      ① **holdout 只在预注册的 sizing(risk=1%)上评估过**。前沿上更高的杠杆档没有
         样本外 holdout 证据 —— 去看它们就是在 holdout 上选点,那会把唯一那段干净数据花掉。
      ② 8% 以上 CAGR 反而下降、且 E[R] 从 +0.379(5%)跳到 +0.693(8%):仓位被现金闸门与
         量能上限截断,定尺寸退化成「能买多少买多少」,成交笔数从 22 掉到 13~19。也就是说
         高杠杆那一端**不是同一策略放大**,而是换了一种被闸门扭曲的行为,数字更脆。
      ③ 同杠杆下 v21 基线在 risk=12% 也有 +81.88%(回撤 -16.8%),所以「高 CAGR」这个
         数字本身不是本 preset 的功劳 —— 杠杆给谁都是这个量级。

【结论:不得用于实盘,也不得当作选股能力的证据】
    可以说的:相对 v21 基线,过滤器 + 退出几何在两个期间都改善,量级 **+2~6pp**(连续口径
    +1.94pp / holdout +5.6pp),方向一致且可重复。
    不能说的:绝对收益为正。搜索期 +14.19% 在 DSR 0.021 下与噪声不可区分、且被单折主导;
    holdout 上是 **-2.19%**。**holdout 已经被用掉了** —— 再做任何迭代都只能拿搜索期数字
    (in-sample),要么就得等新数据。
    下一步的价值不在继续在这个池子上调参(审计已两次测出「价格派生的 A~E 条件上做参数或
    排序层面的改动不产生可证实的 alpha」),而在换信号源(基本面/事件维度)。
"""

PRESET = {
    "universe": dict(mod_universe.DEFAULT_SPEC),   # PIT 流动性池(见 research/universe.py)
    "universe_note": "PIT 流动性池(截至 T 的 60 日滚动成交额 >= $3e7, 因果) —— "
                     "仍未消除的是**退市名单缺失**:DB 里没有已退市标的,所以绝对收益应读作上界",
    # ---- 执行语义(与 v1 同一套)---------------------------------------
    "intraday_tiebreak": "sl_first",
    "max_volume_participation": costs.MAX_VOLUME_PARTICIPATION,
    "cost_bps_roundtrip": costs.COST_BPS_ROUNDTRIP,
    # 价差是**尺寸无关**的那一半成本，此前完全没建模（审计 E1）。
    # 用 Corwin-Schultz 从日线 high/low 估计（逐日、因果、系统性低估 ⇒ 是下限）。
    "spread_model": "corwin_schultz",
    # 止损穿价系数：0 = 按触发价成交（既有行为）。**为什么不取 0.5**：实测穿价中位
    # 约 1% / p90 约 1.8%（见 broker 的 sl_penetration 读数），θ 从 0 到 1 会让
    # 头条差 1~3.5pp —— 也就是说这条假定值钱。但 θ 的正确值需要**实盘成交**来标定
    # （台账 G7 的前向验证闭环），在没有标定数据之前拍一个 0.5 就是凭空造假设。
    # 所以：默认 0、穿价**逐笔记录并进指标**，读者自己按 θ∈{0,0.5,1} 跑敏感性。
    "sl_overshoot": 0.0,
    "atr_slip_scale": costs.ATR_SLIP_SCALE,   # 入场与出场两侧都建模(§4(c))
    # ---- 仓位:sizing 是**预注册写定**的,不是搜出来的 -------------------
    "position_sizing": "risk_budget",
    "position_fraction": 1.0,
    "risk_per_trade": 0.01,          # 预注册 1%:让选参不被杠杆主导
    "max_positions": 2,
    **RISK_LIMITS,                     # 组合层风险政策（见上方 RISK_LIMITS）
    # ---- 退出几何:WFV 训练窗选出 --------------------------------------
    "atr_sl_mult": 2.0,              # 旧 target preset 是 1.5(网格角点)
    "atr_tp_mult": 3.0,              # 旧的是 6.0 —— 18 天持有期内几乎不可达
    "max_hold": 10,                  # 旧的是 18
    # ---- 信号:V21 骨架 + WFV 训练窗选出的两个过滤层 --------------------
    "signal": {**V21_SIGNAL_PARAMS, "min_dist_high252": -0.1, "max_stoch_k": 70.0},
    "regime": {"ma_window": 20, "breadth_window": 20, "min_breadth": 0.45,
               "require_rising": False},
    # A 18 + B 64 + C 48 + D 8 = 138。⚠️ 这是**本次搜索**的口径决策数;preset 家族的累计
    # 还要算上 us_uptrend_pullback_v1/12m_target 的 17+107,所以这里的 DSR 是**乐观下界**。
    "n_comparisons": 138,
    "validated": False,
        "notes": NOTES,
}

VERDICT = (
    '**不得用于实盘。** 预注册 WFV 搜索(wfv_search.py)选出的候选。搜索期池化测试窗 CAGR +14.19%(v21 基线同窗 +3.69%,⇒ +10.5pp),但那个均值被 4 折里的第一折(+49.2%)主导;**换成连续净值口径(一个账户跑完搜索期)优势只剩 +1.94pp(+8.15% vs +6.21%)**。DSR(k=138)只有 0.021 —— 与噪声不可区分;四窗的随机入场百分位是 100/45.5/76.5/58,三窗落在随机分布之内。**唯一一段未参与选择的数据(holdout 2025-10-14..2026-10-13)上它不赚钱:CAGR -2.19%**(v21 基线同窗 -7.78%,所以相对改善 +5.6pp 是真实的,但绝对水平为负)。holdout 已被用掉,再做迭代只能拿 in-sample 数字。结论:过滤器相对基线有可重复的 +5~10pp 改善,**没有**可证实的绝对收益。详见 presets.WFV_SEARCH_NOTE。'
)

# 窗口取**全可评估期**，让行为锁覆盖尽量多的分支（它的预注册搜索期是
# 2024-04-09..2025-10-13，holdout 是 2025-10-14..2026-10-13）。
LOCK_WINDOW = ("2024-04-09", "2026-10-14")
LOCK_IS_CLI_DEFAULT = False

EXPORTS = {"WFV_SEARCH_NOTE": NOTES}
FIXTURE_EXEMPT_REASON = "合成市场上该配置成交为 0，锁会变成空断言"
