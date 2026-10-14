"""US preset —— uptrend_pullback 思路的美股版。

⚠️ 这个 preset 的 `validated = False`。它是**如实交付的产物,不是可交易的策略**。
   按 hiagent CLAUDE.md §5 与 preset_search 的纪律,负面证据必须与参数同时记录
   (他们自己的 v33_long_reverse_v21 就是这么写的:"必须同时记录的负面证据,不得
   只报好的一面")。这里照做。

基线结论(完整数据见 results/BASELINE.md,复现命令见 README.md):

  A) v21 参数**原样**搬过来:holdout 总收益 −86.04%,同期等权基准 +20.09%,
     超额 −106 个百分点。止损 2.1% 在信号日 atr_pct 中位 6.79% 的池子里只有
     0.31 ATR —— 落在噪声之内,94% 的仓位被止损。

  B) 止损改为 ATR 归一化(k ATR)、TP 保持 hiagent 的 12.5:1 几何:迭代期
     E[R] 转正(+0.15 ~ +0.20),但 holdout 塌到 **−0.59 ~ −0.99**。
     且 12.5:1 的几何在 ATR 空间里等于 TP = +85%,18 天持有期内几乎不可达 ——
     holdout 22 笔里 **21 笔止损、0 笔 TP**,E[R] ≈ −1 是设计的必然后果。

  C) 仓位从 all-in 改为风险预算式(本 preset 采用):消除了「连亏 20 次 −75%」
     的毁灭性路径,但**没有创造边缘**。E[R] 与胜率对 risk_per_trade 完全不变
     (它是纯敞口旋钮,不是边缘旋钮),holdout 仍是 E[R] −0.59、超额 −24pp。

  三组数字同向:**这个信号族在美股这个池子上没有可证实的正 alpha。**
  迭代期 E[R] 为正、holdout 为负,是「无边缘 + 样本极小(n=15~22)」的典型形态。

hiagent 自己在 docs/uptrend_pullback_preset_search.md §6 已经写下这个结论,
这里是在**美股数据上复现了它**:
  > 该 preset 家族在非趋势市里期望为负;在趋势市里的正收益与市场 beta 高度相关。
  > 若要取得可证实的 alpha,应**更换信号源**(引入基本面/事件维度),而不是继续在
  > 价格派生的 A~E 条件上做参数或排序层面的改动。

本 preset 的价值不在收益,而在:**它是一台经过验证的证伪装置**。任何人再提出
「把 v21 搬到美股试试」,现在有一条命令和一组数字可以回答。
"""
from __future__ import annotations

import copy

# v33_long_reverse_v21 的原始信号参数,**一个都没改**(基线的意义就在于此)
V21_SIGNAL_PARAMS: dict[str, object] = {
    "min_down_streak": 3,
    "max_down_streak": 10,
    "pct_chg_low": -0.088,
    "pct_chg_high": -0.02,
    "min_amount": 3e7,
    "max_amount": 3e8,
    "min_above_ma60_ratio": 0.6693,
    "close_ma60_buffer": 0.0121,
    "min_mom120": 0.18,
}

# hiagent v21 的盈亏几何 tp/sl = 0.2628/0.021 = 12.514。保留它,是为了只改
# **一个**变量(止损的尺度),而不是同时改几何 —— 否则无法归因。
V21_PAYOFF_GEOMETRY: float = 0.2628 / 0.021

# 权益风险预算。**先验取 1%,不是从回测里挑出来的。**
# 理由:实测 E[R] 与胜率对 risk_per_trade 完全不变(见模块 docstring C 组),
# 它只线性缩放 CAGR 与敞口 —— 属于「风险偏好」,不属于「策略边缘」。
# 拿它去最大化 CAGR 就是拿风险偏好冒充 alpha。
DEFAULT_RISK_PER_TRADE: float = 0.01

PRESETS: dict[str, dict] = {
    "us_uptrend_pullback_v1": {
        # ---- 数据口径 ----------------------------------------------------
        "universe": "oi100",              # 596 只;今日快照,含幸存者偏差(见 BASELINE.md)
        "universe_note": "今日 OI>=100 快照,非 PIT —— 面板本身偏乐观",
        "price_adjust": "total",          # 拆股+分红复权;多头回测必须用复权价

        # ---- 执行语义(CLAUDE.md §2/§4 的可移植部分)----------------------
        "intraday_tiebreak": "sl_first",
        "max_volume_participation": 0.10,
        "cost_bps_roundtrip": 5.0,        # 美股无印花税;IBKR Lite 佣金 $0 + SEC/TAF

        # ---- 仓位政策(偏离 hiagent §2 的 all-in,理由见下)----------------
        "position_sizing": "risk_budget",
        "risk_per_trade": DEFAULT_RISK_PER_TRADE,
        "max_positions": 1,
        # ⚠️ 偏离 hiagent §2「All-In Sizing Policy」。该政策与「窄止损」耦合:
        #    §2 原话 "Tail risk is managed entirely by exit logic (TP/SL), never by
        #    reserving cash at entry" 只在止损很窄时成立。A 股 2.1% 止损下 all-in
        #    每次亏 2.1% 权益;止损加宽到 1 ATR(美股中位 6.79%)后每次亏 6.79%,
        #    连亏 20 次从 −34.6% 变 −75.5%。风险预算式让单笔亏损与止损宽度脱钩。

        # ---- 退出(ATR 归一化,覆盖固定百分比)----------------------------
        # k=1.0 是**机制选择,不是 CAGR 选择**:1 ATR 是「一天典型波幅」的自然单位,
        # 且是能逃出噪声带的最小值;迭代期 E[R] 在 k∈[1.0,2.0] 上基本持平
        # (+0.197 / +0.190 / +0.146),没有梯度的形状就不该去挑峰。
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 1.0 * V21_PAYOFF_GEOMETRY,
        "max_hold": 18,
        # ⚠️ 已知设计缺陷(未修,因为修它需要再一轮搜索,而边缘本就不存在):
        #    TP = 12.514 × 1 ATR ≈ +85%,在 18 天持有期内几乎不可达。
        #    结果:holdout 22 笔中 21 笔走止损、0 笔走 TP,E[R] ≈ −1 是几何的
        #    必然后果而非市场发现。12.5:1 这个比例同样是 A 股校准量。

        # ---- 信号 --------------------------------------------------------
        "signal": dict(V21_SIGNAL_PARAMS),

        # ---- 大盘择时(compute_regime 默认参数,不调参)--------------------
        "regime": {
            "ma_window": 20,
            "breadth_window": 20,
            "min_breadth": 0.45,
            "require_rising": False,      # preset_search §5 M4 已证伪,不要打开
        },

        # ---- §5 记账 -----------------------------------------------------
        # 2(流动性口径候选) × 3(成本档敏感性) = 6
        # + 6(k 的 ATR 阶梯:0.5/1.0/1.5/2.0/3.0/4.0)
        # + 1(风险预算式仓位这一条政策检验)
        # + 1(H1 财报窗口过滤)
        # + 1(H2 移动止损)
        # = 15;本文件的写作、窗口起止的选定本身也计入,故再 +1。
        "n_comparisons": 16,

        # ---- H1(财报窗口过滤)已检验并被证伪 ------------------------------
        # 表面看有效:E[R] −0.288 → +0.109,总收益 −12.54% → +3.04%,剔除 30% 信号。
        # 但**机制检验推翻了它**:被剔除批的 sl_gap 占比 6.7%,保留批 6.9% ——
        # 几乎相同。若财报真通过"跳空穿透止损"造成伤害,前者必须显著更高。
        # 15 笔里 14 笔是普通 sl。所以 H1 只是移除了亏损样本里 34% 的份额,
        # 在 34~44 笔上无法与"有因果的筛选"区分。且 holdout 仍为 E[R] −0.695。
        # → 不作为默认开启。相关工具保留(signals.filter_earnings_window)。
        "avoid_earnings_forward_calendar_days": 26,
        "earnings_filter_validated": False,

        # ---- H2(移动止损)已检验,不成立 -----------------------------------
        # 结果:全期 E[R] −0.288 → −0.061、胜率 13.6% → 33.7%,看似改善;
        # 但**两个期间方向相反**(迭代期 +0.197 → +0.097 变差,holdout −0.990 →
        # −0.238 变好),且成交 44 → 104、平均持有 7.05 → 2.27 天、出场退化成
        # SL 104 / TP 0 / hold 0 —— 不是更有效率,是变成了高频 churn。
        # 同名现象见 §1B 的 k 阶梯:改善总与样本膨胀同步出现。
        # → 默认关闭。机制与因果性测试保留(tests/test_broker.py 的 H2 四条)。
        "trailing_sl": False,
        "trailing_sl_validated": False,

        # ---- 有效性声明(不要删)------------------------------------------
        "validated": False,
        "verdict": (
            "无可证实的正 alpha。五组假设(A v21 原样 / B 止损 ATR 归一化 / "
            "C 风险预算仓位 / H1 财报窗口 / H2 移动止损)逐一被排除,详见 "
            "results/BASELINE.md §3。全期超额 −113pp,TP 出场 0 笔。"
            "不得用于实盘,不得当作选股能力的证据。"
        ),
    },
}


# ============================================================================
# 2026-10-14:为"最近 12 个月 CAGR >= 30%"这个目标搜索出来的配置。
# 它**确实达到了目标**,但必须先读完下面这段再决定用不用。
# ============================================================================
US_12M_SEARCH_NOTE = """
找到的配置(3 个都达到 >=30%,取最好):
    max_positions = 2, risk_per_trade = 0.05, atr_sl_mult = 1.5, atr_tp_mult = 6.0
    + 过滤层 min_dist_high252 <= -0.1 且 stoch_k <= 60
    -> 最近 12 个月 CAGR +44.5%,最大回撤 -18.9%,成交 13 笔,平均敞口 31%

但同一张表里还有三件必须一起看的事:

1. **CAGR 是杠杆做出来的,不是信号做出来的。**
   把 risk_per_trade 从 1% 提到 5%(信号与止损止盈一字不改):
       risk=1%  -> CAGR  +7.9%   回撤  -5.0%   敞口  6%
       risk=2%  -> CAGR +15.9%   回撤  -9.8%   敞口 12%
       risk=5%  -> CAGR +40.6%   回撤 -21.7%   敞口 29%
   而 E[R] 在三个杠杆下**恒为 +0.89**。E[R] 是风险归一化的,它不变说明
   "信号质量"没变;变的是倍数。**拿到 30% 靠的是把每笔风险放大 5 倍。**

2. **max_positions 超过 3 完全不起作用。**
   12 个月只有 13~18 笔、持有 <=18 天,同时在场的仓位很少,所以 mp=3/5/10
   的结果逐位相同。"分散化提高 CAGR"这个假设在这么少的交易数下是空操作。

3. **同杠杆下的随机入场能达到同样的数字。**
   同样的止损止盈、同样的笔数、同样的杠杆,随机选入场点 200 次:
       mp=1 risk=5%: 随机中位 -2.5%  p90 +17.7%  p95 +29.6%  最大 +60.6%
                     随机达到 >=30% 的概率 = 4%
       mp=2 risk=5%: 随机中位 -3.0%  p90 +37.6%  p95 +42.3%  最大 +100.4%
                     随机达到 >=30% 的概率 = 14%
   跑 1000 次后的精确百分位(不是"附近"这种含糊说法):
       mp=1 risk=1%: 策略 +7.9%  位于随机分布第 95.1 百分位
       mp=1 risk=5%: 策略 +40.6% 位于第 95.6 百分位
       mp=2 risk=5%: 策略 +44.5% 位于第 93.0 百分位
   即**约 4%~7% 的随机策略做得和它一样好或更好**。这个百分位比我先前口头
   描述的"p90 附近"要高 —— 先前那句偏保守,以这里的实测值为准。
   但 n_trials 已累计 106 次,按 §5 的选择偏差折算后门槛还要更高。

4. **换个窗口,过滤层的贡献消失了(2026-10-14 补测)。**
   在**前一个 12 个月**(2024-10-14..2025-10-14,选择时未看)上:
       找到的配置   CAGR +26.1%  回撤 -14.5%  E[R] +0.41  超额 -39.2pp
       v21 基线     CAGR +25.5%  回撤 -40.0%  E[R] +0.31  超额 -39.7pp
       (该窗口基准 CAGR +65.3%,是被选窗口 +15.7% 的四倍)
   两件事:① 它没塌掉,这点要说公道;② 但它与 v21 基线几乎无差别
   (+26.1% vs +25.5%),说明那些过滤层在两个窗口之间的贡献从 **45.5pp 掉到 0.6pp**
   —— **过滤层的价值完全集中在被选中的那 12 个月里**。
   同时两个窗口基准相差 4 倍而策略表现没有跟着环境变 —— 变的是市场,不是信号。

结论:这个 preset 达到了指定的数字,但**没有证据表明它优于杠杆化的运气**,
且**换一个 12 个月窗口后它相对基线的优势就消失了**。
交付它是因为你要的是"达到 30% 的 preset",而它确实达到;
不推荐使用它是因为达到的方式不构成可重复的边缘。
"""
PRESETS["us_uptrend_pullback_12m_target"] = {
    "universe": "oi100",
    "price_adjust": "total",
    "intraday_tiebreak": "sl_first",
    "max_volume_participation": 0.10,
    "cost_bps_roundtrip": 5.0,
    "position_sizing": "risk_budget",
    "risk_per_trade": 0.05,          # <= 这一个是 CAGR 的真正来源(1% -> 5% 杠杆)
    "max_positions": 2,
    "atr_sl_mult": 1.5,
    "atr_tp_mult": 6.0,
    "max_hold": 18,
    "signal": {
        **V21_SIGNAL_PARAMS,
        "min_dist_high252": -0.1,
        "max_stoch_k": 60.0,
    },
    "regime": {"ma_window": 20, "breadth_window": 20, "min_breadth": 0.45,
               "require_rising": False},
    # 2(流动性)+ 3(成本) + 37(stage1) + 45(stage2) + 15(分散化) + 3(对照)
    "n_comparisons": 106,   # +1: 前一个 12 个月的 OOS 检验
    "validated": False,
    "verdict": ("达到 12 个月 CAGR >=30% 的目标(+44.5%),但来源是 5 倍杠杆而非信号:"
                "E[R] 在各杠杆下恒为 +0.89。同杠杆随机入场有 14% 概率达到同样水平,"
                "随机最大 +100.4%。落在随机分布 p90 附近,不构成边缘证据。"
                "最大回撤 -18.9% 是 1% 风险版本的 4.3 倍。前一个 12 个月 OOS:与 v21 基线无差别(+26.1% vs +25.5%),过滤层贡献从 45.5pp 掉到 0.6pp,且跑输基准 39.2pp。"),
    "notes": US_12M_SEARCH_NOTE,
}


def get_preset(name: str = "us_uptrend_pullback_v1") -> dict:
    """返回 preset 深拷贝;未知名抛 ValueError。"""
    if name not in PRESETS:
        raise ValueError(f"未知 preset: {name!r}。可用: {sorted(PRESETS)}")
    p = copy.deepcopy(PRESETS[name])
    if p.get("n_comparisons", 0) < 1:
        raise ValueError(f"preset {name!r}: n_comparisons 必须 >= 1(CLAUDE.md §5)")
    return p
