"""策略族的**掩码函数** —— 一个族 = 一个判定函数（2026-10-17 族注册表重构）。

这个模块放的是**共享的机制实现**；族的**声明**（ID / 外围固定轴 / 预注册网格 /
排序分 / 状态门）在 `research/family_defs/<id>.py`。

为什么把掩码单独放一层：15 个族只用到 10 个函数 —— `idio*` 四条臂共用同一个掩码、
`str`/`str_inv` 共用、`cap`/`cap_inv` 共用。把函数体逐族复制进定义体会得到 5 份
必须手工同步的副本（§2(b)「常数恰好定义一次」；这次是一整个函数）。
**新机制不必写到这里** —— 直接写在 `family_defs/<新族>.py` 里即可。

每个掩码的 docstring 是**机制叙述**，`family_search` 的报告会取它的第一行当族名，
所以改动它等于改报告 —— 不是自由文本。

纪律（与 signals.py 相同）：只用 ≤T 的数据；不引 shift(-x) / bfill / 全样本分位阈值；
信号日 T 收盘 → T+1 开盘成交（执行在 broker，本模块不碰）。
"""
from __future__ import annotations

import pandas as pd

from research import signals
from research.universe import above_price_floor

# 流动性下界:与 uptrend-pullback 的 E 条件同一个数,保证「可交易的池子」一致
MIN_AMOUNT60 = signals.V21_SIGNAL_PARAMS["min_amount"]


def mask_short_reversal(ind: pd.DataFrame, p: dict) -> pd.Series:
    """族 1:**短周期反转**(str,short-term reversal)。

    机制:1~5 日内的抛压把价格暂时压到均衡之下,随后均值回归。这是最老的异象之一
    (Lehmann 1990 / Jegadeesh 1990),在大盘股上大幅衰减、在小盘股上更强、且与流动性
    风险绑定。**「低估」在这里的定义**:相对**自身 252 日区间**处于低位(不是相对同池)。

    与 uptrend-pullback 的区别(机制层面,不是阈值):
      * 它**不要求上升趋势** —— 可以买正在下跌的票(只要离 52 周高点有足够距离、
        且离 52 周低点还有空间,避免退市边缘);
      * 它不看 MACD、不看连阴天数;
      * 它的持有期更短(反转的衰减很快)。
    """
    # side="drop" = 买 5 日跌得多的(正题);side="rise" = 买 5 日涨得多的(对照臂)。
    # 其余条件一字不改 —— 这样两条臂的差异**只有方向**,是干净的反转机制检验:
    # 若两个方向都赚钱,说明抓到的不是反转,而是「被打过的票会弹」这种 beta。
    side = p.get("side", "drop")
    m = (
        ((ind["ret5"] <= -float(p["drop5"])) if side == "drop"
         else (ind["ret5"] >= float(p["drop5"])))
        # ⚠️ 用 dd120(120 日包络)而不是 dist_high252:后者要 252 根预热,而评测窗从
        #    第 120 根起算 ⇒ 会白白吃掉前 ~6 个月(实测第一折训练窗信号数为 0)。
        #    dd120 的预热与评测起点重合,零额外样本成本;语义仍是「相对自身历史便宜」。
        & (ind["dd120"] <= -float(p["min_drawdown"]))
        & (ind["ss_lo120"] >= float(p.get("min_off_low") or 0.0))
        & above_price_floor(ind)
        & (ind["amount60"] >= MIN_AMOUNT60)
        & ind["ret5"].notna() & ind["dd120"].notna()
        & ind["ss_lo120"].notna() & ind["amount60"].notna()
    )
    if p.get("max_atr_pct") is not None:
        m &= (ind["atr_pct"] <= float(p["max_atr_pct"])) & ind["atr_pct"].notna()
    if p.get("max_rvol20") is not None:
        m &= (ind["rvol20"] <= float(p["max_rvol20"])) & ind["rvol20"].notna()
    return m

def mask_upb_ref(ind: pd.DataFrame, p: dict) -> pd.Series:
    """**参考族**(恒等检查用,不是新策略思想)。

    逐字复刻 uptrend-pullback 的 A~F + 两个过滤层。用途只有一个:验证
    `family_search` 用它自己的 `F.select` 构造出来的信号表,与 `signals.select_entries`
    构造出来的**逐位相同** —— 否则「新族 vs uptrend-pullback」的比较不是同一台机器跑的,
    结论无效(见 tests/test_families.py::test_family_path_reproduces_select_entries)。

    ⚠️ 不要把它当候选族放进横向比较(它等于基准本身)。
    """
    sp = signals.V21_SIGNAL_PARAMS
    m = (
        (ind["close"] > ind["ma60"])
        & (ind["above_ma60_ratio"] >= sp["min_above_ma60_ratio"])
        & (ind["mom120"] >= sp["min_mom120"])
        & (ind["down_streak"] >= sp["min_down_streak"])
        & (ind["down_streak"] <= sp["max_down_streak"])
        & (ind["ret1"] >= sp["pct_chg_low"])
        & (ind["ret1"] <= sp["pct_chg_high"])
        & (ind["macd_dif"] > 0) & (ind["macd_dea"] > 0)
        & (ind["macd_bar"].abs() < ind["macd_bar_prev"].abs())
        & (ind["amount60"] >= sp["min_amount"]) & (ind["amount60"] <= sp["max_amount"])
        & (ind["close"] >= ind["ma60"] * (1 + sp["close_ma60_buffer"]))
    )
    for c in ("ma60", "above_ma60_ratio", "amount60", "macd_bar",
              "macd_bar_prev", "mom120"):
        m &= ind[c].notna()
    if p.get("min_dist_high252") is not None:
        m &= ind["dist_high252"].notna() & (ind["dist_high252"] >= float(p["min_dist_high252"]))
    if p.get("max_stoch_k") is not None:
        m &= ind["stoch_k"].notna() & (ind["stoch_k"] <= float(p["max_stoch_k"]))
    return m

def mask_volshock(ind: pd.DataFrame, p: dict) -> pd.Series:
    """族:**波动率恐慌的吸收确认**(volshock)。

    机制(打分最高的提案):一次 ≥2.5σ 的暴跌只有两种来源 —— 新信息,或一次性的
    流动性/仓位冲击。二者的差别**不在跌得多深,而在波动率的寿命**:信息型冲击会让波动率
    持续(明天还会抖),一次性冲击不跟进。所以信号 = 「S 日标准化暴跌」+「T 日波动率不跟进」
    +「恐慌低点守住」。

    「低估」在本族的定义:价格被一次**非信息性**冲击打到偏离,而冲击的指纹(波动率)
    已经消退 —— 便宜是因为有人被迫卖,不是因为有坏消息。

    与 uptrend-pullback 的区别:它要求上升趋势 + 连阴 + MACD 衰减(路径几何);
    本族要求**两根 bar 的对比**(昨日暴跌 vs 今日波动率骤降),不要求趋势、不看 MACD。
    """
    def col(name):
        v = p.get(name)
        return None if v in (None, "None") else float(v)

    m = (
        (ind["shock_z_S"] <= float(p["shock_z_max"]))
        & (ind["ret1_S"] <= float(p["ret1_floor"]))
        & (ind["prev_down_streak_S"] == 0)          # 恐慌 bar 之前不是连阴(排除阴跌段)
        & (ind["amount60_S"] >= MIN_AMOUNT60)
        & ind["shock_z_S"].notna() & ind["ret1_S"].notna()
    )
    if col("tr_decay_max") is not None:             # 机制对照臂:None = 关闭「波动率不跟进」确认
        m &= (ind["tr_decay_rel"] <= col("tr_decay_max")) & ind["tr_decay_rel"].notna()
    if col("hold_buffer") is not None:              # None = 不要求守住恐慌低点
        m &= (ind["low"] >= ind["low_S"] * (1.0 - col("hold_buffer")))
    m &= (ind["close"] >= ind["low_S"]) & ind["low_S"].notna()
    if col("mkt_floor") is not None:                # 特异性:昨天不是全场一起跌
        m &= (ind["mkt_ret1_S"] >= col("mkt_floor")) & ind["mkt_ret1_S"].notna()
    if p.get("regime_gate", True):                  # 大盘门(与 uptrend-pullback 同一套 F)
        m &= ind["regime_ok"].fillna(False).astype(bool)
    return m

def mask_state_flip(ind: pd.DataFrame, p: dict) -> pd.Series:
    """族:**状态翻转的语义门**(state_flip)。

    机制:同一个成交量读数,在不同市场状态下含义相反 ——
      * 恐慌态(市场波动高、指数在 MA20 下)放量杀跌 = **被迫抛售**(为补保证金/赎回而卖,
        与价格无关)→ 该买;
      * 平静态(指数在 MA20 上、宽度足)缩量阴跌 = **没人接**(没有任何人愿意出价)→
        个体性问题,不该买。
    所以 `vol_ratio` 在格 A 要求 ≥1.5、在格 B 要求 ≤1.0;残差收益(个股 − 市场)在格 A 要求
    接近 0(跌是大家造成的)、格 B 要求显著为负(跌是它自己造成的)。**两处符号翻转就是本族
    的全部内容。**

    与 uptrend-pullback 的区别:它的 F 条件是**许可门**(门开了规则照旧,只是允许交易);
    本族把状态升级成**语义门**(同一个值在不同状态下含义相反,规则本身变了)。
    """
    base = above_price_floor(ind)
    base &= ind["amount60"] >= MIN_AMOUNT60
    base &= ind["atr_pct"].notna() & ind["ret1"].notna()

    stress = (ind["mkt_vol20"] >= float(p["v_stress"])) & (ind["ew_index"] < ind["ew_ma"])
    calm = (ind["ew_index"] > ind["ew_ma"]) & (ind["breadth_sm"] >= 0.50)

    a = (stress
         & (ind["ret1"] <= -float(p["r_a"]))
         & (ind["mkt_ret1"] <= -0.005)
         & ((ind["ret1"] - ind["mkt_ret1"]) >= -0.05)     # 残差护栏:排除个股暴雷
         & (ind["vol_ratio"] >= float(p["vr_a"]))
         & (ind["dd120"] >= -0.60))                       # 不接自由落体(120 日包络,不吃样本)
    b = (calm
         & (ind["ret1"] <= -float(p["r_b"]))
         & ((ind["ret1"] - ind["mkt_ret1"]) <= -0.02)     # 市场没跌,跌的是它自己
         & (ind["vol_ratio"] <= float(p["vr_b"]))         # 缩量:没有增量卖方,只是没人接
         & (ind["dd120"] >= -0.60)
         & (ind["above_ma60_ratio"] >= 0.40))             # 中期结构未坏
    return base & (a | b)

def mask_capitulation(ind: pd.DataFrame, p: dict) -> pd.Series:
    """族:**巨量崩跌的盘中承接**(cap)。

    机制:放量暴跌里混着两类卖方 —— 信息型(一路卖到收盘)与被迫型(与价格无关,必须卖)。
    二者留下的疤痕不同:前者收盘趴在最低端(clv 低);后者砸出坑之后被愿意在该价成交的
    买方接住(收盘显著高于低点)。所以信号 = 「量能爆发 + 跌幅离群 + 收盘在振幅上半段」。

    「低估」在本族的定义:相对**成交价值中枢**(惨跌前的 20 日成交量加权均价)折价,
    而折价的原因是被迫卖出、且在该价位看得见买盘(长下影)。
    """
    m = (
        (ind["climax_ratio"] >= float(p["climax_ratio_min"]))
        & (ind["ret1"] <= -float(p["drop_atr"]) * ind["atr_pct"])
        # clv_side="min" = 要求收盘强(有人接);"max" = 要求收盘趴低(持续卖压)。
        # ⚠️ 两份提案对同一个量的符号读法**相反**:吸收族要 clv 高,恐慌臂要 clv 低。
        # 这正是「clv 到底携带什么信息」的 A/B —— 哪一侧有边缘,答案就在哪一侧。
        & ((ind["clv"] >= float(p["clv_min"])) if p.get("clv_side", "min") == "min"
           else (ind["clv"] <= float(p["clv_max"])))
        & (ind["close"] < ind["ma60"])
        & (ind["above_ma60_ratio"] <= float(p["bad_ratio_max"]))
        & (ind["amount60"] >= 2e7)
        & ind["clv"].notna() & ind["climax_ratio"].notna()
        & ind["atr_pct"].notna() & ind["above_ma60_ratio"].notna()
    )
    if p.get("px_disc_max") not in (None, "None"):
        m &= (ind["px_disc"] <= float(p["px_disc_max"])) & ind["px_disc"].notna()
    return m

def mask_xs_flow(ind: pd.DataFrame, p: dict) -> pd.Series:
    """族:**横截面流量折价反转**(xs_flow)。

    机制:同一天全池里最弱的那十分位,如果它的跌幅**不是来自隔夜跳空**(即不是消息驱动的
    重定价,而是盘中流量冲击)、中期结构未破、且放量止跌,那么它是「相对错位」而不是
    「重新定价」。买它,赚的是同一时点上相对定价的收敛。

    「低估」在本族的定义:**相对同池**最便宜(横截面百分位),而不是相对自己的历史。
    这与 str 族(相对自身 120 日区间)是两种不同的便宜观。

    与 uptrend-pullback 的区别:它按 mom120 降序**优先最强的**;本族按折价深度优先**最弱的**,
    方向相反;而且它不看趋势(只看中期结构未破)。
    """
    m = (
        (ind["amount60"] >= MIN_AMOUNT60)
        & (ind["xs_ret5"] <= float(p["xs_ret5_max"]))
        & (ind["close"] > ind["ma120"]) & (ind["ma60"] > ind["ma120"])
        & (ind["gap5_ratio"] <= float(p["gap5_max"]))
        & (ind["vol_z"] >= float(p["vol_z_min"]))
        & (ind["ret1"] >= -0.02) & (ind["ret1"] <= 0.05)
        & ind["xs_ret5"].notna() & ind["gap5_ratio"].notna() & ind["vol_z"].notna()
        & ind["ma120"].notna() & ind["amount60"].notna() & ind["atr_pct"].notna()
    )
    return m

def mask_long_horizon(ind: pd.DataFrame, p: dict) -> pd.Series:
    """族:**长周期折价 + 短趋势已转向**(lhr)。

    机制:120 日尺度上的深度折价(用**该股自身**的正常日均波幅归一化 —— 折价 12% 对公用事业
    是极端位移,对比特币式波动股是噪声),在最短的趋势(站回 MA20 上方)刚翻过来时买入。
    长周期反转(De Bondt–Thaler)在个股上比短周期反转更持久,但需要「已止跌」才不会接飞刀。

    「低估」在本族的定义:相对自身 120 日价格包络的位移,按自身波幅标准化后异常大。
    """
    # ⚠️ 流动性门用项目统一值 $3e7,不是提案的 $1e8:族间口径必须一致,否则比的是
    #    「谁挑到了流动性更好的票」。$1e8 在这个池子上让全池 0 信号(实测)。
    m = (
        (ind["dd120"] <= -float(p["min_dd120"]))
        & (ind["dd_norm"] <= -float(p["dd_norm_k"]))
        & (ind["close"] >= ind["ma20"] * float(p["ma20_buffer"]))
        & (ind["amount60"] >= MIN_AMOUNT60)
        & ind["dd_norm"].notna() & ind["ma20"].notna() & ind["amount60"].notna()
        & ind["atr_pct"].notna()
    )
    if p.get("xs_mom120_max") not in (None, "None"):
        m &= (ind["xs_mom120"] <= float(p["xs_mom120_max"])) & ind["xs_mom120"].notna()
    return m

def mask_sf_stress(ind: pd.DataFrame, p: dict) -> pd.Series:
    """状态臂 A：恐慌态的**放量**杀跌。

    ⚠️ **与臂 B 不止差"状态"这一个变量**（2026-10-17 审计 F3，已核对代码）：
    两臂共 6 处不同 —— ① 状态门（A: `mkt_vol20 ≥ v_stress ∧ ew_index < ew_ma`；
    B: `ew_index > ew_ma ∧ breadth ≥ 0.50`）② 市场项（A 要 `mkt_ret1 ≤ −0.005`；
    B 无）③ 残差项（A 要 `resid ≥ −0.05`（**不**特异）；B 要 `resid ≤ −resid_max`（特异））
    ④ 量能（A `vol_ratio ≥ min` 放量；B `≤ max` 缩量）⑤ 收盘位置（A 要 `clv ≤ clv_max`；
    B 无）⑥ 趋势质量（B 要 `above_ma60_ratio ≥ 0.40`；A 无）。

    ⇒ **"两臂只差状态"这个说法不成立**，因此"状态语义承重"这句结论**撤回**：
    两臂之间差 6 样东西，任何差异都无法归因到其中某一样。要重新支撑那个结论，
    必须把两臂重写成**只差状态一个变量**（并作为一次新搜索重跑、n_trials 记账）。
    """
    base = above_price_floor(ind) & (ind["amount60"] >= MIN_AMOUNT60)
    stress = (ind["mkt_vol20"] >= float(p["v_stress"])) & (ind["ew_index"] < ind["ew_ma"])
    a = (stress
         & (ind["ret1"] <= -float(p["ret1_max"]))
         & (ind["mkt_ret1"] <= -0.005)
         & ((ind["ret1"] - ind["mkt_ret1"]) >= -0.05)
         & (ind["vol_ratio"] >= float(p["vol_ratio_min"]))
         & (ind["clv"] <= float(p["clv_max"]))          # ← 收盘趴低 = 卖压持续(与 cap 族的符号相反)
         & (ind["dd120"] >= -0.60))
    return base & a & ind["atr_pct"].notna() & ind["ret1"].notna()

def mask_sf_calm(ind: pd.DataFrame, p: dict) -> pd.Series:
    """状态臂 B：平静态的**缩量**阴跌（个体性、没人出价）。

    ⚠️ 见 `_mask_sf_stress` 的说明：与臂 A **不止差状态**（共 6 处不同），
    所以这一对**不能**用来隔离"状态语义"这个机制，相关结论**撤回**。
    （此处原先写的"区别就是两处符号"是**错的**，审计 F3 逐条核对过代码。）
    """
    base = above_price_floor(ind) & (ind["amount60"] >= MIN_AMOUNT60)
    calm = (ind["ew_index"] > ind["ew_ma"]) & (ind["breadth_sm"] >= 0.50)
    b = (calm
         & (ind["ret1"] <= -float(p["ret1_max"]))
         & ((ind["ret1"] - ind["mkt_ret1"]) <= -float(p["resid_max"]))
         & (ind["vol_ratio"] <= float(p["vol_ratio_max"]))   # ← 缩量:没有增量卖方,只是没人接
         & (ind["dd120"] >= -0.60)
         & (ind["above_ma60_ratio"] >= 0.40))
    return base & b & ind["atr_pct"].notna() & ind["ret1"].notna()

def mask_idio(ind: pd.DataFrame, p: dict) -> pd.Series:
    """族:**特质性弱势的均值回归**(idio)—— 把两个幸存者(str / sf_calm)的共同机制抽出来。

    上一步的横向比较留下两个幸存者,它们的条件看起来不同,但共同点是:
    **个股自己跌(不是因为市场跌)+ 不放量 + 相对自身 120 日包络便宜 + 中期结构未坏**。
    本族把这条共同机制**显式**写出来,并用因子式对照臂逐条隔离哪一条在承重:

        mode="idio"      : 正题
        mode="mktdown"   : 把「市场没跌」翻成「市场也在跌」  → 测「个体性」是否承重
        mode="highvol"   : 把「不放量」翻成「放量」          → 测「没有增量卖方」是否承重
        mode="highrvol"  : 加「自身波动率高」(rvol20 ≥ rv)  → 测「安静票」这个偏好是否承重

    机制(为什么这可能是真的):市场没跌而个股独自缩量下跌,最可能是**流动性/情绪性的错位**
    (有人需要卖、但没人认为公司变了),而不是信息驱动的重定价;信息重定价通常伴随放量
    (有人真的在重新配置)与市场同向(系统性因子)。这是与 uptrend-pullback **不同的一笔钱**。
    """
    mode = p.get("mode", "idio")
    m = (above_price_floor(ind) & (ind["amount60"] >= MIN_AMOUNT60)
         & ind["atr_pct"].notna() & ind["ret1"].notna() & ind["mkt_ret1"].notna())
    # A 个股自己跌
    m &= (ind["ret1"] <= -float(p["ret1_max"]))
    m &= ((ind["ret1"] - ind["mkt_ret1"]) <= -float(p["resid_max"]))
    # B 市场没跌(对照臂 mktdown 翻成「市场也在跌」)
    if mode == "mktdown":
        m &= (ind["mkt_ret1"] <= -0.01)
    else:
        m &= (ind["mkt_ret1"] >= -float(p["mkt_floor"]))
    # C 不放量(对照臂 highvol 翻成「放量」)
    if mode == "highvol":
        m &= (ind["vol_ratio"] >= float(p.get("vol_min", 1.5))) & ind["vol_ratio"].notna()
    else:
        m &= (ind["vol_ratio"] <= float(p["vol_max"])) & ind["vol_ratio"].notna()
    # D 相对自身便宜 / E 中期结构未坏
    m &= (ind["dd120"] <= -float(p["min_dd120"])) & ind["dd120"].notna()
    m &= (ind["above_ma60_ratio"] >= 0.40) & ind["above_ma60_ratio"].notna()
    # 附加:安静票 vs 高波动票(对照臂 highrvol)
    if mode == "highrvol":
        m &= (ind["rvol20"] >= float(p["rvol_min"])) & ind["rvol20"].notna()
    return m
