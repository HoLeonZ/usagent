"""其它价格行为策略族 —— 「从价格行为里挖低估机会」的候选实现。

每个族 = 一个**掩码函数**:给面板(已算好指标)返回一个 bool Series(该行是否发出买入信号)。
接口统一,便于用**同一套** WFV 协议横向比较(见 family_search.py),也便于把
「uptrend-pullback 的当前最佳」当作一条要跨过去的线。

纪律(与 signals.py 相同,必须逐条守住):
  * 只用 ≤T 的数据。所有现成指标都是因果的;族自身不得引入 shift(-x) / bfill /
    全样本分位阈值 / 未来才知道的池子。
  * 信号日 T 收盘 → T+1 开盘成交(由 broker 执行,本模块不碰执行)。
  * 空信号是**合法结果**,返回空表,由调用方按 no_signals 处理 —— 不得让它变成 0 收益。

预注册:参数网格写在 GRIDS 里,**跑之前定稿**。改它 = 一次新的搜索,n_trials 要跟着涨。
本文件的族与 uptrend-pullback 的关系必须写在各自的 docstring 里 —— 「只换个阈值」不算新族。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research import signals
from research import universe as mod_universe

logger = logging.getLogger(__name__)

# 流动性下界:与 uptrend-pullback 的 E 条件同一个数,保证「可交易的池子」一致
MIN_AMOUNT60 = signals.V21_SIGNAL_PARAMS["min_amount"]
MIN_PRICE = 1.0          # 价格下限:低于 $1 的票在美股里有额外的交易机制噪声(应声股/退市)

# ---------------------------------------------------------------- 新增指标(最多 2 个/族)
# 全部因果:只用 ≤T 的数据。


def add_family_indicators(ind: pd.DataFrame) -> pd.DataFrame:
    """给面板补各族需要的新指标 —— **全部因果**(只用 ≤T 的数据,无 shift(-x)/bfill)。

    一次算完(harness 只调一次),各族只读自己要的列。

    * ret5 / ret21 / ss_lo252 —— 短中期跌幅、距 252 日最低价的距离
    * clv            —— T 日收盘在当日振幅内的位置 ∈[0,1](≈1 收在最高)
    * vol_base60     —— **前 60 日**的平均量(shift(1) 把当日排除,否则恐慌 bar 自己
                        会把基准抬高、门槛自我削弱)
    * climax_ratio   —— volume / vol_base60(量能爆发倍数)
    * vwap_pre       —— 恐慌前价值中枢:截至 T-5 的 20 根 bar 的成交量加权均价
                        (尾部留 5 根,让中枢不被恐慌本身污染 —— 这正是它与 ma20 的差别)
    * px_disc        —— close / vwap_pre - 1(相对成交价值中枢的折价,负值)
    * tr_decay_rel   —— tr(T) / tr(T-1)("波动率是否跟进"的两个 bar 比值)
    * shock_z        —— ret1 / rvol20(标准化跌幅);_S 后缀列是 T-1 的取值
    * mkt_ret1 / mkt_vol20 —— 按**日期**聚合的等权市场收益与其 20 日标准差
                        (⚠️ 必须是按 date 聚合,按 symbol 分组会算成 0)
    * hh120 / dd120  —— 120 日价格包络上沿与相对它的折价
    * prior_low50    —— T **之前**的 50 日最低价(shift(1),不含当日)
    """
    out = ind.copy()
    g = out.groupby("symbol", group_keys=False)

    out["ret5"] = out["close"] / g["close"].shift(5) - 1.0
    out["ret21"] = out["close"] / g["close"].shift(21) - 1.0
    # 距 252 日最低价的距离。⚠️ 它要 252 根预热 ⇒ 在评测窗的前 ~6 个月整列是 NaN
    # (实测第一折训练窗 0 个可用行)。因此族里改用 ss_lo120(120 日,与评测起点重合);
    # ss_lo252 只作稳健性变体保留。
    ll252 = g["low"].transform(lambda s: s.rolling(252, min_periods=252).min())
    out["ss_lo252"] = out["close"] / ll252 - 1.0
    ll120 = g["low"].transform(lambda s: s.rolling(120, min_periods=120).min())
    out["ss_lo120"] = out["close"] / ll120 - 1.0

    # 日内承接度:收盘在当日振幅的位置。分母下限 0.01 防 high==low 的零振幅 bar。
    rng = (out["high"] - out["low"]).clip(lower=0.01)
    out["clv"] = (out["close"] - out["low"]) / rng

    # ⚠️ shift 必须写在 transform **里面**:写在外面是一个全局位移,每只票的第一行
    #    会取到上一只票最后一行 —— 静默前视/串票。本文件所有 .shift() 都遵守这条。
    out["vol_base60"] = g["volume"].transform(
        lambda s: s.rolling(60, min_periods=60).mean().shift(1))
    out["climax_ratio"] = out["volume"] / out["vol_base60"].replace(0.0, np.nan)

    num = (out["volume"] * out["close"]).groupby(out["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=20).sum().shift(5))
    den = out["volume"].groupby(out["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=20).sum().shift(5))
    out["vwap_pre"] = num / den.replace(0.0, np.nan)
    out["px_disc"] = out["close"] / out["vwap_pre"] - 1.0

    out["tr_decay_rel"] = out["tr"] / g["tr"].shift(1).replace(0.0, np.nan)
    out["shock_z"] = out["ret1"] / out["rvol20"].replace(0.0, np.nan)
    # 波动率恐慌族要跨两根 bar:S 日(恐慌)的读数在 T 日判定时取 shift(1)。
    for c in ("ret1", "rvol20", "low", "amount60", "tr", "shock_z", "prev_down_streak"):
        out[f"{c}_S"] = g[c].shift(1)   # groupby.shift:不跨标的
    out["volshock_score"] = -out["shock_z_S"]     # 越极端的恐慌越优先抢仓位槽

    # 按日期聚合的市场层(因果:只用当日及之前)
    # ⚠️ 市场层必须先在**日期轴**(每个交易日一行)上算完再映射回来。直接在行序上
    #    rolling 会变成「某只票连着 20 天」的标准差 —— 完全不是市场波动。
    by_date = out.groupby("date")["ret1"].mean().sort_index()
    mkt_vol = by_date.rolling(20, min_periods=20).std(ddof=1)
    out["mkt_ret1"] = out["date"].map(by_date)
    out["mkt_vol20"] = out["date"].map(mkt_vol)
    # 市场层的滞后同样必须在**日期轴**上做 shift,再映射回面板 —— 在行序上 shift 会变成
    # 「同一只票的上一行」,而市场层是日期的函数(实测踩过)。
    out["mkt_ret1_S"] = out["date"].map(by_date.shift(1))
    out["_resid"] = out["ret1"] - out["mkt_ret1"]      # 个股当日收益相对市场的那一块

    # 隔夜跳空与它在 5 日跌幅里的占比(横截面折价族用)
    out["gap1"] = out["open"] / out["prev_close"].replace(0.0, np.nan) - 1.0
    gap5 = out.groupby("symbol")["gap1"].transform(
        lambda s: s.rolling(5, min_periods=5).sum())
    out["gap5_ratio"] = (gap5 / out["ret5"].abs().replace(0.0, np.nan)).clip(0.0, 1.5)
    # 同日全池的 ret5 百分位(0 = 当日最弱)——与 xs_mom120 同构,只换源列
    out["xs_ret5"] = out.groupby("date")["ret5"].rank(pct=True)
    # 自身正常日均波幅(长周期折价族用它把「折价」标准化)
    out["atr_pct_ma120"] = g["atr_pct"].transform(
        lambda s: s.rolling(120, min_periods=120).mean())

    hh120 = g["high"].transform(lambda s: s.rolling(120, min_periods=120).max())
    out["hh120"] = hh120
    out["dd120"] = out["close"] / hh120 - 1.0
    out["dd_norm"] = out["dd120"] / out["atr_pct_ma120"].replace(0.0, np.nan)
    out["prior_low50"] = g["low"].transform(
        lambda s: s.rolling(50, min_periods=50).min().shift(1))
    # 排序分(降序抢仓位槽):折价越极端越优先
    out["lhr_score"] = -(out["dd120"] / out["atr_pct_ma120"].replace(0.0, np.nan))
    return out


# ---------------------------------------------------------------- 族的实现
def _mask_short_reversal(ind: pd.DataFrame, p: dict) -> pd.Series:
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
        & (ind["close"] >= MIN_PRICE)
        & (ind["amount60"] >= MIN_AMOUNT60)
        & ind["ret5"].notna() & ind["dd120"].notna()
        & ind["ss_lo120"].notna() & ind["amount60"].notna()
    )
    if p.get("max_atr_pct") is not None:
        m &= (ind["atr_pct"] <= float(p["max_atr_pct"])) & ind["atr_pct"].notna()
    if p.get("max_rvol20") is not None:
        m &= (ind["rvol20"] <= float(p["max_rvol20"])) & ind["rvol20"].notna()
    return m


def _mask_upb_ref(ind: pd.DataFrame, p: dict) -> pd.Series:
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




def _mask_volshock(ind: pd.DataFrame, p: dict) -> pd.Series:
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


def _mask_state_flip(ind: pd.DataFrame, p: dict) -> pd.Series:
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
    base = ind["close"] >= MIN_PRICE
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


def _mask_capitulation(ind: pd.DataFrame, p: dict) -> pd.Series:
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


def _mask_xs_flow(ind: pd.DataFrame, p: dict) -> pd.Series:
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


def _mask_long_horizon(ind: pd.DataFrame, p: dict) -> pd.Series:
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


def _mask_sf_stress(ind: pd.DataFrame, p: dict) -> pd.Series:
    """状态臂 A:**恐慌态**的放量杀跌(系统性、被强制)。语义翻转的一半。"""
    base = (ind["close"] >= MIN_PRICE) & (ind["amount60"] >= MIN_AMOUNT60)
    stress = (ind["mkt_vol20"] >= float(p["v_stress"])) & (ind["ew_index"] < ind["ew_ma"])
    a = (stress
         & (ind["ret1"] <= -float(p["ret1_max"]))
         & (ind["mkt_ret1"] <= -0.005)
         & ((ind["ret1"] - ind["mkt_ret1"]) >= -0.05)
         & (ind["vol_ratio"] >= float(p["vol_ratio_min"]))
         & (ind["clv"] <= float(p["clv_max"]))          # ← 收盘趴低 = 卖压持续(与 cap 族的符号相反)
         & (ind["dd120"] >= -0.60))
    return base & a & ind["atr_pct"].notna() & ind["ret1"].notna()


def _mask_sf_calm(ind: pd.DataFrame, p: dict) -> pd.Series:
    """状态臂 B:**平静态**的缩量阴跌(个体性、没人出价)。语义翻转的另一半。

    与臂 A 的区别就是两处符号:vol_ratio 要**低**(缩量)、残差要**显著为负**(跌是它自己造成的)。
    """
    base = (ind["close"] >= MIN_PRICE) & (ind["amount60"] >= MIN_AMOUNT60)
    calm = (ind["ew_index"] > ind["ew_ma"]) & (ind["breadth_sm"] >= 0.50)
    b = (calm
         & (ind["ret1"] <= -float(p["ret1_max"]))
         & ((ind["ret1"] - ind["mkt_ret1"]) <= -float(p["resid_max"]))
         & (ind["vol_ratio"] <= float(p["vol_ratio_max"]))   # ← 缩量:没有增量卖方,只是没人接
         & (ind["dd120"] >= -0.60)
         & (ind["above_ma60_ratio"] >= 0.40))
    return base & b & ind["atr_pct"].notna() & ind["ret1"].notna()


def _mask_idio(ind: pd.DataFrame, p: dict) -> pd.Series:
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
    m = ((ind["close"] >= MIN_PRICE) & (ind["amount60"] >= MIN_AMOUNT60)
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


MASK = {
    "idio": _mask_idio,
    "idio_mktdown": _mask_idio,      # 对照臂:市场也在跌
    "idio_highvol": _mask_idio,      # 对照臂:放量
    "idio_highrvol": _mask_idio,     # 对照臂:高波动票
    "str": _mask_short_reversal,
    "str_inv": _mask_short_reversal,     # 对照臂:买 5 日涨得多的
    "cap_inv": _mask_capitulation,       # 对照臂:要收盘趴低
    "sf_stress": _mask_sf_stress,
    "sf_calm": _mask_sf_calm,
    "xs_flow": _mask_xs_flow,
    "lhr": _mask_long_horizon,
    "volshock": _mask_volshock,
    "state_flip": _mask_state_flip,
    "cap": _mask_capitulation,
    "upb_ref": _mask_upb_ref,        # 参考族:不参与横向比较
}

# 每族的排序分(同日信号多于 max_positions 时用它抢仓位槽)。
# 族间不一致是**有意的**:排序是各族策略的一部分。默认 mom120 只为与 uptrend-pullback 可比。
#
# ⚠️ **符号约定**（这里是本文件出过错的地方，2026-10-17 审计 F2）：
#   排序一律**降序**（score 越大越先抢到仓位槽），所以"越小越优先"的族**必须取负**。
#   `idio*` 的机制是"残差越负（跌得越'自己'）越优先"，掩码要求 `_resid ≤ -resid_max`；
#   直接把裸 `__resid` 当 score 会让降序排在最前的是**残差最不负**（最像市场）的那次下跌
#   ——与声明的机制**相反**。`cap_inv` 同理：它要求 clv ≤ clv_max（收盘趴低），
#   却沿用了 cap 的 clv 降序 ⇒ 在低位集合里挑的是**最不极端**的一批。
#   `volshock_score`/`lhr_score` 在指标层已经取过负（见 add_family_indicators），
#   所以它们**不再**出现在下面的取负集合里 —— 两处都取负就会抵消。
SCORE_COL = {"volshock": "volshock_score", "cap": "clv", "cap_inv": "clv",
             "idio": "__resid", "idio_mktdown": "__resid",
             "idio_highvol": "__resid", "idio_highrvol": "__resid",
             "lhr": "lhr_score"}

# 这些族的 score 需要**取负**：它们的机制是"该列越小越优先"。
SCORE_NEGATED = frozenset({"idio", "idio_mktdown", "idio_highvol", "idio_highrvol",
                           "cap_inv"})

# 需要 regime 帧(状态列 ew_index/ew_ma/breadth_sm/regime_ok)的族
NEEDS_REGIME = frozenset({"state_flip", "volshock", "sf_stress", "sf_calm"})

# 被**固定**的外围轴(不参与搜索;取值来自提案给的参考值)。
# 写在代码里而不是只写在注释里 —— 否则掩码会因为拿不到这些键而报错(实测踩过)。
FIXED: dict[str, dict] = {
    "volshock": {"ret1_floor": -0.04, "mkt_floor": -0.010, "regime_gate": True},
    "cap": {"bad_ratio_max": 0.30, "px_disc_max": None},
    "lhr": {"ma20_buffer": 1.02},
    "idio": {"mode": "idio"},
    "idio_mktdown": {"mode": "mktdown"},
    "idio_highvol": {"mode": "highvol"},
    "idio_highrvol": {"mode": "highrvol"},
    "str": {},
    "str_inv": {"side": "rise"},
    "cap_inv": {"bad_ratio_max": 0.30, "px_disc_max": None, "clv_side": "max"},
    "sf_stress": {"v_stress": 0.012},
    "sf_calm": {"calm_breadth": 0.50},
    "state_flip": {},          # 提案里那几个常量直接写死在掩码里(它们是定义的一部分)
    "str": {},
}

# 参与横向比较的**新族**(参考族除外)
REFERENCE_FAMILIES = frozenset({"upb_ref"})
NEW_FAMILIES = [k for k in MASK if k not in REFERENCE_FAMILIES]

# ---------------------------------------------------------------- 预注册网格
# ⚠️ 网格收缩声明:提案给的完整网格(有的到 243 组)会把 n_trials 推到 DSR 归零。
#    我按「只搜机制核心的那几个轴、外围轴固定在提案给的参考值」收缩到 ≤48 组/族。
#    这是**跑之前**做的收缩(不是看到结果之后改的);被固定的轴写在每族的注释里。
GRIDS: dict[str, dict[str, list]] = {
    # 核:标准化恐慌强度 × 波动率是否跟进 × 恐慌低点是否守住。
    # 固定:ret1_floor=-0.04、mkt_floor=-0.010、regime_gate=on(提案的参考值)
    "volshock": {
        "shock_z_max": [-2.0, -2.5, -3.0],
        "tr_decay_max": [None, 0.4, 0.6, 0.8],     # None = 关闭「波动率不跟进」= 机制对照臂
        "hold_buffer": [None, 0.0, 0.005, 0.02],
    },
    # 核:5 个轴各 2 值 = 32(提案自己推荐的「可判定」版本,而非 243)
    # 固定:v_stress 之外的 6 个常量按提案
    "state_flip": {
        "v_stress": [0.009, 0.012],
        "r_a": [0.03, 0.06],
        "vr_a": [1.2, 2.0],
        "r_b": [0.02, 0.05],
        "vr_b": [0.8, 1.2],
    },
    # 核:量能爆发 × 跌幅(ATR 倍数) × 承接度。固定:bad_ratio_max=0.30、px_disc_max=None
    "cap": {
        "climax_ratio_min": [2.5, 3.0, 4.0, 6.0],
        "drop_atr": [1.0, 1.5, 2.0, 3.0],
        "clv_min": [0.40, 0.50, 0.60],
    },
    # 核:横截面便宜度 × 跳空占比 × 量能门。固定:结构门(close>ma120 ∧ ma60>ma120)、
    # ret1 带 [-2%,+5%]。⚠️ 提案另有 max_positions=10 / risk=0.5%,本比较统一用
    # risk=1% / mp=2(与 uptrend-pullback 一字不差),否则杠杆会被误读成边缘 —— 见报告。
    "xs_flow": {
        "xs_ret5_max": [0.05, 0.10, 0.20],
        "gap5_max": [0.20, 0.34],
        "vol_z_min": [0.0, 0.5],
    },
    # 核:折价倍数 × 站回 MA20 的缓冲 × 中期动量上限(不设 = 不要求长周期仍弱)
    # 固定:dd120 ≤ -0.25、amount60 ≥ 1e8、不开大盘门
    "lhr": {
        "dd_norm_k": [12.0, 14.0, 18.0],
        "xs_mom120_max": [None, 0.30, 0.50],
        "min_dd120": [0.20, 0.30],
    },
    # 正题:残差阈值 × 市场门 × 便宜度 × 量能上限 = 3×2×2×2 = 24
    "idio": {
        "ret1_max": [0.03, 0.04, 0.05],
        "resid_max": [0.02, 0.04],
        "mkt_floor": [0.0, 0.005],
        "min_dd120": [0.15, 0.25],
        "vol_max": [1.0],
    },
    # 三条对照臂:只动各自那一条轴,其余取正题的中位值
    # 对照臂也在各自那条轴上调两档(≥4 组),否则单一配置的「对照」没有说服力
    "idio_mktdown": {
        "ret1_max": [0.04, 0.05], "resid_max": [0.04], "min_dd120": [0.15, 0.25],
        "vol_max": [1.0],
    },
    "idio_highvol": {
        "ret1_max": [0.04], "resid_max": [0.04, 0.05], "mkt_floor": [0.0],
        "min_dd120": [0.20], "vol_min": [1.2, 2.0],
    },
    "idio_highrvol": {
        "ret1_max": [0.04, 0.05], "resid_max": [0.04], "mkt_floor": [0.0],
        "min_dd120": [0.20], "vol_max": [1.0], "rvol_min": [0.04, 0.06],
    },
    # 对照臂:与 str 同样的跌/涨阈值,只把方向翻过来
    "str_inv": {
        "drop5": [0.05, 0.08, 0.12],
        "min_drawdown": [0.20, 0.35],
        "min_off_low": [0.0],
        "max_atr_pct": [None],
        "max_rvol20": [None],
    },
    # 对照臂:吸收族要 clv 高,这条要 clv 低(clv_max = 1 − clv_min)
    "cap_inv": {
        "climax_ratio_min": [3.0, 4.0],
        "drop_atr": [1.5, 2.0],
        "clv_max": [0.45, 0.55, 0.60],
    },
    # 合成把状态族拆成两臂:恐慌臂 12 组(含 clv 符号轴)、平静臂 18 组
    "sf_stress": {
        "ret1_max": [0.035, 0.045, 0.06],
        "vol_ratio_min": [1.5, 2.0],
        "clv_max": [0.45, 0.55],
    },
    "sf_calm": {
        "ret1_max": [0.035, 0.04, 0.05],
        "vol_ratio_max": [0.6, 0.7, 0.8],
        "resid_max": [0.03, 0.05],
    },
    "str": {
        "drop5": [0.05, 0.08, 0.12, 0.18],
        "min_drawdown": [0.20, 0.35, 0.50],
        "min_off_low": [0.0, 0.05],
        "max_atr_pct": [None, 0.10],
        "max_rvol20": [None, 0.06],
    },
}


def grid_size(family_id: str) -> int:
    n = 1
    for v in GRIDS[family_id].values():
        n *= len(v)
    return n


def select(ind: pd.DataFrame, family_id: str, params: dict, lo: str, hi: str,
           reg: pd.DataFrame | None = None, *, use_regime: bool = False) -> pd.DataFrame:
    """跑一个族,返回与 signals.ENTRY_COLS 同构的信号表(左闭右开 [lo, hi))。

    use_regime 默认 **False** —— 大盘门是 uptrend-pullback 的条件,新族默认不继承它,
    否则「新族」实际上还挂在旧族的框架上。要开就显式开(并计入该族的口径决策)。
    """
    if family_id not in MASK:
        raise KeyError(f"未知策略族 {family_id!r};可用:{sorted(MASK)}")
    # 指标只算一次:调用方(harness)通常已准备好带族指标的面板;这里做个幂等检查
    # 免得每次评估都在 425k 行上重算 rolling/shift。
    df = ind if "ret5" in ind.columns else add_family_indicators(ind)
    if family_id in NEEDS_REGIME:
        if reg is None:
            raise ValueError(f"{family_id} 需要 regime 帧(状态门),调用方必须传 reg=")
        need = [c for c in reg.columns if c != "date" and c not in df.columns]
        if need:
            df = df.merge(reg[["date", *need]].drop_duplicates("date"), on="date", how="left")
    params = {**FIXED.get(family_id, {}), **params}   # 固定轴先铺底,网格轴覆盖它
    m = MASK[family_id](df, params)
    # 池资格（§3 PIT Mandate）：族也只在**可投资集**里找信号，缺列 fail-fast。
    m &= mod_universe.require_eligible_mask(df)
    m &= (df["date"] >= pd.Timestamp(lo)) & (df["date"] < pd.Timestamp(hi))
    hits = df[m]
    if use_regime and reg is not None and not hits.empty:
        ok = set(pd.to_datetime(reg.loc[reg["regime_ok"], "date"]).tolist())
        hits = hits[hits["date"].isin(ok)]
    if hits.empty:
        return pd.DataFrame(columns=signals.ENTRY_COLS)

    hits = hits.copy()
    # 排序分:族间的排序口径要一致,否则「超容量当 T 丢弃」这一步不可比。
    # 统一用 mom120(与 uptrend-pullback 相同):分数只决定同日多信号时的先后,
    # 不参与是否入选 —— 这样族与族之间的差异只在**信号集**,不在容量分配。
    _sc = SCORE_COL.get(family_id, "mom120")
    _raw = hits["_resid"] if _sc == "__resid" else hits[_sc]
    # 符号约定见 SCORE_COL 的注释：排序是降序，所以"越小越优先"的族要取负。
    hits["score"] = -_raw if family_id in SCORE_NEGATED else _raw
    hits["sig_close"] = hits["close"]
    hits["sig_ma20"] = hits["ma20"]
    hits["sig_ma60"] = hits["ma60"]
    hits["sig_down_streak"] = hits["down_streak"]
    hits["sig_pullback"] = hits["pullback"]
    hits["sig_mom120"] = hits["mom120"]
    hits["sig_amount60"] = hits["amount60"]
    return hits[signals.ENTRY_COLS].sort_values(
        ["date", "score"], ascending=[True, False]).reset_index(drop=True)
