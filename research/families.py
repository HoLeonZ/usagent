"""其它价格行为策略族 —— 「从价格行为里挖低估机会」的候选实现。

每个族 = 一个**掩码函数**:给面板(已算好指标)返回一个 bool Series(该行是否发出买入信号)。
接口统一,便于用**同一套** WFV 协议横向比较(见 family_search.py),也便于把
「uptrend-pullback 的当前最佳」当作一条要跨过去的线。

纪律(与 signals.py 相同,必须逐条守住):
  * 只用 ≤T 的数据。所有现成指标都是因果的;族自身不得引入 shift(-x) / bfill /
    全样本分位阈值 / 未来才知道的池子。
  * 信号日 T 收盘 → T+1 开盘成交(由 broker 执行,本模块不碰执行)。
  * 空信号是**合法结果**,返回空表,由调用方按 no_signals 处理 —— 不得让它变成 0 收益。

预注册:参数网格写在 `family_defs/<id>.py` 的 `GRID` 里,**跑之前定稿**。改它 = 一次新的搜索,n_trials 要跟着涨。
本文件的族与 uptrend-pullback 的关系必须写在各自的 docstring 里 —— 「只换个阈值」不算新族。

**本模块现在是「注册表 + 兼容门面」**（2026-10-17 族注册表重构）：一个族的**声明**
（ID / 外围固定轴 / 预注册网格 / 排序分 / 状态门 / 对照臂）在 `research/family_defs/<id>.py`，
**掩码函数**在 `research/family_masks.py`（15 个族共用 10 个掩码，逐族复制会得到 5 份
必须手工同步的副本）。本模块只扫描定义体、按契约校验（`research/family_registry.py`）、
把既有名字原样导出 —— 外部调用点一行未改。**新增一个族 = 新增一个文件。**
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research import signals
from research import universe as mod_universe
from research.family_registry import load_dir
# 价格下限（审计 A4）搬到了 universe.py —— 它本来就是"某标的某天算不算机会集"的一部分，
# 放在那里也让族定义体可以 import 它而不产生循环依赖。这里按原名字再导出。
from research.universe import MIN_PRICE, above_price_floor  # noqa: F401


# ---------------------------------------------------------------- 口径说明
# 声明（族/网格/排序分）已经搬到 `family_defs/<id>.py`，但**应用**它们的地方
# 仍在本模块（`select` 怎么用排序分、怎么铺固定轴），所以这几段留在这里。

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


# 被**固定**的外围轴(不参与搜索;取值来自提案给的参考值)。
# 写在代码里而不是只写在注释里 —— 否则掩码会因为拿不到这些键而报错(实测踩过)。


# ---------------------------------------------------------------- 预注册网格
# ⚠️ 网格收缩声明:提案给的完整网格(有的到 243 组)会把 n_trials 推到 DSR 归零。
#    我按「只搜机制核心的那几个轴、外围轴固定在提案给的参考值」收缩到 ≤48 组/族。
#    这是**跑之前**做的收缩(不是看到结果之后改的);被固定的轴写在每族的注释里。
#
# ⚠️⚠️ 但**实际没做到**（2026-10-17 审计 F1，如实记录而不是改口径）：
#      `str` 的网格是 **96 组**（4×3×2×2×2），超过自己声明的 48。
#      而它唯一的方向对照臂 `str_inv` 只有 **6 组** ⇒ 两者搜索强度差 **16 倍**。
#      "方向有意义"这条结论正是靠这条对照得出的：搜得狠 16 倍的那一侧**更容易撞上
#      好参数**，所以那个 A/B 里混着"搜索强度"这个混淆项。
#
#      处置（本轮）：把**对照臂的网格与正题对齐**（见 family_defs/str_inv.py / family_defs/cap_inv.py 的 GRID 注释），
#      让 A/B 只差"方向"这一个变量；`GRID_CAP` 如实写成 96（= 实际最大值），
#      并加测试钉住"对照臂与正题的网格大小必须相等"。⚠️ 这**扩大了对对照臂的搜索**
#      （str_inv 6→96、cap_inv 12→48）⇒ 按 §5 迭代记账，这是新一轮、n_trials 要涨。


# ---------------------------------------------------------------- 注册表 + 门面
_DEFS_DIR = Path(__file__).resolve().parent / "family_defs"
_REGISTRY: dict[str, dict] = {}
load_dir(_REGISTRY, _DEFS_DIR)          # fail-fast：违规在 import 时就抛
if not _REGISTRY:
    raise ValueError(str(_DEFS_DIR) + " 里没有任何族定义 —— 空注册表会让横向比较静默跑空")

MASK = {k: r["mask"] for k, r in _REGISTRY.items()}
# 只收**声明了**固定轴的族：空 `FIXED` 等于"没有固定轴"，`set(FIXED)` 的含义
# （哪些族有外围固定轴）与重构前逐字一致；`select` 本来就按 `.get(fid, {})` 取。
FIXED = {k: r["fixed"] for k, r in _REGISTRY.items() if r["fixed"]}
# 没有网格的族（参考族）不进 GRIDS —— 进了会让 grid_size() 对一个不该搜的族给出 1
GRIDS = {k: r["grid"] for k, r in _REGISTRY.items() if r["grid"]}
SCORE_COL = {k: r["score_col"] for k, r in _REGISTRY.items() if r["score_col"]}
SCORE_NEGATED = frozenset(k for k, r in _REGISTRY.items() if r["score_negated"])
NEEDS_REGIME = frozenset(k for k, r in _REGISTRY.items() if r["needs_regime"])
CONTROL_ARM = {k: r["control"] for k, r in _REGISTRY.items() if r["control"]}
REFERENCE_FAMILIES = frozenset(k for k, r in _REGISTRY.items() if r["reference"])
NEW_FAMILIES = [k for k in MASK if k not in REFERENCE_FAMILIES]
# ⚠️ 顺序 = 定义体文件的**文件名排序**（此前是 MASK 字典的书写顺序）。它决定
# `family_search --families` 不给时的默认族序 ⇒ 报告里族的先后会变，数字不变。
# 写成排序而不是"碰巧的顺序"，是因为后者会在新增文件时无声改变报告。

GRID_CAP = 96

# 原符号名再导出（`tests/test_families.py` 的 F3 护栏按旧名引用掩码；
# 不改那两条测试 —— 它们守的是"两臂不止差状态"这个结论，与本次重构无关）
for _rec in _REGISTRY.values():
    for _alias in _rec["legacy_names"]:
        globals()[_alias] = _rec["mask"]
del _rec, _alias


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
