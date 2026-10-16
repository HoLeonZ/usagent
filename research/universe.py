"""PIT（时间点）股票池资格 —— 机构里这是数据层的第一条硬要求（§3 PIT Mandate）。

现状（2026-10-17 审计 A2）:

  * preset 里声明了 `universe`，但**引擎不读它**（`core/preset_contract.py` 的
    docstring 自己承认"universe 是元数据，引擎不读它"）；
  * `panel.load_full_panel(universe_path=...)` 这个形参**从未被使用**（死参数）——
    正是本仓库为根除"声明了却没人读"而建契约的那类缺陷的残余；
  * DB 的填充名单来自 `data/universe_oi100.csv`，那是一份**今日**「期权 OI≥100 +
    价差可接受」的快照。把"今天还在 Nasdaq 上市、今天还有期权 OI"倒着投影到 3 年
    历史上，等于同时引入两种偏差：已退市/被并购/流动性枯竭的名字整批缺席
    （**幸存者偏差**），而名单本身是未来信息（**前视**）。

本模块只负责一件事：**把「某标的在某天算不算我的机会集」定义成一个因果可判定的函数**。
它不删行（删行会缩短指标预热、改变截面排名的分母），而是给出逐 (symbol, date) 的
`eligible` 布尔列，由下游各取所需（信号选择、横截面排名、基准构成）。

三种模式，全部**因果**（T 日的资格只依赖 ≤T 的信息）:

  * `membership` —— 显式成分区间表 `(symbol, start, end)`。拿到真实成分历史时用这个，
    它是对"历史可投资集"最忠实的表达。
  * `liquidity`  —— 用**截至 T** 的滚动成交额门槛定义"此刻可交易"。不需要历史成分
    数据，且因果性来自构造：窗口右端就是 T。这是本项目当前数据条件下能做到的
    最好的非幸存者口径。
  * `all`        —— 不分池（= 现状）。必须**显式**声明，且会打 warning：
    它等于"用今天的名单回测过去"，是幸存者偏差最重的口径。

未声明（spec=None）等价于 `all`，但同样会打 warning —— 静默沿用最差口径，
正是本模块要消灭的东西。
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MODE_MEMBERSHIP = "membership"
MODE_LIQUIDITY = "liquidity"
MODE_ALL = "all"
MODES = (MODE_MEMBERSHIP, MODE_LIQUIDITY, MODE_ALL)

# liquidity 模式的默认值：60 个交易日的滚动成交额 ≥ $3e7。
# 门槛的来源不是调参：hiagent 的 E 条件本来就用 amount60 ≥ 3e7（A 股口径），
# 这里把同一条门槛**改成因果的逐日判定**（原来是"全窗已实现"的池子筛选）。
DEFAULT_LOOKBACK = 60
DEFAULT_MIN_DOLLAR_VOLUME = 3e7

# 本项目的默认 PIT 口径（presets 里各 preset 显式写同一份声明，便于逐 preset 覆盖）。
# 为什么默认是 liquidity 而不是 all：'all' = 用"今天还在榜上"的名单回测过去，
# 幸存者偏差最重。liquidity 不需要历史成分数据，且因果性来自构造（窗口右端就是 T）。
DEFAULT_SPEC: dict = {"mode": MODE_LIQUIDITY,
                      "min_dollar_volume": DEFAULT_MIN_DOLLAR_VOLUME,
                      "lookback": DEFAULT_LOOKBACK}


class UniverseError(RuntimeError):
    """股票池声明不合法 —— fail-fast（§0）。"""


@dataclass(frozen=True)
class UniverseSpec:
    """一份取池声明。`mode` 决定用哪种资格函数。"""
    mode: str
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME
    lookback: int = DEFAULT_LOOKBACK
    path: str | None = None            # membership 模式的区间表路径
    min_price: float | None = None     # 可选的因果价格下限（用 close(T)）
    note: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise UniverseError(f"universe.mode 必须是 {MODES} 之一，得到 {self.mode!r}")
        if self.mode == MODE_MEMBERSHIP and not self.path:
            raise UniverseError("membership 模式必须给 path（成分区间表）")
        if self.mode == MODE_LIQUIDITY:
            if self.lookback < 2:
                raise UniverseError(f"lookback 至少 2 个交易日，得到 {self.lookback}")
            if self.min_dollar_volume < 0:
                raise UniverseError(
                    f"min_dollar_volume 必须 ≥ 0，得到 {self.min_dollar_volume}")
        if self.min_price is not None and self.min_price < 0:
            raise UniverseError(f"min_price 必须 ≥ 0，得到 {self.min_price}")


def parse(spec: dict | str | None) -> UniverseSpec:
    """把 preset 里的 `universe` 字段解析成 UniverseSpec。

    兼容历史写法（字符串）：`"all"` / `"oi100"` / `"tight"` —— 但这三个名字都只是
    「截至生成日的一张静态名单」，一律映射到 `all` 并打 warning：
    它们**不是** PIT 声明，不能假装是。
    """
    if spec is None:
        return UniverseSpec(mode=MODE_ALL, note="未声明")
    if isinstance(spec, str):
        logger.warning(
            "universe=%r 是历史字符串写法（静态名单，非 PIT）—— 按 'all' 处理。"
            "要真正的 PIT 口径请写成 dict：{'mode': 'liquidity', ...} 或 "
            "{'mode': 'membership', 'path': ...}", spec)
        return UniverseSpec(mode=MODE_ALL, note=f"静态名单 {spec}")
    if not isinstance(spec, dict):
        raise UniverseError(f"universe 必须是 dict/str/None，得到 {type(spec).__name__}")
    known = {"mode", "min_dollar_volume", "lookback", "path", "min_price", "note"}
    unknown = sorted(set(spec) - known)
    if unknown:
        raise UniverseError(
            f"universe 含未知键 {unknown} —— 拼错的键不会被静默忽略（§0 Fail-Fast）。"
            f"可用：{sorted(known)}")
    return UniverseSpec(**spec)


def describe(spec: UniverseSpec) -> str:
    """一句话口径说明（进报告与日志）。"""
    if spec.mode == MODE_MEMBERSHIP:
        return f"PIT 成分区间表 {spec.path}"
    if spec.mode == MODE_LIQUIDITY:
        out = (f"PIT 流动性池:截至 T 的 {spec.lookback} 日滚动成交额 "
               f"≥ ${spec.min_dollar_volume:,.0f}")
        if spec.min_price is not None:
            out += f",且 close(T) ≥ ${spec.min_price:g}"
        return out
    return "全池(未过滤) —— 幸存者偏差最重的口径"


def load_membership(path: str | Path) -> pd.DataFrame:
    """读成分区间表：列 = symbol, start, end（含端点，闭区间）。

    `end` 允许为空 —— 表示"至今仍在"。表缺失或列不对是配置错误，fail-fast。
    """
    p = Path(path)
    if not p.exists():
        raise UniverseError(f"成分区间表不存在：{p}")
    rows = []
    with p.open() as fh:
        rd = csv.DictReader(fh)
        missing = {"symbol", "start", "end"} - set(rd.fieldnames or [])
        if missing:
            raise UniverseError(f"{p} 缺列 {sorted(missing)}；需要 symbol,start,end")
        for r in rd:
            sym = (r["symbol"] or "").strip().upper()
            if not sym:
                continue
            start = (r["start"] or "").strip()
            if not start:
                raise UniverseError(f"{p}:{sym} 的 start 为空 —— 区间必须有起点")
            rows.append({"symbol": sym,
                         "start": pd.Timestamp(start),
                         "end": pd.Timestamp(r["end"].strip()) if (r["end"] or "").strip()
                         else pd.Timestamp.max})
    if not rows:
        raise UniverseError(f"{p} 里没有任何成分区间")
    out = pd.DataFrame(rows)
    bad = out[out["end"] < out["start"]]
    if len(bad):
        raise UniverseError(f"{p} 里有 {len(bad)} 条 end < start 的区间（例："
                            f"{bad.iloc[0].to_dict()}）")
    return out


def eligibility(panel: pd.DataFrame, spec: UniverseSpec,
                membership: pd.DataFrame | None = None) -> pd.Series:
    """逐行的资格布尔列。**因果**：第 i 行的取值只依赖 ≤ 该行日期的信息。

    ⚠️ 滚动窗口必须在 **(symbol, date) 有序**的视图上算：`Series.rolling` 按行的物理
       顺序取窗口，面板若未排序（或调用方先做过 filter/sort），窗口就会串到别的日期上
       —— 那既是错的数值，也是一条**看未来的路径**。所以这里不假设调用方排过序，
       内部自己排、算完再按原索引还原。返回的 Series 与输入行序一一对应。

    panel 需含 [symbol, date, amount]（amount = volume × as-traded close，
    见 panel.to_panel_frame）。min_price 需要 [close]。
    """
    if panel.empty:
        return pd.Series([], dtype=bool, index=panel.index)
    for col in ("symbol", "date"):
        if col not in panel.columns:
            raise UniverseError(f"panel 缺列 {col!r}")

    if spec.mode == MODE_ALL:
        ok = pd.Series(True, index=panel.index)

    elif spec.mode == MODE_LIQUIDITY:
        if "amount" not in panel.columns:
            raise UniverseError("liquidity 模式需要 panel 的 amount 列")
        # 有序视图 → 算 → 还原（顺序无关性由 test_universe 钉住）
        ordered = panel.sort_values(["symbol", "date"], kind="mergesort")
        g = ordered.groupby("symbol", sort=False)["amount"]
        # min_periods=lookback：不足一个完整窗口 ⇒ 不可判定 ⇒ 不入选。
        # 这条对**新上市/新变流动**的名字是保守的，也是正确的：那时你还不知道它的量。
        adv = g.transform(lambda s: s.rolling(spec.lookback,
                                              min_periods=spec.lookback).mean())
        ok = (adv >= spec.min_dollar_volume).reindex(panel.index)

    else:  # membership
        mem = membership if membership is not None else load_membership(spec.path)
        ok = pd.Series(False, index=panel.index)
        dates = pd.to_datetime(panel["date"])
        syms = panel["symbol"]
        for r in mem.itertuples(index=False):
            ok |= (syms == r.symbol) & (dates >= r.start) & (dates <= r.end)

    if spec.min_price is not None:
        if "close" not in panel.columns:
            raise UniverseError("min_price 需要 panel 的 close 列")
        # 用 close(T)：T 日收盘后即可判定，信号在 T 收盘生成、T+1 开盘执行 ⇒ 因果。
        ok &= panel["close"] >= spec.min_price

    return ok.fillna(False).astype(bool)


def require_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """行级资格掩码。**缺列即 fail-fast**，不是静默当作全合格。

    为什么必填而不是可选：可选掩码 = 存在一条"忘了带池子"的路径能跑出数字，
    而那正是本项目原来的状态（池子的声明与消费完全脱钩）。必填之后，
    "这段回测用的是哪个池子"在类型层面就绕不过去。
    """
    if "eligible" not in df.columns:
        raise UniverseError(
            "面板缺 eligible 列 —— 池资格是**必填**口径（§3 PIT Mandate）。"
            "用 panel.load_full_panel(universe=...) / load_panel_for_preset() 载入；"
            "手拼面板时必须显式带上这一列（合成 fixture 见 research/tests/conftest.py）。")
    return df["eligible"].fillna(False).astype(bool)


def attach(panel: pd.DataFrame, spec: UniverseSpec | None) -> pd.DataFrame:
    """给 panel 挂上 `eligible` 列（不删行）。

    为什么不删行：删行会缩短每只票的指标预热，也会改变横截面排名的分母。
    "能不能交易"与"要不要把它的历史算进指标"是两件事，混在一起会让
    warmup 与排名口径都变得不可解释。
    """
    s = parse(spec) if not isinstance(spec, UniverseSpec) else spec
    out = panel.copy()
    out["eligible"] = eligibility(out, s)
    n = int(out["eligible"].sum())
    logger.info("universe[%s]: %d/%d 行合格（%s）", s.mode, n, len(out), describe(s))
    if s.mode == MODE_ALL:
        logger.warning(
            "universe 模式为 'all'（%s）—— 回测池 = DB 里恰好有的标的，"
            "存在幸存者/前视偏差，不能当作机会集口径", s.note or "未声明")
    return out
