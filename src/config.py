"""usagent 数据层配置。

所有可调参数集中在这里，脚本不要各自硬编码。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "usagent.db"

MASTER_UNIVERSE = DATA / "universe_nasdaq_all.json"
UNIVERSE_OI100 = DATA / "universe_oi100.csv"
UNIVERSE_TIGHT = DATA / "universe_tight.csv"

# 历史窗口：3 年
HISTORY_YEARS = 3

# 期权链快照只保留这些范围，避免把全市场全链落成几十 GB
#
# 注意：不要用「现货 ±X%」当唯一筛选条件 —— 低价股的行权价间距可达 $2.50，
# 而 ±10% 的 $4.5 股票只有 ±$0.45，会把整条链滤成空集（实测 100 只全空，
# 99/100 是股价 <$10）。所以主口径是「最接近现货的 N 档」，百分比只做上限保护。
CHAIN_MAX_STRIKES = 20        # 每个到期保留最接近现货的 N 档行权价
CHAIN_STRIKE_BAND = 0.30      # 同时要求落在现货 ±30% 内（防止行权价间距极大时抓到离谱的档）
CHAIN_DTE_MIN = 7
CHAIN_DTE_MAX = 120
CHAIN_MAX_EXPIRIES = 3        # 每次快照保留最接近 ATM 的 N 个到期

# 复权口径：按报告结论存「未复权 + 公司行动」，读取时现算，绝不落库前复权序列
OHLCV_AUTO_ADJUST = False
OHLCV_ACTIONS = True

# 抓取节流（Yahoo 已两次限流，这里保守取）
BATCH_SIZE = 10               # OHLCV 批量下载的每批标的数（每只约 3-5s，每批约 40s）
MIN_INTERVAL = 0.35           # 单次请求之间的最小间隔（秒）
MAX_RETRY = 4
BACKOFF_BASE = 3.0            # 退避基数：3s / 9s / 27s / 81s


def history_start(today: date | None = None) -> date:
    t = today or date.today()
    return t - timedelta(days=365 * HISTORY_YEARS)


@dataclass(frozen=True)
class Universe:
    name: str
    symbols: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.symbols)


def load_universe(name: str = "oi100") -> Universe:
    """name: all | oi100 | tight"""
    if name == "all":
        rows = json.loads(MASTER_UNIVERSE.read_text())
        return Universe("all", tuple(r["symbol"] for r in rows))
    path = {"oi100": UNIVERSE_OI100, "tight": UNIVERSE_TIGHT}.get(name)
    if path is None:
        raise ValueError(f"未知 universe: {name}")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return Universe(name, tuple(r["symbol"] for r in rows))
