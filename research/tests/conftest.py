"""pytest 配置:让 research 包可从仓库根导入。

pytest 的 prepend 导入模式会把测试文件所在目录放进 sys.path;本仓库的
research/ 不是 src/ 那样的扁平目录,import 需要**仓库根**在 path 上。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BARS = 360


def make_synth_panel(symbols: tuple[str, ...] = ("AAA", "BBB"), bars: int = BARS) -> pd.DataFrame:
    """确定性合成 panel —— 无随机数,所以因果性断言是可复现的。

    价格用「正弦震荡 + 缓慢上行」,保证同时存在上涨日与下跌日(down_streak 才有意义);
    不同 symbol 用不同相位,避免退化成同一序列。
    """
    dates = pd.bdate_range("2020-01-01", periods=bars)
    frames = []
    for k, sym in enumerate(symbols):
        i = np.arange(bars, dtype=float)
        phase = k * 1.7
        close = 100.0 + 8.0 * np.sin(i / 5.0 + phase) + 0.05 * i
        open_ = close * (1.0 + 0.002 * np.sin(i / 3.0 + phase))
        high = np.maximum(open_, close) * 1.004
        low = np.minimum(open_, close) * 0.996
        volume = 1_000_000.0 + 1_000.0 * i + 50_000.0 * k
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates,
            "open": open_, "high": high, "low": low, "close": close,
            # as-traded 价与复权价相同(合成分红为 0)⇒ 双价格体系对这份 panel 是恒等操作。
            "raw_open": open_, "raw_high": high, "raw_low": low, "raw_close": close,
            "volume": volume, "amount": volume * close, "dividends": 0.0,
            # 池资格是**必填**列（§3 PIT Mandate）：合成面板的执行语义是
            # "全员随时可交易"，所以这里恒为 True。要测取池行为请显式改这一列。
            "eligible": True,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "date"]).reset_index(drop=True)


@pytest.fixture
def synth_panel() -> pd.DataFrame:
    return make_synth_panel()
