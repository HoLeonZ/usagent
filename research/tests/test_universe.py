"""PIT 取池资格 —— 因果性是这一层唯一的硬要求。

动机（2026-10-17 审计 A2）：`preset.universe` 引擎不读、`load_full_panel(universe_path=)`
是死参数、回测池 = DB 里恰好有什么（而 DB 名单来自**今日**的期权 OI 快照）
⇒ 幸存者偏差 + 前视。本文件钉住"资格函数是因果的"这件事，而不是钉住某个具体池子。

最关键的三条：
  1. **不看未来**：只在之后才变流动的名字，在之前必须不合格；
  2. **顺序无关**：面板没排序时，滚动窗口不得串到别的日期上（否则既错又前视）；
  3. **不删行**：`attach` 只挂列 —— 删行会缩短预热、改变截面排名的分母。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import universe as U  # noqa: E402


def _panel(symbols_amounts: dict[str, list[float]], start="2024-01-01",
           price=10.0) -> pd.DataFrame:
    frames = []
    for sym, amts in symbols_amounts.items():
        dates = pd.bdate_range(start, periods=len(amts))
        frames.append(pd.DataFrame({"symbol": sym, "date": dates,
                                    "amount": amts, "close": price}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 声明解析

def test_parse_defaults_to_all_with_a_reason():
    s = U.parse(None)
    assert s.mode == U.MODE_ALL and "未声明" in s.note


def test_parse_legacy_string_maps_to_all(caplog):
    """历史字符串写法（oi100/tight）是静态名单，不是 PIT 声明 —— 不得假装是。"""
    with caplog.at_level("WARNING"):
        s = U.parse("oi100")
    assert s.mode == U.MODE_ALL
    assert any("非 PIT" in r.message for r in caplog.records)


def test_parse_rejects_unknown_keys():
    with pytest.raises(U.UniverseError, match="未知键"):
        U.parse({"mode": "liquidity", "lookbackk": 60})


def test_parse_validates_mode_and_params():
    with pytest.raises(U.UniverseError):
        U.parse({"mode": "nope"})
    with pytest.raises(U.UniverseError):
        U.parse({"mode": "membership"})                  # 缺 path
    with pytest.raises(U.UniverseError):
        U.parse({"mode": "liquidity", "lookback": 1})
    with pytest.raises(U.UniverseError):
        U.parse({"mode": "liquidity", "min_dollar_volume": -1})


def test_describe_is_human_readable():
    d = U.describe(U.parse({"mode": "liquidity", "min_dollar_volume": 3e7, "lookback": 60}))
    assert "60" in d and "30,000,000" in d
    assert "幸存者偏差最重" in U.describe(U.parse(None))


# ---------------------------------------------------------------- 成分区间表

def test_membership_roundtrip(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("symbol,start,end\nAAPL,2020-01-01,2024-06-30\nMSFT,2021-01-01,\n")
    mem = U.load_membership(p)
    assert len(mem) == 2
    assert mem.loc[mem["symbol"] == "MSFT", "end"].iloc[0] == pd.Timestamp.max


def test_membership_errors(tmp_path):
    with pytest.raises(U.UniverseError, match="不存在"):
        U.load_membership(tmp_path / "nope.csv")
    p = tmp_path / "bad.csv"
    p.write_text("symbol,start\nAAPL,2020-01-01\n")
    with pytest.raises(U.UniverseError, match="缺列"):
        U.load_membership(p)
    p2 = tmp_path / "rev.csv"
    p2.write_text("symbol,start,end\nAAPL,2024-01-01,2020-01-01\n")
    with pytest.raises(U.UniverseError, match="end < start"):
        U.load_membership(p2)
    p3 = tmp_path / "empty.csv"
    p3.write_text("symbol,start,end\n")
    with pytest.raises(U.UniverseError, match="没有任何成分区间"):
        U.load_membership(p3)


def test_membership_intervals_are_inclusive_and_bounded(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("symbol,start,end\nAAA,2024-01-05,2024-01-10\n")
    panel = _panel({"AAA": [1e9] * 10})
    s = U.UniverseSpec(mode=U.MODE_MEMBERSHIP, path=str(p))
    ok = U.eligibility(panel, s)
    got = panel.loc[ok, "date"].dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]  # 含端点、跳过周末


# ---------------------------------------------------------------- 因果性（核心）

def test_liquidity_matches_an_independent_naive_reimplementation():
    """逐日资格必须与"只用手算的、显然因果的"朴素实现在每一点上一致。

    为什么用朴素实现当参照而不是写死一个期望值：写死的期望值只能钉住我想到的那一种
    情形，而这里要钉住的是"实现没有偷偷用未来"。朴素实现逐点扫描 ≤T 的窗口，
    它看未来在构造上就是不可能的。
    """
    n = 120
    amts = [1e5] * (n // 2) + [1e9] * (n // 2)
    lookback, thr = 20, 3e7
    panel = _panel({"AAA": amts})
    got = U.eligibility(panel, U.UniverseSpec(
        mode=U.MODE_LIQUIDITY, min_dollar_volume=thr, lookback=lookback)).tolist()
    want = []
    for t in range(n):
        window = amts[max(0, t - lookback + 1): t + 1]      # 只用 ≤T
        want.append(len(window) == lookback and sum(window) / lookback >= thr)
    assert got == want
    assert not any(want[:n // 2]), "放量之前的任何一根都不该合格"
    assert want[n // 2], "第一根放量当天就该合格（它的成交量在收盘时已知）"


def test_liquidity_short_history_is_ineligible():
    """不足一个完整窗口 ⇒ 不可判定 ⇒ 不入选（对新上市的名字是保守且正确的）。"""
    panel = _panel({"AAA": [1e9] * 5})
    s = U.UniverseSpec(mode=U.MODE_LIQUIDITY, min_dollar_volume=3e7, lookback=20)
    assert not U.eligibility(panel, s).any()


def test_eligibility_is_order_independent():
    """面板行序被打乱后，逐 (symbol,date) 的资格必须一模一样。

    滚动窗口按物理行序取窗口；若不先排序，窗口会串到别的日期上 —— 那既是错的数值，
    也是一条看未来的路径。
    """
    panel = _panel({"AAA": [1e5, 1e8, 1e8, 1e9, 1e9], "BBB": [1e9] * 5})
    s = U.UniverseSpec(mode=U.MODE_LIQUIDITY, min_dollar_volume=3e7, lookback=3)
    base = panel.assign(ok=U.eligibility(panel, s))
    shuffled = panel.sample(frac=1.0, random_state=7).reset_index(drop=True)
    got = shuffled.assign(ok=U.eligibility(shuffled, s))
    key = ["symbol", "date"]
    a = base.sort_values(key)[key + ["ok"]].reset_index(drop=True)
    b = got.sort_values(key)[key + ["ok"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_min_price_is_causal_on_close():
    panel = _panel({"AAA": [1e9] * 5})
    panel.loc[panel.index[:2], "close"] = 0.5
    s = U.UniverseSpec(mode=U.MODE_LIQUIDITY, min_dollar_volume=3e7, lookback=2,
                       min_price=1.0)
    ok = U.eligibility(panel, s)
    assert not ok.iloc[:2].any() and ok.iloc[2:].all()


def test_missing_columns_fail_fast():
    panel = _panel({"AAA": [1e9] * 5}).drop(columns=["amount"])
    with pytest.raises(U.UniverseError, match="amount"):
        U.eligibility(panel, U.parse({"mode": "liquidity"}))
    with pytest.raises(U.UniverseError, match="symbol"):
        U.eligibility(pd.DataFrame({"date": [1], "amount": [1.0]}), U.parse(None))


# ---------------------------------------------------------------- attach

def test_attach_adds_a_column_without_dropping_rows(caplog):
    panel = _panel({"AAA": [1e5] * 30, "BBB": [1e9] * 30})
    out = U.attach(panel, {"mode": "liquidity", "min_dollar_volume": 3e7, "lookback": 10})
    assert len(out) == len(panel), "attach 不得删行（会缩短预热、改变截面排名的分母）"
    assert "eligible" in out.columns
    assert set(out.loc[out["eligible"], "symbol"]) == {"BBB"}


def test_attach_warns_loudly_for_all_mode(caplog):
    with caplog.at_level("WARNING"):
        U.attach(_panel({"AAA": [1e9] * 5}), None)
    assert any("幸存者/前视偏差" in r.message for r in caplog.records), \
        "沿用最差口径必须出声，不能静默"


# ---------------------------------------------------------------- 连续性（A3）

def test_continuity_requires_showing_up_on_most_sessions():
    """出勤率是"此刻可交易"的一部分：一个月里只交易 10 天的名字，不该进机会集。

    实测（2026-10-17）池内有 10 只标的在自身历史里缺一半以上的交易日，它们近期
    成交额够大所以过了 ADV 门槛，但**不是每个交易日都有成交** —— 引擎会在无 bar 的
    日子里按陈价重估、持仓天数照走（审计 A3）。
    """
    axis = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for i, d in enumerate(axis):
        rows.append(("A", d, 1e9))
        if i % 3 == 0:
            rows.append(("B", d, 1e9))          # B 每 3 天才有 1 天有 bar
    panel = pd.DataFrame(rows, columns=["symbol", "date", "amount"])
    panel["raw_close"] = panel["close"] = 10.0
    panel["splits"] = 0.0
    ok = U.continuity_ok(panel, 10, 0.9)
    assert ok[panel["symbol"] == "A"].any(), "每天都有 bar 的名字应当合格"
    assert not ok[panel["symbol"] == "B"].any(), "三缺二的名字不该合格"


def test_continuity_counts_against_the_global_axis_not_row_count():
    """按行数算出勤永远是满勤 —— 必须对着**全局交易日**数。"""
    rows = [("A", d, 1e9) for d in pd.bdate_range("2024-01-01", periods=20)]
    rows += [("B", d, 1e9) for d in pd.bdate_range("2024-01-01", periods=5)]
    panel = pd.DataFrame(rows, columns=["symbol", "date", "amount"])
    panel["raw_close"] = panel["close"] = 10.0
    panel["splits"] = 0.0
    ok = U.continuity_ok(panel, 10, 0.9)
    assert not ok[panel["symbol"] == "B"].any(), \
        "B 只有 5 根 bar（对它自己而言是'满勤'），但对着全局轴只有 25% 出勤"


def test_liquidity_eligibility_requires_both_adv_and_continuity():
    axis = pd.bdate_range("2024-01-01", periods=30)
    rows = [("A", d, 1e9) for d in axis] + [("B", d, 1e9) for d in axis[::3]]
    panel = pd.DataFrame(rows, columns=["symbol", "date", "amount"])
    panel["raw_close"] = panel["close"] = 10.0
    panel["splits"] = 0.0
    spec = U.parse({"mode": "liquidity", "lookback": 10, "min_dollar_volume": 0.0})
    ok = U.eligibility(panel, spec)
    assert ok[panel["symbol"] == "A"].sum() > 0
    assert ok[panel["symbol"] == "B"].sum() == 0, "成交额够大但不出勤 ⇒ 仍不合格"


def test_describe_mentions_the_coverage_requirement():
    d = U.describe(U.parse({"mode": "liquidity"}))
    assert "出勤率" in d
