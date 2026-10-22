"""池资格必须是**承重**的，而不是装饰。

动机（2026-10-17 审计 A2，机构里最主要的一类偏差）：preset 声明了 `universe`，
但引擎不读它，`load_full_panel(universe_path=)` 是死参数，回测池 = "DB 里恰好有什么"
（而 DB 名单来自**今日**的期权 OI 快照）⇒ 幸存者偏差 + 前视。

这一层修好之后，"这段回测用的是哪个池子"必须在**三条**消费路径上都生效：
  1. 信号选择（signals.select_entries）—— 不在池里的票不出信号；
  2. 策略族选择（families.select）—— 同上；
  3. 基准构成（regime.compute_regime）—— 等权指数只算池内的票
     （这一条同时修掉审计 C1：基准与策略的机会集不同构）。
且**缺列即 fail-fast**：不允许存在"忘了带池子"却照样出数字的路径。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import families as F  # noqa: E402
from research import indicators, panel as mod_panel, regime, signals  # noqa: E402
from research import universe as U  # noqa: E402
from research.tests.conftest import make_synth_panel  # noqa: E402

# 放宽 A~F 里除"回调"之外的阈值，让合成面板上真的出信号 ——
# 否则"eligible=False ⇒ 0 信号"会因为本来就没信号而变成空断言。
WIDE = dict(min_down_streak=1, max_down_streak=10, pct_chg_low=-0.5, pct_chg_high=-0.001,
            min_amount=0.0, max_amount=None, min_above_ma60_ratio=0.0, min_mom120=-1.0,
            close_ma60_buffer=0.0)
STR_WIDE = {"drop5": 0.0, "min_drawdown": 0.0, "min_off_low": 0.0,
            "max_atr_pct": None, "max_rvol20": None}


@pytest.fixture
def ind():
    return indicators.compute_all(make_synth_panel(("AAA", "BBB"), bars=360))


def _signals(ind, **over):
    kw = {**WIDE, **over}
    return signals.select_entries(ind, start_date="2020-06-01", end_date="2021-01-01", **kw)


# ---------------------------------------------------------------- 信号路径

def test_baseline_signals_exist(ind):
    assert len(_signals(ind)) > 0, "夹具本身必须出信号，否则下面的断言是空的"


def test_ineligible_names_cannot_generate_signals(ind):
    assert len(_signals(ind)) > 0
    gated = ind.assign(eligible=False)
    assert len(_signals(gated)) == 0, "池外的票不该出信号"


def test_partial_pool_only_yields_that_pool(ind):
    only_aaa = ind.assign(eligible=ind["symbol"] == "AAA")
    got = _signals(only_aaa)
    assert set(got["symbol"]) <= {"AAA"}
    assert len(got) > 0, "另一半池子仍在 ⇒ 仍应有信号（否则是整体被关掉，不是取池生效）"


def test_signals_fail_fast_without_the_column(ind):
    with pytest.raises(U.UniverseError, match="eligible"):
        _signals(ind.drop(columns=["eligible"]))


# ---------------------------------------------------------------- 族路径

def test_family_respects_the_pool(ind):
    ind_f = F.add_family_indicators(ind)
    assert len(F.select(ind_f, "str", STR_WIDE, "2020-06-01", "2021-01-01")) > 0
    gated = ind_f.assign(eligible=False)
    assert len(F.select(gated, "str", STR_WIDE, "2020-06-01", "2021-01-01")) == 0
    only_aaa = ind_f.assign(eligible=ind_f["symbol"] == "AAA")
    got = F.select(only_aaa, "str", STR_WIDE, "2020-06-01", "2021-01-01")
    assert set(got["symbol"]) <= {"AAA"}


def test_family_fails_fast_without_the_column(ind):
    with pytest.raises(U.UniverseError, match="eligible"):
        F.select(ind.drop(columns=["eligible"]), "str", STR_WIDE, "2020-06-01", "2021-01-01")


# ---------------------------------------------------------------- 基准路径

def _regime_panel() -> pd.DataFrame:
    """两只票：AAA 温和上涨，BBB 暴涨暴跌。池子只含 AAA 时，指数不该被 BBB 带动。"""
    dates = pd.bdate_range("2024-01-01", periods=40)
    a = [100.0 * (1.0 + 0.002 * i) for i in range(40)]
    b = [100.0 * (1.0 + 0.15 * ((-1) ** i)) for i in range(40)]
    frames = []
    for sym, closes in (("AAA", a), ("BBB", b)):
        s = pd.Series(closes)
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates, "close": closes,
            "ret1": s.pct_change().fillna(0.0).values,
            "above_ma20": True, "eligible": sym == "AAA",
        }))
    return pd.concat(frames, ignore_index=True)


def test_benchmark_uses_only_the_pool():
    df = _regime_panel()
    only_a = regime.compute_regime(df)
    both = regime.compute_regime(df.assign(eligible=True))
    # AAA 是单调小幅上行(日收益 +0.2%)，BBB 是 ±15% 的摆动。
    assert only_a["ew_index"].iloc[-1] == pytest.approx(both["ew_index"].iloc[-1], rel=0.25)
    assert only_a["ew_index"].pct_change().abs().max() < 0.05, \
        "池外那只 ±15% 的票被算进了基准 —— 基准必须与策略同机会集"
    assert both["ew_index"].pct_change().abs().max() > 0.05


def test_regime_fails_fast_without_the_column():
    with pytest.raises(U.UniverseError, match="eligible"):
        regime.compute_regime(_regime_panel().drop(columns=["eligible"]))


def test_regime_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="可投资集为空"):
        regime.compute_regime(_regime_panel().assign(eligible=False))


# ---------------------------------------------------------------- 取池入口

def test_load_panel_for_preset_forwards_the_declared_universe(monkeypatch):
    """preset 的 `universe` 声明必须被真的传给载入层（曾经是死参数）。"""
    seen = {}

    def fake_load_full_panel(symbols=None, *, adjust="total", universe=None):
        seen["universe"] = universe
        return pd.DataFrame({"symbol": [], "date": []})

    monkeypatch.setattr(mod_panel, "load_full_panel", fake_load_full_panel)
    spec = {"mode": "liquidity", "min_dollar_volume": 1e7, "lookback": 30}
    mod_panel.load_panel_for_preset({"universe": spec})
    assert seen["universe"] == spec
    mod_panel.load_panel_for_preset({})                     # 未声明 ⇒ None（attach 会大声警告）
    assert seen["universe"] is None


def test_load_panel_for_preset_rejects_non_dict():
    with pytest.raises(mod_panel.PanelError):
        mod_panel.load_panel_for_preset(None)


# ---------------------------------------------------------------- 取池口径本身

def test_preset_declarations_are_pit_and_not_static_lists():
    """仓库里所有 preset 都必须声明 PIT 口径，不得回退到静态名单（"oi100"）。"""
    from research import presets
    for name, p in presets.PRESETS.items():
        spec = p.get("universe")
        assert isinstance(spec, dict), f"{name} 的 universe 仍是静态名单写法：{spec!r}"
        parsed = U.parse(spec)
        assert parsed.mode in (U.MODE_LIQUIDITY, U.MODE_MEMBERSHIP), \
            f"{name} 声明了 all（幸存者偏差最重的口径）"
