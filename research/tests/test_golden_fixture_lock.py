"""行为锁的**离线版** —— 没有真库时它照样跑（审计 G1）。

`test_golden_trades` 锁的是真数据上的交易列表，而真库不入库 ⇒ 全新 clone / CI 上
那把锁静默退化成 skip。本文件用确定性合成市场（`fixture_market.py`）补上这一环：
它不替代真库锁，它的职责是让"交易列表漂移"在**任何机器上**都能被抓住。

重生成（改了口径/引擎时）：`GOLDEN_FIXTURE_REGEN=1 python -m pytest
research/tests/test_golden_fixture_lock.py`，然后把新的指纹连同**成因与量级**
写进 commit message（§6 对真库锁的同一套要求）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import db as dbmod  # noqa: E402
from research import golden_baseline as G  # noqa: E402
from research import holdout as mod_holdout  # noqa: E402
from research import indicators, panel as panel_mod, presets  # noqa: E402
from research.tests import fixture_market as FX  # noqa: E402

BASELINE_DIR = ROOT / "research" / "tests" / "baselines"
REGEN = bool(os.environ.get("GOLDEN_FIXTURE_REGEN"))

# 窗口取**绝对值**（与真库锁同一纪律）：合成市场是确定性的，所以它不会自己动。
# 起点取第 130 根 K 线之后（mom120 预热 121 根 + dd120 预热 120 根）。
#
# 用例**从注册表派生**（2026-10-17 注册表重构）：窗口与"入不入锁"的表态写在
# `preset_defs/<name>.py` 里 —— 有 `FIXTURE_WINDOW` 就入锁；只有
# `FIXTURE_EXEMPT_REASON` 就不入锁（理由必须非空）；两者都没有由装载器在 import 期
# 直接抛错。此前这里是手写清单：新增 preset 时漏掉它，新 preset 就静默不在离线锁里,
# 而离线锁正是"没有真库的机器上唯一能跑的那道锁"。
def _cases() -> list[dict]:
    out = []
    for name, rec in sorted(presets.REGISTRY.items()):
        fw = rec["fixture_window"]
        if not fw:
            continue
        out.append({"name": f"fixture_{name}", "preset": name,
                    "start": fw[0], "end": fw[1]})
    return out


CASES = _cases()


def test_the_offline_exemption_list_is_explicit():
    """豁免表不许无声变长 —— 与真库锁的 `LOCK_EXEMPT` 同一条纪律。"""
    exempt = {n for n, r in presets.REGISTRY.items() if r["fixture_exempt"]}
    # 2026-10-17 事件维度两条（noearn / postearn）也在这张表里：wfv 那一系在**合成市场**
    # 上出场/入场条件咬合不上 ⇒ 0 笔成交，锁会退化成空断言（这正是这张表存在的理由）。
    assert exempt == {"us_uptrend_pullback_wfv", "us_uptrend_pullback_wfv_noearn",
                      "us_uptrend_pullback_wfv_postearn"}, \
        f"离线锁的豁免名单变了：{sorted(exempt)} —— 要么给它 FIXTURE_WINDOW，" \
        f"要么在这里显式写明为什么它锁不了"


def _path(name: str) -> Path:
    return BASELINE_DIR / f"{name}.golden"


def _read(name: str) -> dict:
    p = _path(name)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _write(name: str, case: dict, info: dict) -> None:
    _path(name).write_text(
        "# golden baseline（**合成市场**版，§6 行为锁）—— 无需真库，任何机器都能跑。\n"
        "# 只放哈希与计数，不放任何收益数字（§7）。\n"
        "# 重生成:GOLDEN_FIXTURE_REGEN=1 python -m pytest "
        "research/tests/test_golden_fixture_lock.py\n"
        "# 重生成 = 承认交易列表漂移，必须在 commit message 里说明原因与量级。\n"
        f"case={name}\n"
        f"preset={case['preset']}\n"
        f"start={case['start']}\n"
        f"end={case['end']}\n"
        f"vintage={info['vintage']}\n"
        f"n_trades={info['n_trades']}\n"
        f"trades_sha256={info['trades_sha256']}\n"
        f"equity_sha256={info['equity_sha256']}\n")


@pytest.fixture(scope="module")
def fixture_chain(tmp_path_factory):
    """建合成库 → 把 `db.DB_PATH` 指过去 → 载入面板与指标。"""
    tmp = tmp_path_factory.mktemp("fixture_market")
    db_file = FX.build(tmp / "fixture.db")
    old = dbmod.DB_PATH
    dbmod.DB_PATH = db_file
    # 台账指向临时文件：合成市场的窗口不该混进真实 holdout 台账
    old_ledger = mod_holdout.LEDGER_PATH
    mod_holdout.LEDGER_PATH = tmp / "ledger.jsonl"
    try:
        panel = panel_mod.load_panel_for_preset(presets.get_preset(CASES[0]["preset"]))
        yield panel, indicators.compute_all(panel)
    finally:
        dbmod.DB_PATH = old
        mod_holdout.LEDGER_PATH = old_ledger


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_trade_list_is_locked(fixture_chain, case):
    panel, ind = fixture_chain
    info = G.run_case(case, panel, ind)
    assert info["n_trades"] > 0, (
        "合成市场必须让这条路径**真的出信号** —— 0 笔成交的锁几乎测不到东西"
        "（形状是按条件反推出来的，见 fixture_market 的 docstring）")
    if REGEN:
        _write(case["name"], case, info)
        pytest.skip(f"{case['name']}：已重生成基线（GOLDEN_FIXTURE_REGEN=1）")
    base = _read(case["name"])
    assert base, (f"缺基线 {_path(case['name'])}\n"
                  f"  重生成：GOLDEN_FIXTURE_REGEN=1 python -m pytest "
                  f"research/tests/test_golden_fixture_lock.py（并在 commit 里说明漂移）")
    assert info["vintage"] == base["vintage"], (
        f"合成市场的 vintage 变了（夹具改过？）—— 这是数据换了，不是逻辑变了：\n"
        f"  基线 {base['vintage']}\n  当前 {info['vintage']}")
    assert info["trades_sha256"] == base["trades_sha256"], (
        f"交易列表漂移（§6）：\n  基线 {base['n_trades']} 笔  {base['trades_sha256'][:16]}\n"
        f"  当前 {info['n_trades']} 笔  {info['trades_sha256'][:16]}\n"
        f"  改了引擎/口径就按上面的命令重生成，并在 commit message 里写清成因与量级")
    assert info["equity_sha256"] == base["equity_sha256"], "权益曲线漂移"


def test_the_lock_is_not_vacuous(fixture_chain):
    """整把锁最少要覆盖两种信号路径（普通 preset 与族型 preset）。"""
    panel, ind = fixture_chain
    counts = {c["name"]: G.run_case(c, panel, ind)["n_trades"] for c in CASES}
    assert sum(counts.values()) >= 10, f"合成市场的成交量太少，锁的覆盖面不足：{counts}"
    assert all(v > 0 for v in counts.values()), f"有路径完全没成交：{counts}"


def test_fixture_is_deterministic():
    """夹具本身必须逐位确定（否则锁会自己漂）。"""
    a, b = FX.rows(), FX.rows()
    assert a == b
    assert len(a) == FX.N_SYMBOLS * FX.N_DAYS
    df = pd.DataFrame(a, columns=["symbol", "date", "open", "high", "low", "close",
                                  "volume", "dividends", "splits", "fetched_at", "source"])
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all(), \
        "夹具自己不能造出不自洽的 OHLC（引擎会拒绝在上面成交）"
    assert set(df["source"]) == {"fixture"}
