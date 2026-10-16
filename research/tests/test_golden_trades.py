"""Golden-baseline 锁 —— 交易列表行为必须被机械钉住(CLAUDE.md §6)。

CLAUDE.md §6 原文:
    Golden-baseline hashes lock trade-list behavior for every engine; any logic
    change that alters the trade list MUST regenerate the baseline and justify
    the drift in the PR.

为什么需要这条(2026-10-15 审计的结论):research/tests 的 89 个用例全是「机制级单点
断言」——止损在同 bar 双触发时取最坏情况、权益 == cash + qty×close、量能上限咬合……
它们保的是「机制存在」,不保「输出不变」。而本项目的交付结论是以**具体交易列表**为载体的,
于是任何没被单点断言覆盖的逻辑改动,都能在 89 个用例全绿的情况下悄悄换掉整个交易列表。
提交 6bac26d / e5e8af4 / 091cae5 就发生过这种事,当时只能靠 commit message 的散文交代。

纪律(见 research/golden_baseline.py 的模块 docstring):
  * 本地 DB 缺失 ⇒ skip(§6)。
  * 基线文件缺失 ⇒ **fail**(它是入库产物,缺了就是契约破了;信息里带重生成命令)。
  * 基线里只有哈希与计数,没有任何收益数字(§7)。
"""
from __future__ import annotations

import pytest

import pandas as pd

from research import golden_baseline as G
from research import indicators, panel as panel_mod, presets


@pytest.fixture(scope="module")
def real_chain():
    """真库 → 指标。只加载一次(425k 行,逐用例重载没必要)。

    regime **不在这里算** —— 它必须按每个 case 自己的 preset 参数算,所以放在
    G.run_preset 里(复核实测:此前用函数默认值算,载体锁的是另一条曲线)。
    """
    if not G.db_available():
        pytest.skip(f"本地 DB 不在({G.db_path()}),跳过 golden 基线比对(§6:skip 不 error)")
    try:
        panel = panel_mod.load_full_panel()
    except Exception as exc:                                  # noqa: BLE001
        pytest.skip(f"本地 DB 不可读,跳过 golden 基线比对:{type(exc).__name__}: {exc}")
    return panel, indicators.compute_all(panel)


@pytest.mark.parametrize("case", G.CASES, ids=[c["name"] for c in G.CASES])
def test_trade_list_matches_golden_baseline(real_chain, case):
    panel, ind = real_chain
    base = G.load_baseline(case["name"])
    if base is None:
        pytest.fail(
            f"golden 基线缺失:{G.baseline_path(case['name'])}\n"
            f"  重生成:python -m research.golden_baseline --regen\n"
            f"  重生成 = 承认交易列表漂移,必须在 commit message 里写清原因与量级(§6)。"
        )

    cur = G.run_case(case, panel, ind)

    assert cur["vintage"] == base["vintage"], (
        "数据 vintage 变了 —— 这是数据换了,不是回测逻辑变了:\n"
        f"  基线 {base['vintage']}\n"
        f"  当前 {cur['vintage']}"
    )
    assert cur["trades_sha256"] == base["trades_sha256"], (
        "交易列表漂移(§6 要求重生成基线并在 commit message 里说明原因):\n"
        f"  基线 {base['n_trades']} 笔  {base['trades_sha256'][:16]}\n"
        f"  当前 {cur['n_trades']} 笔  {cur['trades_sha256'][:16]}"
    )
    assert cur["equity_sha256"] == base["equity_sha256"], (
        "权益曲线漂移(记账语义变了 —— 不是交易列表本身变了):\n"
        f"  基线 {base['equity_sha256'][:16]}\n"
        f"  当前 {cur['equity_sha256'][:16]}"
    )


def test_golden_harness_reproduces_the_cli(real_chain):
    """golden 载体本身不得与 CLI 分叉。

    这条测试防的是「两个入口给 broker 喂不同的东西」这一类缺陷 —— 2026-10-15 审计
    实测 render_report 的 7 键白名单与 CLI 的整包透传曾让同一个 preset 跑出差 45.5pp
    的两条曲线。golden 基线若与 CLI 分叉,它就锁错了东西。
    """
    import argparse
    from research import run_baseline

    panel, ind = real_chain
    case = G.CASES[0]
    assert case.get("is_cli_default"), "CASES[0] 必须标注它等于 CLI 默认窗口"
    default = run_baseline.run(argparse.Namespace(
        preset=case["preset"], start=case["start"], end=case["end"],
        cost_bps=float(G.preset_cost(case["preset"])),
        # 手工搭的 Namespace 必须与 CLI 的字段集一致：缺一个字段就等于
        # 用另一套成本口径跑同一份 preset —— 正是 §6 要消灭的"两个入口分叉"。
        spread_model=None, windows=False,
        avoid_earnings=False))

    cur = G.run_case(case, panel, ind)
    assert cur["n_trades"] == default["metrics"]["trades"], (
        f"golden 载体与 CLI 的成交笔数不一致:{cur['n_trades']} vs {default['metrics']['trades']}"
    )
    # 只比笔数是不够的:换掉一笔(笔数不变)会被漏掉。比指纹。
    assert cur["trades_sha256"] == default["trades_sha256"], (
        f"golden 载体与 CLI 的交易列表不一致:{cur['trades_sha256'][:16]} vs "
        f"{default['trades_sha256'][:16]} —— 两个入口又分叉了(§6)"
    )


@pytest.mark.parametrize("case", G.CASES, ids=[c["name"] for c in G.CASES])
def test_entries_never_use_same_day_proceeds(real_chain, case):
    """§2(d) 在**真实交易列表**上的不变量:入场只用上一交易日收盘的现金与槽位。

    test_broker.py 的四条是合成 panel 上的单点断言;这条盯住交付 preset 的实际成交 ——
    2026-10-15 审计实测旧实现下 12m preset 有 3/13 笔必须靠同日卖出所得才能成交,
    其中 MTSI 用的是一小时多之后才发生的 PCVX 盘中止损回款。

    注意「入场日 == 某笔出场日」本身**不是**违规(只要用的是更早的现金),所以这里
    断言的是**依赖关系**,不是日期相等。
    """
    panel, ind = real_chain
    res, _ = G.run_preset(case, panel, ind)
    tr = res["trades"]
    eq = res["equity"].set_index("date")
    max_positions = int(presets.get_preset(case["preset"]).get("max_positions", 1))

    cash_violations, slot_violations = [], []
    for _, t in tr.iterrows():
        idx = eq.index.get_loc(t["entry_date"])
        assert idx > 0, "窗口首日不应有入场(信号来自更早的交易日)"
        prev = eq.iloc[idx - 1]
        notional = float(t["entry_price"]) * int(t["size"])
        if notional > float(prev["cash"]) + 1e-6:
            cash_violations.append((str(t["entry_date"].date()), t["symbol"], notional))
        if float(prev["n_pos"]) >= max_positions:
            slot_violations.append((str(t["entry_date"].date()), t["symbol"], int(prev["n_pos"])))

    assert not cash_violations, f"入场用了同日卖出所得:{cash_violations}"
    assert not slot_violations, f"入场用了同日释放的槽位:{slot_violations}"


# ------------------------------------------------- 入口一致性(§2(b)/§6)
# 2026-10-15 审计抓到的两处「同一条链路、两个入口、两份参数」:
#   * preset 的 cost_bps_roundtrip 在 CLI 跑数路径上从未被读取(只有 HTML 报告读它),
#     于是改 preset 的成本假设不会反映到 CLI 数字上;
#   * render_report 的 7 键白名单对 broker 实际读的键静默丢弃(丢 5 个,含 trailing_sl),
#     实测同一个 preset 跑出差 45.5pp / 4 笔成交的两条曲线。
# 修法都是「唯一的展开入口 = core.preset_contract.resolve()」;这两条测试盯住它。

def _cli_run(monkeypatch, case, **over):
    """按 CLI 语义跑一条用例(成本走 preset 口径)。"""
    import argparse
    from research import run_baseline

    p = presets.get_preset(case["preset"])
    p.update(over)
    monkeypatch.setitem(presets.PRESETS, case["preset"], p)
    return run_baseline.run(argparse.Namespace(
        preset=case["preset"], start=case["start"], end=case["end"],
        cost_bps=None, spread_model=None, windows=False, avoid_earnings=False))


def test_cli_honours_the_preset_cost(real_chain, monkeypatch):
    from core import costs

    case = G.CASES[1]
    cheap = _cli_run(monkeypatch, case, cost_bps_roundtrip=costs.COST_BPS_ROUNDTRIP)
    dear = _cli_run(monkeypatch, case, cost_bps_roundtrip=100.0)

    assert dear["cost_bps"] == 100.0, "CLI 没有采用 preset 的成本口径"
    assert dear["metrics"]["total_return"] < cheap["metrics"]["total_return"], (
        "把成本从 5bp 提到 100bp 之后总收益没变 —— 说明跑数路径根本没读 preset 的成本"
    )


def test_report_and_cli_agree_on_the_same_preset(real_chain, monkeypatch):
    """报告与 CLI 必须跑出同一条曲线 —— 包括 preset 里那些「不常用的」执行键。"""
    from research import render_report

    case = G.CASES[1]
    p = presets.get_preset(case["preset"])
    p["trailing_sl"] = True                    # 白名单时代就是被它丢掉的
    monkeypatch.setitem(presets.PRESETS, case["preset"], p)

    from research.broker import trade_fingerprint

    res, *_ = render_report.run(case["start"], case["end"], case["preset"],
                                  control_runs=0)   # 跳过 200 次运气对照(测试不需要)
    cli = _cli_run(monkeypatch, case, trailing_sl=True)
    assert len(res["trades"]) == cli["metrics"]["trades"], (
        f"报告 {len(res['trades'])} 笔 vs CLI {cli['metrics']['trades']} 笔 —— 两个入口又分叉了(§6)"
    )
    # 只比笔数不够:换掉一笔(笔数不变)会漏。比交易列表指纹。
    assert trade_fingerprint(res["trades"]) == cli["trades_sha256"], (
        "报告与 CLI 的交易列表不一致 —— 报告正是交付物(K 线卡/权益数字)的数据源(§6)"
    )


def test_cli_default_window_start_is_derived_from_the_data(real_chain):
    """预热边界与 vintage 必须由数据算出/记录,不能是写死的常量。

    审计实测:此前 `FIRST_SIGNAL_FLOOR = "2024-04-09"` 恰好等于当前这份数据的第 121
    根 K 线,而算出它的 `panel.first_signal_date` 是**死代码**。换一份数据(窗口起点
    漂移 3 个月)后仍拿常量当起点,预热期就被算进 CAGR 的分母,且没有任何断言会报警。
    """
    import argparse
    from research import panel as panel_mod, run_baseline

    panel, ind = real_chain
    out = run_baseline.run(argparse.Namespace(
        preset="us_uptrend_pullback_v1", start=None, end=None,
        cost_bps=None, spread_model=None, windows=False, avoid_earnings=False))

    assert out["start"] == str(panel_mod.first_signal_date(ind).date())
    # 还要把「CASES[0] 就是 CLI 默认窗口」这件事真正连起来 —— 旧版只断言了
    # G.CASES[0]["is_cli_default"] 这个**字面量 True**,对窗口本身零校验。
    case = G.CASES[0]
    assert (case["start"], case["end"]) == (out["start"], out["end"]), (
        f"golden 用例的窗口 {case['start']}..{case['end']} 与 CLI 默认窗口 "
        f"{out['start']}..{out['end']} 不一致 —— is_cli_default 这个标注就成了空话"
    )
    assert out["end"] == str(ind["date"].max().date() + pd.Timedelta(days=1)), (
        "默认终点必须由数据算出(库内最大日期 + 1 天),不能是写死的常量"
    )
    assert out["data_vintage"] == panel_mod.data_vintage(panel), (
        "结果 JSON 必须带上数据 vintage —— 否则「当前代码 + 一条命令」复现不出这些数字"
    )


def test_golden_carrier_uses_the_preset_regime(real_chain, monkeypatch):
    """载体必须用 **preset 声明的** regime 参数,而不是 compute_regime 的函数默认值。

    2026-10-15 复核实测:此前载体用默认值、CLI 用 preset 值,于是把 preset 的 regime
    改掉之后,载体锁住的仍是旧曲线 —— 护栏对着一条谁都不会跑的配置。
    """
    case = G.CASES[1]
    before = G.run_case(case, *real_chain)["n_trades"]

    p = presets.get_preset(case["preset"])
    p["regime"] = {**p["regime"], "min_breadth": 0.90}      # 窗口内几乎没有可做多的日子
    monkeypatch.setitem(presets.PRESETS, case["preset"], p)

    assert G.run_case(case, *real_chain)["n_trades"] != before, (
        "改了 preset 的 regime 但载体结果不变 —— 它没读 preset 的 regime 参数"
    )


def test_family_path_reproduces_select_entries_on_real_data(real_chain):
    """两条路径在**真实数据**上必须给出逐位相同的信号表。

    为什么必须有这条:family_search 用 families.select 自己构造信号表,而 search_12m 走
    signals.select_entries。「新族 vs uptrend-pullback」的结论只有在两条路径构造出的表
    逐位相同时才成立 —— 否则比的是两台机器。参考族 upb_ref 逐字复刻 A~F + 两个过滤层。

    ⚠️ 断言里必须包含「信号数 > 0」:合成面板上 A~F 常常 0 信号,那时 0==0 是空断言。
    """
    from research import families as F, signals

    panel, ind = real_chain
    ind = F.add_family_indicators(ind)
    lo, hi = "2024-04-09", "2025-10-13"
    p = {"min_dist_high252": -0.1, "max_stoch_k": 70.0}

    via_family = F.select(ind, "upb_ref", p, lo, hi)
    via_signals = signals.select_entries(
        ind, start_date=lo, end_date=hi, **{**signals.V21_SIGNAL_PARAMS, **p})

    assert len(via_signals) >= 20, (
        f"窗口内信号太少({len(via_signals)})—— 空信号时两条路径都是 0 个,断言 0==0 是空断言")
    assert len(via_family) == len(via_signals)
    assert via_family["symbol"].tolist() == via_signals["symbol"].tolist()
    assert via_family["date"].tolist() == via_signals["date"].tolist()
    assert via_family["score"].tolist() == via_signals["score"].tolist()
