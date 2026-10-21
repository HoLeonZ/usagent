"""回测明细 → 自包含 HTML 报告（**入口层**）。

**呈现**在 `research/html_report.py` —— 那是公共能力：版式、卡片、口径头、自检
对任何策略完全相同，**变的只是策略**。本文件只负责把一条策略跑成逐笔明细
（含口径自报、运气对照），再交给它渲染 + 自检。

用法（三种，形式一致，只有策略不同）：

    python -m research.render_report                        # 默认 preset（兼容旧行为）
    python -m research.render_report --preset us_reversal_dip_v1 \
        --lo 2024-04-09 --hi 2026-10-14 --out results/dip.html
    python -m research.render_report --family xs_flow \
        --params '{"xs_ret5_max": 0.1}' --out results/xs_flow.html

⚠️ 报告里的数字一律**现算**，一个字都不写死；判读依据是**策略自己声明的 VERDICT**
（§5 要求负面证据与参数同时记录）—— 所以换一条策略，报告不会变成假陈述。
"""
from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from core.preset_contract import describe_engine_spec, exit_scheme, resolve
from research import (broker, indicators, metrics, panel as panel_mod, presets,
                      regime, signals)
# 呈现层（公共能力）。这些名字在本模块**再导出**一次：外部（老代码/测试）仍可按
# `render_report.render_html` 之类引用，而实现只有一处。
from research.html_report import (RenderError, condition_rows,  # noqa: F401
                                  family_explain_rows, render_html, selfcheck)

logger = logging.getLogger("render")

# 默认渲染哪个 preset —— **从注册表读，不在代码里写死配置**。
# 早先这里写死过一份 BEST_CFG（mp=1/risk=1%），而真正注册的那条 preset 不是它 ——
# 报告与注册表分叉过一次。现在默认值只是一个名字，内容全从定义体读。
DEFAULT_PRESET = "us_uptrend_pullback_12m_target"
WIN_LO, WIN_HI = "2025-10-14", "2026-10-14"
OUT_HTML = Path(__file__).resolve().parent / "results" / "trades_12m.html"

def control_arm(ind: pd.DataFrame, preset: dict, n_trades: int, lo: str, hi: str,
                reg, n_runs: int, own_cagr: float) -> dict:
    """**运气对照**：同止损止盈、同笔数、随机入场 n_runs 次 ⇒ 本配置落在第几百分位。

    为什么必须是一个函数：这条判词是把"运气"和"本事"分开的**唯一**手段
    （`family_search` 的对照随机的是**日子**，这里随机的是**入场**），而 CLI（`--control-runs`）
    与本地平台都要出它 —— 两处各写一份必然分叉（本仓库为此建过账）。

    `n_runs <= 0` ⇒ 返回 NaN 并让渲染层**不出那条判词**（绝不能留着 NaN 渲染成
    "nan 百分位"）。返回的四个键直接 `summary.update(...)`。
    """
    if n_runs <= 0:
        return {"ctl_runs": 0, "ctl_failed": 0, "ctl_p90": float("nan"),
                "ctl_max": float("nan"), "ctl_pct": float("nan")}
    from research import search_12m as _S
    # 对照配置从**已解析的 broker preset** 反推，保证与策略同杠杆同止损止盈。
    # ⚠️ 按 preset **实际用的那套几何**建对照（审计 G5）：契约允许 ATR 归一化或固定
    # 百分比两套，直接读 `preset["atr_sl_mult"]` 会让一个合法的 pct preset KeyError。
    cfg = {"max_hold": preset["max_hold"], "max_positions": preset["max_positions"],
           "risk_per_trade": preset["risk_per_trade"]}
    # 保持 `_scheme = exit_scheme(...)` + `if _scheme == "atr"` 这个形状：
    # `test_contract_gaps` 的源码级护栏钉的正是它（"任何 atr_* 的直接读取都必须在
    # 按几何分支之后"）。改成内联调用会让那条护栏失效 —— 护栏对了，改代码。
    _scheme = exit_scheme(preset)
    if _scheme == "atr":
        cfg.update({"sl_mult": preset["atr_sl_mult"], "tp_mult": preset["atr_tp_mult"]})
    else:
        cfg.update({"sl_pct": preset["sl_pct"], "tp_pct": preset["tp_pct"]})
    # 必须把 reg 传进去：对照要在**策略自己的机会集**里抽（含 regime 门），
    # 否则比的是两个不同的池子（见 search_12m._control_pool）。
    ctl = _S.random_control(ind, cfg, n_trades, n_runs=n_runs, lo=lo, hi=hi, seed=0, reg=reg)
    return {"ctl_runs": ctl["n_runs"], "ctl_failed": ctl["n_failed"],
            "ctl_p90": ctl["p90"], "ctl_max": ctl["max"],
            "ctl_pct": float((ctl["cagrs"] < own_cagr).mean() * 100.0)}


def _strategy(preset_name: str = DEFAULT_PRESET, *, family: str | None = None,
              params: dict | None = None, base_preset: str | None = None) -> tuple[dict, str]:
    """策略选择 → (已注册的 preset 字典, 显示名)。**run 与 prepare_chain 共用这一段**。"""
    P = presets.get_preset(preset_name)
    if family:
        # 族型策略：退出/仓位口径取自一条**已注册**的 preset（默认那条 str 族 preset），
        # 只有入场掩码换成目标族 —— 与 rank_strategies / family_search 同一做法。
        from research import families as _F0
        _fid = str(family)
        if _fid not in _F0.MASK:
            raise RenderError(f"未知策略族 {_fid!r}；可用：{sorted(_F0.MASK)}")
        _grid = _F0.GRIDS.get(_fid) or {}
        P = {**presets.get_preset(base_preset or "us_reversal_dip_v1"),
             "family": _fid,
             "family_params": dict(params or {g: v[len(v) // 2] for g, v in _grid.items()})}
        # 那条 preset 的 VERDICT 是**另一条策略**的证据，不能当成本策略的 ⇒ 摘掉。
        # 于是报告会显示"本策略没有声明负面证据"（除非 --note 补一句）—— 这是有意的。
        P.pop("verdict", None)
        return P, "family:" + _fid
    return P, preset_name


def prepare_chain(preset_name: str = DEFAULT_PRESET, *, family: str | None = None,
                  params: dict | None = None, base_preset: str | None = None) -> tuple:
    """构建 run() 需要的那条**链路**（面板 / 指标 / regime），供重复回测复用。

    为什么单独暴露它：一次"载入面板"要 35–66 秒（3028 只票逐只查询），而本地平台会连着
    点好几次 ⇒ 必须能缓存。**只有一处接线**：`run()` 在没给 chain 时调的就是这个函数，
    不是第二份实现（本仓库最忌讳的两条路径）。
    """
    P, _ = _strategy(preset_name, family=family, params=params, base_preset=base_preset)
    return _chain_from(P)


def _chain_from(P: dict) -> tuple:
    """由**已解析的 preset** 建链路。run() 与 prepare_chain() 都走这里，只有一份实现。"""
    p = panel_mod.load_panel_for_preset(P)
    ind = indicators.compute_all(p)
    # 卡片上的均线也要落在 **as-traded** 轴上(与蜡烛、止损止盈线、买卖标记同轴)——
    # 见 candles_svg 的注释:混轴会让图上出现从未发生的止损击穿。
    ind = ind.assign(
        raw_ma20=ind.groupby("symbol")["raw_close"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()),
        raw_ma60=ind.groupby("symbol")["raw_close"].transform(
            lambda s: s.rolling(60, min_periods=60).mean()),
    )
    # ⚠️ 必须把 preset 的 regime 参数传进去:此前这里用函数默认值做择时,而 CLI 用
    #    preset 值 —— preset 的 regime 一改,报告与 CLI 就分叉(审计实测差 13 笔成交)。
    reg = regime.compute_regime(ind, **(P.get("regime") or {}))
    return p, ind, reg


def run(lo: str, hi: str, preset_name: str = DEFAULT_PRESET,
        control_runs: int = 200, *, family: str | None = None,
        params: dict | None = None, base_preset: str | None = None,
        title: str | None = None, note: str = "",
        chain: tuple | None = None) -> tuple:
    """窗口 + 策略 ⇒ 逐笔明细。见模块 docstring 的三种用法。

    preset_name 与 kw（family/params/base_preset/title/note）都只影响
    **跑哪条策略**；呈现一律由 `research/html_report.py` 负责。

    `chain=(panle, ind, reg)` 用于**重复回测**（本地平台）：跳过"读 + 算"这一段，
    其余一字不改。给了 chain 就不再重新载面板 —— 否则缓存等于白做。
    """
    P, preset_name = _strategy(preset_name, family=family, params=params,
                               base_preset=base_preset)
    if chain is None:
        # 直接用**已解析**的 P（不要再用 preset_name 二次解析：族型的显示名是
        # "family:xxx"，不是注册名 —— 实测会把 KeyError 抛在缓存未命中那条路径上）
        p, ind, reg = _chain_from(P)
    else:
        p, ind, reg = chain
        _miss = {"raw_ma20", "raw_ma60"} - set(ind.columns)
        if _miss:
            raise RenderError(f"chain 里的指标表缺 {sorted(_miss)} —— 请用 prepare_chain() "
                              f"预载（缺列只会让 K 线卡静默少画均线）")

    # 信号:preset 里声明的 signal 字典直接透传(含可选过滤层)。
    # preset 是**单一事实来源** —— 不在这里重新拼参数,否则报告展示的配置
    # 会与 presets.py 里注册的那个悄悄分叉。
    if P.get("family"):
        # 族型策略（2026-10-17 起支持）：入口与 golden 载体**同一个** `families.select`
        # ⇒ 报告画的与行为锁锁的是同一条曲线。此前这里直接 NotImplementedError，
        # 理由是"条件清单是 A~F 表、对族没意义" —— 那个理由对**呈现**不成立，
        # 所以现在改为：呈现不变，条件清单换成族自己的（见 explain 钩子）。
        from research import families as _F
        _fid = str(P["family"])
        sig = _F.select(ind, _fid, dict(P.get("family_params") or {}), lo, hi, reg=reg)
        explain = partial(family_explain_rows, mask=_F.MASK[_fid],
                          params=P.get("family_params") or {})
    else:
        sig = signals.select_entries(ind, start_date=lo, end_date=hi,
                                     regime_df=reg if P.get("regime") else None,
                                     **P["signal"])
        explain = condition_rows

    # 执行:**整包透传**。此前这里是一份 7 键白名单,对 broker 实际会读的键静默丢弃
    # (position_fraction / sl_pct / tp_pct / atr_slip_scale / trailing_sl),
    # 实测同一个 preset 在报告与 CLI 上跑出差 45.5pp 的两条曲线。
    # 现在唯一的展开入口是契约的 resolve(),CLI 与报告读到的是同一份参数。
    preset = resolve(P)
    res = broker.simulate(preset, sig, ind, start=lo, end=hi)

    # 每笔对应的信号日:入场日是信号日之后的第一个交易日,取 ≤ 入场日 的最后一个信号
    sig_dates = {}
    for _, r in res["trades"].iterrows():
        c = sig[(sig["symbol"] == r["symbol"]) & (sig["date"] < r["entry_date"])]
        sig_dates[r["symbol"]] = c["date"].max() if len(c) else r["entry_date"]

    eq = res["equity"]
    idx = reg.set_index("date")["ew_index"]
    b = idx[(idx.index >= pd.Timestamp(eq["date"].iloc[0])) &
            (idx.index <= pd.Timestamp(eq["date"].iloc[-1]))]
    bench = (b / b.iloc[0]).reindex(eq["date"]).ffill().bfill()

    ts = metrics.tail_stats(res["trades"])
    summary = {
        "preset_name": title or preset_name, "preset": preset,
        # signal 表：signal 型给 A~F 的参数，族型给族参数 —— 报告里那一节照旧显示"配置"
        "signal": P.get("signal") or P.get("family_params") or {},
        # 这组参数**来自哪儿**（报告的「配置」那一节据此写前缀：族型 preset 的族网格轴
        # 不能被标成 "signal.*" —— 那是同一件事的两种说法）
        "signal_source": ("signal" if P.get("signal") else "family_params"),
        # 判读依据 = **策略自己声明的负面证据**（§5：负面证据必须与参数同时记录）。
        # 此前这里是写死"30% 目标"那段文字 —— 只对某一条 preset 成立，换一条就成了假陈述
        # （本文件的历史注释里记着两次翻车）。现在任何策略都渲染它自己的 verdict。
        "verdict": (P.get("verdict") or "").strip() or None,
        "note": note or None,
        "lo": lo, "hi": hi,
        "cagr": metrics.cagr(eq["equity"], eq["date"]),
        "max_dd": float(res["metrics"]["max_dd"]),
        "win": float(res["metrics"]["win_rate"]) if len(res["trades"]) else float("nan"),
        "e_r": float(ts["expectancy_r"]), "exposure": float(res["metrics"]["avg_exposure"]),
        "sig_dates": sig_dates, "n_trials": P.get("n_comparisons", 106),
    }

    # 运气对照:**真跑**,不硬编码。同止损止盈、同交易笔数的随机入场 200 次,
    # 再算本配置落在该分布的第几百分位。渲染一份报告多花 ~1 分钟,换的是
    # "报告里的每个数字都可复现"。
    from research import search_12m as _S
    # 对照配置:从**已解析的 broker preset** 反推,保证对照与策略同杠杆同止损止盈。
    # (search_12m.make_preset 读的是 sl_mult/tp_mult 这套键名,broker preset 用的是
    #  atr_sl_mult/atr_tp_mult —— 名字不同,值必须一致,否则对照就不成立。)
    # ⚠️ 按 preset **实际用的那套几何**建对照（审计 G5）：契约允许 ATR 归一化或
    # 固定百分比两套，而这里此前直接读 `preset["atr_sl_mult"]` —— 一个合法的
    # pct preset 会让它 KeyError。用 `exit_scheme()` 显式分支，两套都支持。
    summary.update(control_arm(ind, preset, len(res["trades"]), lo, hi, reg,
                               control_runs, summary["cagr"]))
    # 第 6 项是**策略相关**的钩子：呈现层拿它去画信号日清单。
    return res, eq, ind, bench, summary, explain


def main() -> int:
    ap = argparse.ArgumentParser(description="逐笔 HTML 报告（呈现与策略分离）")
    ap.add_argument("--preset", default=None, help="preset 名（与 --family 二选一）")
    ap.add_argument("--family", default=None, help="策略族 id（如 xs_flow / str / idio）")
    ap.add_argument("--params", default=None,
                    help="族参数 JSON（缺省=该族网格中点那一组）")
    ap.add_argument("--base-preset", default=None,
                    help="族型的退出/仓位口径取自哪条 preset（默认 us_reversal_dip_v1）")
    ap.add_argument("--lo", default=WIN_LO)
    ap.add_argument("--hi", default=WIN_HI)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--note", default="", help="额外写进报告的一句说明（可选）")
    ap.add_argument("--control-runs", type=int, default=200,
                    help="运气对照次数（0 = 跳过，快很多；报告里会写明未跑）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if a.preset and a.family:
        print("--preset 与 --family 只能给一个", file=sys.stderr)
        return 2

    params = None
    if a.params:
        import json
        params = json.loads(a.params)
    name = a.family or a.preset or DEFAULT_PRESET
    out = Path(a.out or (OUT_HTML if (a.preset in (None, DEFAULT_PRESET) and not a.family)
                         else Path(__file__).resolve().parent / "results"
                         / f"report_{name.replace(':', '_')}.html"))

    res, eq, ind, bench, summary, explain = run(
        a.lo, a.hi, a.preset or DEFAULT_PRESET, a.control_runs,
        family=a.family, params=params, base_preset=a.base_preset,
        title=a.title, note=a.note)
    n = len(res["trades"])
    doc = render_html(res["trades"], eq, ind, summary["signal"], bench, summary,
                      explain=explain)

    problems = selfcheck(doc, n)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"策略 {summary['preset_name']} ｜ 窗口 {a.lo}..{a.hi}")
    print(f"成交 {n} 笔  CAGR {summary['cagr']*100:+.2f}%  "
          f"基准 {bench.iloc[-1]/bench.iloc[0]-1 if len(bench) else float('nan'):+.2%}")
    print(f"已写入 {out}  ({out.stat().st_size/1024:.0f} KB,自包含、无外部依赖)")
    if problems:
        for p in problems:
            print(f"  [自检失败] {p}")
        return 1
    print(f"  [自检通过] {n} 张卡片各有 K线/止损/止盈/入场/出场 5 个元素,无 NaN,无外部依赖")
    return 0


if __name__ == "__main__":
    sys.exit(main())
