"""Golden-baseline 锁件 —— 交易列表的机械校验(CLAUDE.md §6)。

CLAUDE.md §6 原文:
    Golden-baseline hashes lock trade-list behavior for every engine; any logic
    change that alters the trade list MUST regenerate the baseline and justify
    the drift in the PR.

为什么需要:research/tests 的 89 个用例全是「机制级单点断言」(止损在同 bar 双触发时取
最坏情况、权益 == cash + qty×close、量能上限咬合……)。它们保的是「机制存在」,不保
「输出不变」。而本项目的交付结论以**具体交易列表**为载体:一次没被单点断言覆盖的逻辑
改动,可以在 89 个用例全绿的情况下悄悄换掉整个交易列表。提交 6bac26d / e5e8af4 /
091cae5 就发生过这种事,当时只能靠 commit message 的散文交代。

三条纪律:
  1. **本地 DB 缺失 ⇒ skip**(§6 原文:测试在没有本地 DB 或运行产物时应当 skip 而不是
     error)。但**基线文件**缺失 ⇒ **fail**:基线是入库产物,缺它就是契约本身破了,
     而这条 fail 的信息里带着重生成命令。
  2. **数据 vintage 一并锁住。** 若只有 vintage 变了,说明是数据换了、不是逻辑变了 ——
     两者分开报,免得把「重下过一次数据」误读成「回测语义漂移」。
  3. 基线里**只有哈希与计数,没有任何收益数字**(§7:仓库里不得有历史业绩主张)。
     `n_trades` 是计数不是业绩;CAGR / Sharpe / 胜率一律不进基线文件。

重生成:
    python -m research.golden_baseline --regen     # 写入 research/tests/baselines/
    python -m research.golden_baseline --show      # 只打印当前哈希,不写盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker, indicators, presets, regime as regime_mod, signals
from research import holdout as mod_holdout
from research.broker import TRADE_LOCK_COLS, trade_fingerprint
from research.panel import data_vintage, load_panel_for_preset

BASELINE_DIR = Path(__file__).resolve().parent / "tests" / "baselines"

EQUITY_KEY_COLS = ["date", "equity", "cash", "n_pos"]

# 锁哪两个用例。窗口**写死绝对值**,不用 CLI 的默认解析 —— 默认解析会随数据漂移,
# 而 golden 基线必须钉在一个不会自己动的坐标上(vintage 另有一道断言看住数据)。
#
# 2026-10-17 注册表重构：用例**从注册表派生**（窗口写在 `preset_defs/<name>.py` 里）。
# 此前这里是手写清单 —— 新增 preset 时漏掉它**什么都不会发生**，新 preset 只是从此
# 不在任何护栏里；而"漏了也没人知道"正是本仓库反复吃亏的那类缺陷。
#
# 豁免表：**空 = 全都锁**。它能被写大，但一次性写大的代价是被
# `test_lock_coverage.test_exempt_table_cannot_grow_silently` 当场抓住 ——
# 豁免名单一旦能无声扩大，这条护栏就等于没有。
LOCK_EXEMPT: dict[str, str] = {}


def _cases_from_registry() -> list[dict]:
    out = []
    for name, rec in sorted(presets.REGISTRY.items()):
        if name in LOCK_EXEMPT:
            continue
        out.append({"name": name, "preset": name,
                    "start": rec["window"][0], "end": rec["window"][1],
                    "is_cli_default": rec["is_cli_default"]})
    return out


CASES: list[dict] = _cases_from_registry()


# ---------------------------------------------------------------- 环境

def db_path() -> str:
    try:
        import config
        return str(config.DB_PATH)
    except Exception:                                          # noqa: BLE001
        return "(config 不可导入)"


def db_available() -> bool:
    """本地 DB 在不在。不在就让调用方 skip,不在这里抛。"""
    try:
        import config
    except Exception:                                          # noqa: BLE001
        return False
    return Path(str(config.DB_PATH)).is_file()


# ---------------------------------------------------------------- 规范化

def _fmt(v: object) -> str:
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "nan" if not math.isfinite(f) else f"{f:.6f}"
    return str(v)


def canonical_equity(equity: pd.DataFrame) -> str:
    e = equity[EQUITY_KEY_COLS].sort_values("date", kind="mergesort")
    return "\n".join(",".join(_fmt(v) for v in row) for row in e.itertuples(index=False))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 跑一个用例

def preset_cost(name: str) -> float:
    return float(presets.get_preset(name)["cost_bps_roundtrip"])


def run_preset(case: dict, panel: pd.DataFrame,
               ind: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """跑一条用例的完整链路,返回 (broker 结果, 信号表)。

    这是**唯一**的接线点:golden 指纹与不变量测试都走它,免得两处接线各自漂移。

    ⚠️ regime 必须按**每个 case 自己的 preset** 参数算。2026-10-15 复核实测:此前
       这里用的是 `compute_regime(ind)` 的**函数默认值**,而 CLI 与报告用的是 preset
       声明的值 —— 于是 preset 的 regime 一改,载体锁住的仍是旧曲线,护栏形同虚设。
    ⚠️ 它与 run_baseline._build/_signals/run 是同一形态的两份实现,重复本身是风险。
       护栏是 test_golden_trades.py::test_golden_harness_reproduces_the_cli(现在比指纹)。
    """
    preset = presets.get_preset(case["preset"])
    reg = regime_mod.compute_regime(ind, **(preset.get("regime") or {}))
    if preset.get("family"):
        from research import families
        sig = families.select(ind, str(preset["family"]),
                              dict(preset.get("family_params") or {}),
                              case["start"], case["end"], reg=reg)
    else:
        sig = signals.select_entries(
            ind, start_date=case["start"], end_date=case["end"],
            regime_df=reg if preset.get("regime") else None, **preset["signal"])
    res = broker.simulate(preset, sig, ind, start=case["start"], end=case["end"],
                          cost_bps_roundtrip=preset_cost(case["preset"]))
    return res, sig


def run_case(case: dict, panel: pd.DataFrame, ind: pd.DataFrame) -> dict:
    """跑完整链路并返回指纹。

    台账登记为 `kind="lock"`（2026-10-17 审计 B1）：本用例会读到 holdout 窗口，
    但**重跑一次行为锁不是拿 holdout 出结论**，所以只登记、不烧窗口。
    台账因此能回答"这段窗口被谁读过几次"，而不会把护栏本身算成一次研究迭代。
    """
    mod_holdout.consume(case["start"], case["end"], script="golden_baseline",
                        tag=case["name"], kind=mod_holdout.KIND_LOCK)
    res, _ = run_preset(case, panel, ind)
    return {
        "vintage": data_vintage(panel),
        "n_trades": int(len(res["trades"])),
        "trades_sha256": trade_fingerprint(res["trades"]),
        "equity_sha256": _sha(canonical_equity(res["equity"])),
    }


# ---------------------------------------------------------------- 基线读写

def baseline_path(name: str) -> Path:
    return BASELINE_DIR / f"{name}.golden"


def load_baseline(name: str) -> dict | None:
    p = baseline_path(name)
    if not p.is_file():
        return None
    out: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    out["n_trades"] = int(out.get("n_trades", 0))
    return out


def save_baseline(name: str, case: dict, info: dict) -> Path:
    p = baseline_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# golden baseline —— 交易列表与权益曲线的行为锁(CLAUDE.md §6)",
        "# 只放哈希与计数,不放任何收益数字(§7:仓库里不得有历史业绩主张)。",
        "# 重生成:python -m research.golden_baseline --regen",
        "# 重生成 = 承认交易列表漂移,必须在 commit message / PR 里说明原因与量级。",
        f"case={name}",
        f"preset={case['preset']}",
        f"window={case['start']}..{case['end']}",
        f"vintage={info['vintage']}",
        f"n_trades={info['n_trades']}",
        f"trades_sha256={info['trades_sha256']}",
        f"equity_sha256={info['equity_sha256']}",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------- CLI

def _run_all() -> list[tuple[dict, dict]]:
    # 行为锁跑的是一个**面板**，所以所有用例必须声明同一个取池口径；
    # 声明不一致 ⇒ 这个锁本身是歧义的，fail-fast（§0），绝不挑一个默认的用。
    specs = {json.dumps(presets.get_preset(c["preset"]).get("universe"), sort_keys=True)
             for c in CASES}
    if len(specs) != 1:
        raise ValueError(
            f"CASES 里的 preset 声明了 {len(specs)} 种不同的 universe —— "
            f"行为锁必须跑在同一个池子上；请统一声明或拆成两把锁：{sorted(specs)}")
    panel = load_panel_for_preset(presets.get_preset(CASES[0]["preset"]))
    ind = indicators.compute_all(panel)
    return [(c, run_case(c, panel, ind)) for c in CASES]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="交易列表 golden 基线(CLAUDE.md §6)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regen", action="store_true", help="重写基线文件")
    g.add_argument("--show", action="store_true", help="只打印当前指纹")
    a = ap.parse_args(argv)

    for case, info in _run_all():
        print(f"\n{case['name']}  [{case['start']}..{case['end']}]")
        print(f"  vintage    {info['vintage']}")
        print(f"  n_trades   {info['n_trades']}")
        print(f"  trades     {info['trades_sha256']}")
        print(f"  equity     {info['equity_sha256']}")
        if a.regen:
            print(f"  → 已写入 {save_baseline(case['name'], case, info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
