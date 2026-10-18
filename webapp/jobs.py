"""回测任务与结果 —— 平台的服务端核心：**跑哪条策略、跑出什么、跑过几次**。

三件事，都刻意做得很小：
  1. **任务**：`submit()` 收一份请求（策略 + 窗口），后台线程跑，前端轮询 `status()`。
  2. **链路缓存**：面板/指标/regime 是 35–66 秒的重活，缓存起来 ⇒ 第二次点击只要几秒。
     缓存与 `render_report.run()` 共用**同一个** `prepare_chain()`，不是第二份接线。
  3. **台账**：每一次点击都写 `results/web_runs.jsonl` 与 `logs/runs.jsonl`，并把它当作
     §5 的迭代记账 —— 页面上显示"这条策略已被跑过 N 次、DSR 的 k 随之上涨"。
     一个能一键重跑的界面天然会诱发 p-hacking；纪律的做法不是禁止点击，而是**让次数可见**。

结果指标与仓库别处**同一套口径**（`metrics` / `overlay`）：曝光匹配超额、beta/alpha、
尾部统计（E[R]/胜率/PF/top-decile 占比）、DSR（k = 声明试验数 + 平台已跑次数）。
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import obs                                                        # noqa: E402
from core.preset_contract import resolve                           # noqa: E402
from research import broker, families, html_report as H, metrics  # noqa: E402
from research import overlay, panel as panel_mod, presets         # noqa: E402
from research import render_report as RR, signals                 # noqa: E402

RESULTS = ROOT / "research" / "results"
RUNS_LEDGER = RESULTS / "web_runs.jsonl"

_LOCK = threading.Lock()          # 重活串行：面板载入与回测都在这个锁里
_JOBS: dict[str, dict] = {}
_CHAIN: dict[str, tuple] = {}     # universe_key -> (panel, ind, reg)；目前只有一种池子口径


def prior_runs(strategy_id: str) -> int:
    """这条策略在平台上被跑过几次（§5 迭代记账的**可见化**）。"""
    if not RUNS_LEDGER.exists():
        return 0
    n = 0
    for line in RUNS_LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("strategy_id") == strategy_id:
                n += 1
        except json.JSONDecodeError:
            continue
    return n


def recent_runs(limit: int = 20) -> list[dict]:
    if not RUNS_LEDGER.exists():
        return []
    lines = RUNS_LEDGER.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for ln in reversed(lines):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _chain_for(P: dict) -> tuple:
    """按**取池口径**缓存链路。键取 preset 的 universe 声明 —— 换池子必须换链路。"""
    key = json.dumps(P.get("universe") or {}, sort_keys=True, default=str)
    if key not in _CHAIN:
        _CHAIN[key] = RR._chain_from(P)
    return _CHAIN[key]


def _metrics(res: dict, eq: pd.DataFrame, reg, lo: str, hi: str, k: int) -> dict:
    """与仓库别处同一套口径（不是页面自己的第二套算法）。"""
    ov = overlay.excess_vs_pool(eq, reg, lo, hi)
    ba = metrics.beta_attribution(eq["equity"].pct_change().fillna(0.0),
                                  overlay.pool_daily_returns(reg), eq["exposure"])
    rets = eq["equity"].pct_change().dropna()
    folds = np.array_split(rets, 4)
    ts = metrics.tail_stats(res["trades"]) if len(res["trades"]) else {"n": 0}
    dsr = metrics.deflated_sharpe_from_returns(rets, n_trials=max(int(k), 1))
    return {
        "n_trades": int(len(res["trades"])),
        "own_ret": float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0),
        "cagr": float(metrics.cagr(eq["equity"], pd.Series(eq.index))),
        "exposure": float(eq["exposure"].mean()),
        "bench_cagr": float(ov["bench"]),
        "excess": float(ov["excess"]),
        "excess_per_exposure": (float(ov["excess"] / ov["exposure"]) if ov["exposure"] else None),
        "beta": float(ba["beta"]), "alpha_annual": float(ba["alpha_annual"]),
        "t_alpha": float(ba["t_alpha"]),
        "maxdd": float((eq["equity"] / eq["equity"].cummax() - 1).min()),
        "folds_positive": int(sum(1 for f in folds if len(f) and float((1 + f).prod() - 1) > 0)),
        "E_R": float(ts.get("expectancy_r", float("nan"))),
        "win_pct": float(ts.get("win_pct", float("nan"))),
        "profit_factor": float(ts.get("profit_factor", float("nan"))),
        "top_decile_share": float(ts.get("top_decile_share", float("nan"))),
        "dsr": float(dsr.get("deflated_sharpe", float("nan"))), "dsr_k": int(max(int(k), 1)),
    }


def submit(req: dict) -> dict:
    """收一份回测请求，立刻返回任务句柄（前端轮询 status）。"""
    job_id = uuid.uuid4().hex[:12]
    # 初始 step 就把"最慢的那一步"说出来（首次冷启动 30–70 秒在界面上不能是假死）
    job = {"id": job_id, "status": "queued",
           "step": "取链路（首次约 30–70 秒；命中缓存近 0 秒）", "steps": [],
           "request": req, "metrics": None, "report": None, "error": None,
           "started": obs.now_iso(), "seconds": None}
    _JOBS[job_id] = job
    threading.Thread(target=_run, args=(job_id,), daemon=True).start()
    return dict(job)


def status(job_id: str) -> dict | None:
    j = _JOBS.get(job_id)
    return dict(j) if j else None


def _step(job: dict, name: str, t0: float) -> None:
    job["steps"].append({"name": name, "seconds": round(time.perf_counter() - t0, 2)})
    job["step"] = name


def _run(job_id: str) -> None:
    job = _JOBS[job_id]
    req = job["request"]
    t_all = time.perf_counter()
    try:
        with _LOCK:                                   # 重活串行，避免面板/CPU 互抢
            job["status"] = "running"
            kind, ident = req["kind"], req["id"]
            lo, hi = req["lo"], req["hi"]
            if lo >= hi:
                raise ValueError(f"窗口非法：{lo} >= {hi}")
            t0 = time.perf_counter()
            if kind == "preset":
                P, name = RR._strategy(ident)
                chain = _chain_for(P)
            elif kind == "family":
                P, name = RR._strategy("us_reversal_dip_v1", family=ident,
                                       params=req.get("params") or None)
                chain = _chain_for(P)
            else:
                raise ValueError(f"未知策略类型 {kind!r}")
            _step(job, "取链路（命中缓存则近 0 秒）", t0)
            panel, ind, reg = chain

            t0 = time.perf_counter()
            if P.get("family"):
                fid = str(P["family"])
                sig = families.select(ind, fid, dict(P.get("family_params") or {}), lo, hi, reg=reg)
                hint = "本族网格 %.0f 组 ⇒ DSR 的 k 从这里起算" % (families.grid_size(fid) or 1)
            else:
                sig = signals.select_entries(ind, start_date=lo, end_date=hi,
                                             regime_df=reg if P.get("regime") else None,
                                             **P["signal"])
                hint = "该 preset 声明 n_comparisons=%d" % int(P.get("n_comparisons") or 1)
            _step(job, f"计算信号（{len(sig)} 个候选）", t0)

            t0 = time.perf_counter()
            res = broker.simulate(P, sig, ind, start=lo, end=hi)
            eq = res["equity"].set_index("date")
            _step(job, f"回测（成交 {len(res['trades'])} 笔）", t0)

            # DSR 的 k：声明/网格数 + **平台已跑次数**（§5 迭代记账，页面也显示它）
            prior = prior_runs(f"{kind}:{ident}")
            # k = 声明/网格数 + **平台已跑次数**（§5：手动迭代也是优化，必须记账）
            declared = (int(families.grid_size(ident) or 1) if kind == "family"
                        else int(P.get("n_comparisons") or 1))
            k = declared + prior
            m = _metrics(res, eq, reg, lo, hi, k)
            m["k_breakdown"] = {"declared_or_grid": declared, "platform_runs_before": prior}

            t0 = time.perf_counter()
            # ⚠️ render_html 返回的是**字符串**（不是 (doc, n) 二元组）—— 实测踩过
            doc = H.render_html(res["trades"], eq, ind,
                                P.get("signal") or P.get("family_params") or {},
                                _bench_series(reg, eq),
                                _summary(P, res, sig, eq, name, lo, hi),
                                explain=_explain_for(P))
            problems = H.selfcheck(doc, len(res["trades"]))
            report = RESULTS / f"web_report_{job_id}.html"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(doc, encoding="utf-8")
            _step(job, f"渲染报告（{report.stat().st_size/1024:.0f} KB）", t0)

            t0 = time.perf_counter()
            rec = {"ts": obs.now_iso(), "git_sha": obs.git_sha(), "kind": kind,
                   "strategy_id": f"{kind}:{ident}", "name": name, "window": [lo, hi],
                   # 记**实际生效**的参数：族型 runner 会用网格中点那一组，
                   # 只记 req["params"]（可能是 None）等于没记（实测发现的）
                   "params": req.get("params") or (P.get("family_params") or None),
                   "n_trials_k": k, "prior_runs": prior,
                   "n_trades": m["n_trades"], "excess": m["excess"], "dsr": m["dsr"],
                   # ⚠️ job_id 必须记：报告文件名就是 web_report_<job_id>.html，不记它这条台账
                   # 就是**死胡同** —— 想再看一眼那次结果只能重跑，而重跑会把 k 推高
                   # （拿统计预算买回已经付过费的结果）。审计实测：盘上 6 份报告、界面 0 个入口。
                   "job_id": job_id, "report": f"web_report_{job_id}.html",
                   "seconds": job["seconds"], "note": req.get("note", "")}
            RUNS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with RUNS_LEDGER.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            obs.log_run("平台回测", True, seconds=time.perf_counter() - t_all,
                        strategy=rec["strategy_id"], window=f"{lo}..{hi}",
                        n_trades=m["n_trades"], excess=round(m["excess"], 6), k=k)
            _step(job, "登记台账", t0)

            job.update({"status": "done", "metrics": m, "hint": hint,
                        "report": f"/report/{job_id}", "report_path": str(report),
                        "selfcheck": problems})
    except Exception as exc:                          # noqa: BLE001 —— 页面要看到原因
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        obs.log_run("平台回测", False, seconds=time.perf_counter() - t_all,
                    strategy=f"{req.get('kind')}:{req.get('id')}", detail=job["error"])
    finally:
        job["seconds"] = round(time.perf_counter() - t_all, 2)


def _bench_series(reg: pd.DataFrame, eq: pd.DataFrame) -> pd.Series:
    """基准序列（与 render_report.run 的取法一致：同窗、首日归一到 1）。"""
    idx = reg.set_index("date")["ew_index"]
    b = idx[(idx.index >= pd.Timestamp(eq.index[0])) & (idx.index <= pd.Timestamp(eq.index[-1]))]
    return (b / b.iloc[0]).reindex(eq.index).ffill().bfill()


def _summary(P: dict, res: dict, sig: pd.DataFrame, eq: pd.DataFrame,
             name: str, lo: str, hi: str) -> dict:
    ts = metrics.tail_stats(res["trades"]) if len(res["trades"]) else {"expectancy_r": float("nan")}
    return {
        "preset_name": name, "preset": resolve(P),
        "signal": P.get("signal") or P.get("family_params") or {},
        "lo": lo, "hi": hi,
        "cagr": float(metrics.cagr(eq["equity"], pd.Series(eq.index))),
        "max_dd": float(res["metrics"]["max_dd"]),
        "win": float(res["metrics"]["win_rate"]) if len(res["trades"]) else float("nan"),
        "e_r": float(ts["expectancy_r"]),
        "exposure": float(res["metrics"]["avg_exposure"]),
        "sig_dates": _signal_dates(res, sig),
        "n_trials": int(P.get("n_comparisons") or 1),
        "verdict": (P.get("verdict") or "").strip() or None,
        "note": None, "ctl_runs": 0, "ctl_failed": 0, "ctl_p90": float("nan"),
        "ctl_max": float("nan"), "ctl_pct": float("nan"),
    }


def _signal_dates(res: dict, sig: pd.DataFrame) -> dict:
    """每笔的信号日（**与 render_report.run 同一口径**：入场日之前最后一个信号日）。

    这张表是卡片上"信号日"那一行的来源；口径若与报告不同，图与表就会各说一套。
    """
    out: dict = {}
    for _, r in res["trades"].iterrows():
        if sig is not None and len(sig):
            c = sig[(sig["symbol"] == r["symbol"]) & (sig["date"] < r["entry_date"])]
            out[r["symbol"]] = c["date"].max() if len(c) else r["entry_date"]
        else:
            out[r["symbol"]] = r["entry_date"]
    return out


def _explain_for(P: dict):
    if P.get("family"):
        fid = str(P["family"])
        from functools import partial
        return partial(H.family_explain_rows, mask=families.MASK[fid],
                       params=P.get("family_params") or {})
    return H.condition_rows
