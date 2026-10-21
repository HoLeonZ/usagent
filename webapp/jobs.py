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

# 平台上的窗口下限（交易日）。CLI 不受这条约束 —— 它不做统计判读。
MIN_WINDOW_DAYS = 5


class Cancelled(Exception):
    """用户取消。**粗粒度**：只在步骤之间检查 —— 取链路与回测都是单次调用，中途打断不了。"""


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


def params_by_job() -> dict[str, dict | None]:
    """job_id → 那一次**实际生效**的族参数（报告列表要显示"这份是哪组参数跑的"）。

    为什么按 job_id 回读台账，而不是从报告 HTML 里正则抓：参数是台账的既定字段
    （写在 rec["params"]），报告里根本没有它 —— 从渲染产物里反推事实是"一份事实两处"。
    """
    out: dict[str, dict | None] = {}
    if not RUNS_LEDGER.exists():
        return out
    for line in RUNS_LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("job_id"):
            out[r["job_id"]] = r.get("params")
    return out


# 台账里读不出来的行数（由 all_runs 顺手记下）。**不能静默跳过**：
# 少读一行 = 少记一次试验 = DSR 的分母 k 偏小，而这个数是"我到底试了多少次"的唯一载体。
# 仓库自己的纪律台账（research/holdout.py）遇到坏行是**直接抛 LedgerCorrupt** 并要求去修
# （"台账坏了必须修，不能跳过"）—— 平台的这一本不该比它松。
_SKIPPED = 0


def skipped_lines() -> int:
    """上一次 all_runs() 跳过了几行（0 = 台账完好）。"""
    return _SKIPPED


def all_runs() -> list[dict]:
    """台账**全部**记录（时间正序）。导 CSV 与"共几条"都从这里来 ——
    页面上的表只显示最近 30 条，若导出也只用那 30 条，"导出"这两个字就是在骗人。

    坏行跳过但**记数**（见 `skipped_lines`）：页面据此警告"台账有行读不出来，记账可能偏小"，
    而不是让读者以为那就是全部。
    """
    global _SKIPPED
    _SKIPPED = 0
    if not RUNS_LEDGER.exists():
        return []
    out = []
    for ln in RUNS_LEDGER.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            _SKIPPED += 1
            continue
    return out


def recent_runs(limit: int = 20) -> list[dict]:
    return list(reversed(all_runs()[-limit:]))


def last_by_strategy() -> dict[str, dict]:
    """每条策略**最近一次**运行的读数（全文台账）。

    与 `counts_by_strategy` 同理：卡片的「已跑 N 次」与「最近一次」是同一个量的两面，
    若一个按全台账数、另一个按"最近 30 条"取，第 31 次运行之后同一张卡会自相矛盾
    （审计实测：卡片同时说「已跑 31 次」和「最近一次」停在更早那次）。
    """
    out: dict[str, dict] = {}
    for r in all_runs():
        k = r.get("strategy_id")
        if k:
            out[k] = r                      # 正序覆盖 ⇒ 最后留下的就是最近一次
    return out


def counts_by_strategy() -> dict[str, int]:
    """每条策略**全部**运行次数。

    为什么不能拿 recent_runs(30) 现数：卡片上的「已跑 N 次」与 DSR 的 k 用的是同一个
    计数（`prior_runs` 读全文），而页面只取最近 30 条 —— 到第 31 次运行时，界面会说
    "已跑 30" 而 k 已经是 31（同一个量的两个数字，本仓库最忌讳的那种）。
    """
    out: dict[str, int] = {}
    for r in all_runs():
        k = r.get("strategy_id")
        if k:
            out[k] = out.get(k, 0) + 1
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
        # 窗口伸到池子覆盖之外时，超额的比较区间被裁过 —— 必须让面板说得出来
        # （否则用户以为算的是整段窗口）。实测：起点早于池子覆盖的窗口，修复前会报
        # −15.09pp 的假超额，现在报 0 并标明"只比了 2024-01-11.."
        "excess_clamped": bool(ov.get("clamped")),
        "excess_compared_from": ov.get("compared_from"),
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
    job = {"id": job_id, "status": "queued", "cancel": False,
           "step": "取链路（首次约 30–70 秒；命中缓存近 0 秒）", "steps": [],
           "request": req, "metrics": None, "report": None, "error": None,
           "started": obs.now_iso(), "seconds": None}
    _JOBS[job_id] = job
    threading.Thread(target=_run, args=(job_id,), daemon=True).start()
    return dict(job)


def cancel(job_id: str) -> bool:
    """请求取消一次运行。返回「这次请求是否被受理」。

    诚实边界：当前这一步**跑完才会停**（载面板 30–70 秒、回测数秒，都是单次调用，
    打断不了）—— 界面上也这么说。取消的运行**不进台账**：没有结果被看到，
    它就不该计入 §5 的迭代次数（k 是"看过的试验数"，不是"花掉的 CPU"）。
    """
    j = _JOBS.get(job_id)
    if not j or j["status"] in ("done", "error", "cancelled"):
        return False
    j["cancel"] = True
    return True


def inflight() -> list[dict]:
    """还在跑（queued/running）的任务清单 —— 页面**刷新之后**靠它把状态找回来。

    为什么必须有它（2026-10-18 审计的 blocking #1，两名核验者独立复现）：job id 只活在被刷掉的
    那份 DOM 里 —— 没有 localStorage/sessionStorage 落它，服务端也只按 id 查、没有枚举入口。
    于是运行中刷新页面就会：徽标消失、面板消失、取消入口消失、页面还写着「还没跑过」（假话，
    服务端此刻仍在跑）。用户在这句假话下的自然反应是**再点一次回测** —— 那会把 k 再推高一次。
    """
    out = []
    for jid, j in list(_JOBS.items()):
        if j.get("status") not in ("queued", "running"):
            continue
        req = j.get("request") or {}
        out.append({"id": jid, "status": j["status"], "step": j.get("step"),
                    "kind": req.get("kind"), "strategy": req.get("id"),
                    "lo": req.get("lo"), "hi": req.get("hi"),
                    "steps": j.get("steps") or [],
                    "seconds": None if j.get("seconds") is None else float(j["seconds"])})
    return out


def status(job_id: str) -> dict | None:
    j = _JOBS.get(job_id)
    return dict(j) if j else None


def _check_cancel(job: dict) -> None:
    if job.get("cancel"):
        raise Cancelled("已取消")


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
            _check_cancel(job)                        # 排队期间被取消：连面板都不载
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
            _check_cancel(job)
            panel, ind, reg = chain

            # ---- 窗口体检（**在**信号与回测之前）----
            # 审计实测：一个"窗口里没有交易日"的输入，要先跑完信号（7–11 秒）才由 broker 报错，
            # 而报错还不告诉你合法范围。链路已经在手上了，交易日轴几乎不值钱 —— 就在这里问，
            # 答案里带上这个池子真正可评估的那段。
            axis = pd.Index(sorted(panel["date"].unique()))
            lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
            n_days = int(len(axis[(axis >= lo_ts) & (axis < hi_ts)]))
            # 范围在**拒收之前**就写进 job：只失败过、还没成功过的会话里，窗口旁那句
            # 「池子可评估范围」否则会一直是占位文案 —— 而正确范围刚被算出来、就在错误信息里
            job["pool_range"] = [str(panel_mod.first_signal_date(panel).date()),
                                 str(axis[-1].date())]
            if not n_days:
                raise ValueError(
                    f"窗口 {lo}..{hi} 里没有任何交易日 —— 这个池子的数据覆盖 "
                    f"{axis[0].date()} .. {axis[-1].date()}（指标预热完之后可评估的是 "
                    f"{panel_mod.first_signal_date(panel).date()} .. {axis[-1].date()}）")
            # 太短的窗口此前会**一路跑完**信号与回测，最后在 overlay 里以一条没法行动的错误失败
            # （审计实测）。这里当场拦下并说清为什么需要这么多天。
            if n_days < MIN_WINDOW_DAYS:
                raise ValueError(
                    f"窗口 {lo}..{hi} 只有 {n_days} 个交易日 —— 至少要 {MIN_WINDOW_DAYS} 个："
                    f"入场、出场各要一天，结算与敞口统计也各要一天；再短的窗口跑出来的"
                    f"DSR/超额是在噪声上算的。（要极致短窗请用命令行 render_report。）")

            # k 的规则**只写一次**，提示与 DSR 的 k 都用它 —— 分头写必然各自漂移（下面那条注释
            # 记的就是这么漂出去的一次）。
            declared = (int(families.grid_size(ident) or 1) if kind == "family"
                        else int(P.get("n_comparisons") or 1))
            t0 = time.perf_counter()
            if P.get("family"):
                fid = str(P["family"])
                sig = families.select(ind, fid, dict(P.get("family_params") or {}), lo, hi, reg=reg)
                # ⚠️ 「用族掩码」**不等于**「k 从网格起算」：已交付的 preset 内部照样用掩码，
                # 但它的 k 是**声明的 n_comparisons**（那批网格早已计入那次搜索）。验收实测
                # 这里曾报「本族网格 96 组 ⇒ k 从这里起算」，而同一份报告的 k 是 1622（1614+8）。
                hint = (f"本族网格 {declared} 组 ⇒ DSR 的 k 从这里起算" if kind == "family"
                        else f"该 preset 声明 n_comparisons={declared}"
                             + (f"（内部用族 {fid} 的掩码；那 {families.grid_size(fid)} 组"
                                f"已计入上面这次声明）" if families.grid_size(fid) else ""))
            else:
                sig = signals.select_entries(ind, start_date=lo, end_date=hi,
                                             regime_df=reg if P.get("regime") else None,
                                             **P["signal"])
                hint = f"该 preset 声明 n_comparisons={declared}"
            _step(job, f"计算信号（{len(sig)} 个候选）", t0)
            _check_cancel(job)

            t0 = time.perf_counter()
            res = broker.simulate(P, sig, ind, start=lo, end=hi)
            eq = res["equity"].set_index("date")
            _step(job, f"回测（成交 {len(res['trades'])} 笔）", t0)
            _check_cancel(job)

            ctl_runs = int(req.get("control_runs") or 0)
            cagr_now = float(metrics.cagr(eq["equity"], pd.Series(eq.index)))
            t0 = time.perf_counter()
            ctl = RR.control_arm(ind, resolve(P), len(res["trades"]), lo, hi, reg,
                                 ctl_runs, cagr_now)
            if ctl_runs:
                # 单独一步 + **真实**耗时（此前把这一步的时间记在渲染步里，且标签上的估时
                # 是照抄另一条 preset 的测量 —— 实测同一条 preset 只要 3 秒，估时却写 26 秒。
                # 数字要么是量出来的，要么给区间）。
                _step(job, f"运气对照：{ctl_runs} 次随机入场（每次约 15–130 毫秒，视笔数）", t0)

            # DSR 的 k：声明/网格数 + **平台已跑次数**（§5 迭代记账，页面也显示它）
            # k 的组成**只有这一处算法**（面板、台账、测试都读它）
            kb = k_accounting(P, kind, ident, declared)
            k = kb["k"]
            prior = kb["platform_runs_before"]
            m = _metrics(res, eq, reg, lo, hi, k)
            m["k_breakdown"] = kb

            t0 = time.perf_counter()
            # ⚠️ render_html 返回的是**字符串**（不是 (doc, n) 二元组）—— 实测踩过
            doc = H.render_html(res["trades"], eq, ind,
                                P.get("signal") or P.get("family_params") or {},
                                _bench_series(reg, eq),
                                _summary(P, res, sig, eq, name, lo, hi, ctl),
                                explain=_explain_for(P))
            problems = H.selfcheck(doc, len(res["trades"]))
            report = RESULTS / f"web_report_{job_id}.html"
            try:
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(doc, encoding="utf-8")
            except OSError as exc:
                raise _artifact_failure(report, "报告", exc) from None
            _step(job, f"渲染报告（{report.stat().st_size/1024:.0f} KB）", t0)

            t0 = time.perf_counter()
            trades_csv = RESULTS / f"web_trades_{job_id}.csv"
            n_csv = write_trades_csv(trades_csv, res["trades"], job_id=job_id, name=name,
                                     lo=lo, hi=hi, spec=resolve(P), n_trials=k,
                                     declared=declared, prior=prior)
            _step(job, f"导出逐笔 CSV（{n_csv} 行）", t0)

            t0 = time.perf_counter()
            # ⚠️ sha 与"脏不脏"必须**一起**记（obs.git_dirty 的 docstring 讲得很清楚：
            # 脏工作区上 rev-parse HEAD 照样返回同一个 sha ⇒ 只记 sha 答不出"跑的是不是
            # 这份代码"）。台账里已经混着 12 个不同的 sha 了，而页面上此前一个字都不显示。
            rec = {"ts": obs.now_iso(), "git_sha": obs.git_sha(),
                   "git_dirty": obs.git_dirty(), "kind": kind,
                   "strategy_id": f"{kind}:{ident}", "name": name, "window": [lo, hi],
                   # 记**实际生效**的参数：族型 runner 会用网格中点那一组，
                   # 只记 req["params"]（可能是 None）等于没记（实测发现的）。
                   # 信号型 preset 的参数在 P["signal"] 里（不是 family_params）—— 漏了它
                   # 台账会对这类 preset 一律显示"（在定义体里）"，而参数其实就在手边。
                   # 与 `_summary` 的取法是同一个并集：族参数与 signal 参数在这一层是同一件事。
                   "params": (req.get("params") or P.get("family_params")
                              or P.get("signal") or None),
                   "n_trials_k": k, "prior_runs": prior,
                   # 这条候选从哪条策略复制而来（§5：派生候选要把父策略的平台运行次数并进 k）
                   "derived_from": P.get("derived_from"),
                   "n_trades": m["n_trades"], "excess": m["excess"], "dsr": m["dsr"],
                   # 台账里多记几个**动态指标** —— 否则"跑完两条候选怎么比"只能人去翻两份报告
                   # （或从渲染好的 HTML 里反推事实，那是一份事实两处）。有了它们，平台就能
                   # 直接给出逐项对比，并在 k 不同时**明说 DSR 不可比**（2026-10-20 的教训）。
                   "cagr": m["cagr"], "exposure": m["exposure"], "bench_cagr": m["bench_cagr"],
                   "maxdd": m["maxdd"], "win_pct": m["win_pct"], "e_r": m["E_R"],
                   "profit_factor": m["profit_factor"], "folds_positive": m["folds_positive"],
                   "alpha_annual": m["alpha_annual"], "t_alpha": m["t_alpha"],
                   # ⚠️ job_id 必须记：报告文件名就是 web_report_<job_id>.html，不记它这条台账
                   # 就是**死胡同** —— 想再看一眼那次结果只能重跑，而重跑会把 k 推高
                   # （拿统计预算买回已经付过费的结果）。审计实测：盘上 6 份报告、界面 0 个入口。
                   "job_id": job_id, "report": f"web_report_{job_id}.html",
                   "control_runs": ctl_runs,
                   # ⚠️ 不能写 job["seconds"]：它由 finally 赋值，而这一行在 finally **之前**
                   # ⇒ 台账 9 条全是 null（记了却永远用不上的字段，审计实测）。就地算。
                   "seconds": round(time.perf_counter() - t_all, 2),
                   "note": req.get("note", "")}
            try:
                RUNS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
                with RUNS_LEDGER.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            except OSError as exc:
                raise _artifact_failure(RUNS_LEDGER, "台账", exc) from None
            obs.log_run("平台回测", True, seconds=time.perf_counter() - t_all,
                        strategy=rec["strategy_id"], window=f"{lo}..{hi}",
                        n_trades=m["n_trades"], excess=round(m["excess"], 6), k=k)
            _step(job, "登记台账", t0)

            job.update({"status": "done", "metrics": m,
                        "hint": hint + (f"；运气对照已跑 {ctl_runs} 次" if ctl_runs
                                       else "；运气对照**未跑**（报告里会写明）"),
                        "report": f"/report/{job_id}", "report_path": str(report),
                        "trades_csv": f"/api/trades/{job_id}.csv",
                        "selfcheck": problems})
    except Cancelled:
        # 取消**不写台账**：没有结果被看到，就不算一次被看过的试验（k 只数看过的）
        job["status"] = "cancelled"
        # 面板会打「已取消 —— {这句}」，所以这里不要再重复"已取消"
        job["error"] = "未记账：这一跑没有结果被看到，因此不计入 k"
        obs.log_run("平台回测", False, seconds=time.perf_counter() - t_all,
                    strategy=f"{req.get('kind')}:{req.get('id')}", detail="cancelled")
    except Exception as exc:                          # noqa: BLE001 —— 页面要看到原因
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        obs.log_run("平台回测", False, seconds=time.perf_counter() - t_all,
                    strategy=f"{req.get('kind')}:{req.get('id')}", detail=job["error"])
    finally:
        job["seconds"] = round(time.perf_counter() - t_all, 2)


TRADES_CSV_COLS = ["entry_date", "exit_date", "symbol", "exit_reason", "entry_price",
                   "exit_price", "size", "hold_days", "gross_pnl", "dividends", "fees",
                   "net_pnl", "net_return", "atr_pct", "exit_trigger", "sl_width",
                   "tp_width", "risk_amount", "sizing_cap", "risk_frac", "sl_penetration"]


def _artifact_failure(path: Path, kind: str, exc: OSError) -> RuntimeError:
    """产物写不进去时给人话，而不是裸 PermissionError（实测：只读目录下用户只看到
    `PermissionError: [Errno 13] Permission denied: '/tmp/…/web_report_xxx.html'` —— 既不说
    是磁盘满还是目录只读，也不说该去看哪儿）。"""
    return RuntimeError(
        f"写不进{kind}：{path.name} —— {exc.strerror or exc}\n"
        f"  常见原因：磁盘满 / results 目录只读 / 权限不对。\n"
        f"  产物目录：{path.parent}")


def write_trades_csv(path: Path, trades: pd.DataFrame, *, job_id: str, name: str,
                     lo: str, hi: str, spec: dict, n_trials: int,
                     declared: int | None = None, prior: int | None = None) -> int:
    """把逐笔写成 CSV —— 报告是给人看的，**这份是给脚本/表格看的**。

    为什么必须有：报告是 1.9MB 的自包含 HTML，逐笔那 24 列只存在于它的 `<table>` 里；
    想把这些数字拿去做别的计算（换个口径统计、画别的图、和别的策略对齐）此前只能从 HTML
    里抠。数据本来就在手上（`res["trades"]`），存一份成本几乎为零。

    口径必须随文件走：前面几行 `#` 注释写清窗口、成本、价格口径与 k —— 一份没有口径的
    数字表会被当成"可以直接算的东西"，而这里的成交价是 **as-traded**（§3(b) 的双轨），
    拿复权价去对会差一截。（`pandas.read_csv(..., comment="#")` 能直接吃掉这些注释行。）
    """
    cols = [c for c in TRADES_CSV_COLS if c in trades.columns]
    head = [
        f"# usagent 逐笔明细 · {name}",
        f"# 任务 {job_id} ｜ 窗口 {lo}..{hi}（左闭右开）｜ 成交 {len(trades)} 笔",
        f"# 价格口径：成交价/止损止盈触发/盈亏一律 as-traded（§3(b) 双轨）；"
        f"atr_pct 等指标列为当日 as-traded 派生量",
        f"# 成本：双边 {spec.get('cost_bps_roundtrip')!r}bp ｜ 滑点 scale={spec.get('atr_slip_scale')!r}",
        # ⚠️ 这里必须写清 k 是**哪一个**数：报告里那句「N 次试验里选出来的」用的是
        # **声明**的 n_comparisons，而这一列是 DSR 的分母 = 声明 + 平台已跑次数 ——
        # 同一个词 n_trials 在两份产物里指两个数（审计实测：报告 1614 / 同一 job 的 CSV 1627）。
        f"# DSR 分母 k={int(n_trials)}"
        + (f"（= 声明 n_comparisons {int(declared)} + 平台在本平台已跑 {int(prior)} 次）"
           if declared is not None and prior is not None else ""),
        # ⚠️ 刻度也要写：这一份是**小数**（0.1188 = 11.88%），而报告里同名列印百分数。
        "# 刻度：*_pct / net_return 等比率列一律给**小数**（0.1188 就是 11.88%）；"
        "报告 HTML 里那些数字印的是百分数 —— 同一列两处刻度不同，别混用",
        f"# 这是机器可读件；人看的版本是同一 job 的报告 HTML（含 K 线/指标/判词）",
        "# 读法：pandas.read_csv(path, comment='#')",
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(head) + "\n")
            trades[cols].to_csv(fh, index=False)
    except OSError as exc:
        raise _artifact_failure(path, "逐笔 CSV", exc) from None
    return len(trades)


def k_accounting(P: dict, kind: str, ident: str, declared: int) -> dict:
    """DSR 分母 k 的组成 —— **一处算法**：面板、台账、测试都读它。

    k = 声明值/网格数 + **继承自父策略的平台运行次数** + 本条策略在本平台的运行次数。

    为什么要"继承"这一项（2026-10-20 用平台**真做**一次迭代时撞出来的）：平台按**策略名**各自
    数运行次数，而"复制一条 preset 改个参数"得到的新名字 prior 恒为 0 ⇒ 它的 DSR 分母偏小、
    判词偏乐观 —— 可它其实是**同一片搜索空间**里的又一次试验。实测：基线 k=20（17+3），
    兄弟记成 17（17+0），实际应为 17+3+1=21。

    机制：候选在定义体里声明 `derived_from: "<父策略 id>"`（已进 METADATA_KEYS）。
    **父策略的声明值不继承** —— 那该由作者写进本条候选自己的 n_comparisons（它属于"这片空间
    试过多少次"）；平台只补**运行期才知道**的那一半：父策略在这台机器上被跑过几次。
    """
    parent = P.get("derived_from")
    inherited = prior_runs(f"preset:{parent}") if parent else 0
    own = prior_runs(f"{kind}:{ident}")
    return {"declared_or_grid": int(declared), "inherited_from_parent": inherited,
            "inherited_parent": (str(parent) if parent else None),
            "platform_runs_before": own,
            "k": int(declared) + inherited + own}


def _bench_series(reg: pd.DataFrame, eq: pd.DataFrame) -> pd.Series:
    """基准序列（与 render_report.run 的取法一致：同窗、首日归一到 1）。"""
    idx = reg.set_index("date")["ew_index"]
    b = idx[(idx.index >= pd.Timestamp(eq.index[0])) & (idx.index <= pd.Timestamp(eq.index[-1]))]
    return (b / b.iloc[0]).reindex(eq.index).ffill().bfill()


def _summary(P: dict, res: dict, sig: pd.DataFrame, eq: pd.DataFrame,
             name: str, lo: str, hi: str, ctl: dict | None = None) -> dict:
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
        "note": None,
        **{k: v for k, v in (ctl or {}).items()},
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
