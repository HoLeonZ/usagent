"""数据校验：证明落盘的数据能用，而不是假设它能用。

每一类数据都做「结构性断言 + 分布检查」，不通过就非零退出，便于挂进 cron 告警。

用法：python src/validate.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

import db
from config import COLLECT_OPTIONS, load_universe
from trading_calendar import expected_sessions, last_session_on_or_before

CHECKS: list[tuple[str, bool, str]] = []

# 「池内」= 最近 60 个交易日的成交额均值 ≥ 研究池门槛（与 research/universe.DEFAULT_SPEC
# 同一个量级）。用它把"可疑但反正不会交易的名字"与"会进研究结论的名字"分开 ——
# 前者若也报警，运维层会训练出"反正每天都红"的疲劳（审计已吃过这个教训）。
POOL_MIN_ADV = 3e7


def _pool_symbols(con) -> set[str]:
    rows = con.execute(
        "SELECT symbol, AVG(a) FROM (SELECT symbol, volume*close AS a, "
        "ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) rn FROM ohlcv) "
        "WHERE rn <= 60 GROUP BY symbol HAVING AVG(a) >= ?", (POOL_MIN_ADV,)).fetchall()
    return {r[0] for r in rows}


def _split_pool(items, pool: set[str], all_symbols: set[str]) -> tuple[int, int, bool]:
    """(池内个数, 池外个数, 池是否为空)。

    ⚠️ 池为空时**按全部标的判**（返回的第三项告知调用方），否则这类"池内才判失败"的
    检查会在算不出流动性时**静默通过** —— 那是比误报更坏的失败模式（合成复核就抓过
    "判据形同虚设"这一类）。
    """
    if not pool:
        return sum(1 for _ in items), 0, True
    return (sum(1 for x in items if x in pool),
            sum(1 for x in items if x not in pool), False)


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    uni = load_universe("oi100")
    print(f"目标池: oi100 ({len(uni)} 只)\n")
    db.init()

    with db.session() as con:
        print("── OHLCV ──")
        n, syms, d0, d1 = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) FROM ohlcv").fetchone()
        check("有数据", n > 0, f"{n} 行 / {syms} 只 / {d0} → {d1}")
        if not n:
            return _summary()

        days = (pd.Timestamp(d1) - pd.Timestamp(d0)).days
        check("时间跨度 ≥ 3 年", days >= 365 * 3 - 10, f"{days} 天")

        # OHLC 算术完整性
        bad = con.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE high < MAX(open,close)-1e-4 "
            "OR low > MIN(open,close)+1e-4").fetchone()[0]
        check("High/Low 与 Open/Close 自洽", bad == 0, f"违规 {bad} 行")

        # 非正价格 / 空值
        bad = con.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE close IS NULL OR close<=0 OR volume IS NULL").fetchone()[0]
        check("无空价/非正价", bad == 0, f"违规 {bad} 行")

        # 每只标的的行数：要区分「真的缺数据」和「本来就晚上市」。
        # 做法是拿每只标的自己的 [首日, 末日] 区间里应有几个交易日来比，而不是跟
        # 全期行数比 —— 后者会把 IPO 新股误判为缺口。
        rows = con.execute(
            "SELECT symbol, COUNT(*) c, MIN(date) a, MAX(date) b FROM ohlcv GROUP BY symbol").fetchall()
        gappy = []
        for s, c, a, b in rows:
            # 期望值取自**独立**的交易所日历，不取自库里出现过的日期。
            # 旧写法（拿 DISTINCT date 当日历）是自指的：全市场都缺的那一天对它是隐形的。
            expected = len(expected_sessions(date.fromisoformat(a), date.fromisoformat(b)))
            if expected and c < expected * 0.9:
                gappy.append((s, c, expected))
        gappy.sort(key=lambda x: x[1] / x[2])
        pool = _pool_symbols(con)
        g_in, g_out, pool_empty = _split_pool([s for s, _c, _e in gappy], pool, set())
        check("池内个股无中间缺口（池外稀疏交易是正常的）", g_in == 0,
              ("⚠️ 算不出流动性池 ⇒ 按**全部标的**判；" if pool_empty else "")
              + f"池内缺档 {g_in} 只 / 池外 {g_out} 只（池外多为低价低流动、并非每日成交的名字）"
              + (f"（例：{[(s, c, e) for s, c, e in gappy if s in pool][:4]}）" if g_in else ""))

        starts = sorted(r[2] for r in rows)
        late = sum(1 for s in starts if s > d0)
        print(f"     其中 {late} 只首次上市晚于窗口起点（正常）")

        # ── 新鲜度：用独立日历判「这库到底更新了没有」──
        # 动机（2026-10-17 审计）：取数失败此前会被吞成 exit 0（run_daily 只看 exit code），
        # 于是「数据已更新」与「数据一个月没动」在运维层不可区分。新鲜度是唯一能
        # 分辨两者的判据，而它必须由**外部日历**给出（不能问 ohlcv「你新不新」）。
        today = date.today()
        last_expected = last_session_on_or_before(today)
        stale = expected_sessions(date.fromisoformat(d1) + timedelta(days=1), last_expected)
        # 容差 1 个交易日：每日任务在北京 10:00 跑（= 美东前一日 21:00~22:00），
        # 此时「今天」这个美东交易日尚未收盘，最新一根 bar 自然对应上一交易日。
        check("数据新鲜（缺的交易日 ≤ 1）", len(stale) <= 1,
              f"最新 {d1}；其后应有 {len(stale)} 个交易日"
              + (f"（{stale[0]} … {stale[-1]}）" if stale else ""))

        # ── 全市场缺口：非自指的完整性检查 ──
        # 逐个「应有交易日」数有几只标的有数据；显著低于中位数 ⇒ 那天整批没落盘。
        per_day = dict(con.execute("SELECT date, COUNT(DISTINCT symbol) FROM ohlcv GROUP BY date").fetchall())
        sess = expected_sessions(date.fromisoformat(d0), date.fromisoformat(d1))
        counts = [per_day.get(s.isoformat(), 0) for s in sess]
        if counts:
            med = sorted(counts)[len(counts) // 2]
            holes = [(s.isoformat(), per_day.get(s.isoformat(), 0))
                     for s in sess if med and per_day.get(s.isoformat(), 0) < med * 0.5]
            check("无全市场缺口（不该有 data 的日子）", not holes,
                  f"疑似缺口 {len(holes)} 天" + (f"（例：{holes[:4]}）" if holes else ""))
        unexpected = sorted(set(per_day) - {s.isoformat() for s in sess})
        check("无「不该有数据」的日期（周末/休市）", not unexpected,
              f"多余 {len(unexpected)} 天" + (f"（例：{unexpected[:4]}）" if unexpected else ""))

        # ── 序列尾部截断（审计 A3）──
        # 此前"某只票的数据在窗口中间就断了"在 validate 里**完全隐形**：缺口检查只看
        # 每只标的自己的 [首日,末日] 区间内部。而尾部截断的后果正好落在回测上 ——
        # 引擎在无 bar 时按最近一次收盘无限期重估，持仓天数照走，于是那段权益是"陈价"。
        # 实测（2026-10-17 全量）：2921/3028 无尾缺，池内**没有**一只缺 ≥5 天 ⇒ 今天不咬合，
        # 但判据必须在（下一份数据可能就咬合了）。
        tails = con.execute(
            "SELECT symbol, MAX(date) m FROM ohlcv GROUP BY symbol").fetchall()
        last_panel = date.fromisoformat(d1)
        cut = {}
        for sym, m in tails:
            gap = len(expected_sessions(date.fromisoformat(m) + timedelta(days=1), last_panel))
            if gap >= 5:
                cut[sym] = gap
        c_in, c_out, _ = _split_pool(list(cut), pool, set())
        check("池内无序列尾部截断（缺 ≥5 个交易日）", c_in == 0,
              f"池内 {c_in} 只 / 池外 {c_out} 只"
              + (f"（例：{sorted(((s, g) for s, g in cut.items() if s in pool), key=lambda x: -x[1])[:4]}）"
                 if c_in else ""))

        # ── 取数源：同一标的不得混源（审计 G9）──
        # 两个源的价格口径相同但**事件来源不同**（Yahoo 给显式 dividends/splits，
        # 备用源是从 (a,c) 反解的）。混在同一条序列里，两个口径会无痕地拼在一起 ——
        # 而这种污染没有任何别的检查能发现。
        mixed = con.execute(
            "SELECT symbol, COUNT(DISTINCT source) k FROM ohlcv "
            "GROUP BY symbol HAVING k > 1").fetchall()
        check("同一标的不混源", not mixed,
              f"混源 {len(mixed)} 只" + (f"（例：{[m[0] for m in mixed[:5]]}）" if mixed else ""))
        unknown = con.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE source IS NULL OR source=''").fetchone()[0]
        check("每行都有取数源", unknown == 0, f"缺失 {unknown} 行")

        # ── 疑似基数断层：厂商重述遗留（增量刷新时代的病灶）──
        # 判据：单日倍数**恰好**接近一个整数 ≥2（容差 1%，且当天没有拆股事件）。
        # 真实行情的暴涨暴跌几乎不会精确等于整数倍，而"厂商按 k:1 重述整段历史、
        # 增量刷新只重写了窗口内那一天"必然精确等于整数倍。
        breaks = con.execute("""
            WITH s AS (
              SELECT symbol, date, close, COALESCE(splits,0) AS sp,
                     LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev
              FROM ohlcv)
            SELECT symbol, date, prev, close FROM s
            WHERE prev > 0 AND close > 0 AND sp = 0 AND (
              (close/prev >= 1.98 AND ABS(close/prev - ROUND(close/prev)) / ROUND(close/prev) < 0.01)
              OR (prev/close >= 1.98 AND ABS(prev/close - ROUND(prev/close)) / ROUND(prev/close) < 0.01))
            ORDER BY symbol, date""").fetchall()
        b_in, b_out, _ = _split_pool([s for s, *_ in breaks], pool, set())
        check("池内无疑似价格基数断层", b_in == 0,
              f"池内 {b_in} 只 / 池外 {b_out} 只（池外多为仙股/破产壳：实测有标的在 "
              f"$0.00001 价位上刷出 110 倍的假跳变，成交量仅几十股）"
              + (f"（例：{[s for s, *_ in breaks if s in pool][:4]}）" if b_in else ""))

        # 覆盖率
        cov = {r[0] for r in con.execute(
            "SELECT symbol FROM coverage WHERE kind='ohlcv'")}
        missing = sorted(set(uni.symbols) - cov)
        check("覆盖目标池", not missing,
              f"缺 {len(missing)} 只" + (f"（例：{missing[:8]}）" if missing else ""))

        # 公司行动是否真的抓到（3 年内必然有分红/拆股）
        ev = con.execute("SELECT COUNT(*) FROM ohlcv WHERE dividends>0 OR splits>0").fetchone()[0]
        check("公司行动已捕获", ev > 0, f"{ev} 条事件")

        print("\n── 派生层（日线 → 周线）──")
        import series  # 延迟导入，避免无数据时也加载
        ok, msg = series.source_is_split_adjusted(con)
        check("数据源复权状态已确认", ok, msg)

        probe = "NVDA" if con.execute(
            "SELECT 1 FROM ohlcv WHERE symbol='NVDA' LIMIT 1").fetchone() else (
            con.execute("SELECT DISTINCT symbol FROM ohlcv LIMIT 1").fetchone() or [None])[0]
        if probe:
            d = series.daily(probe, con=con)
            w = series.weekly(probe, con=con)
            wc = series.weekly_complete(probe, con=con)
            if len(d):
                check("周线可合成且压缩比合理", 3.5 <= len(d) / max(len(w), 1) <= 6.0,
                      f"{probe}: 日线 {len(d)} → 周线 {len(w)}（比 {len(d)/max(len(w),1):.2f}）")
                bad = int(((w["high"] < w[["open", "close"]].max(axis=1) - 1e-6) |
                           (w["low"] > w[["open", "close"]].min(axis=1) + 1e-6)).sum())
                check("周线 OHLC 自洽", bad == 0, f"违规 {bad} 行")
                check("能剔除未走完的当周", len(wc) <= len(w), f"{len(w)} → {len(wc)}")

        if not COLLECT_OPTIONS:
            print("\n── 期权链 ──")
            print("  [SKIP] 未采集（config.COLLECT_OPTIONS=False）—— 本项目只做标的数据")
            print("         如需启用:置 True 并跑 src/snapshot_options.py")
        else:
            print("\n── 期权链快照 ──")
            n, snaps, csyms = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT snapshot_date), COUNT(DISTINCT symbol) "
                "FROM option_chain").fetchone()
            check("有快照", n > 0, f"{n} 行 / {snaps} 个交易日 / {csyms} 只")
            if n:
                print(f"     快照日: {[r[0] for r in con.execute('SELECT DISTINCT snapshot_date FROM option_chain ORDER BY 1')][-5:]}")

                # IV 哨兵值：报告 §4.4 实测 1e-05 是「求解失败」的占位符
                sent = con.execute(
                    "SELECT COUNT(*) FROM option_chain WHERE implied_volatility IS NOT NULL "
                    "AND implied_volatility <= 1e-4").fetchone()[0]
                tot = con.execute(
                    "SELECT COUNT(*) FROM option_chain WHERE implied_volatility IS NOT NULL").fetchone()[0]
                pct = sent / tot * 100 if tot else 0
                check("IV 哨兵值比例 < 5%", pct < 5, f"{sent}/{tot} = {pct:.1f}%")

                # IV 上界（深度虚值的离谱值）
                hi = con.execute(
                    "SELECT COUNT(*) FROM option_chain WHERE implied_volatility > 3.0").fetchone()[0]
                check("无 IV > 300% 的不可能值", hi == 0, f"{hi} 行")

                # 报价清零检测 —— Yahoo 隔夜清零会让 bid/ask 全为 0。
                # 这是本项目最容易踩的坑：表面"抓成功"，实际反解不出任何 IV。
                tot_all = con.execute("SELECT COUNT(*) FROM option_chain").fetchone()[0]
                zero = con.execute(
                    "SELECT COUNT(*) FROM option_chain WHERE bid<=0 AND ask<=0").fetchone()[0]
                ratio = zero / tot_all if tot_all else 1.0
                check("报价未清零（bid/ask 非全 0）", ratio < 0.20,
                      f"{zero}/{tot_all} = {ratio*100:.0f}% 为 0"
                      + ("（快照抓在了 Yahoo 清零窗口！检查 cron 时刻）" if ratio >= 0.20 else ""))

                # 行权价带（口径是「最接近现货的 N 档」，百分比只是上限保护）
                outband = con.execute(
                    "SELECT COUNT(*) FROM option_chain "
                    "WHERE ABS(strike/underlying_price - 1) > 0.301").fetchone()[0]
                check("行权价在现货 ±30% 内", outband == 0, f"越界 {outband} 行")

                # 每个 (标的, 到期) 必须至少有一档行权价 —— 否则 IV 无法反解
                noatm = con.execute(
                    "SELECT COUNT(*) FROM (SELECT symbol, expiry FROM option_chain "
                    "GROUP BY symbol, expiry HAVING COUNT(DISTINCT strike)=0)").fetchone()[0]
                check("每个到期至少一档行权价", noatm == 0, f"{noatm} 个空到期")

                # 覆盖率：有多少标的当天拿到了 ≥2 个到期（插值需要）
                got = con.execute(
                    "SELECT COUNT(*) FROM (SELECT symbol FROM option_chain "
                    "WHERE snapshot_date=(SELECT MAX(snapshot_date) FROM option_chain) "
                    "GROUP BY symbol HAVING COUNT(DISTINCT expiry)>=2)").fetchone()[0]
                check("当天 ≥2 个到期的标的数 ≥ 400", got >= 400, f"{got} 只")

                # bid/ask 可用性（平值附近）
                r = con.execute(
                    "SELECT COUNT(*) FROM option_chain "
                    "WHERE ABS(strike/underlying_price-1)<=0.03 "
                    "AND (bid IS NULL OR ask IS NULL)").fetchone()[0]
                check("平值±3% 有双边报价", r == 0, f"缺报价 {r} 行")

                # 到期日是否落在 DTE 窗口
                r = con.execute(
                    "SELECT COUNT(*) FROM option_chain WHERE dte<7 OR dte>120").fetchone()[0]
                check("DTE 在 7-120 窗口内", r == 0, f"越界 {r} 行")

            print("\n── IV 历史 ──")
            n = con.execute("SELECT COUNT(*) FROM iv_history").fetchone()[0]
            if n == 0:
                print("  [SKIP] 尚未生成（先跑 src/derive_iv.py）")
            else:
                check("有 IV 序列", True, f"{n} 行")
                r = con.execute("SELECT MIN(iv_rank), MAX(iv_rank), MIN(iv_percentile), "
                                "MAX(iv_percentile) FROM iv_history").fetchone()
                check("IV Rank/Percentile 在 0-100 内",
                      r[0] is not None and 0 <= r[0] and r[1] <= 100 and 0 <= r[2] and r[3] <= 100,
                      f"Rank {r[0]:.1f}-{r[1]:.1f}, Pctile {r[2]:.1f}-{r[3]:.1f}" if r[0] is not None else "")

    return _summary()


def _summary() -> int:
    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'='*60}\n{len(CHECKS)-len(failed)}/{len(CHECKS)} 通过")
    if failed:
        print("失败项:")
        for n, _, d in failed:
            print(f"  - {n}: {d}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
