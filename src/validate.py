"""数据校验：证明落盘的数据能用，而不是假设它能用。

每一类数据都做「结构性断言 + 分布检查」，不通过就非零退出，便于挂进 cron 告警。

用法：python src/validate.py
"""
from __future__ import annotations

import sys

import pandas as pd

import db
from config import COLLECT_OPTIONS, load_universe

CHECKS: list[tuple[str, bool, str]] = []


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
        cal = {r[0] for r in con.execute("SELECT DISTINCT date FROM ohlcv")}
        rows = con.execute(
            "SELECT symbol, COUNT(*) c, MIN(date) a, MAX(date) b FROM ohlcv GROUP BY symbol").fetchall()
        gappy = []
        for s, c, a, b in rows:
            expected = sum(1 for d in cal if a <= d <= b)
            if expected and c < expected * 0.9:
                gappy.append((s, c, expected))
        gappy.sort(key=lambda x: x[1] / x[2])
        check("个股无中间缺口（已排除新股晚上市）", not gappy,
              f"缺档 {len(gappy)} 只" + (f"（例：{[(s, c, e) for s, c, e in gappy[:4]]}）" if gappy else ""))

        starts = sorted(r[2] for r in rows)
        late = sum(1 for s in starts if s > d0)
        print(f"     其中 {late} 只首次上市晚于窗口起点（正常）")

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
