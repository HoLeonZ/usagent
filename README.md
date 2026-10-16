# usagent — 美股数据层

「Nasdaq 上市股票 → 周线趋势 → 单腿期权」这条链路的数据底座。**只做取数与落盘，不做交易。**

数据源全部是合规的公开接口（Yahoo Finance via yfinance + marketdata.app），不碰券商通道。

---

## 这套东西解决的问题

期权链**只有当前快照、没有历史版本**。Yahoo 的 `option_chain()` 的 `date` 参数是「到期日」不是「截至某日」，
所以今天的期权链今天不落盘，就永久消失、事后无法回补。

因此这个数据层的核心不是「下载历史」（历史大多能回补），而是**从今天开始把每天的期权链钉在硬盘上**。

---

## 目录

```
usagent/
  data/
    usagent.db                 # SQLite，全部数据
    universe_nasdaq_all.json   # Nasdaq 全量普通股 3,030 只（含市值/价格）
    universe_oi100.csv         # 期权 OI>=100 的 596 只  ← 默认工作池
    universe_tight.csv         # 再叠加价差<=3tick 且 <=15% 的 49 只
  src/
    config.py           # 全部可调参数集中在这里
    db.py               # schema + 幂等写入
    yfutil.py           # Yahoo 节流 + 指数退避
    download_ohlcv.py   # 日线 OHLCV（3 年 + 增量两种模式）
    snapshot_options.py # 每日期权链快照（含飞行前检查）
    derive_iv.py        # 从 mid 反解 IV，算 IV30 / HV20 / IV Rank
    series.py           # 读取层：日线 → 周线，复权检测与计算
    validate.py         # 数据校验（可挂告警）
    run_daily.sh        # 每日任务入口
  docs/                 # 数据源选型报告
```

---

## 环境

```bash
# 解释器**不写死某一台机器的路径**（审计 G2：文档曾把 /opt/homebrew/... 当环境契约，
# 而那台机器上的路径在本机早就失效了）。依赖版本钉在 requirements.txt。
python3 -m pip install -r requirements.txt
PY=${PY:-python3}
$PY -c "import yfinance, pandas; print(yfinance.__version__, pandas.__version__)"
# 期望：1.7.0 3.0.5（版本必须与 requirements.txt 一致 —— §0 的"同输入⇒同输出"
# 在没有依赖锁定时做不到，而数字变了会被误读成策略的问题）
```

落盘用 SQLite（Python 内置），**不用 parquet**（不引额外依赖）。

---

## 首次运行

```bash
cd <仓库根>

# 1) 3 年日线全量回补（596 只约 35 分钟；每只 ~3-5s，这是 Yahoo 的固有速度）
$PY -u src/download_ohlcv.py --universe oi100

# 2) 抓一次当日期权链快照（596 只约 110 分钟）
$PY -u src/snapshot_options.py --universe oi100

# 3) 算 IV / IV Rank
$PY -u src/derive_iv.py

# 4) 校验
$PY -u src/validate.py
```

**中途被限流或断网没关系** —— 全部写入都是幂等的（`INSERT OR REPLACE`），
重跑会自动跳过已覆盖的标的（`coverage` 表记录覆盖区间）。

---

## 每日运行

```bash
bash src/run_daily.sh
```

挂 cron（**先确认时点，见下**）：

```bash
crontab -e
# 北京 10:00，周一至周五（机器时区须为北京）
0 10 * * 1-5  cd <仓库根> && bash src/run_daily.sh >> logs/daily.log 2>&1
```

### ⚠️ 运行时刻是这套东西唯一真正会「做错」的地方

**Yahoo 在美东深夜会把期权报价整体清零。** 实测（2026-10-14，本机）：

| 美东时刻 | 北京时刻 | NVDA/AAPL 平值档 |
|---|---|---|
| 23:48 | 11:48 | bid 10.40 / ask 10.55 ✅ |
| 00:51 | 12:51 | **bid 0.00 / ask 0.00**，IV = 0.0000 ❌ |
| 01:10 | 13:10 | 仍然全 0 ❌ |

在这个窗口抓，会得到两万行 bid=0 的废数据 —— **表面"抓成功"，实际反解不出任何 IV**。
本项目的第一次快照就踩了这个坑，**整个 2026-10-13 的期权链报废，且无法回补**。

**有效窗口 = 美股收盘（16:15 ET）→ 约 23:45 ET：**

| | 夏令时 | 冬令时 |
|---|---|---|
| 北京时刻 | 04:15 – 11:45 | 05:15 – 12:45 |
| 交集（全年可用） | **05:15 – 11:45** | |

所以 cron 排 **北京 10:00** —— 两个 DST 制式下都在窗口内，而且是白天，不用通宵开机。

`snapshot_options.py` 里有**飞行前检查**：用 AAPL 平值档做探针，发现报价清零就拒绝写入并返回 3。
所以就算 cron 时刻设错，也是响亮报错，而不是静默往库里落垃圾。`validate.py` 里也有一道对应的断言。


---

## 数据口径（这些设定是刻意的，改之前先读理由）

| 设定 | 取值 | 为什么 |
|---|---|---|
| **粒度** | **存日线；周线由 `series.py` 合成** | 单一数据源，避免两套数据打架。周线用 `W-FRI` 聚合 |
| **拆股复权** | **不需要做 —— 数据源已经做了** | 见下面单独一节，这是本项目踩过的最贵的一个坑 |
| **分红复权** | 默认不做；`adjust="total"` 时读取现算 | 默认口径 = 拆股已复权、分红未复权，适合趋势/形态；总收益口径用于含股息的收益分析 |
| IV | **从 mid 反解 Black-Scholes**，不用厂商的 IV 字段 | 实测 yfinance 的 put IV 在 ITM/ATM 只有 18-24% 落在自算值 ±5% 内（ask 反解系统性 +3.4 vp） |
| r / q | **固定 4.0% / 0.0%**，全程不变 | 逐日变的 r/q 会往 IV 序列注入台阶（±50bp ≈ ±0.3-0.5 vp）。IV Rank 是相对度量，固定值的常数偏移会对消 |
| 30 天 IV | 在**方差域**插值（σ²T 对 T 线性） | 直接对 IV 线性插值是错的；期权到期固定在每月第三个周五，不插值会被到期日轮换污染 |
| 期权链筛选 | **最接近现货的 N 档（N=20）**，外加 ±30% 上限 | 用「现货 ±X%」当唯一条件会在低价股上变成空集：ABEO 现货 $4.52、行权价间距 $2.50，最近一档 $5.00 是 +10.6%，±10% 带内一档都没有（实测 100 只全空） |
| 到期日优先 | **标准月度（第三个周五）** | 实测同一 DTE 下，MSFT 假期周 9.83% vs 标准月度 3.33%；OI 21 vs 3,233 |

### ⚠️ 关于拆股复权：数据源已经做过了，不要再做一次

这是本项目踩过的最贵的一个坑，值得单独说：

```
yfinance auto_adjust=False, NVDA:
  2024-06-07   Close 120.888
  2024-06-10   Close 121.790   ← 这天标了 10:1 拆股，但价格和前一天同级
  2024-06-11   Close 120.910
```

**Yahoo 返回的 `Close` 本身就已经是拆股复权后的价位**（真实未复权价应在 $1,200 量级）。
`Adj Close` 与 `Close` 的比值在整个窗口恒定 0.9972，差额只来自分红 —— 也就是说
**`Close` = 拆股已复权 + 分红未复权**，`Adj Close` = 两者都复权。

如果读取层再乘一次 1/split，就会在拆股次日造出一根 **+892% 的假巨阳**。这个 bug 我写出来又修了两次才对。
`series.source_is_split_adjusted()` 现在会**检测**这件事（对 56 次 3:1 以上大拆股算「实际跳幅 / 隐含跳幅」的吻合度，
中位 0.01 → 已复权）。**换数据源（marketdata.app / EODHD）时必须重跑这个检测**，因为那些源可能返回真正未复权的价格。

### 周线怎么用

```python
import sys; sys.path.insert(0, "src")
from series import weekly, weekly_complete, panel_weekly

w  = weekly("NVDA")                    # 周线，W-FRI
wc = weekly_complete("NVDA")           # 丢掉未走完的当周（做突破判断用这个）
wt = weekly("NVDA", adjust="total")    # 再叠加分红 → 总收益口径
df = panel_weekly(["NVDA","AAPL"])     # 多标的长表，做横截面/相对强度

# 自检（含复权状态检测 + 拆股日逐日对照）
#   python src/series.py NVDA WMT AAPL
```

实测：751 根日线 → 157 根周线（压缩比 4.78），OHLC 自洽 0 违规。


---

## 表结构

| 表 | 主键 | 说明 |
|---|---|---|
| `ohlcv` | (symbol, date) | 未复权 OHLCV + dividends + splits |
| `option_chain` | (snapshot_date, symbol, expiry, strike, right) | 快照日 = 数据所反映的**美东交易日**，不是本地日期 |
| `iv_history` | (date, symbol) | iv30 / hv20 / vrp / iv_rank / iv_percentile |
| `fetch_log` | — | 每次抓取的结果，用于排查 |
| `coverage` | (kind, symbol) | 覆盖区间，用于断点续跑 |

常用查询：

```sql
-- 某标的的 IV 位置
SELECT date, ROUND(iv30*100,1) iv30_pct, ROUND(iv_rank,1) rank, ROUND(iv_percentile,1) pctile
FROM iv_history WHERE symbol='NVDA' ORDER BY date DESC LIMIT 20;

-- 快照里某个到期的价差与流动性
SELECT strike, right, bid, ask, ROUND((ask-bid)/((ask+bid)/2)*100,2) sp_pct,
       open_interest, volume, ROUND(implied_volatility*100,1) iv_pct
FROM option_chain
WHERE symbol='NVDA' AND snapshot_date=(SELECT MAX(snapshot_date) FROM option_chain)
  AND expiry='2026-11-20' AND ABS(strike-230)<10
ORDER BY strike, right;
```

---

## 已知限制（别指望它做到这些）

1. **历史 IV 无法回补。** 只从你开始每天跑的那天起积累。IV Rank 需要 252 个交易日才有意义，
   所以前一年里 `iv_rank` / `iv_percentile` 会是 NULL（`derive_iv.py` 里 `LOOKBACK=252` 的控制）。
   → 想立刻有 IV Rank，见下面「想要 3 年历史期权数据」。

2. **财报未过滤。** 30 天窗口里夹着财报时 IV 会结构性抬高，一年里约 1/4 的时间会给出错误的
   「IV 高位」信号。`derive_iv.py` 目前不剔除，需要时再加 earnings 表。

3. **只保留 ±10% 行权价、3 个到期。** 做多腿、跨式、日历价差需要更宽的行权价带 —— 改
   `config.CHAIN_STRIKE_BAND` / `CHAIN_MAX_EXPIRIES` 后重跑（但**历史部分补不回来**，只能从现在开始）。

4. **欧式 BS 反解美式期权有偏差。** ATM 近月影响小，深实值 put 影响大。

5. **Yahoo 不是生产级数据源。** 无 SLA、ToS 仅限个人自用、限流机制不公开。
   本层已做幂等 + 退避 + 超时 + 断点续跑，但**不要把它接到无人值守的自动交易上**。

6. **997 只微盘股未覆盖。** 全量扫描时被 Yahoo 限流（失败 100% 集中在市值 <$100M 那一层）。
   对 `oi100` 工作池（596 只）无影响 —— 失败率为 0。

---

## 想要 3 年历史期权数据 / 立刻有 IV Rank

免费层只有 1 年，且**所有厂商任何档位都不提供历史 IV**（marketdata.app 官方原文：
"Greeks and implied volatility are not stored historically ... on every plan"）。
所以历史 IV 必须自己从 mid 反解 —— 这正是 `derive_iv.py` 在本地做的事，只是它只能从今天往后攒。

想一步到位拿到过去的期权链（用来回补 IV Rank）：

| 方案 | 价格 | 数据 | 说明 |
|---|---|---|---|
| marketdata.app Starter | **$12/月（年付）** | 5 年历史期权链（bid/ask/OI/volume，历史 IV 为 null） | 正好覆盖你要的 3 年；免费档只有 1 年不够 |
| ThetaData 免费档 | $0 | EOD 自 2023-06-01（≈3.3 年） | 刚好够，但需本地跑 Theta Terminal（Java） |

回补脚本还没写 —— **先把每天的抓取挂上，这个的优先级更高**，因为回补随时能做，而每天的链错过就没了。

---

## 与 docs/ 报告的关系

`docs/2026-10-14-美股期权数据源选型.md` 是选型依据与证据链；
本目录是它的可执行落地。报告里所有「口径」类结论都体现在 `config.py` 的注释里。
