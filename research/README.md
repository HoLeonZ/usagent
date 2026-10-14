# research —— uptrend_pullback 美股判定

把 hiagent `uptrend_pullback` 的思路搬到 usagent 的 3 年美股数据上,回答一个问题:
**这个 preset 还剩多少,以及那些收益是不是 beta?**

> 本文按 hiagent CLAUDE.md §7 撰写:**只写口径与复现方法,不含历史指标。**
> 带数字的结果在 `results/BASELINE.md`(运行记录,故意含数字)。

设计文档:`docs/superpowers/specs/2026-10-14-uptrend-pullback-us-baseline-design.md`

**本目录不修改 `src/`。** 数据层不被交易逻辑污染(见 2026-10-14 审计)。

---

## 复现

```bash
cd /Users/holeon/code/usagent
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

# 单元测试(合成 fixture;不读真库、不联网)
$PY -m pytest research/tests -q

# 基线(默认口径,全可评估窗口)
$PY -m research.run_baseline --preset us_uptrend_pullback_v1

# 指定区间 + 逐窗口表 + 落 JSON
$PY -m research.run_baseline --preset us_uptrend_pullback_v1 \
    --start 2024-04-09 --end 2025-10-01 --windows --out research/results/iter.json

# 成本敏感性
$PY -m research.run_baseline --preset us_uptrend_pullback_v1 --cost-bps 20
```

---

## 口径(改之前先读理由)

| 项 | 取值 | 为什么 |
|---|---|---|
| 价格 | `series.daily(adjust="total")` | 多头回测必须用复权价:除息日裸价跳空下跌会被误判为止损(hiagent `data.py` 原话)。**复用 `src/series.py`,不重写复权数学** |
| `amount` | `volume × close` | `ohlcv` 无成交额列。代理量,支撑 hiagent 的 E 条件 |
| 价格完整历史一次载入 | 非逐窗口载入 | 逐窗口时每个窗的 MA120 只拿到本窗+400 自然日;全历史载入预热更充分。**是等价变换**:全体算完再切片 == 带足预热的逐窗算(由 `test_causality` 钉住) |
| 可评估窗口 | `2024-04-09` 起 | `mom120` 需 121 根 K 线,前 120 个交易日是预热。**预热期不能算进 CAGR 分母** |
| 成本 | 双边 5bp | 美股无印花税。IBKR Lite 佣金 $0;SEC §31 ≈0.278bp(卖出);FINRA TAF $0.000166/股上限 $8.30。另报 {0,5,20} 敏感性 |
| 纪律 | `n_comparisons = 13` | CLAUDE.md §5 Iterative-Tuning Accounting |

---

## 执行语义(可移植部分 + 明确删除的 A 股件)

**保留**:信号 T 收盘 → T+1 开盘入场;`intraday_tiebreak = "sl_first"`(同 bar 双触发按最坏情况);
gap 跳过止损按开盘价成交;成交量上限 `Bar_Volume × 10%`,超出 IOC 丢弃;
`max_positions = 1`;排序 `score = mom120` 降序、超容量当 T 丢弃不排队。

**删除**(理由见设计文档 §3):T+1 可卖性 / 涨跌停不可成交 / 100 股整手 / 印花税 /
三池保证金。

**偏离 hiagent §2 的一处**:`all-in` → `risk_budget`。§2 的 all-in 与「窄止损」耦合,
详见 `presets.py` 的注释与 `results/BASELINE.md` §1C。

---

## 模块

| 文件 | 职责 | 它守住的不变量 |
|---|---|---|
| `panel.py` | 读数 → 长表 | 全历史载入;预热边界诚实 |
| `indicators.py` | `compute_indicators` 移植 | **因果性**:截断到 T 重算,T 处逐位相同 |
| `regime.py` | 合成等权指数 + breadth 门 | 只用 ≤T 数据 |
| `signals.py` | A~F 六条件 | 参数区间 fail-fast;信号集为空要抛错不静默 |
| `broker.py` | 执行语义 | **权益 = cash + qty×close**(hiagent 曾把建仓名义额计入两次) |
| `metrics.py` | 窗口 / E[R] / DSR / beta 归因 | DSR 单位一致性(年化 vs 逐期) |
| `presets.py` | **交付物** | `n_comparisons` 与负面证据必须同时存在 |
| `run_baseline.py` | CLI | R 退化时拒绝报告 t 值 |

---

## 已知方法学限制(不要当优点读)

1. **没有独立期。** 库只有 3 年(可评估 2.51 年),不存在一段"从未参与迭代"的长期历史。
   最多只能 train/holdout 对分 —— 能防过拟合,防不了"整段窗口恰好是牛市"。
2. **幸存者偏差未消除。** 股票池是今日快照;绝对 CAGR 应读作上界。
3. **基准是日度再平衡等权指数**,非严格 buy-and-hold;它高估的是对照项。
4. **`t_expectancy` 在 R 退化时无判别力**(固定止损让每次止损 ≈ −1R)。`run_baseline.py`
   会检出并拒绝该行 t 值。

---

## 交付物

`presets.py::PRESETS["us_uptrend_pullback_v1"]` —— 但注意它 **`validated = False`**。
按 hiagent §5,负面证据与参数同时记录。结论见 `results/BASELINE.md`。
