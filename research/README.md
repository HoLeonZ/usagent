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

# 0) 取数。data/*.db 不入库 —— 不先跑这一步,下面每条命令都会
#    sqlite3.OperationalError: no such table: ohlcv。
#    --as-of 把窗口右端**钉住**:不给它 = 用今天,同一份代码在不同日期会下到
#    不同的数据,于是复现不出同一个数字(§0 Identical inputs ⇒ identical outputs)。
#    本文写定时(2026-10-15)的 vintage:rows=425332;symbols=596;from=2023-10-16;to=2026-10-13
$PY src/download_ohlcv.py --universe oi100 --as-of 2026-10-14

# 单元测试(合成 fixture;除 golden 外不读真库、不联网)
$PY -m pytest research/tests -q

# 基线(默认口径,全可评估窗口)
$PY -m research.run_baseline --preset us_uptrend_pullback_v1

# 指定区间 + 逐窗口表 + 落 JSON(先看一个不同窗口稳不稳)
$PY -m research.run_baseline --preset us_uptrend_pullback_v1 \
    --start 2024-10-01 --end 2025-10-01 --windows

# 交易列表 golden 基线(§6):锁住成交清单,漂移就红
$PY -m research.golden_baseline --show

# 成本敏感性
$PY -m research.run_baseline --preset us_uptrend_pullback_v1 --cost-bps 20
```

---

## 口径(改之前先读理由)

| 项 | 取值 | 为什么 |
|---|---|---|
| 价格 | **双价格体系**(§3(b)):`close` 等用分红回补价喂指标;`raw_*` 用 as-traded 价做止损/跳空/市值/现金 | 复权价只对**比例型**指标无害;止损止盈触发与现金记账必须用物理价格。审计实测:全部跑在复权价空间时,复权因子(含**未来**分红)使 VNOM 一笔被乐观 0.92pp |
| `amount` | `volume × raw_close` | 代理量。**必须用 as-traded 价**:`ohlcv` 无成交额列,而 E 条件拿它比**绝对美元阈值**,用复权价会让「T 日能不能进池」变成 T 之后分红的函数(实测 740 个就绪格判定被翻转) |
| 价格完整历史一次载入 | 非逐窗口载入 | 逐窗口时每个窗的 MA120 只拿到本窗+400 自然日;全历史载入预热更充分。**是等价变换**:全体算完再切片 == 带足预热的逐窗算(由 `test_causality` 钉住) |
| 可评估窗口 | 由**数据算出**(`panel.first_signal_date`) | `mom120` 需 121 根 K 线,前 120 个交易日是预热。**边界不能写死常量** —— 写死之后换一份数据就会把预热期算进 CAGR 分母,且无人报警 |
| 成本 | 双边 5bp,常数在 `core/costs.py` **唯一定义** | 美股无印花税。IBKR Lite 佣金 $0 + SEC §31 + FINRA TAF。CLI 的 `--cost-bps` 只做覆盖,默认读 preset(§2(b) 要求常数恰好定义一次) |
| 滑点 | `core/costs.py` 的 `ATR_SLIP_SCALE`,preset 显式声明 | §4(c) 要求滑点是 **f(ATR, 参与率)**:`slip = scale × atr_pct × sqrt(该单股数/本 bar 成交量)`。此前是**零滑点**,且公式里参与率被写成字面量 1.0 |
| 分红 / 拆股 | 分红**显式入账**到 Free_Cash,并计入该笔净盈亏;拆股不重复缩放 | §3(c):现金分红必须触发一笔物理现金存入。库里的价本身已是拆股后价位(`series.source_is_split_adjusted` 实测),再乘一次就是造 +892% 假巨阳 |
| 纪律 | `n_comparisons` 在 `presets.py`(单一事实来源) | CLAUDE.md §5 Iterative-Tuning Accounting。本文不复述数值 —— §7 要求文档只写怎么复现、不写曾观测到什么 |

---

## 执行语义(可移植部分 + 明确删除的 A 股件)

**保留**:信号 T 收盘 → T+1 开盘入场;`intraday_tiebreak = "sl_first"`(同 bar 双触发按最坏情况);
gap 跳过止损按开盘价成交;成交量上限 `Bar_Volume × 10%`,超出 IOC 丢弃、现金立即回 Free_Cash;
排序 `score = mom120` 降序、超容量当 T 丢弃不排队;`max_positions` 由 preset 显式声明。

**结算隔离(§2(d),2026-10-15 修复)**:**当日入场的预算与容量取自「上一交易日收盘」的
账本快照** —— 同日卖出所得、同日出场释放的仓位槽,当日一律不可用。主循环是「先平仓
(盘中价)再开仓(开盘价)」,开盘在时间上早于盘中;不隔离就等于用几小时后才产生的钱付
几小时前的账。审计实测旧实现下 12m preset 有 3/13 笔靠这笔钱成交,隔离后它的 CAGR 从
交付的 +44.5% 掉到 +9.51%(低于它自称达到的 30% 目标)。

**删除**(理由见设计文档 §3):涨跌停不可成交 / 100 股整手 / 印花税。
⚠️ 此前把「T+1 可卖性」与「三池保证金」一并列为删除项 —— 那一并删掉的正是**现金隔离**,
是上面那条修复要补回来的东西。美股语境下要保留的是「未结算资金不得当日复用」,
不是 A 股 T+1 的可卖性约束。

**偏离 hiagent §2 的一处**:`all-in` → `risk_budget`(preset 里 `risk_per_trade` 在场即生效)。
§2 的 all-in 与「窄止损」耦合,详见 `presets.py` 的注释与 `results/BASELINE.md` §1C。
`position_fraction` 现在是**必填契约字段**(它是关掉风险预算后的兜底预算)。

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
| `golden_baseline.py` | 交易列表/权益曲线的 sha256 基线 | **§6 的行为锁**:任何改动只要改了成交清单就会红 |
| `run_baseline.py` | CLI | R 退化时拒绝报告 t 值;结果 JSON 带数据 vintage |
| `core/costs.py` | 费用/流动性/滑点常数 | **§2(b):全仓唯一定义** |
| `core/preset_contract.py` | 「引擎读哪些键」的声明 | §2(a):preset 缺键/未知键 ⇒ fail-fast;所有入口共用 `resolve()` |

---

## 筛选器 `screen.py`

把 596 只收窄成人能过目的候选。**但它同时在测量自己。**

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

# 1) 先看各层条件到底有没有携带信息(先做这个)
$PY -m research.screen --measure

# 2) 出候选清单(输出末尾**永远**附带该层实测基准率)
$PY -m research.screen --tier trend_pullback --top 20
$PY -m research.screen --measure --horizon 5     # 换持有期
```

### 门槛按族规模折减(§5(c))

判词门槛是 **Bonferroni 校正**后的:一次 `--measure` 要对 4 个层级各下一次判词,
门槛因此从 1.96 抬到 2.498(默认族规模 = 被检验的层级数 4)。换 `--horizon` / 换窗口
再跑一次,族就更大 —— 用 `--family-size` 把它累加进去(例如 4 层 × 3 个 horizon → 12,
门槛 2.865)。输出里会同时印出当前族规模与门槛。

### 这个工具不做什么

**它不预测方向。** 判据是「相对全池基准、并已按重叠采样折减的 t 值」。截至 2026-10-14,
四个层级都落在「与随机选票无统计差别」区间内 —— **具体数值请自己跑 `--measure`**,
本文不复述(§7:文档写怎么复现,不写曾经观测到什么)。所以:
清单的价值在于把 596 只收窄到几十只供你过目,**方向判断得由你补**。

### 两条口径陷阱(输出里也写了)

1. **绝不看"均值是否 > 0"。** 样本覆盖一段上涨市,全池基准的绝对均值必然为正 ——
   任何纯多头筛选的均值都会是正的。判据只有"vs 基准",而基准本身随窗口变,
   所以它同样不该被写死进文档。
2. **重叠采样必须折减。** h 日持有期逐日采样,独立观测数约为 n/h,均值的
   标准误被低估约 sqrt(h) 倍。不折减的 t 会虚高(报告 §3.5 有这张膨胀系数表)。
   `screen.py` 同时给出 `t_raw` 与按 sqrt(h) 折减后的 `t_adj`,**判读只看后者**。

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
