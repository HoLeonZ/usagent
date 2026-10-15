# uptrend_pullback → 美股 1 preset:基线设计与纪律

> 状态:待实现(设计已确认)
> 目标产物:**1 个 preset**(dict,与 `hiagent/uptrend_pullback/presets.py` 同构)+ 其基线回测结果。
> 本文按 hiagent CLAUDE.md §7 撰写:只写口径与复现方法。历史指标入 `research/results/BASELINE.md`,不入本文。
>
> ⚠️ **这是一份定稿于 2026-10-14 的规格文档,不要当现状读。** 2026-10-15 的审计修复改动了
> 其中若干执行语义假设(§2(d) 结算隔离、§3(b) 双价格体系、§4(c) 滑点、§3(c) 分红入账),
> 现状以 `research/README.md` 的「执行语义 / 口径」两节为准。文中描述 **hiagent** 的数字
> 是外部引用,本仓库不复核;描述本仓库运行结果的数字一律不入本文(§7)。

---

## 0. 这次要回答的唯一问题

hiagent 的 `v33_long_reverse_v21`(未经任何改动)**原样**搬到 usagent 的 3 年美股数据上,还剩多少?
以及那些收益有多少是 **beta / 幸存者偏差**而不是 alpha。

**为什么这是真正的样本外**:v21 的参数是在 A 股 2020–2026 上调出来的,usagent 的美股 3 年从未
参与过它的任何一次迭代。所以基线不需要"训练/留出"妥协 —— 整段 3 年对它都是 OOS。
这是先做基线而不是先做搜索的全部理由:基线是**证伪装置**。若基线已等于等权指数的 1.x 倍 beta,
则任何后续择优都只是在拟合同一件东西。

**明确不做**:参数搜索(阶段 B,待基线结论后再定)、交易系统、期权/IV、实盘、基准数据采购。

---

## 1. 数据适配(3 个决定,含代价)

### 1.1 价格口径
`series.daily(symbol, adjust="total")` —— 拆股已由数据源复权,这里再叠分红 → 总收益口径。

理由(引自 `hiagent/uptrend_pullback/data.py` 模块 docstring):「做多策略必须用前复权价:
除权除息日 v_daily 的裸价会凭空跳空下跌,在多头回测里会被误判为止损。」

**复用 `src/series.py`,不重写读取层。**

### 1.2 `amount` 代理
`ohlcv` 无 `amount`(成交额)列,只有 `volume`。代理:`amount = volume × close`。

hiagent 的 E 条件 `amount60 ∈ [3e7, 3e8]` 是 A 股主板口径(上界防小盘/涨停票)。
对 596 只 Nasdaq OI≥100 的票,该上界几乎全被击穿。

**决定**:「照搬 [3e7, 3e8]」与「移除上界」两个候选**都跑**,显式记 `n_trials`。
删上界是**移除**不是调参,但不假装它免费 —— 让结果回答"这条 A 股筛选能不能搬"。

### 1.3 成本模型
美股无印花税。默认 `cost_bps_roundtrip = 5.0`(双边各 2.5bp)。
另报 `{0, 5, 20}` 三档敏感性。

参考量级(不用于替代回测,仅用于说明 5bp 不是随手取的):IBKR Lite 佣金 $0;
SEC §31 ≈ 0.278bp(卖出侧);FINRA TAF $0.000166/股、上限 $8.30。

**不假装零成本,也不假装精确。**

---

## 2. 目录与模块

位置:`/Users/holeon/code/usagent/research/`。**不动 `src/`**(审计结论:数据层不被交易逻辑污染)。

```
research/
  __init__.py
  panel.py         # series.daily(adjust="total") → 长表 panel(含预热)
  indicators.py    # compute_indicators 移植(MA/MACD/连阴/回调/ATR/流动性)
  regime.py        # 合成等权指数 + breadth 择时门
  signals.py       # A~F 六组条件
  broker.py        # 执行语义(portfolio.py 的美股子集)
  metrics.py       # 窗口生成 / CAGR / E[R] / DSR / beta 归因
  presets.py       # **交付物**:US preset dict(s)
  run_baseline.py  # CLI 入口
  README.md        # 口径 + 复现命令(不含历史指标)
  results/         # 结果落盘(BASELINE.md + JSON);将来 git init 时必须 gitignore
  tests/           # 合成 fixture;不读真库、不联网
```

每个模块一个职责,且都有一个**必须独立验证的不变量**(见 §6)。

### 接口约定

```python
# panel.py
def load_panel(start: str, end: str, universe: tuple[str, ...] | None = None,
               warmup_days: int = 400) -> pd.DataFrame
    # columns = [symbol, date, open, high, low, close, volume, amount]
    # 按 (symbol, date) 排序;amount = volume × close

# indicators.py
def compute_indicators(panel: pd.DataFrame) -> pd.DataFrame
    # 追加 ma5/10/20/60/120, above_ma60_ratio, ret1, prev_close,
    # down_streak/prev_down_streak, high20/pullback, mom120, amount60,
    # vol_ma20/vol_ratio, tr/atr14/atr_pct, macd_dif/dea/bar/bar_prev
    # 全部 rolling/shift 按 symbol 分组

# regime.py
def compute_regime(panel_ind: pd.DataFrame, *, ma_window=20, breadth_window=20,
                   min_breadth=0.45, require_rising=False, rising_lookback=5
                   ) -> pd.DataFrame      # [date, ew_index, ew_ma, breadth_sm, regime_ok]

# signals.py
def select_entries(panel_ind, *, start_date, end_date, regime_df=None, **params) -> pd.DataFrame
    # columns = ENTRY_COLS(见 §4),按 (date, score desc) 排序

# broker.py
def simulate(preset: dict, panel_ind: pd.DataFrame, regime_df, *, start, end,
             cost_bps_roundtrip: float = 5.0) -> dict   # {"trades": df, "metrics": dict, "equity": df}

# metrics.py
def monthly_windows(start_month, end_month, window_months=2, step_months=1) -> list[tuple[str,str]]
def tail_stats(trades: pd.DataFrame, sl_pct: float) -> dict
def deflated_sharpe_ratio(...) -> dict      # 移植 dna_stats/deflated.py(含单位一致性修复)
def beta_attribution(equity: pd.Series, benchmark: pd.Series) -> dict
```

---

## 3. 执行语义(移植 + 明确删除的 A 股件)

| 保留 | 出处 |
|---|---|
| 信号 T 收盘 → **T+1 开盘**入场 | `signals.py` docstring |
| `intraday_tiebreak = "sl_first"`(同 bar 双触发按最坏情况) | CLAUDE.md §4 |
| gap 跳过止损 → **按开盘价**成交(悲观) | CLAUDE.md §4 Gap-Through-Stop |
| 成交量上限 `Bar_Volume × 0.10`,超出 IOC 丢弃、余量现金立即回 Free_Cash | CLAUDE.md §4 |
| `max_positions = 1`、`position_fraction = 1.0`(all-in) | CLAUDE.md §2 |
| 排序 `score = mom120` 降序;超容量**当 T 丢弃不排队** | CLAUDE.md §2 单发子弹 |

| 删除 | 理由 |
|---|---|
| T+1 可卖性 | 纯 EOD 回测无关(2024-05-28 改的是结算 T+1,不是可卖性) |
| 涨停/跌停不可成交 | 美股无日涨跌幅限制 |
| 100 股整手 | 美股 `lot_size = 1` |
| 印花税、三池保证金(Free/Locked/Settling) | A 股制度产物;美股长仓现金账户 |

**已证伪、不重测的 5 条**(引用 `hiagent/docs/uptrend_pullback_preset_search.md` §5):
ATR% 排序、回调深度排序、ATR 归一化 TP/SL、regime 加 `require_rising`、延迟入场(T+2)。

---

## 4. 信号条件(A~F,逐条移植)

| 组 | 条件 |
|---|---|
| A 趋势 | `close > ma60` ∧ `ma20 > ma60` ∧ `above_ma60_ratio ≥ min_above_ma60_ratio` ∧ `mom120 ≥ min_mom120` ∧(可选)`close ≥ ma60×(1+close_ma60_buffer)` |
| B 回调 | v33_long_mirror 形态:`down_streak ∈ [min_down_streak, max_down_streak]` ∧ `ret1 ∈ [pct_chg_low, pct_chg_high]` |
| C(并入 B) | 上述 `ret1` 区间即"当日跌幅可控" |
| D 未破位 | `abs(macd_bar) < abs(macd_bar_prev)` ∧ `macd_dif > 0` ∧ `macd_dea > 0` |
| E 流动性 | `amount60 ∈ [min_amount, max_amount]`(见 §1.2) |
| F 择时 | `regime_df.regime_ok`(可选;仅当 preset 有 `regime` 块) |

`ENTRY_COLS`(与 hiagent 同构):
`date, symbol, score, sig_close, sig_ma20, sig_ma60, sig_down_streak, sig_pullback, sig_mom120, sig_amount60, atr_pct`

v21 信号参数(基线原样值):
`min_down_streak=3, max_down_streak=10, pct_chg_low=-0.088, pct_chg_high=-0.02, min_amount=3e7, max_amount=3e8, min_above_ma60_ratio=0.6693, close_ma60_buffer=0.0121, min_mom120=0.18`

---

## 5. 窗口与评估口径

**窗口**:滚动 2 个月 / 步长 1 个月,覆盖 `2023-11 .. 2026-10` → 约 34 个窗口。
与 hiagent 记录口径(2m/1m)一致。预热 400 自然日由 `panel.load_panel` 内部处理,不占窗口。

**输出字段(每一项都有对应理由)**:

| 指标 | 为什么必须有 |
|---|---|
| **CAGR** | 用户要的头条数字,照报 |
| **E[R]**(R-multiple 期望) | 主判据。该族由右尾驱动(具体比例引 hiagent 记录,本仓库不复核:按 §7,不能由当前代码 + 一条命令再生的数字不入文档)→ 均值类指标惩罚创造边缘的那条尾巴 |
| **beta vs 合成等权指数 + 超额** | 判决"alpha 还是 beta" |
| **曝光匹配 beta** | all-in 单持仓只在少数日子在场,绝对 CAGR 被 exposure 稀释;不做曝光匹配的 beta 比较是假的 |
| **n_trades / 时在市比例** | 样本量诚实底线 |
| **DSR(n_trials 显式)+ Bonferroni** | §5 合规 |
| **null 对照:同期 596 只等权 buy-and-hold** | 没有这个,"跑赢"无意义 |
| win% / payoff / top-decile share / maxDD | 分布量是判据的一部分 |

**幸存者偏差处理**:用 596 只自身的等权指数当基准,偏差在**分子与分母里同时存在**,做差时
大部分对消。故 **alpha 估计比绝对 CAGR 可信;绝对 CAGR 应读作上界**。

**`n_comparisons` 记账(§5 Iterative-Tuning Accounting)**:
基线阶段 `n_trials = 2(流动性口径候选) × 3(成本档) = 6`。
本文件的写作、窗口起止的选择、以及任何后续改动都计入。

---

## 6. 错误处理与不变量测试

**fail-fast(§0)**:空 panel、空 universe、窗口内信号为空 → 显式抛错/显式记 0,**不静默跑出 0 笔交易**。

测试 ~20–25 个,合成 fixture,**不读真库、不联网**:

| 测试 | 钉住的不变量 |
|---|---|
| **因果性(最重要)** | panel 截断到 T 后重算指标,T 处数值逐位相同 —— 未来函数防线 |
| regime 因果性 | 只用 ≤T 数据;breadth 不跨股票泄漏 |
| 信号逐条 | 合成 panel 逐条打开/关闭 A~B~D~E |
| broker: SL-first | 同 bar 双触发按 SL |
| broker: gap | 开盘跳过止损 → 按开盘成交 |
| broker: 量能上限 | 超出 10% 的余量被丢弃且现金立即回 Free_Cash |
| broker: all-in | `floor(cash / (px×(1+slip)))`;现金不足不入场 |
| metrics: E[R] | `R_trade = sl_pct × entry_price × size`;不用 `atr_pct` 做分母(hiagent 踩过) |
| metrics: DSR 单位 | 年化 Sharpe 比逐期阈值 —— **hiagent 正是在此踩过**(年化 0.84/k=19 被判 0.000,正确约 0.44) |

---

## 7. 复现命令

```bash
cd /Users/holeon/code/usagent
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

# 基线(两个流动性候选 × 三档成本)
$PY -m research.run_baseline --preset us_v21_baseline --n-trials 6

# 单元测试(不读真库、不联网)
$PY -m pytest research/tests -q
```

---

## 8. 交付物清单

1. `research/presets.py` 中 **≥1 个 preset dict**(与 hiagent 同构,含 `n_comparisons` 与 TP/SL compliance header 注释)
2. `research/results/BASELINE.md` —— 含数字的结论(头条 CAGR + beta 归因 + 诚实结论)
3. `research/README.md` —— 口径与复现方法(不含历史指标)
4. 测试套通过
