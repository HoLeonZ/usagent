# preset / family 注册表重构 —— 设计与验收

> 状态：设计已确认（2026-10-17），待实现。
> 本文按仓库惯例只写**口径与验收判据**，不含任何"改完之后数字变成多少"（§7）。

---

## 0. 要解决的问题

新增一条 preset 今天要**改现有文件**，而不是加文件：

- preset 定义在 `research/presets.py` 的 `PRESETS` dict 里（4 条，其中 3 条用
  `PRESETS["名字"] = {...}` 追加）；
- **族型** preset（`family: "str"` 这类）还额外要改 `research/families.py` 的 4 处：
  `MASK`（mask 函数）、`FIXED`（固定轴）、`GRIDS`（预注册网格）、
  以及视情况的 `SCORE_COL` / `SCORE_NEGATED` / `NEEDS_REGIME`；
- 漏掉 `GRIDS` 会在 `grid_size()` 上直接 `KeyError`；漏掉 `SPEC_LABELS`
  （新增**引擎键**时）会被 E5 那条护栏拦住；而漏掉 `golden_baseline.CASES`
  **什么都不会发生**——新 preset 就是不被行为锁覆盖。

后果不是"麻烦"，而是**新东西默认不在护栏里**：`CASES` 是手写清单（4 条），
`test_presets.py` 的纪律断言只测默认 preset。这与本仓库反复吃亏的另一类缺陷同源：
**声明了却没人读 / 有两条路径**。

## 1. 目标

1. **新增一条 preset = 新增一个文件**（族型也一样），不改任何现有文件；
2. **每个 preset 要么在行为锁里、要么在显式豁免表里**——由测试强制；
3. 现有 20 多处调用点**一行不改**（这是硬约束，见 §4）。

**非目标**（明确不做）：不引入插件发现（entry points）、不做热重载、
不把 `signal`/`universe` 等字段改成新语义、不动任何策略参数。

## 2. 目录与命名

**不能**用 `research/presets/` 当包名：它与 `presets.py` 撞名，且 `research.presets`
会被包遮蔽，`from research import presets` 的语义随之改变。取：

```
research/preset_defs/<name>.py     # 一条 preset 一个文件
research/family_defs/<id>.py       # 一个族一个文件
```

扫描用 `pkgutil.iter_modules`（按包枚举，不手写文件名清单）。

## 3. 单文件契约

### 3.1 preset 文件

```python
NAME = "us_reversal_dip_v1"   # 必须与文件名一致
PRESET: dict = {...}          # 就是今天的 preset dict，**不引入新键**
VERDICT = """…"""             # 负面证据（§5：必须与参数同时记录）
LOCK_WINDOW = ("2024-04-10", "2026-10-17")   # 行为锁用的**绝对**窗口（必填）
LOCK_IS_CLI_DEFAULT = False   # 只有一条 preset 可以为 True（见 §5）
```

`LOCK_WINDOW` **必填**（自审时补的）：行为锁的坐标纪律是"写死绝对值，不用会随数据
漂移的默认解析"。若允许"没写就从数据派生"，新 preset 的锁就会**静默换成另一条曲线**
——而锁的全部意义就在于它不会自己动。所以窗口跟着 preset 走（一并满足"一个文件"
与"坐标不自漂"两条纪律）。

`LOCK_IS_CLI_DEFAULT` 最多一条为 True：CLI 的默认窗口是**由数据算出**的
（`panel.first_signal_date`），既有测试断言"标注为 CLI 默认的那条 preset 的窗口
必须等于算出来的窗口"。派生 CASES 之后这条测试的语义必须保持。

装载规则（全部 **fail-fast**，不静默跳过）：

| 违规 | 处置 |
|---|---|
| 文件名 ≠ `NAME` | 抛错（否则"名字与文件对不上"这件事没人发现） |
| 缺 `PRESET` 或 `VERDICT` | 抛错（§5 要求负面证据不能省） |
| 名字重复 | 抛错 |
| `PRESET` 里 `verdict`/`notes` 字段与模块级 `VERDICT` 冲突 | 抛错（一份事实只能有一处） |

### 3.2 族文件

```python
ID = "str"
def mask(ind, p): ...          # 原 `_mask_*`，签名与语义不变
FIXED = {...}                  # 原 FIXED[id]
GRID = {...}                   # 原 GRIDS[id]（预注册网格）
SCORE_COL = None               # 原 SCORE_COL.get(id)
SCORE_NEGATED = False          # 是否属于 SCORE_NEGATED
NEEDS_REGIME = False
CONTROL = "str_inv"            # 对照臂声明（可选）
DOC = """机制叙述"""            # 原 mask 的 docstring
```

## 4. 兼容门面（硬约束）

`presets.py` / `families.py` 退化为**注册表 + 门面**，继续导出下列名字——
实测这些是外部（`research/` 其它模块与 `tests/`）真实引用到的：

* `presets`：`PRESETS`、`get_preset`、`DEFAULT_RISK_PER_TRADE`、`V21_*`、
  `FAMILY_SEARCH_NOTE`、`WFV_SEARCH_NOTE`、`RISK_LIMITS`
* `families`：`MASK`、`GRIDS`、`FIXED`、`SCORE_COL`、`NEEDS_REGIME`、
  `NEW_FAMILIES`、`REFERENCE_FAMILIES`、`CONTROL_ARM`、`GRID_CAP`、
  `grid_size`、`select`、`add_family_indicators`、`above_price_floor`

**族文件的符号名很关键**：`tests/test_families.py` 引用了 `F._mask_sf_stress` /
`F._mask_sf_calm`（F3 那两条护栏）。**已定**：门面按原符号名把 `mask` **再导出一次**
（`_mask_sf_stress = ...`），**不改那两条测试**——它们守的是"两臂不止差状态"这个
结论，与本次重构无关，动它们只会把两件事混在一起。

## 5. 行为锁的覆盖判据

`CASES` 从注册表**派生**：默认每个 preset 一条，窗口沿用既有纪律（绝对坐标）。

```python
LOCK_EXEMPT: dict[str, str] = {  # name -> 为什么不锁
    # 例：只在特定 vintage 下才有意义 / 已被取代
}
```

测试断言：

1. 每个 preset ⇒ 要么在派生的 `CASES` 里，要么在 `LOCK_EXEMPT` 里；
2. `LOCK_EXEMPT` 的每一项都必须有非空理由；
3. 豁免表**不许悄悄变长**（断言其内容等于一份显式清单——沿用仓库既有做法：
   豁免名单一旦能无声扩大，护栏就等于没有）。

## 6. 验收判据（重构唯一的硬证据）

| # | 判据 | 怎么验 |
|---|---|---|
| 1 | **行为完全未变** | 7 个 `trades_sha256` 与 7 个 `equity_sha256` **逐字节不变**；离线锁 5 条不变 |
| 2 | 全量测试绿 | `pytest research/tests src/tests` 全绿；只允许改**引用了被移动符号名**的测试 |
| 3 | 对外名字仍在 | 现有测试即为间接证明（`test_contract`/`test_pool_is_load_bearing`/`test_spec_disclosure` 都遍历 `presets.PRESETS`） |
| 4 | **实操验收** | 新增一条 preset / 一个族，**只碰一个新文件**即可跑通（含被行为锁覆盖） |

⚠️ 判据 1 需要**重生成前的对照**：先用当前代码把 7 个哈希记下来，重构后再比。
若哈希变了 ⇒ 说明重构改了行为，必须查根因，**不许直接重生成基线**
（本仓库已吃过一次：两条入口读两个面板，只改数字会把分歧固化成基线）。

## 7. 分三段落地（每段独立可验证、可提交）

1. **preset 注册表**：`preset_defs/` + 门面 + 迁移现有 4 条 + 装载校验测试；
2. **family 注册表**：`family_defs/` + 门面 + 迁移现有 14 个族 + 网格/对照臂一致性沿用现有测试；
3. **锁覆盖**：`CASES` 派生 + `LOCK_EXEMPT` + §5 的三条断言 + 实操验收（判据 4）。

每段结束后跑一次 §6 的判据 1–2；三段都在同一分支上，逐段提交并写明该段动了什么。

## 8. 已知边界（不夸大）

* 这个重构**只消除"改现有文件"**这件事，不消除"必须想清楚"这件事：新 preset 仍要
  显式声明 `n_comparisons`、`validated`、`verdict`；族仍要声明 `GRID`（预注册纪律不变）。
* 它**不**自动保证新 preset 的**统计有效性**——那要靠 §5 的判据与既有的
  `n_trials` 记账，重构不改变这一点。
* `research/tests/test_presets.py` 的纪律断言仍只测默认 preset（不动它，
  避免把"默认 preset"这个概念搅浑）。
