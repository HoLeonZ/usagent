"""US preset —— uptrend_pullback 思路的美股版。

⚠️ 这个 preset 的 `validated = False`。它是**如实交付的产物,不是可交易的策略**。
   按 hiagent CLAUDE.md §5 与 preset_search 的纪律,负面证据必须与参数同时记录
   (他们自己的 v33_long_reverse_v21 就是这么写的:"必须同时记录的负面证据,不得
   只报好的一面")。这里照做。

⚠️ **下面 A/B/C 三组数字测于 2026-10-15 修复之前的引擎**(那次修复改了 §2(d) 结算隔离、
§3(b) 双价格体系、§4(c) 滑点、amount 口径 —— 都会改变交易列表)。结论方向(该信号族在这个
池子上没有可证实的正 alpha)在修复后依然成立,但**具体数值已失效**;当前口径下本 preset 的
数字见它自己的 `verdict`,12m 那个见 `preset_defs/us_uptrend_pullback_12m_target.py` 的 `NOTES`。两者都由一条命令可再生。

原基线结论(完整数据见 results/BASELINE.md,复现命令见 README.md):

  A) v21 参数**原样**搬过来:holdout 总收益 −86.04%,同期等权基准 +20.09%,
     超额 −106 个百分点。止损 2.1% 在信号日 atr_pct 中位 6.79% 的池子里只有
     0.31 ATR —— 落在噪声之内,94% 的仓位被止损。

  B) 止损改为 ATR 归一化(k ATR)、TP 保持 hiagent 的 12.5:1 几何:迭代期
     E[R] 转正(+0.15 ~ +0.20),但 holdout 塌到 **−0.59 ~ −0.99**。
     且 12.5:1 的几何在 ATR 空间里等于 TP = +85%,18 天持有期内几乎不可达 ——
     holdout 22 笔里 **21 笔止损、0 笔 TP**,E[R] ≈ −1 是设计的必然后果。

  C) 仓位从 all-in 改为风险预算式(本 preset 采用):消除了「连亏 20 次 −75%」
     的毁灭性路径,但**没有创造边缘**。E[R] 与胜率对 risk_per_trade 完全不变
     (它是纯敞口旋钮,不是边缘旋钮),holdout 仍是 E[R] −0.59、超额 −24pp。

  三组数字同向:**这个信号族在美股这个池子上没有可证实的正 alpha。**
  迭代期 E[R] 为正、holdout 为负,是「无边缘 + 样本极小(n=15~22)」的典型形态。

hiagent 自己在 docs/uptrend_pullback_preset_search.md §6 已经写下这个结论,
这里是在**美股数据上复现了它**:
  > 该 preset 家族在非趋势市里期望为负;在趋势市里的正收益与市场 beta 高度相关。
  > 若要取得可证实的 alpha,应**更换信号源**(引入基本面/事件维度),而不是继续在
  > 价格派生的 A~E 条件上做参数或排序层面的改动。

本 preset 的价值不在收益,而在:**它是一台经过验证的证伪装置**。任何人再提出
「把 v21 搬到美股试试」,现在有一条命令和一组数字可以回答。

**本模块现在是「注册表 + 兼容门面」**（2026-10-17 注册表重构）：preset 的定义体按
「一条一个文件」放在 `research/preset_defs/`，本模块只扫描它们、按装载契约校验
（见 `research/preset_registry.py`），再把既有名字原样导出 —— 所以外部 20 多处调用点
一行未改。**新增一条 preset = 新增一个文件，不需要改这个文件。**
"""
from __future__ import annotations

import copy
from pathlib import Path

from core.preset_contract import validate
# 共享常量只为**转发**：单一来源分别在 core（costs）、research.signals（V21 信号参数）
# 与 research.preset_common（其余）。搬迁前它们在本模块里各写了一份 —— 同一批常数
# 有两个定义，正是 §2(b)「常数恰好定义一次」禁止的形态；现在收成一处，这里按原名字
# 再导出，故外部读 `presets.RISK_LIMITS` / `presets.V21_SIGNAL_PARAMS` 的行为不变。
from research.preset_common import (DEFAULT_RISK_PER_TRADE, RISK_LIMITS,  # noqa: F401
                                    V21_PAYOFF_GEOMETRY, V21_SIGNAL_PARAMS)
from research.preset_registry import load_dir

_DEFS_DIR = Path(__file__).resolve().parent / "preset_defs"


def _build() -> tuple[dict[str, dict], dict[str, dict]]:
    """扫描 `preset_defs/` 建注册表。任何装载违规都在 **import 期**就抛（fail-fast，§0）。"""
    registry: dict[str, dict] = {}
    load_dir(registry, _DEFS_DIR)
    if not registry:
        raise ValueError(str(_DEFS_DIR) + " 里没有任何 preset 定义 —— "
                         "空注册表会让所有入口静默地跑空，而不是报错")
    # verdict 由模块级 VERDICT 注入（定义体里不许再写一份，见装载器）
    presets = {name: {**rec["preset"], "verdict": rec["verdict"]}
               for name, rec in registry.items()}
    return presets, registry


PRESETS, _REGISTRY = _build()

# 装载记录（含**锁坐标**与 fixture 窗口等元数据）。公开出来是因为"哪些 preset 在
# 行为锁里、用哪个绝对窗口"这件事必须能被测试断言，而 PRESETS 只带参数不带坐标。
REGISTRY: dict[str, dict] = _REGISTRY

# 定义体声明要再导出的名字（长篇负面证据随 preset 走，外部仍按原名字 import 得到）
for _rec in _REGISTRY.values():
    for _nm, _val in (_rec.get("exports") or {}).items():
        globals()[_nm] = _val
del _rec, _nm, _val


def get_preset(name: str = "us_uptrend_pullback_v1") -> dict:
    """返回 preset 深拷贝;未知名抛 ValueError。"""
    if name not in PRESETS:
        raise ValueError(f"未知 preset: {name!r}。可用: {sorted(PRESETS)}")
    p = copy.deepcopy(PRESETS[name])
    validate(p)                     # 执行契约:缺键 / 未知键 / 政策不一致 ⇒ fail-fast
    if p.get("n_comparisons", 0) < 1:
        raise ValueError(f"preset {name!r}: n_comparisons 必须 >= 1(CLAUDE.md §5)")
    return p
