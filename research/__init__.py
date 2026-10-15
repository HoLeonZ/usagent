"""research — uptrend_pullback → 美股基准判定。

把 hiagent 的 uptrend_pullback 思路原样搬到 usagent 的 3 年美股数据上,回答
「这个 preset 还剩多少,以及那些收益是不是 beta」。

设计文档: docs/superpowers/specs/2026-10-14-uptrend-pullback-us-baseline-design.md

本包**不修改 src/**(数据层不被交易逻辑污染)。下面这段 sys.path 垫片是
research 能 import 到 src/ 下扁平模块(series / db / config)与仓库根的 core/
(§2(b) 的常数单一来源)的唯一手段 —— usagent 的 src/ 不是包,
`import series` 依赖 sys.path[0] == src/。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for _p in (ROOT, SRC):                       # 倒序插入 ⇒ src/ 落在 sys.path[0],扁平模块优先
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = ["SRC", "ROOT"]
