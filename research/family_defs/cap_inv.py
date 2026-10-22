"""族定义体：cap_inv（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_capitulation`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_capitulation as mask

ID = "cap_inv"

DOC = """对照臂:要收盘趴低
臂：掩码同正题 `cap`（`family_masks.mask_capitulation`，见其 docstring 的机制叙述）；相对正题只改 FIXED —— 新增 clv_side='max'。"""

FIXED = {"bad_ratio_max": 0.30, "px_disc_max": None, "clv_side": "max"}

GRID = {
    "climax_ratio_min": [2.5, 3.0, 4.0, 6.0],
    "drop_atr": [1.0, 1.5, 2.0, 3.0],
    "clv_max": [0.60, 0.50, 0.40],
}

SCORE_COL = "clv"
SCORE_NEGATED = True          # 该列越小越优先 ⇒ 排序时取负（见 families.py 的符号约定）
