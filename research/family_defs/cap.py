"""族定义体：cap（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_capitulation`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_capitulation as mask

ID = "cap"

DOC = """正题：掩码 `family_masks.mask_capitulation` 与 1 条臂共用（`cap_inv`）；正题与臂的差别全在 FIXED 与各自的 GRID 里。"""

FIXED = {"bad_ratio_max": 0.30, "px_disc_max": None}

GRID = {
    "climax_ratio_min": [2.5, 3.0, 4.0, 6.0],
    "drop_atr": [1.0, 1.5, 2.0, 3.0],
    "clv_min": [0.40, 0.50, 0.60],
}

SCORE_COL = "clv"

CONTROL = "cap_inv"        # 方向对照臂（只差一个变量）
