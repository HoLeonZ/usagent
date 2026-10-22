"""族定义体：str（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_short_reversal`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_short_reversal as mask

ID = "str"

DOC = """正题：掩码 `family_masks.mask_short_reversal` 与 1 条臂共用（`str_inv`）；正题与臂的差别全在 FIXED 与各自的 GRID 里。"""

FIXED = {}

GRID = {
    "drop5": [0.05, 0.08, 0.12, 0.18],
    "min_drawdown": [0.20, 0.35, 0.50],
    "min_off_low": [0.0, 0.05],
    "max_atr_pct": [None, 0.10],
    "max_rvol20": [None, 0.06],
}

CONTROL = "str_inv"        # 方向对照臂（只差一个变量）
