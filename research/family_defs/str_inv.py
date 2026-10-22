"""族定义体：str_inv（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_short_reversal`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_short_reversal as mask

ID = "str_inv"

DOC = """对照臂:买 5 日涨得多的
臂：掩码同正题 `str`（`family_masks.mask_short_reversal`，见其 docstring 的机制叙述）；相对正题只改 FIXED —— 新增 side='rise'。"""

FIXED = {"side": "rise"}

GRID = {
    "drop5": [0.05, 0.08, 0.12, 0.18],
    "min_drawdown": [0.20, 0.35, 0.50],
    "min_off_low": [0.0, 0.05],
    "max_atr_pct": [None, 0.10],
    "max_rvol20": [None, 0.06],
}
