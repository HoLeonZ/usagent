"""族定义体：lhr（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_long_horizon`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_long_horizon as mask

ID = "lhr"

DOC = """族:**长周期折价 + 短趋势已转向**(lhr)。"""

FIXED = {"ma20_buffer": 1.02}

GRID = {
    "dd_norm_k": [12.0, 14.0, 18.0],
    "xs_mom120_max": [None, 0.30, 0.50],
    "min_dd120": [0.20, 0.30],
}

SCORE_COL = "lhr_score"
