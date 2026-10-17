"""族定义体：xs_flow（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_xs_flow`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_xs_flow as mask

ID = "xs_flow"

DOC = """族:**横截面流量折价反转**(xs_flow)。"""

FIXED = {}

GRID = {
    "xs_ret5_max": [0.05, 0.10, 0.20],
    "gap5_max": [0.20, 0.34],
    "vol_z_min": [0.0, 0.5],
}
