"""族定义体：upb_ref（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_upb_ref`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_upb_ref as mask

ID = "upb_ref"

DOC = """参考族:不参与横向比较
**参考族**(恒等检查用,不是新策略思想)。"""

FIXED = {}

GRID_EXEMPT_REASON = "参考族不参与横向比较 —— 没有任何人搜它，所以它没有网格（不是漏写）"
REFERENCE = True
