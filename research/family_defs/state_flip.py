"""族定义体：state_flip（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_state_flip`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_state_flip as mask

ID = "state_flip"

DOC = """族:**状态翻转的语义门**(state_flip)。"""

FIXED = {}   # 提案里那几个常量直接写死在掩码里(它们是定义的一部分)

GRID = {
    "v_stress": [0.009, 0.012],
    "r_a": [0.03, 0.06],
    "vr_a": [1.2, 2.0],
    "r_b": [0.02, 0.05],
    "vr_b": [0.8, 1.2],
}
NEEDS_REGIME = True           # 需要 regime 帧（状态列）
