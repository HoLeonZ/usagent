"""族定义体：sf_stress（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_sf_stress`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_sf_stress as mask

ID = "sf_stress"

DOC = """状态臂 A：恐慌态的**放量**杀跌。"""

FIXED = {"v_stress": 0.012}

GRID = {
    "ret1_max": [0.035, 0.045, 0.06],
    "vol_ratio_min": [1.5, 2.0],
    "clv_max": [0.45, 0.55],
}
NEEDS_REGIME = True           # 需要 regime 帧（状态列）

LEGACY_NAMES = ("_mask_sf_stress",)   # 历史上按这个符号名被引用（F3 那两条护栏）
