"""族定义体：sf_calm（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_sf_calm`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_sf_calm as mask

ID = "sf_calm"

DOC = """状态臂 B：平静态的**缩量**阴跌（个体性、没人出价）。"""

FIXED = {"calm_breadth": 0.50}

GRID = {
    "ret1_max": [0.035, 0.04, 0.05],
    "vol_ratio_max": [0.6, 0.7, 0.8],
    "resid_max": [0.03, 0.05],
}
NEEDS_REGIME = True           # 需要 regime 帧（状态列）

LEGACY_NAMES = ("_mask_sf_calm",)   # 历史上按这个符号名被引用（F3 那两条护栏）
