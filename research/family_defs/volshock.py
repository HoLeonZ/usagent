"""族定义体：volshock（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_volshock`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_volshock as mask

ID = "volshock"

DOC = """族:**波动率恐慌的吸收确认**(volshock)。"""

FIXED = {"ret1_floor": -0.04, "mkt_floor": -0.010, "regime_gate": True}

GRID = {
    "shock_z_max": [-2.0, -2.5, -3.0],
    "tr_decay_max": [None, 0.4, 0.6, 0.8],     # None = 关闭「波动率不跟进」= 机制对照臂
    "hold_buffer": [None, 0.0, 0.005, 0.02],
}

SCORE_COL = "volshock_score"
NEEDS_REGIME = True           # 需要 regime 帧（状态列）
