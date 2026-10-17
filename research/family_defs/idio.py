"""族定义体：idio（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_idio`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_idio as mask

ID = "idio"

DOC = """正题：掩码 `family_masks.mask_idio` 与 3 条臂共用（`idio_mktdown`, `idio_highvol`, `idio_highrvol`）；正题与臂的差别全在 FIXED 与各自的 GRID 里。"""

FIXED = {"mode": "idio"}

GRID = {
    "ret1_max": [0.03, 0.04, 0.05],
    "resid_max": [0.02, 0.04],
    "mkt_floor": [0.0, 0.005],
    "min_dd120": [0.15, 0.25],
    "vol_max": [1.0],
}

SCORE_COL = "__resid"
SCORE_NEGATED = True          # 该列越小越优先 ⇒ 排序时取负（见 families.py 的符号约定）
