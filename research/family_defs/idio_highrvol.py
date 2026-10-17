"""族定义体：idio_highrvol（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_idio`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_idio as mask

ID = "idio_highrvol"

DOC = """对照臂:高波动票
臂：掩码同正题 `idio`（`family_masks.mask_idio`，见其 docstring 的机制叙述）；相对正题只改 FIXED —— mode: 'idio'→'highrvol'。"""

FIXED = {"mode": "highrvol"}

GRID = {
    "ret1_max": [0.04, 0.05], "resid_max": [0.04], "mkt_floor": [0.0],
    "min_dd120": [0.20], "vol_max": [1.0], "rvol_min": [0.04, 0.06],
}

SCORE_COL = "__resid"
SCORE_NEGATED = True          # 该列越小越优先 ⇒ 排序时取负（见 families.py 的符号约定）
