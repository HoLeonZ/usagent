"""族定义体：idio_mktdown（契约见 research/family_registry.py）。

掩码在 `research.family_masks.mask_idio`（机制叙述就在它的 docstring 里 —— `family_search` 的报告取它的第一行当族名）。
"""
from __future__ import annotations

from research.family_masks import mask_idio as mask

ID = "idio_mktdown"

DOC = """对照臂:市场也在跌
臂：掩码同正题 `idio`（`family_masks.mask_idio`，见其 docstring 的机制叙述）；相对正题只改 FIXED —— mode: 'idio'→'mktdown'。"""

FIXED = {"mode": "mktdown"}

GRID = {
    "ret1_max": [0.04, 0.05], "resid_max": [0.04], "min_dd120": [0.15, 0.25],
    "vol_max": [1.0],
}

SCORE_COL = "__resid"
SCORE_NEGATED = True          # 该列越小越优先 ⇒ 排序时取负（见 families.py 的符号约定）
