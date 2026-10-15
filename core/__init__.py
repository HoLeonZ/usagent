"""core —— 引擎层:费用/流动性常数与执行契约的**单一事实来源**。

为什么要有这一层(CLAUDE.md §2(b) 原文):

    commission / stamp-duty / min-fee constants are defined EXACTLY ONCE (in core/).
    Every engine MUST import them; redefining fee constants inside an engine or preset
    is a fail-fast violation.

2026-10-15 审计时本仓库**没有** core/:同一个「双边 5bp」被独立写出 7 处,而 preset 里的
`cost_bps_roundtrip` 在 CLI 跑数路径上从未被读取(`--cost-bps` 的默认值说了算,只有
HTML 报告读 preset 字段)—— 同一份 preset 在两处得到不同的成本假设,没有任何机制会发现。

本层的两个模块:
  * `costs`           —— 常数本体(恰好定义一次);
  * `preset_contract` —— 「引擎会读哪些键」的声明,以及 preset 的校验。
"""
from __future__ import annotations
