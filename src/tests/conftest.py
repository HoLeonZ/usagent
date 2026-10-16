"""src/ 的测试配置。

src/ 是一组**扁平模块**(`import db` / `from config import ...`),不是包,所以
测试必须把 `src/` 本身放进 sys.path —— 与 research/tests/conftest.py 把仓库根
放进 path 是同一个道理。

这些测试**一律不联网**(§6:测试要能在没有外部服务的机器上跑)。Yahoo 调用在
每个用例里被显式替换掉。
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
