"""决策凭证产物（`research/screen.py --out`）的测试 —— 离线合成面板，不读真库。

2026-10-17 机构就绪审计第 ③ 项：这个工具此前只 `print()`（`grep -cE
"to_csv|json.dump|write_text"` = 0），于是从"信号"到"下单"这一段**没有任何可对账的凭证**：
事后答不出"这 20 只是哪份代码 + 哪份数据算的"。这里钉住凭证的硬字段与结构。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import indicators, screen as SC  # noqa: E402
from research.tests.conftest import make_synth_panel  # noqa: E402


def _ind():
    return indicators.compute_all(make_synth_panel(("AAA", "BBB"), bars=360))


def test_artifact_carries_the_three_hard_fields(tmp_path):
    ind = _ind()
    table = pd.DataFrame([{"symbol": "AAA", "mom120": 0.1, "date": "2024-06-03"}])
    out = tmp_path / "artifact.json"
    SC._write_artifact(out, ind=ind, kind="candidates", as_of="2024-06-03",
                       tier="trend_pullback", horizon=18, window=["2024-01-01", "2025-01-01"],
                       table=table, benchmark={"mean": 0.01}, base_mean=0.02, note="为什么跑")
    d = json.loads(out.read_text(encoding="utf-8"))
    m = d["manifest"]
    assert set(m) >= {"ts", "git_sha", "python", "data_vintage"}, "凭证三件套缺一不可"
    assert m["data_vintage"], "没有数据 vintage ⇒ 事后答不出'哪份数据算的'"
    assert d["schema"] == "usagent.screen.artifact/v1"
    assert d["params"]["tier"] == "trend_pullback" and d["params"]["horizon"] == 18
    assert d["note"] == "为什么跑" and d["n_rows"] == 1
    assert d["table"][0]["symbol"] == "AAA"


def test_artifact_is_json_safe_for_numpy_values(tmp_path):
    """清单里带 numpy 标量是常态（pandas 的 to_json 处理不了全部）—— 不许因此崩。"""
    import numpy as np
    ind = _ind()
    table = pd.DataFrame([{"symbol": "AAA", "score": np.float64(1.5),
                           "n": np.int64(3), "flag": np.bool_(True)}])
    out = tmp_path / "a.json"
    SC._write_artifact(out, ind=ind, kind="measure", as_of=None, tier=None, horizon=18,
                       window=["2024-01-01", "2025-01-01"], table=table)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["n_rows"] == 1 and d["table"][0]["symbol"] == "AAA"


def test_artifact_defaults_are_honest_when_absent(tmp_path):
    ind = _ind()
    out = tmp_path / "a.json"
    SC._write_artifact(out, ind=ind, kind="candidates", as_of="2024-06-03",
                       tier="t", horizon=18, window=["a", "b"],
                       table=pd.DataFrame([{"symbol": "AAA"}]))
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["benchmark"] is None and d["base_mean"] is None and d["note"] == ""
