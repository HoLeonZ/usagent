"""holdout 台账 —— 「未参与任何迭代」必须是机器可验证的状态，不是一句承诺。

动机（2026-10-17 审计 B1，CRITICAL）：holdout 此前是 `wfv_search.report()` 里的一段
**无条件代码**，`--run` 跑一次就再消费一次，不留痕迹；`factor_probe --mid` 又把同一条
分界线当判据的设计参数。「该窗口是样本外」只靠文档与注释维持 ——
`family_search.py` 里那句「holdout 已在上一轮用过一次」是人写的注释，不是状态。

钉住的不变量：
  1. 同一窗口第二次被当作 claim 消费 ⇒ **默认拒绝**（HoldoutBurned），不是打条日志继续；
  2. 给了显式理由才放行，且理由本身进台账（可审计）；
  3. `kind="lock"`（行为锁/golden/单测）只登记、不烧窗口；
  4. 台账只增不改；坏行 fail-fast 而不是静默跳过（跳过会让计数偏小）；
  5. 不同窗口互不影响 —— 烧掉的是窗口，不是"holdout 这个概念"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import holdout as H  # noqa: E402

LO, HI = "2025-10-14", "2026-10-13"


@pytest.fixture
def ledger(tmp_path) -> Path:
    return tmp_path / "holdout_ledger.jsonl"


def _consume(ledger, lo=LO, hi=HI, **kw):
    kw.setdefault("script", "test")
    kw.setdefault("tag", "t")
    return H.consume(lo, hi, path=ledger, **kw)


def test_first_claim_is_allowed(ledger):
    prior = _consume(ledger)
    assert prior == 0
    assert H.n_evaluations(path=ledger) == 1
    assert H.claimed_windows(path=ledger) == [H.window_key(LO, HI)]


def test_second_claim_is_refused(ledger):
    _consume(ledger)
    with pytest.raises(H.HoldoutBurned) as e:
        _consume(ledger)
    msg = str(e.value)
    assert H.window_key(LO, HI) in msg
    assert "不再是样本外" in msg
    # 被拒绝的那次**不得**写进台账（否则计数被自己污染）
    assert H.n_evaluations(path=ledger) == 1


def test_forced_claim_records_the_reason(ledger):
    _consume(ledger)
    prior = _consume(ledger, force_reason="数据 vintage 变了,重算一遍")
    assert prior == 1
    recs = H.history(LO, HI, path=ledger)
    assert len(recs) == 2
    assert recs[-1]["force_reason"] == "数据 vintage 变了,重算一遍"
    assert recs[-1]["kind"] == "claim"


def test_a_different_window_is_fresh(ledger):
    _consume(ledger)
    assert _consume(ledger, lo="2024-10-14", hi="2025-10-13") == 0
    assert H.n_windows(path=ledger) == 2
    assert len(H.claimed_windows(path=ledger)) == 2


def test_lock_never_burns_and_never_raises(ledger):
    """行为锁/golden/单测可以反复登记 —— 重跑一次基线不是拿 holdout 出结论。"""
    for _ in range(3):
        assert _consume(ledger, kind=H.KIND_LOCK) == 0
    assert H.n_evaluations(path=ledger) == 3
    assert H.claimed_windows(path=ledger) == []
    # lock 之后再 claim 仍然是第一次 claim
    assert _consume(ledger) == 0
    with pytest.raises(H.HoldoutBurned):
        _consume(ledger)


def test_bad_kind_rejected(ledger):
    with pytest.raises(ValueError):
        _consume(ledger, kind="whatever")


def test_window_must_be_ordered(ledger):
    with pytest.raises(ValueError):
        H.window_key("2026-10-13", "2025-10-14")


def test_corrupt_ledger_fails_fast(ledger):
    """坏行不静默跳过：静默跳过会让 n_trials 偏小，正是台账要消灭的失败模式。"""
    _consume(ledger)
    with ledger.open("a") as fh:
        fh.write("{ this is not json\n")
    with pytest.raises(H.LedgerCorrupt):
        H.n_evaluations(path=ledger)


def test_missing_field_fails_fast(ledger):
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"ts": "x"}) + "\n")
    with pytest.raises(H.LedgerCorrupt):
        H.history(path=ledger)


def test_ledger_is_append_only(ledger):
    _consume(ledger, tag="first")
    _consume(ledger, kind=H.KIND_LOCK, tag="second")
    _consume(ledger, lo="2024-10-14", hi="2025-10-13", tag="third")
    dur = H.history(LO, HI, path=ledger)[0]
    backlog = H.history(lo="2024-10-14", hi="2025-10-13", path=ledger)[0]
    assert dur["tag"] == "first" and backlog["tag"] == "third"
    assert len(ledger.read_text().strip().splitlines()) == 3


def test_metrics_and_git_are_recorded(ledger):
    _consume(ledger, metrics={"cagr": 0.1})
    rec = H.history(LO, HI, path=ledger)[0]
    assert rec["metrics"] == {"cagr": 0.1}
    assert set(rec) >= {"ts", "window", "kind", "script", "tag", "git", "force_reason"}
    assert "ts" in rec and rec["window"] == H.window_key(LO, HI)


def test_wfv_main_maps_burned_holdout_to_exit_code_2(monkeypatch):
    """wfv_search 的契约：台账拒绝 ⇒ 非零退出（而不是打条警告继续跑）。"""
    from research import wfv_search as W

    monkeypatch.setattr(W.mod_panel, "load_full_panel", lambda *a, **k: None)
    monkeypatch.setattr(W.indicators, "compute_all", lambda p: None)
    monkeypatch.setattr(W.presets, "get_preset", lambda name=None: {})
    monkeypatch.setattr(W.regime, "compute_regime", lambda *a, **k: None)

    def burn(*a, **k):
        raise H.HoldoutBurned("已消费过，拒绝继续")

    monkeypatch.setattr(W, "report", burn)
    assert W.main(["--run"]) == 2


def test_wfv_main_refuses_via_default_ledger(monkeypatch, tmp_path):
    """不经 --force-holdout 的二次运行必须被台账挡住（默认路径，不是可选项）。"""
    from research import wfv_search as W

    monkeypatch.setattr(H, "LEDGER_PATH", tmp_path / "l.jsonl")
    H.consume(W.HOLDOUT_LO, W.HOLDOUT_HI, script="wfv_search", tag="第一次")
    with pytest.raises(H.HoldoutBurned):
        H.consume(W.HOLDOUT_LO, W.HOLDOUT_HI, script="wfv_search", tag="第二次")


def test_factor_probe_refuses_before_loading_data(monkeypatch, tmp_path, capsys):
    """factor_probe 的判据②把窗口切成两半，后半段就是 holdout ⇒ 二次运行必须被拒。

    快路径要求：拒绝要发生在加载面板**之前** —— 被拒的运行连数据都不该去碰。
    """
    from research import factor_probe as P

    monkeypatch.setattr(P.mod_holdout, "LEDGER_PATH", tmp_path / "l.jsonl")
    # 预先把这段窗口消费掉（模拟"上一轮已经看过它"）
    H.consume("2025-10-14", "2026-10-14", script="factor_probe", tag="上一轮",
              path=tmp_path / "l.jsonl")
    loaded = {"n": 0}

    def must_not_load(*a, **k):
        loaded["n"] += 1
        raise AssertionError("被拒的运行不该去加载面板")

    monkeypatch.setattr(P.mod_panel, "load_full_panel", must_not_load)
    assert P.main(["--run"]) == 2
    assert loaded["n"] == 0, "拒绝必须发生在数据加载之前"
    assert "拒绝继续" in capsys.readouterr().out


def test_factor_probe_allows_a_fresh_window(monkeypatch, tmp_path):
    """换一段从未消费过的窗口 ⇒ 正常放行（台账烧的是窗口，不是"holdout"这个概念）。"""
    from research import factor_probe as P

    monkeypatch.setattr(P.mod_holdout, "LEDGER_PATH", tmp_path / "l.jsonl")
    H.consume("2025-10-14", "2026-10-14", script="factor_probe", tag="另一段",
              path=tmp_path / "l.jsonl")
    monkeypatch.setattr(P.mod_panel, "load_full_panel",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))
    import pytest as _pytest
    with _pytest.raises(SystemExit):          # 说明它**走过了**台账、进入数据加载
        P.main(["--run", "--lo", "2023-10-14", "--mid", "2024-10-14", "--hi", "2025-10-14"])


def test_golden_baseline_registers_as_lock_not_claim(monkeypatch):
    """行为锁读到 holdout 窗口，但只能登记为 lock —— 不得烧掉它。"""
    from research import golden_baseline as G
    import pandas as pd

    seen: list[dict] = []
    monkeypatch.setattr(G, "mod_holdout", type("M", (), {
        "consume": staticmethod(lambda lo, hi, **kw: seen.append({"lo": lo, "hi": hi, **kw}) or 0),
        "KIND_LOCK": H.KIND_LOCK,
        "HoldoutBurned": H.HoldoutBurned,
    }))

    class _Stop(Exception):
        pass

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(G, "run_preset", stop)   # 让流水线在这里停住：只验台账那一步
    case = {"name": "X", "preset": "p", "start": "2025-10-14", "end": "2026-10-14"}
    with pytest.raises(_Stop):
        G.run_case(case, pd.DataFrame(), pd.DataFrame())
    assert len(seen) == 1
    assert seen[0]["kind"] == H.KIND_LOCK, "行为锁必须登记为 lock"
    assert (seen[0]["lo"], seen[0]["hi"]) == ("2025-10-14", "2026-10-14")
