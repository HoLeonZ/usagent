"""validate.py 的三条新判据：新鲜度 / 全市场缺口 / 疑似基数断层。

三者都是**运维层可见性**问题（2026-10-17 审计）：

  * 新鲜度：取数失败曾被吞成 exit 0，于是「数据已更新」与「数据一个月没动」
    不可区分。新鲜度是唯一能分辨的判据，且必须由**外部日历**给出。
  * 全市场缺口：旧判据拿 ohlcv 自己的日期集合当日历 ⇒ 全市场缺的那天是隐形的。
  * 基数断层：增量刷新时代厂商重述历史会留下等于拆股比的假单日跌幅。检测判据是
    「单日倍数**恰好**接近整数 ≥2 且当天没有拆股事件」—— 真实行情几乎不会精确
    等于整数倍，而重述必然精确等于。
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import db  # noqa: E402
import trading_calendar as TC  # noqa: E402
# validate 现在是 CLI（--universe/--json）⇒ 测试必须显式给它 argv，
# 不能让它去读 sys.argv（在 pytest 下那是测试自己的参数）。
import validate  # noqa: E402

# ⚠️ 这份夹具必须**相对今天**构造。断言说的是「数据新鲜（缺的交易日 ≤ 1）」，而夹具此前写死到
# 一个固定日期 —— 日历往前走，它必然变红（2026-10-21 实测：「最新 2026-10-16；其后应有 3 个
# 交易日」）。那不是校验逻辑坏了，是<夹具>过期了：一个"关于现在"的断言，不能用"当时"的数据喂。
_TODAY = date.today()
RECENT = TC.expected_sessions(_TODAY - timedelta(days=45), _TODAY - timedelta(days=1))
OLD = TC.expected_sessions(date(2025, 1, 2), date(2025, 2, 28))


def _seed(tmp_path, monkeypatch, basis: dict[str, list[float]],
          sessions=RECENT, skip_days: set[str] = frozenset()) -> Path:
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "v.db")
    db.init()
    with db.session() as con:
        rows = []
        for sym, closes in basis.items():
            for d, c in zip(sessions, closes):
                if d.isoformat() in skip_days:
                    continue
                rows.append((sym, d.isoformat(), c, c * 1.01, c * 0.99, c, 1e6, 0.0, 0.0,
                             "seed"))
        con.executemany("INSERT INTO ohlcv (symbol,date,open,high,low,close,volume,"
                        "dividends,splits,fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return tmp_path / "v.db"


def _checks() -> dict[str, tuple[bool, str]]:
    return {name: (ok, detail) for name, ok, detail in validate.CHECKS}


def _jump_check() -> tuple[bool, str]:
    """跳变项现在是**非阻断的提示**（引擎已按标的×日期排除），所以断言要看**计数**
    而不是 ok 标志。名字里带前缀，按前缀找，改名字不会让测试静默失效。"""
    hits = [(ok, d) for name, ok, d in validate.CHECKS if "整数倍跳变" in name]
    assert hits, f"没有找到跳变检查项：{[n for n, _, _ in validate.CHECKS]}"
    return hits[0]


def _run(tmp_path, monkeypatch, **kw) -> dict[str, tuple[bool, str]]:
    _seed(tmp_path, monkeypatch, **kw)
    validate.CHECKS.clear()
    validate.main([])
    return _checks()


def _flat(symbols, n) -> dict[str, list[float]]:
    return {s: [100.0 + i * 0.1 for i in range(n)] for s in symbols}


def test_freshness_passes_for_recent_data(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch, basis=_flat(("AAA",), len(RECENT)))
    ok, detail = got["数据新鲜（缺的交易日 ≤ 1）"]
    assert ok, detail


def test_freshness_fails_for_stale_data(tmp_path, monkeypatch):
    """库停在一年前 ⇒ 必须判失败。这就是「数据其实没更新」的告警点。"""
    got = _run(tmp_path, monkeypatch, basis=_flat(("AAA",), len(OLD)), sessions=OLD)
    ok, detail = got["数据新鲜（缺的交易日 ≤ 1）"]
    assert not ok, "陈旧数据必须判失败"
    assert "其后应有" in detail


def test_market_wide_hole_is_detected(tmp_path, monkeypatch):
    """全市场都缺的那一天，旧判据（拿 DISTINCT date 当日历）看不见；新判据必须看见。"""
    hole = RECENT[5].isoformat()
    got = _run(tmp_path, monkeypatch, basis=_flat(("AAA", "BBB", "CCC"), len(RECENT)),
               skip_days={hole})
    ok, detail = got["无全市场缺口（不该有 data 的日子）"]
    assert not ok, "全市场缺口应当被抓到"
    assert hole in detail


def test_no_hole_when_data_complete(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch, basis=_flat(("AAA", "BBB"), len(RECENT)))
    assert got["无全市场缺口（不该有 data 的日子）"][0]


def test_basis_break_is_detected(tmp_path, monkeypatch):
    """厂商按 10:1 重述历史、而增量只重写了后半段 ⇒ 精确 10 倍的假涨幅。"""
    n = len(RECENT)
    closes = [100.0] * (n // 2) + [110.0] * (n - n // 2)
    closes[n // 2] = 1100.0                                    # 断层：正好 10 倍
    _run(tmp_path, monkeypatch, basis={"AAA": closes})
    ok, detail = _jump_check()
    assert "池内 1 只" in detail, f"整数倍断层应当被数出来：{detail}"


def test_reverse_split_break_is_detected(tmp_path, monkeypatch):
    """反向拆股的重述让历史价**上调** ⇒ 倍数方向相反，同样要抓到。"""
    n = len(RECENT)
    closes = [100.0] * (n // 2) + [10.0] * (n - n // 2)
    closes[n // 2] = 1.0 / 15.0 * 100.0                        # 1/15 反向拆股的重述
    _run(tmp_path, monkeypatch, basis={"BBB": closes})
    assert "池内 1 只" in _jump_check()[1]


def test_split_event_explains_the_jump(tmp_path, monkeypatch):
    """有拆股事件解释的整数倍跳变不该被误报（拆股本来就是整数倍）。"""
    n = len(RECENT)
    closes = [1000.0] * (n // 2) + [100.0] * (n - n // 2)
    db_path = _seed(tmp_path, monkeypatch, basis={"AAA": closes})
    with db.session(db_path) as con:
        con.execute("UPDATE ohlcv SET splits=10.0 WHERE symbol='AAA' AND date=?",
                    (RECENT[n // 2].isoformat(),))
    validate.CHECKS.clear()
    validate.main([])
    assert "池内 0 只" in _jump_check()[1], _jump_check()[1]


def test_extra_non_session_date_is_flagged(tmp_path, monkeypatch):
    """周末/休市日出现数据 ⇒ 抓出来（数据源串行或日期解析错了）。"""
    db_path = _seed(tmp_path, monkeypatch, basis=_flat(("AAA", "BBB"), len(RECENT)))
    with db.session(db_path) as con:
        con.execute("INSERT INTO ohlcv (symbol,date,open,high,low,close,volume,dividends,"
                    "splits,fetched_at) VALUES ('AAA','2026-10-17',100,101,99,100,1e6,0,0,'x')")
    validate.CHECKS.clear()
    validate.main([])
    assert not _checks()["无「不该有数据」的日期（周末/休市）"][0]


def test_truncated_tail_is_detected_for_in_pool_names(tmp_path, monkeypatch):
    """序列在窗口中间就断了 ⇒ 引擎会按陈价无限期重估那段权益（审计 A3）。

    此前这类标的在 validate 里**完全隐形**（缺口检查只看每只票自己的区间内部）。
    这里用流动性池内的名字构造截断（池外的不判失败，避免误报噪声）。
    """
    n = len(RECENT)
    basis = {s: [100.0 + i * 0.1 for i in range(n)] for s in ("AAA", "BBB")}
    db_path = _seed(tmp_path, monkeypatch, basis=basis)
    with db.session(db_path) as con:
        # AAA 从倒数第 10 个交易日之后就没有数据了（尾部截断），且给它足够成交量进池
        con.execute("UPDATE ohlcv SET volume=5e6 WHERE symbol='AAA'")
        con.execute("DELETE FROM ohlcv WHERE symbol='AAA' AND date >= ?",
                    (RECENT[-10].isoformat(),))
    validate.CHECKS.clear()
    validate.main([])
    got = _checks()
    ok, detail = got["池内无序列尾部截断（缺 ≥5 个交易日）"]
    assert not ok, detail
    assert "AAA" in detail


# ---------------------------------------------------------------- 覆盖率判据的**承重性**
#
# 2026-10-17 对抗核验抓到：这道检查的第一版写成"缺 coverage 且落在池内才失败"，
# 而 coverage 是从 ohlcv 反推的、池内也是对 ohlcv 分组的 ⇒ pool ∩ missing ≡ ∅
# ⇒ 在真库上**恒绿**（把活的检查改成空断言）。下面两道测试就是防它再退化。

class _FakeUniverse:
    def __init__(self, symbols):
        self.symbols = tuple(symbols)


def _coverage_check() -> tuple[bool, str]:
    hits = [(ok, d) for n, ok, d in validate.CHECKS if "从没尝试过" in n]
    assert hits, f"没有找到覆盖率检查项：{[n for n, _, _ in validate.CHECKS]}"
    return hits[0]


def test_coverage_check_fails_for_a_name_never_attempted(tmp_path, monkeypatch):
    """声明池里有一个"从没被尝试取过"的名字 ⇒ **必须红**（这正是每日刷 596 只、
    声明 3030 只那类事故的形状）。"""
    _seed(tmp_path, monkeypatch, basis=_flat(["AAA"], len(RECENT)))
    with db.session() as con:
        # 取过 AAA（ok），但 BBB 连一条 fetch_log 都没有 ⇒ 从没尝试
        db.log(con, "r1", "2026-10-16T00:00:00+00:00", "ohlcv", "AAA", "ok")
    monkeypatch.setattr(validate, "load_universe",
                        lambda name: _FakeUniverse(["AAA", "BBB"]))
    validate.CHECKS.clear()
    validate.main([])
    ok, detail = _coverage_check()
    assert ok is False, detail
    assert "BBB" in detail and "从没尝试 1" in detail


def test_coverage_check_does_not_red_on_attempted_but_empty(tmp_path, monkeypatch):
    """尝试过但厂商没给数据（退市/改名）⇒ **不判死**，只进 detail。

    这是有意的取舍：否则那些注定没有数据的名字会让这条闸门每天报红，
    运维层很快就会训练出"反正每天都红"的疲劳（本文件顶部那条 doctrine）。
    """
    _seed(tmp_path, monkeypatch, basis=_flat(["AAA"], len(RECENT)))
    with db.session() as con:
        db.log(con, "r1", "2026-10-16T00:00:00+00:00", "ohlcv", "AAA", "ok")
        db.log(con, "r1", "2026-10-16T00:00:00+00:00", "ohlcv", "GONE", "empty")
    monkeypatch.setattr(validate, "load_universe",
                        lambda name: _FakeUniverse(["AAA", "GONE"]))
    validate.CHECKS.clear()
    validate.main([])
    ok, detail = _coverage_check()
    assert ok is True, detail
    # 合成库里 coverage 表是空的 ⇒ AAA 也算"缺数据"；要证的是**没有"从没尝试"** ⇒ 不红
    assert "从没尝试 0" in detail and "尝试过但无数据" in detail


def test_coverage_check_is_informational_on_a_fresh_db(tmp_path, monkeypatch):
    """fetch_log 为空（全新/合成库）⇒ 不判死，但**要写清为什么**（不许静默通过）。"""
    _seed(tmp_path, monkeypatch, basis=_flat(["AAA"], len(RECENT)))
    monkeypatch.setattr(validate, "load_universe",
                        lambda name: _FakeUniverse(["AAA", "BBB"]))
    validate.CHECKS.clear()
    validate.main([])
    ok, detail = _coverage_check()
    assert ok is True and "fetch_log 为空" in detail
