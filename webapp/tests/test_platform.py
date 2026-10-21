"""策略平台的测试 —— **离线**（确定性合成市场 + 真 HTTP 服务在随机端口）。

钉住四件事：
  1. 目录页的数据**来自注册表本身**（数得出来的 preset/族条数、每条都有参数与负面证据）；
  2. 一次任务能从头跑到尾：取链路 → 信号 → 回测 → 渲染（自检通过）→ 台账；
  3. **迭代记账**：同一策略跑第二次，DSR 的 k 会 +1（"让点击次数可见"不是口号）；
  4. 服务端路由与拒绝语义（未知路径 404 / 坏请求 400 / 未知策略 404），且页面自包含。
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db as dbmod                                        # noqa: E402
import obs                                                # noqa: E402
from research import families, indicators, panel as panel_mod, presets  # noqa: E402
from research import render_report as RR                   # noqa: E402
from research.tests import fixture_market as FX             # noqa: E402
from webapp import catalog, jobs, server as srv_mod         # noqa: E402

WIN = ("2024-01-01", "2025-01-01")          # 合成市场的可评估窗

# node 里的最小 DOM 桩（两个 node 测试共用）：够脚本加载并调用纯函数。
# ⚠️ 没有 node 就跳过（CI 上没有 node）—— 与真数据行为锁在本机没库时跳过是同一种诚实边界。
NODE_PRELUDE = r"""
// ---- 最小 DOM 桩：够让脚本加载，并能调用几个纯函数 ----
// ⚠️ 元素**按 id 缓存**（2026-10-21 改）：浏览器里 `$(id)` 每次拿回的是同一个对象，
// 桩若不缓存，"填进去的值"就无从断言（`fillConfig` 把值写进 select，之后要能读回来）。
// 缓存之后这层桩与真实语义更接近 —— 之前每个 id 都返回新对象，等于假设"写入即丢弃"。
const _els = {};
let _made = 0;
const optsFor = (id) => {                    // 族的参数下拉：选项就是预注册网格上的取值
  const fam = ((globalThis.__CAT || {}).families || []).find(f => String(id).startsWith("p-" + f.id + "-"));
  if (!fam) return [];
  const axis = String(id).slice(("p-" + fam.id + "-").length);
  return (fam.grid[axis] || []).map(v => ({ value: JSON.stringify(v) }));
};
const fakeEl = (id) => _els[id] || (_els[id] = { id,
  // 日期输入给日期串；族参数下拉（id 形如 p-<族>-<轴>）必须给**合法 JSON**——
  // currentParams 会对它 JSON.parse（我第一版全给 "2025-01-01" ⇒ JSON.parse 当场抛）
  value: String(id).startsWith("p-") ? "2" : "2025-01-01", checked: false, innerHTML: "", textContent: "",
  style: {}, dataset: {}, disabled: false, children: [], options: optsFor(id),
  classList: { add(){}, remove(){}, contains(){ return false; } },
  appendChild(){}, remove(){}, insertAdjacentHTML(){}, addEventListener(){},
  querySelector: () => null, querySelectorAll: () => [], closest: () => null,
  scrollIntoView(){}, focus(){}, setAttribute(){}, getAttribute: () => null,
  getBoundingClientRect: () => ({ top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }) });
globalThis.document = { getElementById: (id) => fakeEl(id), querySelector: () => null,
  querySelectorAll: () => [], addEventListener(){},
  createElement: (tag) => fakeEl("__made" + (++_made) + "_" + tag),
  body: fakeEl("body"), documentElement: fakeEl("html"), styleSheets: [] };
globalThis.window = globalThis;
globalThis.location = { hash: "", pathname: "/" };
globalThis.navigator = {}; globalThis.alert = () => {};
globalThis.setInterval = () => 0; globalThis.clearInterval = () => {}; globalThis.setTimeout = () => 0;
globalThis.fetch = () => new Promise(() => {});        // 永不 resolve：不触发任何后续流程
globalThis.sessionStorage = (() => {      // 按 id 缓存**要有地方放**：存起来才能断言写进去了
  const m = new Map();
  return {getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)),
          removeItem: (k) => m.delete(k), clear: () => m.clear()};
})();
// 目录的最小形态：够 card()/cliCmd() 走完。脚本自己会 let CAT，所以要在它的作用域里赋值（见尾部探针）
globalThis.__CAT = {
  presets: [{ kind: "preset", id: "us_reversal_dip_v1", name: "n", window: ["2025-01-01", "2025-10-13"],
    engine_spec: [], strategy: { type: "signal", signal_params: { a: 1 } }, n_comparisons: 1,
    validated: false, touches_holdout: false, is_cli_default: false, delivered_as: [], verdict: "",
    grid: {}, grid_size: 0, score_col: "x", score_negated: false, needs_regime: false,
    control_arm: null, fixed: {}, reference: false, mechanism: "m", default_params: {} }],
  // 族也要有一条：currentParams(kind="family") 会去 CAT.families 里找它
  families: [{ kind: "family", id: "xs_flow", name: "xs_flow", mechanism: "m", grid: { a: [1, 2, 3] },
    grid_size: 3, fixed: {}, reference: false, score_col: "x", score_negated: false, needs_regime: false,
    control_arm: null, delivered_as: [], validated: false, default_params: { a: 2 }, window: null }],
  notes: { holdout_presets: 0, multiplicity: "" },
  holdout: ["2025-10-14", "2026-10-13"], train: ["2024-04-09", "2025-10-13"],
  data_through: null, stale_defs: [],
};
"""


# ---------------------------------------------------------------- 目录页

def test_catalog_lists_every_registered_strategy():
    cat = catalog.build()
    assert len(cat["presets"]) == len(presets.PRESETS)
    assert len(cat["families"]) == len(families.MASK)
    for e in cat["presets"]:
        assert e["engine_spec"], "preset 必须能展出引擎口径（与 CLI/报告同一个函数）"
        assert e["n_comparisons"] >= 1
        assert "window" in e
    for f in cat["families"]:
        assert f["mechanism"], "族要能展出机制叙述（掩码 docstring 首行）"
        assert f["grid_size"] >= 1 or f["reference"], "非参考族必须有预注册网格"


def test_family_default_params_are_the_grid_midpoint():
    """族的默认参数必须是**网格中点**（确定性、不看结果）—— 页面上也这么写。"""
    e = catalog.find("family", "str")
    assert e["default_params"] == {k: v[len(v) // 2] for k, v in e["grid"].items()}


def test_data_through_is_cheap_safe_and_never_creates_a_database(tmp_path, monkeypatch):
    """数据新鲜度探针：**只读**、便宜、且绝不凭空建库。

    最后那条是给自己立的规矩：`sqlite3.connect(path)` 会为不存在的路径**建一个空库文件** ——
    如果这里图省事直接 connect，CI 上就会留下一个假的 data/usagent.db，而它看起来像一个
    "空的数据集"（比报错更难查）。所以必须先判文件存在，且用只读 URI 连。
    """
    import db as dbmod
    missing = tmp_path / "nope" / "usagent.db"
    monkeypatch.setattr(dbmod, "DB_PATH", missing)
    assert catalog.data_through() is None
    assert not missing.parent.exists(), "不许凭空建目录/建库"

    # 真库（合成市场）在时：要么给得出，要么老实说没有 —— 给了就必须是这三件事
    db_file = FX.build(tmp_path / "fixture.db")
    monkeypatch.setattr(dbmod, "DB_PATH", db_file)
    got = catalog.data_through()
    if got is not None:
        assert set(got) == {"last_date", "symbols", "days_ago"}, got
        assert got["days_ago"] >= 0 and got["symbols"] >= 0


def test_catalog_flags_definitions_changed_after_the_process_started(tmp_path, monkeypatch):
    """改了定义体、服务没重启 ⇒ 服务读的是旧值，界面必须**主动说**，不能只靠一句提示。

    这条路是静默的，而且实测有更坏的版本：同秒内改动又改回、文件长度不变时，Python 认为
    旧 .pyc 仍有效 —— 进程于是用了**仓库里不存在的值**（我自己的验证脚本踩出过这一幕）。
    """
    d = tmp_path / "research" / "preset_defs"
    d.mkdir(parents=True)
    f = d / "x.py"
    f.write_text("NAME = 'x'\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    # 比"启动时刻"更旧 ⇒ 不算陈旧
    assert catalog.stale_definitions(started_at=time.time() + 10) == []
    # 比"启动时刻"更新 ⇒ 陈旧，且路径是仓库相对形式
    assert catalog.stale_definitions(started_at=0.0) == ["research/preset_defs/x.py"]


def test_the_maintainer_map_points_only_at_things_that_exist():
    """代码地图**不许烂**：里面写的每个路径都要真的在仓库里，写的每个数字都要对得上。

    为什么值得一条测试：这段文案是给"接手的人"看的 —— 他照着去开一个已被改名的文件，
    比没有地图更坏。数字同理（"6 条 preset 的真数据行为锁"这种话，改一次基线就可能变假）。
    所以路径按文件系统验、数字按 `golden_baseline.CASES` 验。
    """
    from research import golden_baseline as G
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    block = re.search(r'<details id="maintain".*?</details>', html, re.S).group(0)

    # ⚠️ 这个洞我自己踩过：原先只扫 <dt> 里的路径，而那句「细则见 CLAUDE.md」写在**正文的
    # <code> 里** —— 护栏验了我心里想的那类，没验我实际写下的全部，于是它放行了一个指向
    # 不存在文件的指路牌（本仓库根本没有 CLAUDE.md）。现在扫**整块里所有**像仓库路径的
    # <code> 片段，并显式列出允许指向外部的例外（例外要少，且每个都要有理由）。
    EXTERNAL = {"CLAUDE.md"}          # §0–§7 的原文写在维护者另一处，本仓库只有引用
    for tok in set(re.findall(r"<code>([^<>{}]+)</code>", block)):
        tok = tok.strip()
        if not re.fullmatch(r"[A-Za-z0-9_./*-]+\.(py|md|sh|html|json|csv)", tok):
            continue
        if tok in EXTERNAL:
            assert "不在本仓库" in block or "另一处" in block, \
                f"{tok} 被列为外部例外，但文案里没说清它是外部的"
            continue
        if tok.endswith("/*.py"):
            assert (ROOT / tok).parent.is_dir(), f"地图指向不存在的目录：{tok}"
        else:
            assert (ROOT / tok).is_file(), f"地图指向不存在的文件：{tok}"
    assert EXTERNAL == {"CLAUDE.md"}, "外部例外被扩大了 —— 每个例外都要单独说理由"

    # 命令行入口清单里的模块与脚本也要**真的在**（改名/搬走时这里会红）
    import importlib
    mods = re.findall(r"<dt>(research\.[a-z_]+)</dt>", block)
    assert len(mods) >= 5, f"命令行入口清单不见了？只找到 {mods}"
    for mod in mods:
        importlib.import_module(mod)          # 真的能导进来（不是靠文件名猜）
    for script in re.findall(r"<dt>(src/[a-z_]+\.py)</dt>", block):
        assert (ROOT / script).is_file(), f"清单里指向不存在的脚本：{script}"

    paths = re.findall(r"<dt>(research/[A-Za-z0-9_./*]+)</dt>", block)
    assert len(paths) >= 5, f"地图太薄：{paths}"
    for rel in paths:
        if rel.endswith("/*.py"):
            assert (ROOT / rel).parent.is_dir(), f"地图指向不存在的目录：{rel}"
        else:
            assert (ROOT / rel).is_file(), f"地图指向不存在的文件：{rel}"

    # 命令也要真的存在（工具被改名时这里会红）
    assert (ROOT / "tools" / "preflight.sh").is_file()
    assert "research.golden_baseline" in block and hasattr(G, "CASES")

    # 地图里那个数字必须与真基线用例数一致（"6 条 preset 的真数据行为锁"）
    assert f"{len(G.CASES)} 条 preset 的交易列表与权益曲线 sha256" in block, \
        f"地图里的条数与真基线（{len(G.CASES)}）对不上"


def test_every_anchor_target_clears_the_sticky_top_bar():
    """锚点是"量"出来的事，不是"有没有写 scroll-margin-top" —— 上一轮那条护栏只断言子串存在，
    于是 64px 这个**不够大的值**从它眼皮下走过去了（审计实测顶栏 78–118px）。

    这条测试做两件事：① 每个 href="#x" 的目标元素都在**会给它设 offset 的那组选择器**里
    （h2 / .card / #run）；② 那组规则的值不小于实测顶栏高度（宽屏 79px，窄屏 118px）。
    真正的像素级验收仍要看浏览器 —— 但值写小了这种错，这里会红。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    # ① 锚点目标必须被覆盖：把 href="#x" 的目标 id 与选择器规则对照
    targets = set(re.findall(r'href="#([a-z][a-z0-9_-]*)"', html))
    covered = {"presets", "families", "run", "runs", "reports", "top"}
    assert targets <= covered | {"presets", "families"}, f"有锚点目标没被规则覆盖：{targets}"
    assert "#run{scroll-margin-top" in html, "#run 是普通 div，必须单独给它 scroll-margin"
    # ② 值必须够大（宽屏顶栏实测 79px；窄屏换行到 118px）
    wide = [int(x) for x in re.findall(r"scroll-margin-top:(\d+)px", html)]
    assert wide and min(wide) >= 96, f"锚点偏移的最小值是 {min(wide) if wide else None}px，不够（顶栏实测 79–118px）"
    assert "scroll-margin-top:132px" in html, "窄屏（顶栏换行到 118px）要有更大的偏移"


def test_the_onboarding_block_is_collapsed_by_default():
    """入门块必须**默认收起**：老手每次打开页面都要多滚一屏，那它就成了负担。

    顺便钉住它讲的是"这一页怎么用、哪些词是什么意思"，而不是又一份营销文案 ——
    六个词里每一个都对应本仓库真在用的口径（k 的构成、holdout 只准读一次、
    曝光匹配超额与报告「超额」不是同一个量、双轨价格）。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    tag = re.search(r"<details[^>]*id=\"firsttime\"[^>]*>", html)
    assert tag, "没有入门块"
    assert "open" not in tag.group(0), "入门块默认必须是收起的"
    for term in ("preset", "族", "k", "holdout", "曝光匹配超额", "价格口径"):
        assert f"<dt>{term}</dt>" in html, f"入门块少了 {term} 这个词"


def test_catalog_publishes_the_windows_from_their_real_sources():
    """页面上的窗口预设必须**取自出处**，不能在 JS 里抄一份。

    HOLDOUT 还担着"与 holdout 重叠"的警示判据 —— 抄错/漂移就会错报（把一次正常回测标红，
    或者更糟：放过一次真的碰了 holdout 的运行）。TRAIN 的出处是 WFV 的搜索期。
    """
    from research import wfv_search as W
    cat = catalog.build()
    assert cat["holdout"] == list(catalog.HOLDOUT), "holdout 必须就是 holdout 台账那个常量"
    assert cat["train"] == [W.SEARCH_LO, W.SEARCH_HI], "训练窗必须取自 wfv_search 的搜索期"

    # 而且页面真的**用了**它（不是发布了但没人读 —— 本仓库为"声明了却没人读"立过规矩）
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    assert "if (CAT.holdout) HOLDOUT = CAT.holdout" in html
    assert "if (CAT.train) TRAIN = CAT.train" in html


def test_touches_holdout_matches_the_engines_half_open_window():
    """判据必须与引擎的 `[lo, hi)` 逐日一致 —— 特别是两端那两天。

    这是"同一件事两处实现"里最容易错的一类：区间开闭。所以测试**自己按引擎的语义**逐日扫一遍，
    再与被测判据逐例对照（不调用 engine，只复刻它选日的那条规则）。
    """
    import datetime as dt
    ho = tuple(catalog.HOLDOUT)
    h0, h1 = dt.date.fromisoformat(ho[0]), dt.date.fromisoformat(ho[1])

    def engine_actually_reads(lo: str, hi: str) -> bool:
        """引擎读 `[lo, hi)` 内的日（自然日是它的上界；边界两天下面单独钉住）。"""
        d, end = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
        while d < end:
            if h0 <= d <= h1:
                return True
            d += dt.timedelta(days=1)
        return False

    for lo, hi in [("2024-09-01", "2025-10-14"),   # 终点正好是 holdout 第一天 ⇒ 一天没读
                   ("2024-09-01", "2025-10-15"),   # 多一天 ⇒ 读了
                   ("2025-10-14", "2026-09-30"),   # 起点在 holdout 第一天 ⇒ 读了
                   ("2026-10-13", "2026-10-20"),   # 起点是 holdout 最后一天 ⇒ 读了
                   ("2026-10-14", "2026-11-01"),   # 整段在 holdout 之后 ⇒ 没读
                   ("2024-01-01", "2025-01-01"),   # 远早于 holdout ⇒ 没读
                   ("2024-01-01", "2026-12-31")]:  # 整段包含 holdout ⇒ 读了
        assert catalog.touches_holdout((lo, hi), ho) == engine_actually_reads(lo, hi), \
            f"{lo}..{hi} 判据与引擎不一致"

    assert catalog.touches_holdout(("2024-09-01", "2025-10-14"), ho) is False, \
        "终点落在 holdout 第一天不该算越界 —— 引擎窗口的右端是排他的（这条曾被误报）"


def test_holdout_touches_counts_what_the_page_warns_about():
    """卡片上写着「只准读一次」，那就得给出**被跨过几次** —— 否则它只是句口号。

    这条计数要覆盖**全台账**（不是显示的那 30 条）：它是纪律计数，不是表的一部分。
    """
    ho = tuple(catalog.HOLDOUT)
    rows = [{"window": [ho[0], ho[1]]},                 # 完全落在 holdout 里
            {"window": ["2024-01-01", "2025-01-01"]},   # 不碰
            {"window": ["2025-10-01", "2025-10-20"]},   # 边跨过去
            {"window": None},                           # 老记录可能没有窗口
            {}]
    assert catalog.holdout_touches(rows) == 2, catalog.holdout_touches(rows)
    assert catalog.holdout_touches([]) == 0


def test_catalog_flags_windows_that_touch_the_holdout():
    ho = tuple(catalog.HOLDOUT)
    assert catalog.touches_holdout(("2025-01-01", "2026-01-01"), ho) is True
    assert catalog.touches_holdout(("2024-01-01", "2025-01-01"), ho) is False
    cat = catalog.build()
    assert any(e["touches_holdout"] for e in cat["presets"]), "应当至少有一条 preset 含 holdout 天数"


# ---------------------------------------------------------------- 任务（离线合成市场）

@pytest.fixture
def offline(monkeypatch, tmp_path):
    """把"链路"换成合成市场的（同一段真代码，只是换了库），并把产物/台账都落到 tmp。"""
    db_file = FX.build(tmp_path / "fixture.db")
    monkeypatch.setattr(dbmod, "DB_PATH", db_file)
    P = presets.get_preset("us_reversal_dip_v1")
    chain = RR._chain_from(P)                     # 真代码路径，只是库是合成的
    monkeypatch.setattr(jobs.RR, "_chain_from", lambda _P: chain)
    monkeypatch.setattr(jobs, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    monkeypatch.setattr(obs, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(obs, "ALERT_LOG", tmp_path / "alerts.jsonl")
    jobs._JOBS.clear()
    return tmp_path


def _wait(job_id: str, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = jobs.status(job_id)
        assert st is not None
        if st["status"] in ("done", "error", "cancelled"):
            return st
        time.sleep(0.2)
    raise AssertionError("任务超时")


def test_every_run_records_which_code_produced_it(offline):
    """**§6 归因**：一次运行必须能回答"它是哪份代码算的"。

    只记 sha 不够 —— 脏工作区上 `rev-parse HEAD` 照样返回同一个 sha（`obs.git_dirty` 的
    docstring 讲的就是这件事），所以脏不脏要一起记。台账里混着多个 sha 是常态：改一次代码
    跑一次就多一个，而"同配置重跑 Δ=0 ⇒ 行为没变"这句判词只有在**同一份代码**下才成立。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    rec = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()][-1]
    assert "git_sha" in rec and rec["git_sha"], rec
    assert "git_dirty" in rec, "脏不脏没记 ⇒ 脏树上算出来的结果标注不了确定版本"
    assert rec["git_dirty"] in (True, False, None)


def test_inflight_route_lists_a_queued_job_and_the_page_can_recover_it(offline, http):
    """刷新后必须能把在途任务找回来 —— 判据是"服务端列得出它"，而不是"页面记得它"。

    此前 job id 只活在被刷掉的那份 DOM 里（没有 localStorage/sessionStorage 落它，服务端也只按
    id 查、没有枚举入口）⇒ 刷新后徽标/面板/取消入口全没了，页面还写着「还没跑过」，而服务端
    仍在跑。用户在这句假话下的自然反应是再点一次回测，k 再 +1。
    """
    jobs._LOCK.acquire()
    try:
        jid = jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                           "lo": WIN[0], "hi": WIN[1]})["id"]
        assert jobs.status(jid)["status"] == "queued"
        d = json.loads(_get(http + "/api/jobs")[1])
        got = {j["id"]: j for j in d["jobs"]}
        assert jid in got, "排队中的任务没被列出来"
        assert got[jid]["kind"] == "preset" and got[jid]["strategy"] == "us_reversal_dip_v1"
        assert got[jid]["status"] == "queued" and got[jid]["lo"] == WIN[0]
        # 面板要靠这几个字段重建标题**与"完整命令"**（否则刷新后的命令是拿卡片当前输入现拼的）
        fam = jobs.submit({"kind": "family", "id": "str", "lo": WIN[0], "hi": WIN[1],
                           "params": {"a": 3}, "control_runs": 200, "note": "n"})["id"]
        d1 = json.loads(_get(http + "/api/jobs")[1])
        g = {j["id"]: j for j in d1["jobs"]}[fam]
        assert g["params"] == {"a": 3} and g["control_runs"] == 200 and g["note"] == "n"
        jobs.cancel(fam)
    finally:
        jobs._LOCK.release()
    st = _wait(jid)
    assert st["status"] == "done", st.get("error")
    # 跑完之后它不该还挂在"在途"里
    d2 = json.loads(_get(http + "/api/jobs")[1])
    assert jid not in {j["id"] for j in d2["jobs"]}, "已结束的任务仍被列为在途"


def test_a_window_with_no_signal_records_zero_trades_and_a_zero_excess(offline):
    """一段窗口里**一笔都没成交**时，跑得完、记得下，而且各方说的 0 是同一个 0。

    为什么值得钉住（2026-10-21 实测）：这种运行会给出一整排 0 —— 收益 0、回撤 0、
    **曝光匹配超额精确为 0**（零敞口时两条腿完全相同，这是仓库的不变量）。在面板上
    「+0.00pp」配上按符号着色的绿色，读起来像"与池子打平"。所以：数字照算（不许为了好看
    去改口径），但要能一眼看出这是"没参与"。这条测试钉的是**数字本身**（页面那半句由
    node 护栏钉：zeroTradesNote / verdictDraft）。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": "2024-01-01", "hi": "2024-02-01"})["id"])
    assert st["status"] == "done", st.get("error")
    m = st["metrics"]
    assert m["n_trades"] == 0
    assert m["exposure"] == 0.0
    assert abs(m["excess"]) < 1e-9, f"零敞口 ⇒ 超额必须精确为 0，实际 {m['excess']}"
    assert m["excess_per_exposure"] is None, "0/0 必须是 None，不能是 NaN 或 0"
    # 步骤里要看得见"信号 0 个 / 成交 0 笔"（这是唯一能区分"跑完了"与"什么都没发生"的线索）
    names = " ".join(s["name"] for s in st["steps"])
    assert "0 个候选" in names and "0 笔" in names, names
    # 接口 JSON 不许有裸 NaN（JS 的 JSON.parse 会拒 —— 那会让整个面板渲染不出来）。
    # 这条专门用"一堆指标是 NaN 的 0 笔运行"再打一次：normal 运行测不出这个边界。
    import urllib.request
    srv = ThreadingHTTPServer(("127.0.0.1", 0), srv_mod.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/job/{st['id']}",
                                    timeout=10) as r:
            raw = r.read().decode("utf-8")
    finally:
        srv.shutdown()
        srv.server_close()
    for bad in ("NaN", "Infinity", "-Infinity"):
        assert bad not in raw, f"0 笔运行的接口 JSON 里有裸 {bad} —— 面板会整块渲染不出来"
    led = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()]
    assert led[-1]["n_trades"] == 0 and led[-1]["excess"] is not None
    # 池子范围也要落台账：页面存的是 sessionStorage（本次会话），**新标签页**会丢掉它 ——
    # 而页面上那句「（池子可评估范围：跑过任意一条后显示）」这时是假话（其实跑过很多次）。
    pr_ = led[-1]["pool_range"]
    assert isinstance(pr_, list) and len(pr_) == 2 and all(pr_), pr_


def test_runs_route_publishes_the_current_code_version(http, tmp_path, monkeypatch):
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    d = json.loads(_get(http + "/api/runs")[1])
    assert d["git_sha"], "当前代码版本没发出来"
    assert "git_dirty" in d
    assert d["git_sha"] == obs.git_sha()


def test_a_run_goes_all_the_way_and_lands_in_the_ledger(offline):
    job = jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                       "lo": WIN[0], "hi": WIN[1], "note": "测试"})
    st = _wait(job["id"])
    assert st["status"] == "done", st.get("error")
    assert [s["name"] for s in st["steps"]][:2] == ["取链路（命中缓存则近 0 秒）",
                                                   st["steps"][1]["name"]]
    assert st["metrics"]["n_trades"] > 0
    assert st["selfcheck"] == [], f"报告自检未通过：{st['selfcheck']}"
    rep = offline / f"web_report_{job['id']}.html"
    assert rep.is_file() and rep.stat().st_size > 10_000
    assert 'class="kchart"' in rep.read_text(encoding="utf-8")
    led = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()]
    assert led[-1]["strategy_id"] == "preset:us_reversal_dip_v1"
    assert led[-1]["note"] == "测试" and led[-1]["n_trades"] == st["metrics"]["n_trades"]


def test_second_run_raises_k_the_multiplicity_is_visible(offline):
    """**迭代记账**：同一策略跑第二次 ⇒ DSR 的 k 从 N+1 起算，页面上显示得出来。"""
    first = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                               "lo": WIN[0], "hi": WIN[1]})["id"])
    second = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                                "lo": WIN[0], "hi": WIN[1]})["id"])
    k1, k2 = first["metrics"]["dsr_k"], second["metrics"]["dsr_k"]
    assert k2 == k1 + 1, f"k 没随点击次数上涨：{k1} → {k2}"
    assert second["metrics"]["k_breakdown"]["platform_runs_before"] == 1


def test_a_queued_run_can_be_cancelled_and_is_not_recorded(offline):
    """取消要真的生效，且**不进台账** —— k 只数"被看过的试验"，不数花掉的 CPU。

    为什么用"排队中"这条路径测：运行本身只要一两秒，抢时序必然 flaky；而排队是确定的 ——
    先握住重活锁，提交的任务必然停在 queued，取消它、再放锁，结果必然 cancelled。
    （粗粒度的诚实边界：正在跑的那一步跑完才停，这条断言只覆盖"还没开始跑"的那段。）
    """
    jobs._LOCK.acquire()
    try:
        jid = jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                           "lo": WIN[0], "hi": WIN[1]})["id"]
        assert jobs.status(jid)["status"] == "queued"
        assert jobs.cancel(jid) is True
        assert jobs.cancel(jid) is True, "重复请求取消应当是幂等的（任务还没结束）"
    finally:
        jobs._LOCK.release()
    st = _wait(jid)
    assert st["status"] == "cancelled", st
    assert "未记账" in st["error"], st["error"]
    # 台账里不能有它留下的任何一行
    led = offline / "web_runs.jsonl"
    assert not led.exists() or "cancelled" not in led.read_text(encoding="utf-8")
    assert jobs.prior_runs("preset:us_reversal_dip_v1") == 0, "取消的运行不该推高 k"


def test_cancel_refuses_unknown_or_finished_jobs(offline):
    assert jobs.cancel("deadbeefdead") is False, "未知任务不该被受理"
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done"
    assert jobs.cancel(st["id"]) is False, "已经跑完的任务不该被受理"


def test_the_hint_about_k_matches_the_k_that_was_actually_used(offline):
    """提示里那句「k 从这里起算」必须与 DSR 真正用的 k 是同一个数。

    验收实测漂移过一次：`us_reversal_dip_v1` 是**内部用族掩码的 preset**，提示按"它用了族"
    报了「本族网格 96 组 ⇒ k 从这里起算」，而 DSR 用的是它声明的 n_comparisons（1614）——
    同一份报告里 k 是 1622。根因是"用哪条信号路径"和"k 怎么算"两件事被写在同一个分支里。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    mb = st["metrics"]["k_breakdown"]
    declared = mb["declared_or_grid"]
    assert f"n_comparisons={declared}" in st["hint"], f"提示与 k 不是同一个数：{st['hint']}"
    assert st["metrics"]["dsr_k"] == declared + mb["platform_runs_before"]
    # 族型请求才该说"网格组数"
    st2 = _wait(jobs.submit({"kind": "family", "id": "str",
                             "lo": WIN[0], "hi": WIN[1]})["id"])
    assert "本族网格" in st2["hint"] and st2["metrics"]["dsr_k"] == \
        st2["metrics"]["k_breakdown"]["declared_or_grid"] + 0


def test_every_run_leaves_a_machine_readable_trades_csv(offline):
    """报告是给人看的；逐笔那 20 列必须还有一份**脚本能直接读**的。

    此前只有 1.9MB 的自包含 HTML —— 想把逐笔拿去做别的计算只能从 HTML 里抠。
    这里连"真的能读"都测了（`read_csv(comment="#")`），因为一份带注释行的 CSV 如果
    注释写得不规矩，等于没导出。
    """
    import pandas as pd
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    csv = offline / f"web_trades_{st['id']}.csv"
    assert csv.is_file(), "没有逐笔 CSV"
    text = csv.read_text(encoding="utf-8")
    head = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert len(head) >= 5 and any("as-traded" in h for h in head), head   # 口径必须随文件走
    assert any("read_csv" in h for h in head), "要告诉读者怎么读"
    df = pd.read_csv(csv, comment="#")
    assert len(df) == st["metrics"]["n_trades"], "CSV 行数与成交笔数不一致"
    for col in ("entry_date", "exit_date", "symbol", "exit_reason", "entry_price",
                "exit_price", "net_pnl", "net_return"):
        assert col in df.columns, f"少了 {col}"
    assert st["trades_csv"] == f"/api/trades/{st['id']}.csv"


def test_trades_csv_route_serves_only_known_jobs(http, tmp_path, monkeypatch):
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    (tmp_path / "web_trades_abc123def456.csv").write_text(
        "# usagent 逐笔明细\nentry_date,exit_date\n2025-01-02,2025-01-03\n", encoding="utf-8")
    code, body = _get(http + "/api/trades/abc123def456.csv")
    assert code == 200 and body.decode().startswith("# usagent")
    assert _get(http + "/api/trades/deadbeefdead.csv")[0] == 404, "不存在的任务要 404"
    assert _get(http + "/api/trades/..%2F..%2Fetc%2Fpasswd.csv")[0] == 404, "路径不许拼"


def test_unknown_strategy_and_bad_window_surface_as_errors(offline):
    st = _wait(jobs.submit({"kind": "preset", "id": "nope_v9",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "error" and "nope_v9" in st["error"]
    st2 = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                             "lo": WIN[1], "hi": WIN[0]})["id"])
    assert st2["status"] == "error" and "窗口非法" in st2["error"]


def test_a_window_with_no_trading_days_fails_fast_and_names_the_range(offline):
    """空窗口要在**信号之前**就被挡下，且报错必须说出合法范围（审计实测两样都没有）。

    原本的路径：链路面板载入 → 算信号（7–11 秒）→ broker 抛「窗口内没有任何交易日」，
    而这句话没告诉读者该填什么。交易日轴在载完面板之后几乎不值钱，就在那一步问。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": "1999-01-01", "hi": "1999-02-01"})["id"])
    assert st["status"] == "error", st
    assert "没有任何交易日" in st["error"] and "覆盖" in st["error"], st["error"]
    assert "1999" in st["error"]
    # 而且**没跑信号**（那一步的名字不该出现在步骤里）
    assert not any("计算信号" in s2["name"] for s2 in st["steps"]), st["steps"]


def test_a_too_short_window_is_refused_up_front_with_a_reason(offline):
    """窗口太短要在**体检那一步**就拦下并说清原因。

    此前只挡"一个交易日都没有"，1–2 个交易日的窗口会一路跑完信号与回测，最后在 overlay
    里以一条没法行动的错误失败（审计实测）—— 而 DSR/超额在那么少的天数上本来也没有意义。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": "2024-06-03", "hi": "2024-06-05"})["id"])
    assert st["status"] == "error", st
    assert "交易日" in st["error"] and str(jobs.MIN_WINDOW_DAYS) in st["error"], st["error"]
    assert not any("计算信号" in s2["name"] for s2 in st["steps"]), "不该跑到信号那一步"


def test_a_run_reports_the_pool_range_so_the_page_can_warn_before_running(offline):
    """跑过一次之后，页面要知道"这个池子哪一段可评估" —— 据此在窗口旁显示范围并当场预警。"""
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    lo, hi = st["pool_range"]
    # 起止都必须是合法日期且起点早于终点（不写死具体年份：合成市场的跨度是 fixture 的事）
    import pandas as pd
    lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
    assert lo_ts < hi_ts, st["pool_range"]
    # 这条范围必须与刚跑过的窗口**重叠**（这正是页面拿它做的那个判断）—— 否则"可评估范围"
    # 与"刚跑出成交的窗口"互相矛盾。注意不能断言"含住窗口"：数据尾端早于窗口右端是常态
    # （合成市场到 2024-12-27，而评测窗到 2025-01-01），那种情况本来就只是"后面没数据"。
    assert lo_ts < pd.Timestamp(WIN[1]) and hi_ts > pd.Timestamp(WIN[0]), st["pool_range"]


def test_family_path_uses_the_registry_template(offline):
    """族型请求走的是"已注册 preset + 换掩码"那条路（与 CLI 报告同一做法）。"""
    st = _wait(jobs.submit({"kind": "family", "id": "str",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    assert st["hint"].startswith("本族网格")
    assert st["metrics"]["n_trades"] > 0


# ---------------------------------------------------------------- 服务端

@pytest.fixture
def http(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), srv_mod.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_serves_the_page_and_the_catalog(http):
    code, body = _get(http + "/")
    assert code == 200 and b"usagent \xe7\xad\x96\xe7\x95\xa5\xe5\xb9\xb3\xe5\x8f\xb0" in body
    assert b"<script" not in body.lower().replace(b"<script>", b"")     # 页面只有一段内联脚本
    code, body = _get(http + "/api/catalog")
    assert code == 200
    cat = json.loads(body)
    assert len(cat["presets"]) == len(presets.PRESETS)


def test_ledger_records_a_real_duration_and_the_params_that_ran(offline):
    """台账的两个"记了却用不上"的字段：seconds 曾恒为 null、params 有值却没渲染。

    seconds 的坑很典型：那一行在 `finally` **之前**读 `job["seconds"]`，而它正是 finally
    才赋的值 ⇒ 9 条台账全是 null（实测）。记了却永远用不上的字段比不记更坏 —— 它看起来
    像"有这件事"，让下一个人以为能拿它做统计。
    """
    st = _wait(jobs.submit({"kind": "family", "id": "str",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    rec = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()][-1]
    assert rec["seconds"] is not None and rec["seconds"] > 0, f"seconds 又变成 {rec['seconds']}"
    assert rec["params"], "族型运行必须记下**实际生效**的那组参数（网格中点）"


def test_server_rejects_unknown_paths_and_bad_requests(http):
    assert _get(http + "/nope")[0] == 404
    assert _get(http + "/report/deadbeef")[0] == 404          # 合法 id 但没有报告文件
    assert _get(http + "/report/../../etc/passwd")[0] == 404  # 不接受路径
    req = urllib.request.Request(http + "/api/run", data=b"{not json",
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("坏 JSON 应当 400")
    except urllib.error.HTTPError as e:
        assert e.code == 400
    req = urllib.request.Request(http + "/api/run", method="POST",
                                 data=json.dumps({"kind": "preset", "id": "nope",
                                                  "lo": WIN[0], "hi": WIN[1]}).encode())
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("未知策略应当 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_no_literal_markdown_leaks_into_visible_copy():
    """页面上的可见文案不许出现 `**` —— 那是 markdown 的强调，浏览器不认识，会原样打出来。

    同一个坑在这个仓库踩过三次（报告层靠 `inline_md` 先转义再做最小 markdown 解决，
    页面层一直没人管）。这条锁只看**标记之外**的部分：`<script>`（JS 注释里可以有）与
    `<style>`（CSS 注释里可以有）先剥掉。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    body = re.sub(r"<script>.*?</script>", "", html, flags=re.S)
    body = re.sub(r"<style>.*?</style>", "", body, flags=re.S)
    offenders = [ln.strip()[:70] for ln in body.splitlines() if "**" in ln]
    # ⚠️ 这个盲区我自己踩过第四次：**JS 生成的文案**（模板字符串拼出来的 HTML）不在静态标记里，
    # 上面的括号扫不到它。所以 script 里也要扫 —— 先剥掉注释（注释里可以有 markdown），
    # 再看剩下的还有没有 `**`。JS 里的 `**` 只可能是幂运算或文案，这个页面没有前者。
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    js = m.group(1) if m else ""
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    js = "\n".join(re.sub(r"(?<![:/])//.*$", "", ln) for ln in js.splitlines())
    offenders += [ln.strip()[:70] for ln in js.splitlines() if "**" in ln]
    assert not offenders, f"可见文案里有 markdown 的 **：{offenders}"


def test_no_tooltip_contains_html_tags():
    """`title="..."` 里不许有 HTML 标签 —— 属性值里的 `<b>` 不是加粗，就是那三个字符本身。

    为什么值得一条锁：HTML **允许**属性值里出现 `<`，所以页面不报错、只是悬停框里多出几个
    字符（读者看到「按它<b>从大到小</b>」这种字样）。我自己两次差点这么写：一次在
    `METRIC_HELP`（那条另有专门的锁，因为它走 `${esc(...)}` —— esc 之后还会再解码一次），
    一次在族卡片的「排序分」。这条扫**全部** title（静态标记与 JS 模板一起），
    所以它在哪儿写都拦得住。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    starts = [m.end() for m in re.finditer(r'title="', html)]
    assert len(starts) >= 20, f"只找到 {len(starts)} 个 title —— 扫描是不是失效了？"
    bad = []
    for start in starts:
        i, depth = start, 0
        while i < len(html):
            c = html[i]
            if c == "$" and html[i + 1:i + 2] == "{":
                depth += 1
            elif c == "}" and depth:
                depth -= 1
            elif c == '"' and not depth:
                break
            i += 1
        val = html[start:i]
        if re.search(r"</?(b|i|code|span|div|br)\b", val):
            bad.append((html[:start].count("\n") + 1, val[:90].replace("\n", " ")))
    assert not bad, f"tooltip 里有 HTML 标签（悬停框会原样印出来）：{bad}"


def test_static_markup_has_no_stray_backticks_or_markdown():
    """静态标记里的文案不许出现反引号 —— 页面上的代码是 `<code>`，反引号会原样打出来。

    为什么单独一条：`**` 那条锁扫的是**标记里**的文案（静态 HTML + 去掉注释的 JS），
    而我在维护者地图里写 `research/preset_defs/` 时就是手滑用了 markdown 的反引号
    （那一段现在是 `<code>`）。这条只扫 `<script>` 之前的部分 —— JS 里反引号是模板串，
    扫不得。当前实测：静态部分反引号 0 个，所以这条锁是空的、不会误报。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    body = html[:html.index("<script>")]
    offenders = [ln.strip()[:70] for ln in body.splitlines() if "`" in ln]
    assert not offenders, f"静态文案里出现了 markdown 反引号：{offenders}"


def test_run_counts_come_from_the_whole_ledger_not_the_shown_window(http, tmp_path, monkeypatch):
    """卡片「已跑 N 次」必须与 DSR 的 k 用**同一个计数**（都按全文台账数）。

    页面上的表只显示最近 30 条，而 k 用的是全文（`prior_runs` 逐行数）。此前卡片次数是从
    `/api/runs` 那 30 条里现数的 ⇒ 到第 31 次运行时界面会说"已跑 30"而 k 已经是 31：
    同一个量的两个数字（本仓库最忌讳的那种）。这里造 35 条，逼出这个分叉。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    lines = [json.dumps({"ts": f"2026-10-0{1+i%9}T00:00:0{i%9}", "strategy_id": "preset:x",
                         "n_trials_k": 10 + i, "excess": 0.01 * i, "window": ["2025-01-01", "2025-10-13"]})
             for i in range(35)]
    (tmp_path / "web_runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    d = json.loads(_get(http + "/api/runs")[1])
    assert d["total"] == 35 and d["shown"] == 30, d
    assert d["by_strategy"] == {"preset:x": 35}, d["by_strategy"]
    # 而且它与 k 的记账读的是同一份东西
    assert jobs.prior_runs("preset:x") == 35


def test_holdout_ledger_state_reports_both_kinds_and_refuses_to_hide_corruption(
        tmp_path, monkeypatch):
    """权威 holdout 台账的状态：claim 与 lock 分开数，坏了**要说出来**。

    为什么页面上必须有它：卡片写着「只准读一次」，而平台此前只显示**自己数的**那个数
    （web_runs.jsonl 里数窗口）。可"哪段 holdout 被真读过"的权威记录是这本台账，由
    wfv_search / family_search 通过 holdout.consume 写 —— **平台运行从不登记进去**。
    两本账不并排显示，读者会把平台那个数当成纪律账本上的次数。
    """
    from research import holdout as H
    led = tmp_path / "holdout_ledger.jsonl"
    led.write_text(
        json.dumps({"window": "2025-10-14..2026-10-01", "kind": "claim",
                    "script": "factor_probe", "tag": "stability"}) + "\n" +
        json.dumps({"window": "2024-01-01..2024-06-30", "kind": "lock"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(H, "LEDGER_PATH", led)
    st = catalog.holdout_ledger_state()
    assert st["claims"] == 1 and st["locks"] == 1, st
    assert st["claimed_windows"] == ["2025-10-14..2026-10-01"], st
    # 规范窗口本身**没**被 claim 过（唯一那条用的是它的子集）—— 这个区别必须能显示出来
    assert st["current_window_claimed"] is False, st
    assert st["last_claim"]["script"] == "factor_probe"
    # 平台越界运行里登记进这本账的条数（自己造两条：一条是那条 claim 的窗口、一条不是）
    st2 = catalog.holdout_ledger_state([
        {"window": ["2025-10-14", "2026-10-01"]},      # 正是被 claim 的那个窗口
        {"window": ["2024-04-09", "2026-10-14"]},      # 越界但没登记
        {"window": ["2024-01-01", "2025-01-01"]},      # 不越界
    ])
    assert st2["platform_crossers"] == 2 and st2["platform_registered"] == 1, st2

    # 台账坏了：要说出来，不能装作没事（H 自己的态度就是"坏了必须修，不能跳过"）
    led.write_text("{ 这不是合法 JSON\n", encoding="utf-8")
    st3 = catalog.holdout_ledger_state()
    assert "error" in st3 and "LedgerCorrupt" in st3["error"], st3


def test_compare_gives_deltas_and_flags_the_discipline_breaks(http, tmp_path, monkeypatch):
    """「跑完两条候选怎么比」—— 逐项差值，并且**明说**这次比是不是同口径。

    这是 2026-10-20 我自己**手工**做过一次的活（curl + 拼表），说明平台缺这个入口。
    比它更有价值的是那两句提醒：k 不同 ⇒ DSR 不可直接比；窗口/参数不同 ⇒ 其它项也不同口径。
    一个只给数字不给口径的对比表，会让人拿两次不同试验次数的 DSR 互比。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    rows = [
        {"ts": "2026-10-20T10:00:00", "job_id": "aaaaaaaaaaaa", "strategy_id": "preset:base",
         "window": ["2025-01-01", "2025-10-13"], "params": {"a": 1}, "n_trials_k": 20,
         "cagr": 0.10, "excess": -0.02, "bench_cagr": 0.30, "exposure": 0.1, "n_trades": 40,
         "win_pct": 0.25, "e_r": -0.2, "profit_factor": 0.7, "maxdd": -0.2, "folds_positive": 1,
         "alpha_annual": -0.01, "dsr": 0.003},
        {"ts": "2026-10-20T11:00:00", "job_id": "bbbbbbbbbbbb", "strategy_id": "preset:cand",
         "window": ["2025-01-01", "2025-10-13"], "params": {"a": 2}, "n_trials_k": 21,
         "cagr": 0.12, "excess": -0.01, "bench_cagr": 0.30, "exposure": 0.09, "n_trades": 38,
         "win_pct": 0.3, "e_r": -0.1, "profit_factor": 0.85, "maxdd": -0.21, "folds_positive": 1,
         "alpha_annual": 0.0, "dsr": 0.01},
    ]
    (tmp_path / "web_runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    d = json.loads(_get(http + "/api/compare/aaaaaaaaaaaa/bbbbbbbbbbbb")[1])
    assert d["a"]["strategy_id"] == "preset:base" and d["b"]["strategy_id"] == "preset:cand"
    cagr = next(r for r in d["rows"] if r["key"] == "cagr")
    assert abs(cagr["delta"] - 0.02) < 1e-12, cagr
    pf = next(r for r in d["rows"] if r["key"] == "profit_factor")
    assert abs(pf["delta"] - 0.15) < 1e-12, pf
    # 三条纪律标志：k 不同（20 vs 21）、窗口同、参数不同
    assert d["k_differs"] is True and d["window_differs"] is False and d["params_differ"] is True
    # 坏输入：id 不存在 / 路径不合法
    assert _get(http + "/api/compare/aaaaaaaaaaaa/deadbeefdead")[0] == 404
    assert _get(http + "/api/compare/nothex/bbbbbbbbbbbb")[0] == 400


def test_compare_flags_a_control_that_only_one_side_ran(http, tmp_path, monkeypatch):
    """只有一次跑了运气对照时，「随机入场百分位」那一行**没有可比值** —— 必须说出来。

    勾选框默认不跑（每次约 3–26 秒），所以"只有一边有"是常态而不是边界。不标出来的话，
    那一行是拿一个数减一个空，读起来像"乙变好了"，实则只是乙多付了一次对照的开销。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    base = {"window": ["2025-01-01", "2025-10-13"], "n_trials_k": 20, "cagr": 0.10,
            "params": {"a": 1}}
    rows = [
        dict(base, ts="2026-10-20T10:00:00", job_id="cccccccccccc", strategy_id="preset:x"),
        dict(base, ts="2026-10-20T11:00:00", job_id="dddddddddddd", strategy_id="preset:y",
             control_runs=20, ctl_pct=83.0),
    ]
    (tmp_path / "web_runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    d = json.loads(_get(http + "/api/compare/cccccccccccc/dddddddddddd")[1])
    assert d["control_mismatch"] is True
    assert d["a"]["ctl_pct"] is None and d["b"]["ctl_pct"] == 83.0
    row = next(r for r in d["rows"] if r["key"] == "ctl_pct")
    assert row["a"] is None and row["b"] == 83.0 and row["delta"] is None, row
    # 两边都跑过 ⇒ 不报那条提醒
    rows[0]["control_runs"], rows[0]["ctl_pct"] = 20, 50.0
    (tmp_path / "web_runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    d2 = json.loads(_get(http + "/api/compare/cccccccccccc/dddddddddddd")[1])
    assert d2["control_mismatch"] is False
    assert next(r for r in d2["rows"] if r["key"] == "ctl_pct")["delta"] == 33.0


def test_the_ledger_records_the_dynamic_metrics_the_compare_view_needs(offline):
    """对比视图要的动态指标必须**记进台账** —— 否则只能去翻两份报告（或从渲染好的 HTML 里
    反推事实，那是一份事实两处）。"""
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    rec = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()][-1]
    for key in ("cagr", "exposure", "bench_cagr", "maxdd", "win_pct", "e_r",
                "profit_factor", "folds_positive", "alpha_annual"):
        assert key in rec, f"台账没记 {key}（对比视图会缺这一项）"
    assert rec["cagr"] == pytest.approx(st["metrics"]["cagr"]), "台账与任务状态的同一个量不一致"


def test_k_accounting_inherits_the_parents_platform_runs(tmp_path, monkeypatch):
    """派生候选的 k 必须继承**父策略**在本平台的运行次数（§5 记账）。

    这条是**用平台真做一次迭代**时撞出来的：平台按**策略名**各自数运行次数，而"复制一条 preset
    改个参数"得到的新名字 prior 恒为 0 ⇒ 它的 DSR 分母偏小、判词偏乐观 —— 可它其实是**同一片
    搜索空间**里的又一次试验。实测：基线 k=20（声明 17 + 平台 3），兄弟记成 17（17+0），
    正确值是 17+3+1=**21**。机制：定义体里声明 derived_from。
    """
    led = tmp_path / "web_runs.jsonl"
    led.write_text("\n".join(json.dumps({"strategy_id": x}) for x in
                             ["preset:base"] * 3 + ["preset:cand"]) + "\n", encoding="utf-8")
    monkeypatch.setattr(jobs, "RUNS_LEDGER", led)

    kb0 = jobs.k_accounting({"n_comparisons": 17}, "preset", "cand", 17)
    assert kb0 == {"declared_or_grid": 17, "inherited_from_parent": 0, "inherited_parent": None,
                   "platform_runs_before": 1, "k": 18}, kb0

    kb1 = jobs.k_accounting({"derived_from": "base"}, "preset", "cand", 17)
    assert kb1["inherited_parent"] == "base" and kb1["inherited_from_parent"] == 3, kb1
    assert kb1["k"] == 21, f"应为 17+3+1=21，实际 {kb1['k']}"

    # 父策略的**声明值**不并进来 —— 那属于作者要写进本条 n_comparisons 的那一半（这里传 10）
    assert jobs.k_accounting({"derived_from": "base"}, "preset", "cand", 10)["k"] == 10 + 3 + 1


def test_derived_from_is_a_known_metadata_key_and_lands_in_the_ledger(offline):
    """`derived_from` 必须被契约接受（§2(a)：声明了就得有人读 —— 读它的是 jobs.k_accounting），
    并且要落进台账（否则事后没人能看出这条候选是从哪来的）。"""
    from core.preset_contract import METADATA_KEYS, validate
    assert "derived_from" in METADATA_KEYS
    base = dict(presets.get_preset("us_reversal_dip_v1"))
    base["derived_from"] = "us_uptrend_pullback_v1"
    validate(base)                                   # 未知键会抛；这里不该抛

    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    assert st["status"] == "done", st.get("error")
    rec = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()][-1]
    assert "derived_from" in rec and rec["derived_from"] is None, rec.get("derived_from")


def test_the_inline_script_parses(tmp_path):
    """内联脚本必须能**解析** —— 一个语法错就让整页变成死的（没有 JS，卡片都渲不出来）。

    为什么值得一条测试：2026-10-19 我自己往 METRIC_HELP 的字符串里嵌了 ASCII 引号，整个
    `<script>` 语法错误、页面全死；而当时**所有静态断言全绿** —— 按钮清单只看字符串在不在、
    markdown 扫描只看 `**`，没有一条会解析语法。这个洞是真实的：我当时已经把它推到了正在
    服务的文件里。

    用 `node --check`：开发机有 node 就真跑；没有就 skip 并说明（与"真数据行为锁在没有库时
    跳过"是同一种诚实边界 —— CI 上也没有 node，这条锁只在开发机生效）。
    """
    import shutil
    import subprocess
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    js = re.search(r"<script>(.*?)</script>", html, re.S)
    assert js, "页面里没有内联脚本？"
    node = shutil.which("node")
    if not node:
        pytest.skip("这台机器没有 node —— 语法锁跑不了（CI 也没有 node）")
    f = tmp_path / "inline.js"
    f.write_text(js.group(1), encoding="utf-8")
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, f"内联脚本语法错（整页会死）：{r.stderr[:500]}"


def test_the_inline_script_actually_runs_its_golden_path_functions(tmp_path):
    """内联脚本必须能**跑起来**，不只是"语法正确"。

    为什么补这条（2026-10-19 我自己捅的 blocking 回归）：我把 `const fp = pr ? …` 写在了
    `const pr = …` **上面** ⇒ TDZ 运行期 ReferenceError ⇒ `cliCmd` 抛错 ⇒ `panel()` 建不出面板
    ⇒ **整条回测流程是死的**（点回测：永远不出现面板）。而当时：
      * `node --check` 通过（TDZ 是运行期错误，不是语法错误）；
      * 按钮/字符串清单断言全绿（它只看字符串在不在）；
      * 685 个测试全绿（没有人执行那段 JS）；
      * 我也"验证"了那一批 —— 但只按 GET 与静态检查验，没再真点一次回测。
    两个独立审计 agent 各自用真实点击复现了它，我才发现。所以这里补一条**执行**级护栏：
    把内联脚本装进最小 DOM 桩里在 node 中执行，并调用黄金路径上的几个纯函数。

    没有 node 就 skip（与真数据行为锁在本机没库时跳过是同一种诚实边界）。
    """
    import json as _json
    import shutil
    import subprocess
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)

    node = shutil.which("node")
    if not node:
        pytest.skip("这台机器没有 node —— 执行级护栏跑不了（CI 也没有 node）")

    prelude = NODE_PRELUDE
    # 尾部探针**在脚本自己的作用域里**跑（所以能拿到它的 let 绑定），调用黄金路径上的纯函数
    probe = r"""
// ---- 探针：黄金路径上那几个"建面板/建卡片/给命令"的纯函数 ----
CAT = globalThis.__CAT;
LAST = {}; BYSTRAT = {}; REPORTS = []; RUNS = []; CUR_SHA = "deadbee";
// ---- 台账「回填这一行」：窗口 + **族参数**（2026-10-21 的缺陷：只填窗口时，
// 点「回测」跑的是网格中点那一组 —— 另一个配置，而且新跑那次还会进台账推高 k）
const onGrid = applyRunConfig({ strategy_id: "family:xs_flow",
  window: ["2024-02-02", "2024-08-08"], params: { a: 3 } });
// ⚠️ 值要**当场**读：下面那次调用会把同一个下拉/输入框再写一遍
const onGridValue = document.getElementById("p-xs_flow-a").value;
const onGridWindow = document.getElementById("lo-family-xs_flow").value;
const offGrid = applyRunConfig({ strategy_id: "family:xs_flow",
  window: ["2024-03-03", "2024-09-09"], params: { a: 99, ghost: 1 } });
const offGridKept = document.getElementById("p-xs_flow-a").value;   // 99 不该被写进去
const presetRow = applyRunConfig({ strategy_id: "preset:us_reversal_dip_v1",
  window: ["2024-04-04", "2024-10-10"], params: { drop5: 0.12 } });
const presetWindow = document.getElementById("lo-preset-us_reversal_dip_v1").value;
// ---- 命令的输入必须能**由调用方给**（刷新后从服务端找回的任务靠它，2026-10-21）----
const cmdExplicit = cliCmdFor("family", "xs_flow", "2024-05-05", "2024-11-11", { a: 3 }, 200, "");
const cmdLive = cliCmd("family", "xs_flow");      // 按卡片当前输入拼（桩里 ctl 未勾、a=2）
const pnl = panel("job9", "family", "xs_flow", "2024-05-05", "2024-11-11", { a: 3 }, 200, cmdExplicit);
// ---- 0 笔成交：面板要单独说清"0 不是不赚不亏而是没参与"，草稿也要带上这句 ----
const zeroNote = zeroTradesNote({ n_trades: 0, excess: 2.2e-14, exposure: 0 });
const someNote = zeroTradesNote({ n_trades: 9, excess: 0.01 });
const zeroDraft = verdictDraft({ metrics: { n_trades: 0, cagr: 0, own_ret: 0, exposure: 0,
  excess: 2.2e-14, excess_clamped: false, dsr_k: 1614, k_breakdown: {}, bench_cagr: 2.0,
  folds_positive: 0, win_pct: null, profit_factor: null, maxdd: 0, dsr: null, own_ret_: 0 },
  window: ["2024-01-01", "2024-02-01"], request: {} });
const out = {
  cli_preset: cliCmd("preset", "us_reversal_dip_v1"),
  cli_family: cliCmd("family", "xs_flow"),
  card: card(CAT.presets[0], 0),
  steps: stepsHtml({ status: "running", step: "取链路", steps: [] }, "job1", 3),
  stepsQueued: stepsHtml({ status: "queued", step: "x", steps: [] }, "job2", 1),
  last: lastRunLine({ kind: "preset", id: "us_reversal_dip_v1" }),
  reasonRows: typeof exit_reason_rows === "undefined" ? "n/a" : "n/a",
};
console.log(JSON.stringify({ ok: true, keys: Object.keys(out),
  cliHasFamily: out.cli_family.includes("--family xs_flow"),
  cliHasFingerprint: out.cli_family.includes("_p"),
  cardLooksReal: out.card.length > 200,
  stepsHaveCancel: out.steps.includes("取消"), queuedSaysQueued: out.stepsQueued.includes("排队中"),
  // 回填：网格上的取值要真的写进下拉；不在网格上的要**留下并报出来**（不是静默取近似）
  fillOk: onGrid.ok && onGrid.applied.join() === "a" && onGrid.skipped.length === 0,
  fillAppliedValue: onGridValue,
  fillWindowSet: onGridWindow === "2024-02-02",
  fillOffGrid: offGrid.skipped.slice().sort().join(),
  fillOffGridKept: offGridKept,
  fillPresetSkipsNothing: presetRow.ok && presetRow.applied.length === 0
    && presetRow.skipped.length === 0,
  fillPresetWindow: presetWindow,
  // 显式输入的命令：运气对照与产物名要跟着**给的**参数走（不是卡片当前状态）
  cliCtlExplicit: cmdExplicit.includes("--control-runs 200"),
  cliCtlLive: cmdLive.includes("--control-runs 0"),
  cliOutFromArgs: cmdExplicit.includes("web_report_cli_xs_flow_2024-05-05_2024-11-11_p"),
  cliCtlNotLeaked: !cmdLive.includes("--control-runs 200"),
  // 面板要印**传进去的**那条命令（刷新后找回的任务走这条），而不是现拼一条
  panelCmd: pnl.innerHTML.includes("web_report_cli_xs_flow_2024-05-05_2024-11-11_p"),
  panelParams: pnl.innerHTML.includes("&quot;a&quot;:3"),
  panelCtl: pnl.innerHTML.includes("运气对照 200 次"),
  zeroNoteShown: zeroNote.includes("一笔都没成交") && zeroNote.includes("没参与"),
  zeroNoteSilentForReal: someNote === "",
  zeroDraftWarns: zeroDraft.includes("一笔都没成交") && zeroDraft.includes("别把它们写进结论当证据"),
  // 「−0.00」不是数字（实测：0 笔成交的窗口里超额算出 −5.1e-15，印成 −0.00pp）
  negZero: [pp(-5.10702591327572e-15), pct(-1e-16), num(-1e-9), pp(-0.000123)],
  keepSmall: [pp(-0.0123), pct(-0.021)],
  // 池子可评估范围：新标签页（sessionStorage 空的）也要能从台账里取回来
  poolSeeded: seedPoolFromRuns([{ pool_range: null },
                                { pool_range: ["2024-01-11", "2026-10-16"] }]),
  poolValue: (POOL || []).join(".."),
  poolNotOverwritten: seedPoolFromRuns([{ pool_range: ["1999-01-01", "1999-12-31"] }]),
  poolValueKept: (POOL || []).join(".."),
  poolStored: sessionStorage.getItem("pool"),
  // 窗口没填全：既不能说"起点必须早于终点"，更不能谎称"与 holdout 重叠"
  emptyWindowFlag: (function () {
    const flag = document.getElementById("flag-preset-us_reversal_dip_v1");
    const lo = document.getElementById("lo-preset-us_reversal_dip_v1");
    const hi = document.getElementById("hi-preset-us_reversal_dip_v1");
    const keepLo = lo.value, keepHi = hi.value;
    lo.value = "";
    revalidate("preset", "us_reversal_dip_v1");
    const cleared = flag.textContent;
    lo.value = keepLo; hi.value = keepHi;
    revalidate("preset", "us_reversal_dip_v1");
    return [cleared, flag.textContent];
  })() }));
"""
    f = tmp_path / "run_inline.js"
    f.write_text(prelude + js + probe, encoding="utf-8")
    r = subprocess.run([node, str(f)], capture_output=True, text=True)
    assert r.returncode == 0, f"内联脚本执行失败（整页会死）：\n{r.stderr[-1200:]}"
    got = _json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])
    assert got["ok"], got
    assert got["cliHasFamily"], "cliCmd 没给出族命令"
    assert got["cliHasFingerprint"], "族命令的产物名缺参数指纹（会导致两组参数撞名）"
    assert got["cardLooksReal"], "card() 渲出来的卡片太短，八成中途抛了"
    assert got["stepsHaveCancel"], "运行中的步骤区没有取消按钮"
    assert got["queuedSaysQueued"], "排队中没如实说「排队中」"
    # 台账回填：窗口与**参数**要一起填，不在网格上的取值要报出来而不是静默取近似
    assert got["fillOk"], f"回填没把网格上的参数写进下拉：{got}"
    assert got["fillAppliedValue"] == "3", f"下拉里不是那一行的取值：{got['fillAppliedValue']!r}"
    assert got["fillWindowSet"], "回填没设窗口"
    assert got["fillOffGrid"] == "a,ghost", f"不在网格上的参数没被报出来：{got['fillOffGrid']!r}"
    assert got["fillOffGridKept"] == "3", "不在网格上的取值不该被写进下拉（那会静默换成别的配置）"
    # preset 行的参数在页面上改不了 ⇒ 既不算"填了"也不算"没填进去"
    assert got["fillPresetSkipsNothing"], f"preset 行不该报 skipped：{got}"
    assert got["fillPresetWindow"] == "2024-04-04", "preset 行的窗口没填上"
    # 命令的输入由调用方给：刷新后找回的面板要靠这条印出**那个任务自己的**配置
    assert got["cliCtlExplicit"] and got["cliCtlNotLeaked"], f"命令没跟着显式输入走：{got}"
    assert got["cliCtlLive"], "卡片上的命令没按当前勾选框拼"
    assert got["cliOutFromArgs"], "产物名没跟着显式窗口/参数走"
    assert got["panelCmd"], "面板没印传进去的那条命令（刷新后会印成另一套配置）"
    assert got["panelParams"] and got["panelCtl"], f"面板标题缺参数/运气对照：{got}"
    # 0 笔成交：0 与"没参与"必须被说成两件事（面板 + 判词草稿两处）
    assert got["zeroNoteShown"], "0 笔时没在面板上说清「0 是没参与，不是不赚不亏」"
    assert got["zeroNoteSilentForReal"], "有成交时不该出现那条说明"
    assert got["zeroDraftWarns"], "0 笔时判词草稿没带警告（草稿是要直接粘进 VERDICT 的）"
    # 显示上的 0 不带负号；但真的不为 0 的小数不能被抹掉
    assert got["negZero"] == ["0.00pp", "0.00%", "0.000", "-0.01pp"], f"负零没清掉：{got['negZero']}"
    assert got["keepSmall"] == ["-1.23pp", "-2.10%"], f"把不该抹的抹了：{got['keepSmall']}"
    # 池子范围能从台账里取回来（否则新标签页上「跑过任意一条后显示」是假话：其实跑过很多次）
    assert got["poolSeeded"] is True and got["poolValue"] == "2024-01-11..2026-10-16", got["poolValue"]
    assert got["poolNotOverwritten"] is False and got["poolValueKept"] == "2024-01-11..2026-10-16"
    assert got["poolStored"] == '["2024-01-11","2026-10-16"]', got["poolStored"]
    # 窗口没填全：说清是"没填全"，且**不许**说成"与 holdout 重叠"（空窗口什么都不碰）
    assert "没填全" in got["emptyWindowFlag"][0], got["emptyWindowFlag"]
    assert "holdout" not in got["emptyWindowFlag"][0], f"空窗口谎称碰 holdout：{got['emptyWindowFlag']}"
    assert got["emptyWindowFlag"][1] == "", f"填回去之后不该还挂着警告：{got['emptyWindowFlag']}"


def test_api_json_never_contains_bare_nan_or_infinity(http, monkeypatch):
    """接口的 JSON 里**不许出现裸 `NaN`/`Infinity`** —— Python 会写，JS 的 `JSON.parse` 会拒。

    2026-10-19 边界实测：一个 0 成交（或退化）的运行，metrics 里有 `t_alpha: NaN` / `E_R: NaN`
    ⇒ `/api/job/<id>` 的响应体里出现裸 `NaN` ⇒ **浏览器直接抛 `SyntaxError`** ⇒ 这次运行的结果
    永远渲染不出来（台账里却已经记下了），而前端还会把它误报成"读不到任务状态"。服务器其实是好的
    —— 所以这条锁钉在**唯一的出口**（`server._json`）上，用 JS 的语义（严格解析）验，而不是
    用 Python 的 `json.loads`（它对 NaN 是宽容的，验不出这个洞）。
    """
    fake = {"id": "abc123abc123", "status": "done", "seconds": 1.0, "selfcheck": [],
            "metrics": {"n_trades": 0, "t_alpha": float("nan"), "E_R": float("nan"),
                        "excess": float("-inf"), "dsr": float("nan"),
                        "k_breakdown": {"declared_or_grid": 1, "platform_runs_before": 0}},
            "hint": "", "report": "/report/abc123abc123"}

    monkeypatch.setattr(jobs, "status", lambda jid: fake if jid == "abc123abc123" else None)

    def _strict(txt: str):
        # 模拟 JS 的 JSON.parse：遇到 NaN/Infinity 直接抛
        return json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"裸 {c}")))

    code, body = _get(http + "/api/job/abc123abc123")
    assert code == 200
    text = body.decode()
    assert "NaN" not in text and "Infinity" not in text, f"响应体里有裸 NaN/Infinity：{text[:200]}"
    parsed = _strict(text)                      # 这一行在修复前会抛 ValueError
    assert parsed["metrics"]["t_alpha"] is None and parsed["metrics"]["E_R"] is None
    assert parsed["metrics"]["excess"] is None, "±Inf 也该变成 null"


def test_runs_limit_can_widen_the_window_and_rejects_garbage(http, tmp_path, monkeypatch):
    """台账的 30 条窗口要能被**界面**要求放宽 —— 否则"第 31 条"只能去 CSV 里看（把人赶出界面）。

    `?limit=` 必须夹住上限（台账是线性的，一个查询不该把它拉爆），且坏值要明说。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    lines = [json.dumps({"ts": f"2026-10-0{1 + i % 9}T00:00:0{i % 9}", "strategy_id": "preset:x",
                         "window": ["2025-01-01", "2025-10-13"]}) for i in range(45)]
    (tmp_path / "web_runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    d0 = json.loads(_get(http + "/api/runs")[1])
    assert d0["total"] == 45 and d0["shown"] == 30, d0
    d1 = json.loads(_get(http + "/api/runs?limit=45")[1])
    assert len(d1["runs"]) == 45 and d1["total"] == 45, len(d1["runs"])
    # 超大值被夹到 500（这里只有 45 条，所以给全）
    assert len(json.loads(_get(http + "/api/runs?limit=99999")[1])["runs"]) == 45
    assert len(json.loads(_get(http + "/api/runs?limit=0")[1])["runs"]) == 1, "下限为 1"
    code, body = _get(http + "/api/runs?limit=abc")
    assert code == 400 and "limit" in body.decode(), (code, body[:60])


def test_a_deleted_report_gives_a_human_page_not_json(http, tmp_path, monkeypatch):
    """报告文件被清理之后，那条链接要给人看的东西 —— 不是一坨 JSON。

    边界实测（把报告文件删掉，模拟 research/results/ 被手工清理）：浏览器里显示的是一行
    `{"error": "找不到 web_report_xxx.html"}` —— 而报告是**人点链接打开**的（新标签页、书签），
    这是最没用的一种脸。现在两条报告路由都回一个自包含的小页面：说清"不在盘上了"、文件名、
    常见原因，并给回台账的入口；**API 路由照旧回 JSON**（那边的读者是程序）。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    code, body = _get(http + "/report/deadbeef0001")            # 合法 id、盘上没有这份文件
    text = body.decode()
    assert code == 404, code
    assert "这份报告已经不在盘上了" in text and "回到平台" in text, text[:200]
    assert not text.lstrip().startswith("{"), "又回 JSON 了"
    code2, body2 = _get(http + "/report-file/web_report_deadbeef0001.html")
    assert code2 == 404 and "不在盘上" in body2.decode()
    # 程序读的那条路仍然是 JSON（不要把两种读者混在一起）
    code3, body3 = _get(http + "/api/job/deadbeef0001")
    assert code3 == 404 and body3.decode().lstrip().startswith("{"), body3[:80]


def test_every_registered_preset_has_a_behavior_lock_baseline():
    """**每条**已注册的 preset 都必须有行为锁基线（§6）。

    这条不只是"记一下"：新候选没有基线时 `test_golden_trades` 会对它红 —— 而那件事此前只有
    跑一次测试才知道（2026-10-20 实测：我建了兄弟候选，测试直接红，得自己反推是 `--regen` 的事）。
    平台现在会在卡片上直接说「无基线」，但**基线缺失本身仍要被抓住**。
    """
    from webapp import catalog as cat
    missing = [e["id"] for e in cat.build()["presets"] if not e["golden_baseline"]["exists"]]
    assert not missing, f"这些 preset 没有行为锁基线：{missing}"


def test_the_verdict_draft_cites_this_runs_own_numbers(tmp_path):
    """VERDICT 草稿必须引用**这次运行**的数字，并写明「尚未判定 ⇒ 不得用于实盘」。

    为什么要有这个功能（2026-10-20 亲自踩的坑）：VERDICT 里的数字是**手抄**的 —— 抄成别的
    候选的、或抄成上一轮的，报告与判词就互相矛盾，而报告本身不会报错（我把基线的 VERDICT 一字
    不差复制给兄弟候选，于是两条报告的「结论行」必然一样 ⇒ 看起来像"结论没变"，其实是没比过）。
    草稿只给数字与口径，结论仍由人写。这里在 node 里**真执行**那段 JS，不用字符串断言糊弄。
    """
    import shutil
    import subprocess
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    node = shutil.which("node")
    if not node:
        pytest.skip("这台机器没有 node —— 草稿内容的执行级测试跑不了")

    probe = r"""
CAT = globalThis.__CAT;
const st = { id: "abc123abc123", request: { kind: "preset", id: "us_reversal_dip_v1",
           lo: "2025-01-01", hi: "2025-10-13" },
  metrics: { n_trades: 35, exposure: 0.222, excess: -0.0597, cagr: 0.1078, bench_cagr: 0.3294,
             dsr: 0.019, dsr_k: 1614, win_pct: 0.316, e_r: -0.12, profit_factor: 1.35,
             maxdd: -0.145, folds_positive: 3,
             k_breakdown: { declared_or_grid: 1614, inherited_from_parent: 0,
                            inherited_parent: null, platform_runs_before: 0 } } };
console.log(JSON.stringify({ draft: verdictDraft(st) }));
"""
    f = tmp_path / "draft.js"
    f.write_text(NODE_PRELUDE + js + probe, encoding="utf-8")
    r = subprocess.run([node, str(f)], capture_output=True, text=True)
    assert r.returncode == 0, f"node 执行失败：{r.stderr[-500:]}"
    draft = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])["draft"]
    assert "35 笔" in draft and "-5.97pp" in draft and "10.78%" in draft, draft[:300]
    assert "k=1614" in draft and "声明 1614" in draft, draft[:300]
    assert "尚未判定" in draft and "不得用于实盘" in draft, draft[:300]
    assert "不要复制别的候选" in draft, draft[:300]


def test_server_binds_localhost_only():
    """**只绑 127.0.0.1**：这个页面能一键跑回测、读本地库、写报告文件。"""
    src = (ROOT / "webapp" / "server.py").read_text(encoding="utf-8")
    assert '("127.0.0.1", port)' in src and "0.0.0.0" not in src


# ---------------------------------------------------------------- 按钮/导航清单（本循环的机器化）

def test_the_page_has_the_affordances_development_and_backtesting_need():
    # 注：「回到平台」的返回条**不在**这张清单里 —— 它在服务端注入（server.py 的 _report），
    # 断在 test_served_reports_get_a_back_to_platform_bar_but_the_file_stays_clean 里。
    # 这张清单只扫 index.html，把只存在于服务端的字符串加进来会红（我自己加错过一次）。
    """页面必须齐备「开发 / 回测」所需的按钮与导航 —— 少一个就红。

    为什么写成清单断言：这一层没有 JS 单测，"按钮还在不在"只能靠读页面源码；
    而**漏掉一个按钮**（例如"复制成可复现命令"）不会让任何别的东西变红 ——
    那正是本仓库最忌讳的"新东西默认不在护栏里"。清单本身就是这份护栏。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    need = {
        "搜索框（21 张卡片里找一条）": 'type="search"',
        "顶栏常驻导航": 'class="top"',
        "分段锚点": 'href="#presets"',
        "卡片锚点（可深链）": 'id="${a}"',
        "窗口预设（训练/全窗/声明/holdout）": "windowPresets(",
        "窗口非法即时提示": "窗口非法（起点必须早于终点）",
        "回测按钮": "onclick=\"run('${e.kind}','${e.id}')\"",
        "复制成可复现命令": "copyCmd(",
        "查看定义体源码": "showSrc(",
        "族参数可编辑（只列预注册网格取值）": 'select class="grid"',
        "台账可排序（对比两次运行）": "sortLedger(",
        "holdout 重叠标红": "与 holdout 重叠",
        "报告下载": "download>",
        "报告内嵌": "<iframe",
        # 2026-10-18 UX 审计补的四件（每条都有实测证据，见 webapp/jobs.py 的注释）
        "历史报告入口（结果可回访，不必重跑）": 'href="#reports"',
        "台账的报告列（回访某一次）": 'href="/report/${esc(r.job_id)}"',
        "在途守卫（双击不静默多花一次 k）": "btn.disabled = true; btn.textContent",
        "结果按 job 分块（并发不互相覆盖）": '"job-" + jobId',
        "已用秒数（冷启动不像假死）": "已用 ",
        "步骤渲染只有一处（排队/运行/终态都走它）": "function stepsHtml(",
        "锚点不被常驻顶栏吃掉（含 #run 这种非 h2 目标）": "scroll-margin-top:96px",
        "运气对照开关（标了耗时，默认关）": 'id="ctl"',
        # 2026-10-18 第二轮 UX 审计（5 视角并行取证）补的（每条都有实测数字）
        "长参数串不再撑破整页（实测 2329px→1200px）": "overflow-wrap:anywhere",
        "宽表只让自己横滚，不顶破整页": 'class="scrollx"',
        "提交失败不再静默（实测原来永远停在「正在回测…」）": "function fail(msg, btn)",
        "读目录失败说得出原因并给重试": "读不到目录：",
        "读台账失败说得出原因": "读台账失败：",
        "轮询连续失败要明说（不是无声重试）": "读不到任务状态（连续",
        "台账参数列（对比的前提）": '["params","参数"]',
        "台账的「超额」写清是曝光匹配口径": '["excess","曝光匹配超额"]',
        "等待文案可被读屏读到": 'role="status"',
        "备注输入框（台账那一列原本没人能填）": 'id="note"',
        "复制的命令带上运气对照开关（否则 CLI 默认跑 200 次）": "--control-runs ${ctl}",
        "复制的命令产物与平台同一命名空间（跑完页面上看得见）": "web_report_cli_",
        "窗口按钮自带「按下去会设成什么」": 'title="${ww[0]} .. ${ww[1]}"',
        "族那个空操作按钮已禁用并说明原因（声明窗是 preset 的属性）": "（族没有声明窗）",
        "两个日期输入可被读屏区分": 'aria-label="窗口起点"',
        "搜索 0 命中时分节标题说实话（原本还写着「6 条」）": "这一节没有命中",
        "窗口旁显示池子可评估范围（跑过一次后就有）": "池子可评估范围",
        "与池子范围不重叠时当场预警（不必白等一次）": "完全不重叠",
        "族卡的批量入口（整张网格一次跑完）": "copyGridCmd(",
        "族卡说清「网格才是一次搜索」（别把 96 个格点各点一次）": "一次搜索的完整空间",
        "台账按策略筛选（对比时要能只看两条）": 'id="lf"',
        # Δ 列的排序键必须是 `_d`（poll/renderLedger 里算出来的字段名）。曾写成 d_excess：
        # 表头照样画、aria-sort 照样发，但点下去一行都不动（审计实测）。
        "台账 Δ 列（同一配置重跑的对比，排序键必须是 _d）": '["_d","Δ超额（同配置）"]',
        "台账完整 CSV 导出（表只有 30 条上限）": '/api/runs.csv',
        "台账窗口可点回填到卡片（窗口 + 参数一起，但不自动跑）": "fillConfig(",
        "台账明说只显示最近 N 条": "表只取最近",
        "卡片区排序（靠移动 DOM，不重渲 ⇒ 不吞用户输入）": 'id="csort"',
        "卡片区结构化过滤（只看跑过的 / 只看含 holdout）": 'id="cf-run"',
        "结构化过滤也算「在筛」（标题不能再说「6 条」）": "filtering = !!(q || onlyRun || onlyHeld)",
        "台账列头是真按钮（键盘可达）": 'class="thbtn"',
        "排序状态可被读屏读到": 'aria-sort=',
        # 2026-10-18 第三轮：滚走不失联 + 窄屏
        "顶栏「有任务在跑」徽标（滚到哪儿都看得见）": 'id="busy"',
        "徽标可点回结果": 'href="#job-',
        "徽标在轮询失败/任务丢失时也会摘掉（不留假在跑）": "trackInflight(jobId, false)",
        "超长机器串可断（实测 390px 下曾顶破 47px）": "color:#cbd5e1;overflow-wrap:anywhere",
        "窄屏压缩顶栏（sticky 的每一行都从正文永久扣走）": "@media (max-width:560px)",
        "运行中可取消（粗粒度，界面写明）": "function cancelJob(",
        "取消的终止态有专门分支（否则会掉进 metrics 渲染）": 'st.status === "cancelled"',
        "取消说清「不进台账」": "已取消的运行<b>不进台账</b>",
        # 2026-10-19 六视角对抗审计验出的「面板说假话」一族
        "刷新后能把在途任务找回来（服务端 GET /api/jobs）": "recoverInflight",
        "找回的面板印的是**那个任务自己的**配置与命令（不是卡片当前输入）": "cliCmdFor(",
        "失败时显示服务端的真实原因（不再一律「任务丢失」）": "st.status === undefined",
        "终态写回唯一的 live region（读屏才听得到）": "function fin(",
        "排队中如实说「排队中」（不再假装在取链路）": "服务端一次只跑一个任务",
        "在途守卫是真在途（不是 1.5 秒定时器）": "function releaseCardBtn(", 
        "卡片显示最近一次读数（开发时第一个问题是「哪条最好」）": "function lastRunLine(",
        "读数写清是「最近一次」而不是策略水平": "不是这条策略的水平",
        "可按最近一次超额排序（没跑过的排最后）": 'data-last=',
        "顶栏「重新读目录」（页面旧 vs 服务旧要分清）": "reloadCatalog()",
        "改完定义体要说清「要重启服务」（注册表启动时读一次）": "srchint-",
        "holdout 的次数要显眼（一句没有计数的纪律等于没有纪律）": "只准读一次",
        "定义体在服务启动后被改过要主动警告（不只是提示）": "CAT.stale_defs",
        "逐笔 CSV 下载（报告是给人看的，这份给脚本）": "st.trades_csv",
        "历史报告列表里的逐笔 CSV 列": "r.has_trades",
        # 边界实测（造 40 行台账 / 60 份报告）抓到的两处半拉子修复
        "报告区的截断要说出来（接口有 total，页面此前从没显示）": "盘上共 <b>${REPORTS_TOTAL}</b> 份",
        "报告的参数列在 job 不在台账里时不说「在定义体里」": "（台账里没有这条运行）",
        "权威 holdout 台账的状态要并排显示（两本账不能混）": "权威 holdout 台账",
        # 用平台真做一次迭代时撞出来的（界面审计看不见）：兄弟候选的 k 会偏小
        "开发指引里写明兄弟候选的记账规则（k 会偏乐观）": "别拿两次 k 不同的",
        "开发指引给出机制（derived_from），不只是规矩": "derived_from",
        "跑完两条怎么比：对比条（选甲/选乙）": 'id="cmpbar"',
        "对比表：逐项差值": "function fmtCell(",
        "对比时提醒 k 不可比（上一轮的教训）": "DSR 不可直接比",
        "按这次运行生成 VERDICT 草稿（避免抄错别的候选的数字）": "copyVerdictDraft(",
        "卡片上标出行为锁基线在不在（交付候选的前置条件）": "行为锁基线",
        "台账可视「载入全部」（此前第 31 条只能在 CSV 里看）": "载入全部（共 ${TOTAL} 条）",
        "平台越界运行里登记进权威账本的条数要说出来": "条登记进了这本账",
        "数据覆盖到哪天（拿多新的数据跑的结论）": "数据覆盖到",
        "数据陈旧时才提示，并给出取数命令": "download_ohlcv.py",
        "首次使用的入门块（三步 + 六个词）": 'id="firsttime"',
        "入门块默认收起（老手不必每次多滚一屏）": 'class="ft">',
        "入门块说清第一次要 30–70 秒（否则像挂了）": "第一次要 30–70 秒",
        "接手者用的代码地图（引擎在哪儿）": 'id="maintain"',
        "地图说明改完跑什么（含 --regen 的边界）": "tools/preflight.sh",
        "台账「代码」列（哪份代码跑的）": '["git_sha","代码"]',
        "台账顶栏写当前代码版本": "当前代码 <b>",
        "旧版本运行与脏树都要标出来": "function codeCell(", 
        "运气对照真跑（共用 control_arm，不是第二份）": "control_runs: controlRuns",
        # 候选生命周期的**第一步**（此前最手工：改名 / n_comparisons+1 / derived_from / 换 VERDICT）
        "复制成新候选（四件易忘的事由平台填好）": "newCandidate(",
        "新候选按钮只出现在 preset 卡上（族没有这几样要改的）": 'e.kind === "preset" ? `<button class="mini" title="拷一份改好的骨架',
        # 判据④的读数此前只藏在报告里（面板上只有"跑过 200 次"这句计数）
        "运气对照的读数上面板（判词是在面板上写的）": "随机入场百分位",
        "只有一边跑了对照就不给可比值": "control_mismatch",
        # 交付的判据此前只写在定义体的 docstring 里：页面上有"delivered/validated"的字样，
        # 但没有任何一处说**四条判据是什么**、以及这一页能答到哪一条
        "交付前置条件（四条判据是什么、这一页能答到哪）": "交付一条候选前（四条判据）",
        "明说哪两条口径本页不出（别拿面板的数去声称过了）": "与 ④ 的逐折口径本页不出",
        "排序分那一行要解释「取负」是什么意思（族间口径）": "按它「从大到小」决定谁先做",
        "需要 regime 那一行要说清与 preset 的同一道门": "否则信号被丢掉",
        # 2026-10-21 实测：一段 0 笔成交的真数据窗口，面板给出一排 0 且按符号涂色
        "0 笔成交要单独说清（0 是「没参与」不是「不赚不亏」）": "function zeroTradesNote(",
        "显示上的 0 不带负号（−0.00pp 是浮点残差不是数字）": "const _z0 = (x, d) =>",
        "池子可评估范围在新标签页也能显示（从台账取回，否则那句话是假话）": "seedPoolFromRuns(",
        "窗口没填全要当场说（空窗口不许谎称碰 holdout）": "✗ 窗口没填全（起点与终点都要",
    }
    missing = [k for k, v in need.items() if v not in html]
    assert not missing, f"页面缺这些按钮/导航：{missing}"


def test_metric_tooltips_are_plain_text():
    """19 个格子悬停时的定义必须是**纯文本** —— 它走 `title="${esc(...)}"`。

    为什么值得一条护栏：`esc()` 会把 `<b>` 转成 `&lt;b&gt;`，属性值解析时再解码回字面
    `<b>` ⇒ 浏览器悬停框里显示的是**这几个字符本身**，不是加粗。我写这条 tooltip 时
    第一版就带了 `<b>`（同时踩了另一个护栏：METRIC_HELP 属于可见文案，不许有 markdown 的 **）。
    """
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    i = html.index("const METRIC_HELP = {")
    block = html[i:html.index("};", i)]
    # 只认"像标签"的 `<`（`<[a-z]` / `</`）—— 数学写法「|t| < 2」是合法文本（实测第一版
    # 把这条也扫红了，那是我把判据写宽了）
    offenders = [l for l in block.splitlines() if re.search(r"</?[a-zA-Z]", l)]
    assert not offenders, f"tooltip 里有标签：{offenders}"
    n = len(re.findall(r'^\s{4}"[^"]+":', block, re.M))
    assert n >= 18, f"只找到 {n} 条 tooltip（应当是全部格子）"


def test_source_route_serves_the_strategy_file_and_refuses_traversal(http):
    """「查看定义体」要有只读路由；且**不许**拼用户输入做路径（遍历必须 404）。"""
    if not (ROOT / "research" / "preset_defs" / "us_reversal_dip_v1.py").is_file():
        pytest.skip("定义体不在（不应发生）")
    code, body = _get(http + "/api/source/preset/us_reversal_dip_v1")
    assert code == 200
    d = json.loads(body)
    assert d["path"] == "research/preset_defs/us_reversal_dip_v1.py"
    assert "PRESET" in d["text"] and d["bytes"] == len(d["text"])
    assert _get(http + "/api/source/family/nope")[0] == 404
    assert _get(http + "/api/source/preset/..%2F..%2Fetc%2Fpasswd")[0] == 404
    assert _get(http + "/api/source/preset/" + "a" * 80)[0] == 404      # 超长名字也拒


def test_candidate_skeleton_is_a_ready_to_register_definition():
    """「复制成新候选」给的骨架必须**当场能注册** —— 四件事都替人填好，缺一件就红。

    为什么这条比"字符串里有 derived_from"强：候选生命周期第一步是手工的，而四种错法
    （NAME 与文件名不一致 → 注册表拒绝；n_comparisons 没 +1 → DSR 分母偏小；derived_from
    漏写 → 少继承父策略的运行次数；VERDICT 抄着父策略那段 → 报告与判词互相矛盾）
    **没有一种会当场报错**。所以这里真的 `exec` 它、走注册表自己的 `TC.validate`，
    把"能装进注册表"这件事验到与真落盘同一层。
    """
    from core import preset_contract as TC
    n_checked = 0
    for name in sorted(presets.REGISTRY):
        sk = catalog.candidate_skeleton(name)
        assert sk is not None, f"{name} 没有可复制的定义体"
        path, text = sk["path"], sk["text"]
        # ① 能编译、能过注册表契约（REQUIRED_KEYS / 未知键白名单都在里面）
        ns: dict = {}
        exec(compile(text, path, "exec"), ns)
        TC.validate(ns["PRESET"])
        # ② NAME 必须等于文件名（注册表会校验这条；我 2026-10-20 就踩过）
        assert ns["NAME"] == Path(path).stem == f"{name}_cand"
        # ③ derived_from 指回父策略，且 n_comparisons 恰好 +1（父那条的声明值）
        assert ns["PRESET"]["derived_from"] == name
        assert int(ns["PRESET"]["n_comparisons"]) == int(
            presets.get_preset(name)["n_comparisons"]) + 1
        # ④ VERDICT 不得留着父策略那段结论（那是针对另一套参数的判断）
        assert "尚未判定" in ns["VERDICT"]
        assert ns["VERDICT"] != presets.REGISTRY[name]["verdict"]
        n_checked += 1
    assert n_checked == len(presets.REGISTRY) and n_checked >= 6


def test_candidate_skeleton_never_proposes_a_name_that_is_taken(tmp_path, monkeypatch):
    """同一条 preset 连点两次，第二次**不许**再建议同一个文件名 —— 那是在帮人覆盖自己的候选。"""
    src = ROOT / "research" / "preset_defs" / "us_reversal_dip_v1.py"
    if not src.is_file():
        pytest.skip("定义体不在（不应发生）")
    d = tmp_path / "preset_defs"
    d.mkdir()
    (d / "us_reversal_dip_v1.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(catalog, "_DEFS", d)          # 函数读的是模块级常量
    first = catalog.candidate_skeleton("us_reversal_dip_v1")
    assert Path(first["path"]).stem == "us_reversal_dip_v1_cand"
    assert first["renamed_because"] is None
    (d / f"{Path(first['path']).stem}.py").write_text(first["text"], encoding="utf-8")
    second = catalog.candidate_skeleton("us_reversal_dip_v1")
    assert Path(second["path"]).stem == "us_reversal_dip_v1_cand2"
    assert second["renamed_because"] == "us_reversal_dip_v1_cand"   # 说清为什么换了名字
    compile(second["text"], second["path"], "exec")
    assert 'NAME = "us_reversal_dip_v1_cand2"' in second["text"]
    # 平台**不写文件**（那是开发者的活）：第一条候选应当原封不动
    assert (d / "us_reversal_dip_v1_cand.py").read_text(encoding="utf-8") == first["text"]
    assert not (d / "us_reversal_dip_v1_cand2.py").exists()


def test_candidate_skeleton_route_serves_the_text_and_refuses_unknown(http):
    """页面的「复制成新候选」要有服务端一半；且族不该有它（族没有 NAME/VERDICT 这几样要改的）。"""
    code, body = _get(http + "/api/candidate_skeleton/us_reversal_dip_v1")
    assert code == 200
    d = json.loads(body)
    assert d["path"] == "research/preset_defs/us_reversal_dip_v1_cand.py"
    assert d["parent"] == "us_reversal_dip_v1"
    assert 'NAME = "us_reversal_dip_v1_cand"' in d["text"]
    # 未知 preset / 遍历尝试 / 族一律 404（族那条走的是同一个正则，所以直接判它不匹配）
    assert _get(http + "/api/candidate_skeleton/nope")[0] == 404
    assert _get(http + "/api/candidate_skeleton/..%2F..%2Fetc%2Fpasswd")[0] == 404
    assert _get(http + "/api/candidate_skeleton/" + "a" * 80)[0] == 404


def test_reports_route_lists_past_reports_and_parses_their_header(http, tmp_path, monkeypatch):
    """`/api/reports` 是"结果可回访"的服务端一半：文件名就是 id，名字与窗口**从报告自身读回**。

    为什么要从报告头部读：报告本来就有 title 与副标题（策略名/窗口），另存一份元数据
    就是"一份事实两处"（本仓库的老毛病）。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    (tmp_path / "web_report_abc123def456.html").write_text(
        "<!doctype html><title>回测明细 · family:xs_flow</title>"
        "<div class='sub'>窗口 2025-01-01 → 2025-10-13 ｜ 成交 28 笔</div>", encoding="utf-8")
    (tmp_path / "not_a_report.html").write_text("<title>别的</title>", encoding="utf-8")
    code, body = _get(http + "/api/reports")
    assert code == 200
    reps = json.loads(body)["reports"]
    assert [r["job_id"] for r in reps] == ["abc123def456"], "只该列 web_report_*"
    assert reps[0]["name"] == "family:xs_flow"
    assert reps[0]["window"] == ["2025-01-01", "2025-10-13"]


def test_runs_route_counts_holdout_touches_over_the_whole_ledger(http, tmp_path, monkeypatch):
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(jobs, "RUNS_LEDGER", tmp_path / "web_runs.jsonl")
    ho = list(catalog.HOLDOUT)
    lines = [json.dumps({"strategy_id": "preset:x", "window": ho}),
             json.dumps({"strategy_id": "preset:x", "window": ["2024-01-01", "2025-01-01"]}),
             json.dumps({"strategy_id": "preset:x", "window": ["2025-11-01", "2025-12-01"]})]
    (tmp_path / "web_runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    d = json.loads(_get(http + "/api/runs")[1])
    assert d["holdout_touches"] == 2, d          # 1 与 3 都落在 holdout 内
    assert d["total"] == 3


def test_served_reports_get_a_back_to_platform_bar_but_the_file_stays_clean(
        http, tmp_path, monkeypatch):
    """返回条只在**托管时**存在：文件本身必须仍然自包含、仍然干净。

    报告是可下载、可归档的产物（自检明令不许有 `https?://`）—— 把返回链接写进文件里，
    自检会红、归档后链接是死的。所以只在服务端响应时注入，并实测"下载到的那一份没有它"。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    f = tmp_path / "web_report_abc123def456.html"
    f.write_text("<!doctype html><title>回测明细 · x</title>"
                 "<div class='sub'>窗口 2025-01-01 → 2025-10-13</div></body></html>", encoding="utf-8")
    code, body = _get(http + "/report/abc123def456")
    served = body.decode()
    assert code == 200
    assert 'class="hostbar"' in served and 'href="/#runs"' in served, "托管时应有返回条"
    assert "job abc123def456" in served
    assert "@media print" in served, "打印时不该把这条浮层打到纸上"
    # ⚠️ 关键：**文件本身**没有被动过
    disk = f.read_text(encoding="utf-8")
    assert "hostbar" not in disk and "回到平台" not in disk, "返回条被写进文件了"
    # 也没有引入外链（自检的那条规矩依旧成立）
    assert "http://" not in served.replace("https://", "")


def test_reports_route_carries_the_params_that_run_used(http, tmp_path, monkeypatch):
    """报告列表要能回答"这份是哪组参数跑出来的" —— 参数**不在报告里**，在台账里。

    所以按 job_id 回读台账，而不是从渲染好的 HTML 里正则抓（那是从产物反推事实，
    一份事实两处）；实测同一族两次不同参数（{0.34,0.5,0.1} vs {0.2,0,0.05}）在列表上
    原本完全无法分辨 —— 而这个平台自己就宣传"两条策略的对比看这里"。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    (tmp_path / "web_report_deadbeef0001.html").write_text(
        "<!doctype html><title>回测明细 · family:xs_flow</title>"
        "<div class='sub'>窗口 2025-01-01 → 2025-10-13 ｜ 成交 33 笔</div>", encoding="utf-8")
    (tmp_path / "web_runs.jsonl").write_text(json.dumps(
        {"job_id": "deadbeef0001", "strategy_id": "family:xs_flow", "seconds": 34.3,
         "params": {"gap5_max": 0.34, "vol_z_min": 0.5}}) + "\n", encoding="utf-8")
    reps = json.loads(_get(http + "/api/reports")[1])["reports"]
    assert [r["job_id"] for r in reps] == ["deadbeef0001"]
    assert reps[0]["params"] == {"gap5_max": 0.34, "vol_z_min": 0.5}


def test_a_report_can_be_opened_by_file_name_not_only_by_job_id(http, tmp_path, monkeypatch):
    """CLI `--out` 的产物不是任务 id ⇒ 必须能按**文件名**打开，且遍历必须被拒。

    「复制命令」与页面做的事已经逐项对齐（--control-runs/--note/同一个命名空间），
    但那条命令产出的文件名不是 12 位任务 id，用不了 `/report/<id>`。少了这条路由，
    复制出来的命令跑完，报告照样在页面上看不见。
    """
    monkeypatch.setattr(srv_mod, "RESULTS", tmp_path)
    made = tmp_path / "web_report_cli_xs_flow_2025-01-01_2025-10-13.html"
    made.write_text("<!doctype html><title>回测明细 · family:xs_flow</title>"
                    "<div class='sub'>窗口 2025-01-01 → 2025-10-13</div>", encoding="utf-8")
    code, body = _get(http + "/report-file/" + made.name)
    assert code == 200 and "xs_flow" in body.decode()
    # 它也会出现在「历史报告」里，并且带 file 字段（表格用它拼链接）
    reps = json.loads(_get(http + "/api/reports")[1])["reports"]
    assert [r["file"] for r in reps] == [made.name]
    # 遍历与越界：一律拒（文件名里不许有 /，也不许跳出 results/）
    assert _get(http + "/report-file/../../etc/passwd")[0] == 404
    assert _get(http + "/report-file/sub/dir.html")[0] == 404
    assert _get(http + "/report-file/nope.html")[0] == 404


def test_control_arm_can_be_switched_on_and_lands_in_the_report(offline):
    """运气对照必须能从平台开起来，并**写进报告** —— 那是把运气与本事分开的唯一判词。

    它此前在平台上结构性开不了（`_summary` 把 ctl_* 写死成 NaN），报告因此永远打印
    "本次未跑"。默认仍然是关（每次约 25 秒），但必须**有入口**。
    """
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1], "control_runs": 20})["id"])
    assert st["status"] == "done", st.get("error")
    rep = (offline / f"web_report_{st['id']}.html").read_text(encoding="utf-8")
    assert "位于随机分布的" in rep, "开了对照却没在报告里出判词"
    assert "运气对照本次未跑" not in rep
    assert any("运气对照" in s["name"] for s in st["steps"]), "步骤里没写这一步"
    led = [json.loads(x) for x in (offline / "web_runs.jsonl").read_text().splitlines()]
    assert led[-1]["control_runs"] == 20, "台账要记下这次有没有跑对照"
    # 对照的**读数**也要到面板与台账（此前只有报告里有 —— 判词是在面板上写的，
    # 写的时候手上没有这个数，就只能抄别的候选的）
    assert st["metrics"]["ctl_runs"] == 20
    assert 0.0 <= st["metrics"]["ctl_pct"] <= 100.0, "百分位没进 metrics ⇒ 面板上没有判据④的数"
    assert led[-1]["ctl_pct"] == st["metrics"]["ctl_pct"], "台账与面板各说一套"


def test_default_run_says_the_control_was_not_run(offline):
    """默认关时，报告必须**明说未跑**（不能留个 NaN 让它渲染成 "nan 百分位"）。"""
    st = _wait(jobs.submit({"kind": "preset", "id": "us_reversal_dip_v1",
                            "lo": WIN[0], "hi": WIN[1]})["id"])
    rep = (offline / f"web_report_{st['id']}.html").read_text(encoding="utf-8")
    assert "运气对照本次未跑" in rep and "nan 百分位" not in rep
