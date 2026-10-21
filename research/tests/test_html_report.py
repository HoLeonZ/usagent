"""HTML 报告**公共能力**的测试 —— 离线（确定性合成市场），任何机器可跑。

它钉住三件事：
  1. **只有策略变，形式不变**：把同一条渲染路径喂给两种完全不同的策略（signal 型 preset
     与族型配置），产出的 CSS / 章节骨架 / 卡片结构**逐字相同**，只有标题与判读依据不同；
  2. 族的条件清单**来自掩码自己**（AST 读出它实际读了哪些列）—— 新增一个族不用改渲染层；
  3. `selfcheck` 有牙（元素个数 / NaN 坐标 / 外链），并钉住"呈现只在 html_report 里定义一次"。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db as dbmod  # noqa: E402
from research import families, html_report as H, indicators, presets  # noqa: E402
from research import panel as panel_mod, render_report as RR             # noqa: E402
from research.tests import fixture_market as FX                          # noqa: E402

WIN = ("2024-01-01", "2025-01-01")     # 合成市场的可评估窗（见 fixture_market）


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """建合成库 → 载面板与指标（与离线行为锁同一套装置，不读真库）。"""
    tmp = tmp_path_factory.mktemp("render_html")
    db_file = FX.build(tmp / "fixture.db")
    old = dbmod.DB_PATH
    dbmod.DB_PATH = db_file
    try:
        panel = panel_mod.load_panel_for_preset(presets.get_preset("us_reversal_dip_v1"))
        yield panel, indicators.compute_all(panel)
    finally:
        dbmod.DB_PATH = old


def _render(chain, *, preset="us_reversal_dip_v1", family=None, explain=None, note=""):
    panel, ind = chain
    res, eq, ind2, bench, summary, auto_explain = RR.run(
        *WIN, preset, 0, family=family, note=note)
    doc = H.render_html(res["trades"], eq, ind2, summary["signal"], bench, summary,
                        explain=explain if explain is not None else auto_explain)
    return doc, len(res["trades"]), summary


SKELETON = ("摘要", "权益曲线 vs 基准", "配置", "全部交易", "逐笔明细")


def _skeleton(doc: str) -> tuple:
    """与策略**无关**的那部分形式：章节骨架 + CSS 长度 + 一个 h1。

    注意**不能**把卡片数放进来：卡片数 = 成交笔数，本来就该随策略变；
    "形式不变"说的是"每笔一张卡"，而不是"卡的数量一样"（第一版就是这么写错的）。
    """
    heads = tuple(h for h in SKELETON if re.search(rf"<h2[^>]*>{re.escape(h)}</h2>", doc))
    css = re.search(r"<style>(.*?)</style>", doc, re.S).group(1)
    return heads, len(css), doc.count("<h1>")


def test_only_the_strategy_changes_the_form_stays(chain):
    d_preset, n_p, s_p = _render(chain, preset="us_reversal_dip_v1")
    d_family, n_f, s_f = _render(chain, family="str")
    assert n_p > 0 and n_f > 0, "合成市场上两条路径都要出成交，否则是空断言"
    # 形式：章节骨架 / CSS 长度 / 每笔一张 K 线卡 / 一个 h1 —— 完全一致
    assert _skeleton(d_preset) == _skeleton(d_family), "两条策略的**形式**不同了"
    assert _skeleton(d_preset)[0] == SKELETON, "章节骨架缺项"
    # 卡片数 = 各自的成交数（"每笔一张卡"这条形式对两条策略都成立）
    assert d_preset.count('class="kchart"') == n_p
    assert d_family.count('class="kchart"') == n_f
    # 策略：标题与判读依据必须不同（族没有 VERDICT ⇒ 报告会明说"没有声明负面证据"）
    assert "us_reversal_dip_v1" in d_preset and "family:str" in d_family
    assert s_p["verdict"] and s_f["verdict"] is None
    assert "（本策略没有声明负面证据" in d_family
    # VERDICT 里的 `**` 会被 inline_md 转成 <b>（否则首屏就会打出星号）⇒
    # 断言落在「去掉标签后的纯文本」上，这样它既不受标签影响、又不绑定具体措辞。
    plain_doc = re.sub(r"<[^>]+>", "", d_preset)
    plain_verdict = s_p["verdict"].replace("**", "").replace("`", "")
    assert plain_verdict[:40] in plain_doc, "策略自己的 VERDICT 没进报告"
    assert "<b>不得用于实盘。</b>" in d_preset, "VERDICT 的粗体没被转换"


def test_report_navigation_and_print_are_present(chain):
    """报告自己要走得动、也带得走（审计实测原本两样都没有）。

    2026-10-18 实测的原始状态：报告 docHeight 87794px / 91 张逐笔卡，5 个 h2 的 id
    **全为 null**、`回到顶部` 0 处、href 只有 `#t{i}`（跳下去回不来）；
    `@media` 规则数 0 —— 深色正文（#e6e8ee）会被打到白纸上。
    这两个缺口都属于"看完了拿不走 / 走不回去"，所以当场钉住。
    """
    doc, n, _ = _render(chain)

    # 1. 每个章节都有 id，且**目录里的每个锚点都命中**（不是"有目录"就算过）
    ids = set(re.findall(r'<h2 id="([^"]+)"', doc))
    assert len(ids) == len(SKELETON), f"章节 id 不全：{sorted(ids)}"
    toc = re.findall(r'class="toc"[^>]*>(.*?)</div>', doc, re.S)[0]
    hrefs = re.findall(r'href="#([^"]+)"', toc)
    assert len(hrefs) == len(SKELETON), f"目录没覆盖全部章节：{hrefs}"
    assert set(hrefs) <= ids | {"top"}, f"目录里有死锚点：{set(hrefs) - ids - {'top'}}"

    # 2. 逐笔卡各自有回去的路（每笔两条：回总表 / 回顶部）
    assert doc.count('class="totop"') == 2 * n, "不是每张逐笔卡都能回到总表"
    assert 'class="totopcard"' in doc, "长页面缺一个常驻「回到顶部」"

    # 3. 打印：必须有 @media（否则白纸上打深色正文），且打印时隐藏浮层与目录
    assert "@media print" in doc, "报告没有打印样式 —— 另存 PDF / 打印会不可读"
    assert re.search(r"@media print\{.*?\.totopcard,\.toc,\.fbar\{display:none\}", doc, re.S), \
        "打印样式没把浮层、目录与筛选条藏起来"

    # 6. 逐笔明细的筛选：只出"只看某一类出场"的单选（radio），计数与卡片数对得上
    if n >= 10:
        assert 'class="fbar"' in doc and 'name="xf"' in doc, "没有筛选条"
        # "全部"那一项不算原因（它的 id 是 xf-all，别把它当成一个出场原因）
        reasons = [r for r in re.findall(r'id="xf-([a-z_]+)"', doc) if r != "all"]
        assert len(reasons) >= 2, f"过滤项太少：{reasons}"
        assert f'<label for="xf-all">全部 {n}</label>' in doc, "「全部」缺总笔数"
        for r_ in reasons:
            # 每一项都要带上"这一类有几笔"，否则读者点了不知道会剩几张卡
            assert re.search(rf'id="xf-{r_}"[^>]*>\s*<label for="xf-{r_}">{r_} (\d+)</label>', doc), \
                f"{r_} 的筛选按钮缺计数"
        # 每张逐笔卡都带 data-reason（筛选靠它；用属性而不是 class —— selfcheck 按
        # class="sl"/class="tp" 数元素，多一个就报错）
        # ⚠️ 不能数全文档里的 `data-reason="` —— CSS 的 7 条过滤规则里也各有一个（实测 43 vs 36）
        cards_with_reason = re.findall(r'<div class="card" id="t\d+" data-reason="([a-z_]+)">', doc)
        assert len(cards_with_reason) == n, f"不是每张卡都能被筛到：{len(cards_with_reason)} != {n}"
        assert set(cards_with_reason) <= set(reasons), "有卡片的出场原因没有对应的筛选项"

    # 4. 加了这些仍然**不许**有脚本或外链（自包含是硬约束，selfcheck 也查一遍）
    assert H.selfcheck(doc, n) == []


def test_exit_reason_summary_matches_the_trades_it_claims_to_summarize(chain):
    """报告里的「按出场原因」必须与它下面那张逐笔表**逐项对得上**。

    为什么单独钉它：这张汇总是"读报告的人用来看整体"的那一块，一旦算错（分组漏了、
    合计拿权益总收益冒充逐笔平均、占比不归一），读者会拿着一个错的结论去调策略，
    而报告看起来毫无异常。所以测试**自己重算一遍**，不调用被测函数。
    """
    panel, ind = chain
    res, eq, ind2, bench, summary, _ = RR.run(*WIN, "us_reversal_dip_v1", 0)
    trades = res["trades"]
    assert len(trades) > 0, "合成市场上要有成交，否则是空断言"
    rows = H.exit_reason_rows(trades)

    # 1. 覆盖完整、不重不漏：每一笔都被分到恰好一类
    assert sum(r["n"] for r in rows) == len(trades)
    assert {r["reason"] for r in rows} == set(trades["exit_reason"].unique())
    # 2. 占比归一
    assert abs(sum(r["share"] for r in rows) - 1.0) < 1e-9
    # 3. 每类的四个数都要与**直接重算**一致（不是"看着像"）
    for r in rows:
        g = trades[trades["exit_reason"] == r["reason"]]
        assert r["n"] == len(g)
        assert abs(r["share"] - len(g) / len(trades)) < 1e-12
        assert abs(r["mean_net_return"] - float(g["net_return"].mean())) < 1e-12
        assert abs(r["sum_net_pnl"] - float(g["net_pnl"].sum())) < 1e-6
        assert abs(r["win_rate"] - float((g["net_pnl"] > 0).mean())) < 1e-12
    # 4. 从多到少（"哪一类出场最多"要一眼可见）
    assert [r["n"] for r in rows] == sorted((r["n"] for r in rows), reverse=True)

    # 5. 报告里要真的画出这张表、并且**只给出现过的原因**配图例
    doc = H.render_html(trades, eq, ind2, summary["signal"], bench, summary,
                        explain=H.condition_rows)
    assert H.selfcheck(doc, len(trades)) == []
    assert "按出场原因（从多到少）" in doc
    for reason in {r["reason"] for r in rows}:
        assert f'<code>{reason}</code>' in doc, f"{reason} 没进报告"
    absent = set(H.EXIT_LEGEND) - {r["reason"] for r in rows}
    for reason in absent:
        assert f'<code>{reason}</code>=' not in doc, f"{reason} 没出现过却配了图例"


def test_note_overrides_the_missing_verdict_fallback(chain):
    """族没有 VERDICT：报告**必须**明说"没有声明负面证据"，除非调用方给了 --note。

    这条是 §5 的机器化形式：负面证据与参数必须同时记录 —— 一条新策略如果没写，
    它的报告会当场把这件事印在头顶，而不是显得"看起来很干净"。
    """
    d0, _, _ = _render(chain, family="str")
    assert "没有声明负面证据" in d0
    d1, _, _ = _render(chain, family="str", note="本次只是演示渲染")
    assert "本次只是演示渲染" in d1 and "没有声明负面证据" not in d1


def test_family_explain_rows_come_from_the_mask_itself(chain):
    """族的条件清单必须来自**掩码源码**（AST），而不是一张手维护的表。"""
    panel, ind = chain
    mask = families.MASK["str"]
    cols = H._mask_read_columns(mask)
    assert cols, "AST 没能从掩码里读出任何列"
    row = ind[(ind["symbol"] == ind["symbol"].iloc[0])].iloc[0]
    rows = H.family_explain_rows(ind, row["symbol"], row["date"], {},
                                mask=mask, params={"drop5": 0.12})
    codes = [c for c, *_ in rows]
    for c in cols:
        assert c in codes, f"掩码读了 {c}，但清单里没有"
    assert "param.drop5" in codes, "本次用的参数必须列出来（否则读者不知道阈值是多少）"
    # 诚实边界：族**不打勾**（打勾要重算一遍掩码实现，那是第二条路径）⇒ 第五项必须是 None
    assert all(ok is None for *_, ok in rows), "族的条件清单出现了 ✓/✗"
    assert all(thr == "—" for _, _, _, thr, _ in rows)


def test_unknown_family_fails_fast(chain):
    with pytest.raises(H.RenderError, match="未知策略族"):
        RR.run(*WIN, "us_reversal_dip_v1", 0, family="nope")


def test_selfcheck_has_teeth(chain):
    doc, n, _ = _render(chain)
    assert H.selfcheck(doc, n) == []
    # 少一张卡 ⇒ 元素计数对不上（这类"几何全对、元素整批消失"的失败只有数数才抓得到）
    problems = H.selfcheck(doc.replace('class="kchart"', "", 1), n)
    assert any("K线卡" in p for p in problems)
    # 外链 / 脚本 ⇒ 不再自包含
    assert H.selfcheck(doc.replace("</body>", '<img src="http://x/y.png"></body>'), n)
    assert H.selfcheck(doc.replace("</body>", "<script>x()</script></body>"), n)


def test_presentation_is_defined_in_exactly_one_place():
    """呈现只在 `html_report` 里定义一次；`render_report` 只是入口。

    这条是"公共能力"这个说法的**机器可检形式**：只要允许第二份实现存在，
    它迟早与第一份分叉（本仓库已有前车之鉴：两条入口读两个面板）。
    """
    rr = (ROOT / "research" / "render_report.py").read_text(encoding="utf-8")
    for name in ("def render_html(", "def candles_svg(", "def equity_svg(",
                 "def selfcheck(", "def condition_rows(", "CSS = "):
        assert name not in rr, f"render_report 里又出现了 {name} —— 呈现被抄了第二份"
    assert "from research.html_report import" in rr
    # 写死"30% 目标"的那段（只对某一条 preset 成立）已由**策略自己的 VERDICT** 取代
    assert "_target_note" not in rr
    assert "verdict" in rr


def test_verdict_and_note_are_escaped_then_minimally_formatted():
    """判读依据那一行插的是**策略的 VERDICT / --note**：先转义、再做最小行内 markdown。

    2026-10-17 修：它此前是**裸插**（本文件其余文案全走 html.escape）—— 一个 `<` 就能
    弄坏文档；而 VERDICT 里本来就写着 `**粗体**`，不转换就会原样打出星号（首屏截图上
    一眼可见）。只做两种行内标记：`**粗体**` 与 `` `代码` ``。
    """
    f = H.inline_md
    assert f("**不得用于实盘。** 见 `overlay.py`") == "<b>不得用于实盘。</b> 见 <code>overlay.py</code>"
    assert f("a<b & c>d") == "a&lt;b &amp; c&gt;d"
    assert "<script" not in f("<script>alert(1)</script>")


def test_a_hostile_note_cannot_break_the_document(chain):
    """带 `<script>` 的 note 进报告后**仍然自包含**（转义生效的证据，不只是断言字符串）。

    用**族**来测：preset 有自己的 VERDICT，note 本来就会被它优先顶掉（那是有意的优先级）。
    """
    doc, n, _ = _render(chain, family="str", note="<script>alert(1)</script>")
    assert H.selfcheck(doc, n) == []
    assert "&lt;script&gt;" in doc


def test_the_report_says_which_excess_it_is_printed(chain):
    """报告头条那个「跑赢/跑输 X pp」是**总收益 − 基准**，报告必须说清它不是哪个量。

    为什么：平台的结果面板里有一格叫「曝光匹配超额」（把闲置资金按等权池计息），
    而报告里的这个差**没有**那一步计息 —— 两者**不是同一个量**，甚至可能方向相反。
    报告此前只写「同口径比较」（说的是同窗），读者会把这个差当成策略的边缘。
    现在报告明说它是什么，并指出另一个口径的数在平台面板里（别互相印证）。
    """
    doc, n, _ = _render(chain)
    assert H.selfcheck(doc, n) == []
    assert "曝光匹配超额" in doc, "报告没提另一个口径的数"
    assert "不要互相印证" in doc, "报告没说清两个数的关系"


def test_the_summary_table_says_which_excess_it_shows(chain):
    """摘要表里那一格不能只写「超额」—— 它是「总收益 − 基准」，不是曝光匹配口径。

    平台的结果面板里那一格叫「曝光匹配超额」（把闲置资金按等权池计息），报告这一格不是它
    （两者方向甚至可能相反）。此前报告只写「超额」，而平台的 tooltip 里明写着"与报告摘要里
    那个 tot−bench 的「超额」不是同一个量" ⇒ 同一件事两处各叫一个名字、只有一端说明。
    """
    doc, n, _ = _render(chain)
    assert H.selfcheck(doc, n) == []
    assert "超额（总收益 − 基准）" in doc, "摘要表里的「超额」没写清是哪个量"
    assert "曝光匹配超额" in doc, "报告里没提另一个口径的数"


def test_the_config_section_says_which_parameters_it_prints(chain):
    """族型 preset 的族网格轴不能被标成 `signal.*` —— 那是**族参数**。

    实测：us_reversal_dip_v1 用的是族 str 的掩码，它的参数住在 `family_params` 里，
    而报告的「配置」那一节一律印 `signal.<键>` ⇒ **族网格的轴**被说成"信号参数"
    （平台卡片上写的是「入场族 str · 参数 {…}」—— 两处同一件事两种说法）。

    顺带：`None` 一律印成「未设」（平台口径是「未设 / 不清算」）—— 报告里印 Python 的
    `None` 会让人以为那是个取值，而不是"这个参数没设、这条条件不生效"。
    """
    doc, n, _ = _render(chain)          # 默认 preset = us_reversal_dip_v1（族型）
    assert H.selfcheck(doc, n) == []
    assert "族参数.drop5" in doc, "族型 preset 的族参数还被标成 signal.*"
    assert "signal.drop5" not in doc, "族网格的轴仍被标成 signal.*"
    assert ">None<" not in doc, "报告里还有 Python 的 None（该印「未设」）"
    assert "未设" in doc


def test_the_report_says_which_code_produced_it(chain):
    """报告页脚要带**凭证头**：时间 / 代码版本（脏树也标）/ 解释器版本（§6 归因）。

    为什么：报告是会被存下来、打印、转发的东西 —— 几个月后没人说得清它是哪份代码跑出来的。
    台账里记着 sha，但**报告本身一个字都没有**；CLI 跑的报告更是哪儿都没有。
    """
    doc, n, _ = _render(chain)
    assert H.selfcheck(doc, n) == []
    assert "代码版本" in doc, "报告页脚没有凭证头（这份结果说不清出自哪份代码）"
    assert "生成于" in doc


def test_family_rows_render_the_third_state_not_a_checkmark(chain):
    """族的行在报告里必须是「·」而不是「✓」—— 一个**假 ✓** 比没有这一列更坏。"""
    doc, _, _ = _render(chain, family="str")
    assert 'class="">·</td>' in doc, "族的行没有渲染成'不适用'"
