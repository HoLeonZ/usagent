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
    heads = tuple(h for h in SKELETON if f"<h2>{h}</h2>" in doc)
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


def test_family_rows_render_the_third_state_not_a_checkmark(chain):
    """族的行在报告里必须是「·」而不是「✓」—— 一个**假 ✓** 比没有这一列更坏。"""
    doc, _, _ = _render(chain, family="str")
    assert 'class="">·</td>' in doc, "族的行没有渲染成'不适用'"
