"""族装载器的契约测试 —— 装载违规必须 fail-fast，不能静默跳过。

为什么这几条值得测：族定义体决定**预注册网格**（`GRID`）与 mask。少一条校验的后果
不是"报错难看"，而是"新族默认不在护栏里/网格可以事后改"——本仓库反复吃亏的那一类。
"""
import pytest

from research import family_registry as R

GOOD = '''
ID = "zz"
DOC = """机制叙述"""
def mask(ind, p):
    return ind["close"] > 0
FIXED = {}
GRID = {"a": [1, 2]}
'''


def _w(tmp_path, name, body):
    p = tmp_path / f"{name}.py"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_one_returns_the_declared_fields(tmp_path):
    got = R.load_one(_w(tmp_path, "zz", GOOD))
    assert got["id"] == "zz" and callable(got["mask"])
    assert got["grid"] == {"a": [1, 2]} and got["needs_regime"] is False
    assert got["control"] is None and got["score_negated"] is False
    assert got["desc"] == "机制叙述"
    assert got["legacy_names"] == () and got["reference"] is False


def test_id_must_match_file_name(tmp_path):
    with pytest.raises(ValueError, match="文件名"):
        R.load_one(_w(tmp_path, "other", GOOD))


def test_mask_is_required_and_callable(tmp_path):
    with pytest.raises(ValueError, match="mask"):
        R.load_one(_w(tmp_path, "zz", GOOD.replace("def mask", "def nope")))


def test_empty_grid_fails_fast(tmp_path):
    with pytest.raises(ValueError, match="GRID"):
        R.load_one(_w(tmp_path, "zz", GOOD.replace('GRID = {"a": [1, 2]}', "GRID = {}")))


def test_load_dir_skips_private_files(tmp_path):
    _w(tmp_path, "_private", GOOD.replace('ID = "zz"', 'ID = "_private"'))
    _w(tmp_path, "zz", GOOD)
    models: dict[str, dict] = {}
    R.load_dir(models, tmp_path)
    assert list(models) == ["zz"]


def test_load_dir_rejects_duplicate_ids(tmp_path):
    # 重复只可能发生在**两次装载之间**（同一个目录里出现重名是撞不上的：ID 必须先
    # 与文件名相符，同名文件只可能有一个）。所以这里直接预置一条同 ID 的记录。
    _w(tmp_path, "zz", GOOD)
    with pytest.raises(ValueError, match="重复"):
        R.load_dir({"zz": {"id": "zz"}}, tmp_path)


def test_legacy_names_and_reference_are_carried(tmp_path):
    body = GOOD + '\nLEGACY_NAMES = ("_mask_zz_old",)\nREFERENCE = True\nSCORE_COL = "mom120"\n'
    got = R.load_one(_w(tmp_path, "zz", body))
    assert got["legacy_names"] == ("_mask_zz_old",)
    assert got["reference"] is True
    assert got["score_col"] == "mom120"
