from dataclasses import replace
from pathlib import Path

import pytest

from hasami.core.geometry import CropBox


def _auto(i=1, **kw):
    b = CropBox(100.0, 150.0, 120.0, 180.0, 0.0, 0.9, **kw)
    b.meta["auto_id"] = f"a{i}"
    return b


def _edit(b, **kw):
    return replace(b, manual=True, meta=dict(b.meta), **kw)


def test_case_settings_are_off_by_default_and_load_back(tmp_path):
    from hasami.settings import AppSettings, load_settings, save_settings

    st = AppSettings()
    assert st.cases.enabled is False and st.cases.folder == ""
    st.cases.enabled, st.cases.folder = True, str(tmp_path / "cc")
    save_settings(st, tmp_path / "s.json")
    back = load_settings(tmp_path / "s.json")
    assert back.cases.enabled is True and back.cases.folder == str(tmp_path / "cc")


def test_folder_of_uses_the_setting_or_documents():
    from hasami import cases
    from hasami.settings import CaseSettings

    assert cases.folder_of(CaseSettings(folder="X:/somewhere")) == Path("X:/somewhere")
    d = cases.folder_of(CaseSettings())
    assert d == cases.default_folder() and d.parts[-2:] == ("Hasami Scan", "Crop cases")


def test_an_untouched_box_is_unchanged():
    from hasami import cases

    a = _auto()
    edits = cases.diff_boxes([a], [_edit(a)])
    assert [e["kind"] for e in edits] == ["unchanged"]
    assert not cases.corrected(edits)


def test_moved_sides_are_signed_outwards():
    from hasami import cases

    a = _auto()
    f = _edit(a, h=a.h + 4.0, cy=a.cy + 2.0)  # bottom 4 mm further out
    e = cases.diff_boxes([a], [f])[0]
    assert e["kind"] == "moved"
    assert e["sides_mm"]["bottom"] == pytest.approx(4.0, abs=1e-6)
    assert all(abs(e["sides_mm"][s]) < 1e-6 for s in ("top", "left", "right"))
    g = _edit(a, w=a.w - 3.0, cx=a.cx - 1.5)  # right side 3 mm in
    assert cases.diff_boxes([a], [g])[0]["sides_mm"]["right"] == pytest.approx(-3.0, abs=1e-6)
    assert cases.corrected(cases.diff_boxes([a], [f]))


def test_tiny_moves_and_turns_stay_unchanged_but_a_turn_counts():
    from hasami import cases

    a = _auto()
    assert cases.diff_boxes([a], [_edit(a, cx=a.cx + 0.05)])[0]["kind"] == "unchanged"
    assert cases.diff_boxes([a], [_edit(a, angle=0.04)])[0]["kind"] == "unchanged"
    e = cases.diff_boxes([a], [_edit(a, angle=-0.7)])[0]
    assert e["kind"] == "moved" and e["angle_deg"] == pytest.approx(-0.7)


def test_deleted_and_drawn_boxes():
    from hasami import cases

    a1, a2 = _auto(1), _auto(2)
    drawn = CropBox.from_xywh(5, 5, 20, 20, manual=True)
    drawn.meta["drawn"] = True
    carried = CropBox.from_xywh(40, 5, 20, 20, manual=True)
    carried.meta.update(drawn=True, carried=True)
    edits = cases.diff_boxes([a1, a2], [_edit(a1), drawn, carried])
    assert [(e["id"], e["kind"]) for e in edits] == [("a1", "unchanged"), ("a2", "deleted"), (None, "drawn")]


def test_summary_wording():
    from hasami import cases

    a, b = _auto(1), _auto(2)
    edits = cases.diff_boxes([a, b], [_edit(a, h=a.h + 4.2, cy=a.cy + 2.1), CropBox.from_xywh(1, 1, 5, 5, manual=True)])
    assert cases.summary(edits) == "bottom edge 4.2 mm out; 1 box deleted; 1 box drawn"
    assert cases.summary(cases.diff_boxes([a], [_edit(a)])) == "no corrections"


import json
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image


def _data(**kw):
    from hasami import cases

    rng = np.random.default_rng(0)
    a = _auto()
    d = dict(preview=rng.integers(0, 255, (300, 220, 3), np.uint8), preview_dpi=50.0,
             preview_area_mm=(0.0, 0.0, 216.0, 297.0),
             scan=rng.integers(0, 255, (500, 400, 3), np.uint8), scan_dpi_saved=150.0, scan_dpi=150.0,
             scan_area_mm=(10.0, 20.0, 67.7, 84.7), auto=[a], final=[_edit(a, h=a.h + 4.0, cy=a.cy + 2.0)],
             scanner_id="demo:booklet", scanner_name="Demo booklet", profile="default",
             crop={"mode": "booklet", "rotate": 0}, created=datetime(2026, 10, 15, 14, 32, 10))
    d.update(kw)
    return cases.CaseData(**d)


def test_write_and_read_a_case(tmp_path):
    from hasami import cases

    path = cases.write_case(tmp_path, _data())
    assert path == tmp_path / "2026-10-15_143210"
    assert sorted(p.name for p in path.iterdir()) == ["case.json", "preview.jpg", "scan.jpg"]
    doc = cases.read_case(path)
    assert doc["format"] == 1 and doc["reason"] == "corrected" and doc["scanner"]["name"] == "Demo booklet"
    assert doc["preview"] == {"dpi": 50.0, "area_mm": [0.0, 0.0, 216.0, 297.0], "file": "preview.jpg"}
    assert doc["scan"]["dpi"] == 150.0 and doc["scan"]["scan_dpi"] == 150.0
    assert doc["auto"][0]["id"] == "a1" and doc["auto"][0]["box_mm"] == [100.0, 150.0, 120.0, 180.0, 0.0]
    assert doc["final"][0]["box_mm"] == [100.0, 152.0, 120.0, 184.0, 0.0]
    assert doc["edits"][0]["kind"] == "moved" and doc["edits"][0]["sides_mm"]["bottom"] == 4.0
    with Image.open(path / "scan.jpg") as im:
        assert im.size == (400, 500) and round(im.info["dpi"][0]) == 150
    with Image.open(path / "preview.jpg") as im:
        assert im.size == (220, 300) and round(im.info["dpi"][0]) == 50
    assert cases.list_cases(tmp_path) == [path]


def test_two_cases_in_one_second_get_distinct_names(tmp_path):
    from hasami import cases

    p1, p2 = cases.write_case(tmp_path, _data()), cases.write_case(tmp_path, _data(reason="saved"))
    assert (p1.name, p2.name) == ("2026-10-15_143210", "2026-10-15_143210_2")
    assert cases.list_cases(tmp_path) == [p2, p1]  # newest first


def test_partial_folders_and_the_share_folder_are_not_cases(tmp_path):
    from hasami import cases

    (tmp_path / "2026-10-15_150000.partial").mkdir()
    (tmp_path / cases.SHARE_DIR).mkdir()
    p = cases.write_case(tmp_path, _data())
    assert cases.list_cases(tmp_path) == [p]
    assert cases.list_cases(tmp_path / "missing") == []


def test_read_case_refuses_broken_and_newer_cases(tmp_path):
    from hasami import cases

    p = cases.write_case(tmp_path, _data())
    doc = json.loads((p / "case.json").read_text(encoding="utf-8"))
    (p / "case.json").write_text(json.dumps(dict(doc, format=2)), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        cases.read_case(p)
    (p / "case.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        cases.read_case(p)
    (p / "case.json").unlink()
    with pytest.raises(ValueError):
        cases.read_case(p)


def test_downscale_to_400_dpi():
    from hasami import cases

    img = np.zeros((1200, 800, 3), np.uint8)
    small, dpi = cases.downscale(img, 1200.0)
    assert dpi == 400.0 and small.shape[:2] == (400, 267)
    same, dpi = cases.downscale(img, 300.0)
    assert same is img and dpi == 300.0


def test_write_case_handles_16_bit_grey_scans(tmp_path):
    from hasami import cases

    grey16 = (np.arange(500 * 400, dtype=np.uint32).reshape(500, 400) % 65536).astype(np.uint16)
    p = cases.write_case(tmp_path, _data(scan=grey16))
    with Image.open(p / "scan.jpg") as im:
        assert im.mode == "L" and im.size == (400, 500)


def test_pack_zips_each_case_with_a_readme_and_splits_big_ones(tmp_path):
    from hasami import cases

    p1 = cases.write_case(tmp_path / "cc", _data())
    p2 = cases.write_case(tmp_path / "cc", _data(created=datetime(2026, 10, 15, 15, 0, 0)))
    zips = cases.pack([p1, p2], tmp_path / "out")
    assert [z.name for z in zips] == ["hasami-case-2026-10-15_143210.zip", "hasami-case-2026-10-15_150000.zip"]
    with zipfile.ZipFile(zips[0]) as zf:
        assert sorted(zf.namelist()) == ["2026-10-15_143210/case.json", "2026-10-15_143210/preview.jpg",
                                         "2026-10-15_143210/scan.jpg", "README.txt"]
        assert "crop case" in zf.read("README.txt").decode()
    split = cases.pack([p1], tmp_path / "out2", limit_mb=0.0001)
    assert [z.name for z in split] == ["hasami-case-2026-10-15_143210-part1.zip",
                                       "hasami-case-2026-10-15_143210-part2.zip"]
    with zipfile.ZipFile(split[0]) as zf:
        assert sorted(zf.namelist()) == ["2026-10-15_143210/scan.jpg", "README.txt"]
