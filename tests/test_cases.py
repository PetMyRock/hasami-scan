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
