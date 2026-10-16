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


def _session(tmp_path, enabled=True, **input_kw):
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.input.scan_dpi = input_kw.get("scan_dpi", 150)
    st.crop.mode = "booklet"
    st.output.folder = str(tmp_path / "scans")
    st.cases.enabled = enabled
    st.cases.folder = str(tmp_path / "cases")
    return Session(st)


def _bottom_out(b, mm=4.0):
    return replace(b, h=b.h + mm, cy=b.cy + mm / 2, manual=True, meta=dict(b.meta))


def test_auto_boxes_get_unique_ids(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    ids = [b.meta["auto_id"] for b in s.boxes_mm]
    assert ids and len(set(ids)) == len(ids)
    assert [a.meta["auto_id"] for a in s.auto_mm] == ids
    s.run_preview()
    assert not set(ids) & {b.meta["auto_id"] for b in s.boxes_mm}  # never reused


def test_a_corrected_scan_writes_one_case(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    notes = []
    outcomes = s.run_scan(status=notes.append)
    assert outcomes[0].paths  # the scan itself saved as usual
    [path] = cases.list_cases(tmp_path / "cases")
    doc = cases.read_case(path)
    assert doc["reason"] == "corrected" and doc["scanner"]["id"] == "demo:booklet"
    moved = [e for e in doc["edits"] if e["kind"] == "moved"]
    assert len(moved) == 1 and moved[0]["sides_mm"]["bottom"] == pytest.approx(4.0, abs=0.01)
    assert doc["preview"]["dpi"] == 50 and doc["scan"]["dpi"] == 150
    assert s.last_case_path == path and s.case_note.startswith(f"Saved crop case {path.name} (bottom edge 4.0 mm out")
    assert s.case_note in notes


def test_an_uncorrected_scan_writes_nothing_until_saved(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    s.run_scan()
    assert cases.list_cases(tmp_path / "cases") == [] and s.case_note == ""
    p = s.save_last_case()
    assert cases.read_case(p)["reason"] == "saved"
    assert s.save_last_case() == p  # once
    assert cases.list_cases(tmp_path / "cases") == [p]


def test_setting_off_keeps_and_writes_nothing(tmp_path):
    s = _session(tmp_path, enabled=False)
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    s.run_scan()
    assert s.last_case is None and s.save_last_case() is None
    assert not (tmp_path / "cases").exists()


def test_a_case_that_cannot_be_written_does_not_break_the_scan(tmp_path):
    s = _session(tmp_path)
    (tmp_path / "blocked").write_text("a file, not a folder")
    s.settings.cases.folder = str(tmp_path / "blocked" / "cases")
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    notes = []
    outcomes = s.run_scan(status=notes.append)
    assert outcomes[0].paths and all(Path(p).exists() for p in outcomes[0].paths)
    assert any(n.startswith("Could not save the crop case:") for n in notes)


def test_the_scan_is_kept_at_400_dpi_at_most(tmp_path):
    from hasami import cases

    s = _session(tmp_path, scan_dpi=600)
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    s.run_scan()
    [path] = cases.list_cases(tmp_path / "cases")
    doc = cases.read_case(path)
    assert doc["scan"]["scan_dpi"] == 600 and doc["scan"]["dpi"] == 400
    with Image.open(path / "scan.jpg") as im:
        assert round(im.info["dpi"][0]) == 400
        w_mm = im.size[0] / 400 * 25.4
    assert w_mm == pytest.approx(doc["scan"]["area_mm"][2], abs=0.5)


def test_a_feeder_batch_writes_one_case(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    dev = s.open()
    one = dev.scan
    dev.scan_batch = lambda req, progress=None, cancel=None: iter([one(req), one(req)])
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    assert len(s.run_scan()) == 2
    assert len(cases.list_cases(tmp_path / "cases")) == 1


def test_off_mode_never_writes_a_case_by_itself(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.settings.crop.mode = "off"
    s.run_preview()
    drawn = CropBox.from_xywh(20, 20, 60, 80, manual=True)
    drawn.meta["drawn"] = True
    s.boxes_mm = [drawn]
    s.run_scan()
    assert cases.list_cases(tmp_path / "cases") == []
    assert s.save_last_case() is not None  # still possible by hand


def test_a_drawn_box_carried_to_the_next_page_is_not_a_new_correction(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    drawn = CropBox.from_xywh(5, 5, 20, 20, manual=True)
    drawn.meta["drawn"] = True
    s.boxes_mm.append(drawn)
    s.run_scan()
    assert len(cases.list_cases(tmp_path / "cases")) == 1  # drawn on this page: a correction
    s.run_preview()  # next page: the drawn box stays
    assert any(b.meta.get("carried") for b in s.boxes_mm)
    s.run_scan()
    assert len(cases.list_cases(tmp_path / "cases")) == 1  # carried over: not a new one


def test_redetect_keeps_the_edited_boxs_auto_original(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    first = s.auto_mm[0]
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    s.detect_on_preview()  # a Crop setting changed / Ctrl+R: edited boxes are kept
    assert first.meta["auto_id"] in [a.meta["auto_id"] for a in s.auto_mm]
    kept = next(a for a in s.auto_mm if a.meta["auto_id"] == first.meta["auto_id"])
    assert (kept.cy, kept.h) == (first.cy, first.h)  # the auto original, not the edit
    ids = [a.meta["auto_id"] for a in s.auto_mm]
    assert len(set(ids)) == len(ids)


def test_opened_images_never_write_a_case(tmp_path):
    # File > Open image goes through the file: backend; its id is the local path (not for Discord zips).
    import cv2

    from hasami import cases
    from hasami.demo import SynthOptions, synth_scan

    sc = synth_scan(SynthOptions(n_photos=2), seed=3)
    png = tmp_path / "my image.png"
    cv2.imwrite(str(png), cv2.cvtColor(sc.image, cv2.COLOR_RGB2BGR))
    s = _session(tmp_path)
    s.settings.input.device = f"file:{png}"
    s.settings.crop.mode = "multi"
    s.run_preview()
    assert s.boxes_mm
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    s.run_scan()
    assert s.last_case is None and s.save_last_case() is None
    assert cases.list_cases(tmp_path / "cases") == []


def test_a_new_preview_drops_the_last_scans_case(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    s.run_scan()
    assert s.last_case is not None
    s.run_preview()  # the next page: "Save this case" would no longer mean what is on screen
    assert s.last_case is None and s.save_last_case() is None


def test_only_case_folders_are_listed(tmp_path):
    from hasami import cases

    p = cases.write_case(tmp_path, _data())
    (tmp_path / "My holiday photos").mkdir()
    (tmp_path / "2026-10-15").mkdir()
    assert cases.list_cases(tmp_path) == [p]


def test_delete_case_removes_only_case_files(tmp_path):
    from hasami import cases

    p = cases.write_case(tmp_path, _data())
    cases.delete_case(p)
    assert not p.exists()
    q = cases.write_case(tmp_path, _data())
    (q / "my notes.txt").write_text("keep me")
    with pytest.raises(OSError):
        cases.delete_case(q)
    assert (q / "my notes.txt").read_text() == "keep me"


def test_summary_names_sides_as_shown_after_rotation():
    from hasami import cases

    a = _auto()
    edits = cases.diff_boxes([a], [_edit(a, h=a.h + 4.2, cy=a.cy + 2.1)])
    assert cases.summary(edits, turns=1) == "left edge 4.2 mm out"  # 90 deg right: the bottom shows on the left
    assert cases.summary(edits, turns=2) == "top edge 4.2 mm out"


def test_the_zip_readme_says_who_can_see_a_posted_case():
    from hasami import cases

    assert "members of that channel can see" in cases.README
    assert "never published" not in cases.README


def test_a_case_records_flags_and_snaps(tmp_path):
    from hasami import cases

    d = _data()
    d.auto[0].meta["check_sides"] = {"bottom": "another edge 1.6 mm further in"}
    d.final[0].meta.update(snapped_sides=["bottom"])
    doc = cases.read_case(cases.write_case(tmp_path, d))
    assert doc["auto"][0]["check_sides"] == {"bottom": "another edge 1.6 mm further in"}
    assert doc["final"][0]["snapped_sides"] == ["bottom"]
    plain = cases.read_case(cases.write_case(tmp_path, _data()))
    assert "check_sides" not in plain["auto"][0] and "snapped_sides" not in plain["final"][0]
