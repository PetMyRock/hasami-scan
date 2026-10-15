"""Candidate lines around a final crop box (generated scans only)."""
import numpy as np
import pytest

from hasami.core.autocrop import AutoCropSettings, detect
from hasami.core.geometry import CropBox
from hasami.demo import BookletOptions, synth_booklet


def _stack_scan(dpi=300):
    """A white page on a grey lid with a page-stack band beyond its right edge: 6 page edges,
    0.4 mm apart (as tests/test_booklet.py)."""
    mm = dpi / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 205, np.uint8)
    x0, y0, x1, y1 = (int(20 * mm), int(20 * mm), int(130 * mm), int(180 * mm))
    img[y0:y1, x0:x1] = 246
    for i in range(6):
        a = x1 + int(i * 0.4 * mm)
        img[y0 + 2:y1 - 2, a:a + int(0.4 * mm)] = 238 - 3 * i
        img[y0 + 2:y1 - 2, a:a + 1] = 200
    img[y0:y1, x1:x1 + 1] = 215
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, (x0, y0, x1, y1), mm


def _page_box(x0, y0, x1, y1):
    return CropBox.from_xywh(x0, y0, x1 - x0, y1 - y0)


def test_paper_side_lines_include_the_top_page_edge_and_the_stack():
    from hasami.core.sidelines import SideLine, side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    lines = side_lines(img, _page_box(x0, y0, x1, y1), 300)
    assert set(lines) == {"top", "right", "bottom", "left"}
    right = [ln.offset_mm for ln in lines["right"]]
    assert right == sorted(right)
    assert any(abs(x) < 0.3 for x in right)  # the top page's own edge, under the side
    assert sum(0.2 < x < 3.0 for x in right) >= 2  # stack sheets outside it
    assert any(abs(ln.offset_mm) < 0.3 for ln in lines["left"])  # the page's edge on the lid
    assert all(isinstance(ln, SideLine) and 0.0 < ln.support <= 1.0 and ln.kind in ("layer", "faint", "strong")
               for ln in lines["right"])


def test_columns_off_the_image_are_ignored():
    from hasami.core.sidelines import side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    img = np.ascontiguousarray(img[:, : x1 + int(3 * mm)])  # the image ends 3 mm right of the top page
    right = [ln.offset_mm for ln in side_lines(img, _page_box(x0, y0, x1, y1), 300)["right"]]
    assert any(abs(x) < 0.3 for x in right)
    assert all(x < 2.9 for x in right)  # nothing at the image's own edge


def test_a_side_off_the_image_has_no_lines():
    from hasami.core.sidelines import side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    box = _page_box(x0, y0, img.shape[1] + 30, y1)  # the right side lies beyond the image
    assert side_lines(img, box, 300)["right"] == []


def test_spine_side_gets_crease_lines():
    from hasami.core.sidelines import side_lines

    mm = 300 / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 245, np.uint8)
    crease = int(100 * mm)
    img[:, crease - 2:crease + 3] = 170  # the fold's dark crease line
    box = CropBox.from_xywh(int(20 * mm), int(20 * mm), crease + 2 * mm - int(20 * mm), int(160 * mm))
    box.meta["keep_sides"] = [(0, 1)]  # the right side is the spine
    right = side_lines(img, box, 300)["right"]
    assert right and all(ln.kind == "crease" for ln in right)
    assert any(abs(ln.offset_mm + 2.0) < 0.3 for ln in right)  # the crease 2 mm inside the side


def test_grey_16_bit_images_work():
    from hasami.core.sidelines import side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    grey16 = img[..., 1].astype(np.uint16) * 257
    right = side_lines(grey16, _page_box(x0, y0, x1, y1), 300)["right"]
    assert any(abs(ln.offset_mm) < 0.3 for ln in right)


def test_lines_closer_than_the_merge_distance_keep_the_best_supported():
    from hasami.core.sidelines import SideLine, _merge

    out = _merge([SideLine(1.0, 0.0, 0.7, "faint"), SideLine(1.1, 0.0, 0.9, "layer"),
                  SideLine(2.0, 0.0, 0.6, "strong"), SideLine(2.1, 0.0, 0.6, "layer")])
    assert [(ln.offset_mm, ln.kind) for ln in out] == [(1.1, "layer"), (2.1, "layer")]


def test_detect_is_unchanged_by_side_lines():
    from hasami.core.sidelines import side_lines

    sc = synth_booklet(BookletOptions(), seed=0)
    st = AutoCropSettings(mode="booklet")
    before = detect(sc.image, st, dpi=200)
    for b in before:
        side_lines(sc.image, b, 200)
    after = detect(sc.image, st, dpi=200)
    assert [(b.cx, b.cy, b.w, b.h, b.angle) for b in before] == [(b.cx, b.cy, b.w, b.h, b.angle) for b in after]


def test_a_line_survives_the_trip_through_the_image():
    from hasami.core.sidelines import SideLine, from_image, to_image

    box = CropBox(400, 300, 300, 200, 3.0)
    for name in ("top", "right", "bottom", "left"):
        p, dd = to_image(SideLine(-1.7, 0.004, 0.9, "layer"), box, name, 7.87)
        off, slope = from_image(p, dd, box, name, 7.87)
        assert off == pytest.approx(-1.7, abs=1e-9) and slope == pytest.approx(0.004, abs=1e-9)
    p, dd = to_image(SideLine(0.5, 0.0, 0.9, "layer"), box, "right", 7.87, inset_mm=0.25)
    assert from_image(p, dd, box, "right", 7.87)[0] == pytest.approx(0.25)  # moved in by the inset


def test_normal_hit():
    from hasami.core.sidelines import normal_hit

    assert normal_hit((10.0, 0.0), (0.0, 1.0), (0.0, 5.0), (1.0, 0.0)) == pytest.approx(10.0)
    assert normal_hit((10.0, 0.0), (1.0, 0.0), (0.0, 5.0), (1.0, 0.0)) is None  # parallel


def test_a_spread_box_gets_its_pages_lines_on_its_own_sides():
    from hasami.core.sidelines import box_lines

    sc = synth_booklet(BookletOptions(fully_visible=True), seed=0)
    [spread] = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0, booklet_output="spread"), dpi=200)
    assert "spread_pages" in spread.meta
    lines = box_lines(sc.image, spread, 200)
    for name in ("top", "right", "bottom", "left"):
        assert any(abs(ln.offset_mm) < 0.3 for ln in lines[name]), name  # the paper edge under each side
    assert all(ln.kind != "crease" for side in lines.values() for ln in side)  # the spine is no spread side


def test_box_lines_of_a_page_are_its_side_lines():
    from hasami.core.sidelines import box_lines, side_lines

    sc = synth_booklet(BookletOptions(), seed=0)
    [page] = detect(sc.image, AutoCropSettings(mode="booklet"), dpi=200)
    assert box_lines(sc.image, page, 200) == side_lines(sc.image, page, 200)


def test_signals_are_measured_from_the_paper_edge_under_the_side():
    from hasami.core.sidelines import SideLine, signals

    lines = [SideLine(0.25, 0.0, 0.9, "layer"), SideLine(-1.35, 0.0, 0.95, "strong"), SideLine(2.25, 0.0, 0.7, "faint")]
    sig = signals(lines, margin_mm=0.25, spine=False)
    assert sig["own_support"] == 0.9
    assert sig["near_strong"] == pytest.approx(1.6) and sig["near_strong_at"] == pytest.approx(-1.6)
    assert sig["crowd"] == 3.0 and sig["no_crease"] == 0.0
    none = signals([], 0.25, spine=False)
    assert none["near_strong"] == 99.0 and none["own_support"] == 0.0 and none["crowd"] == 0.0


def test_a_spine_without_a_crease_line_at_it():
    from hasami.core.sidelines import SideLine, signals

    assert signals([SideLine(1.25, 0.0, 0.8, "crease")], 0.25, spine=True)["no_crease"] == 1.0
    assert signals([SideLine(0.45, 0.0, 0.8, "crease")], 0.25, spine=True)["no_crease"] == 0.0


def test_check_reason_names_the_first_condition_that_fires():
    from hasami.core.sidelines import SideLine, check_reason, signals

    sig = signals([SideLine(0.25, 0.0, 0.9, "layer"), SideLine(-1.35, 0.0, 0.95, "strong")], 0.25, spine=False)
    assert check_reason(sig, ()) is None
    assert check_reason(sig, (("near_strong", 2.0),)) == "another edge 1.6 mm further in"
    assert check_reason(sig, (("own_support", 0.85),)) is None
    assert check_reason(sig, (("own_support", 0.95), ("near_strong", 2.0))) == "the edge here is faint"
    assert check_reason(sig, (("crowd", 2),)) == "2 edges within 3 mm"
    spine = signals([], 0.25, spine=True)
    assert check_reason(spine, (("no_crease", 0.5),)) == "no clear crease at the fold"


def test_no_flag_rule_until_the_measurement_sets_one():
    from hasami.core import sidelines

    assert sidelines.CHECK_RULE == ()
