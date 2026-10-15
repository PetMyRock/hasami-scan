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
