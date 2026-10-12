"""Booklet mode: only the page that is completely on the glass."""
import numpy as np
import pytest

from hasami.core.autocrop import AutoCropSettings, detect
from hasami.core.geometry import match_boxes
from hasami.demo import BookletOptions, synth_booklet


OPTIONS = {
    "white": BookletOptions(),
    "dark-lid": BookletOptions(background="dark"),
    "horizontal": BookletOptions(spine="horizontal"),
    "artbook": BookletOptions(content="artbook"),
    "full-spread": BookletOptions(fully_visible=True),
}
# Known trade-off for the real Epson V600 scans (the maintainer's priority): the
# complete page's blank white margin lies at the image edge, where the
# shadows of a propped-up V600 lid look the same, so the side is pulled in to
# the print.
_MARGIN = "blank white margin at the image edge is taken for lid shadow"
KNOWN = {("horizontal", 0): _MARGIN, ("horizontal", 1): _MARGIN, ("horizontal", 2): _MARGIN}


@pytest.mark.parametrize("name,seed", [
    pytest.param(name, seed, marks=pytest.mark.xfail(strict=True, reason=KNOWN[(name, seed)]))
    if (name, seed) in KNOWN else (name, seed)
    for name in OPTIONS for seed in range(3)
])
def test_booklet_returns_the_complete_page(name, seed):
    sc = synth_booklet(OPTIONS[name], seed=seed)
    boxes = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0), dpi=200)
    assert len(boxes) == len(sc.truth), (name, seed)
    m = match_boxes(boxes, sc.truth)
    assert min(iou for _, _, iou in m) > 0.95, (name, seed)


def test_booklet_spine_side_is_kept_on_refine():
    sc = synth_booklet(BookletOptions(), seed=0)
    boxes = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0), dpi=200)
    assert boxes[0].meta.get("keep_sides")


def test_single_cover_is_one_page():
    # A portrait cover lying on a white lid with nothing cut off stays whole,
    # even with a strong horizontal band in its artwork.
    img = np.full((1400, 1000, 3), 240, np.uint8)
    img[200:1200, 200:800] = (120, 60, 160)
    img[560:600, 200:800] = (30, 20, 40)  # dark band across the cover
    boxes = detect(img, AutoCropSettings(mode="booklet", margin_mm=0), dpi=150)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0].bounding_rect()
    assert abs(x0 - 200) < 8 and abs(x1 - 800) < 8 and abs(y0 - 200) < 8 and abs(y1 - 1200) < 8


# ------------------------------------------------- spread output, top page
def _union_iou(box, truth):
    """IoU of a box with the rectangle around all truth boxes (same angle as the box)."""
    from hasami.core.booklet import spread_box
    from hasami.core.geometry import polygon_iou

    return polygon_iou(box, spread_box(list(truth)))


@pytest.mark.parametrize("seed", range(3))
def test_spread_output_is_one_image_over_both_pages(seed):
    sc = synth_booklet(BookletOptions(fully_visible=True), seed=seed)
    pages = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0), dpi=200)
    spread = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0, booklet_output="spread"), dpi=200)
    assert len(pages) == 2 and len(spread) == 1
    assert _union_iou(spread[0], sc.truth) > 0.95
    assert len(spread[0].meta["spread_pages"]) == 2
    assert not spread[0].meta.get("keep_sides")  # no spine side: both outer edges are paper edges


def test_spread_output_with_one_complete_page_is_that_page():
    sc = synth_booklet(BookletOptions(), seed=0)
    boxes = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0, booklet_output="spread"), dpi=200)
    assert len(boxes) == 1 and "spread_pages" not in boxes[0].meta
    assert min(iou for _, _, iou in match_boxes(boxes, sc.truth)) > 0.95


def test_spread_survives_preview_to_scan_refine():
    from hasami.core.autocrop import refine
    from hasami.core.geometry import polygon_iou

    sc = synth_booklet(BookletOptions(fully_visible=True), seed=1)
    st = AutoCropSettings(mode="booklet", margin_mm=0, booklet_output="spread")
    import cv2

    small = cv2.resize(sc.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    pre = detect(small, st, dpi=100)
    fin = refine(sc.image, [b.scaled(2.0) for b in pre], st, dpi=200)
    assert len(fin) == 1 and "spread_pages" in fin[0].meta
    assert polygon_iou(fin[0], detect(sc.image, st, dpi=200)[0]) > 0.98


def _stack_scan(dpi=300):
    """A white page on a grey lid with a page-stack band beyond its right edge: 6 page edges,
    0.4 mm apart, a little darker than the page (the pages under the top one)."""
    mm = dpi / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 205, np.uint8)
    x0, y0, x1, y1 = (int(20 * mm), int(20 * mm), int(130 * mm), int(180 * mm))
    img[y0:y1, x0:x1] = 246
    for i in range(6):
        a = x1 + int(i * 0.4 * mm)
        img[y0 + 2:y1 - 2, a:a + int(0.4 * mm)] = 238 - 3 * i
        img[y0 + 2:y1 - 2, a:a + 1] = 200  # the shaded edge of each page
    img[y0:y1, x1:x1 + 1] = 215  # the top page's own edge shadow
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, (x0, y0, x1, y1), mm


def test_top_page_leaves_the_page_stack_out():
    from hasami.core.geometry import CropBox
    from hasami.core.toppage import top_page

    img, (x0, y0, x1, y1), mm = _stack_scan()
    outer = x1 + 6 * 0.4 * mm  # the box as a plain edge search would find it: outermost layer
    box = CropBox.from_xywh(x0, y0, outer - x0, y1 - y0)
    out, shifts = top_page(img, box, 300)
    bx0, _by0, bx1, _by1 = out.bounding_rect()
    assert abs(bx1 - x1) < 0.3 * mm, (bx1, x1)
    assert abs(bx0 - x0) < 0.3 * mm
    assert shifts[(0, 1)] < -2.0


def test_top_page_keeps_the_spine_side_and_a_clean_edge():
    from hasami.core.geometry import CropBox
    from hasami.core.toppage import top_page

    img, (x0, y0, x1, y1), mm = _stack_scan()
    box = CropBox.from_xywh(x0, y0, x1 + 6 * 0.4 * mm - x0, y1 - y0)
    out, shifts = top_page(img, box, 300, keep=[(0, 1)], snap_spine=False)
    assert shifts[(0, 1)] == 0.0  # the kept side (as a spine would be) does not move
    assert abs(out.bounding_rect()[2] - box.bounding_rect()[2]) < 2


def test_spine_side_snaps_to_the_crease():
    from hasami.core.geometry import CropBox
    from hasami.core.toppage import top_page

    mm = 300 / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 245, np.uint8)
    crease = int(100 * mm)
    img[:, crease - 2:crease + 3] = 170  # the fold's dark crease line
    img[:, crease - int(4 * mm):crease] = np.linspace(245, 215, int(4 * mm)).astype(np.uint8)[None, :, None]
    box = CropBox.from_xywh(int(20 * mm), int(20 * mm), crease + 2 * mm - int(20 * mm), int(160 * mm))
    out, shifts = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1), (0, -1)], snap=[(0, 1)])
    assert abs(out.bounding_rect()[2] - crease) < 0.3 * mm, (out.bounding_rect(), crease)
    assert shifts[(1, -1)] == shifts[(0, -1)] == 0.0


def test_booklet_mode_crops_a_cover_at_the_top_page():
    img, (x0, y0, x1, y1), mm = _stack_scan(200)
    boxes = detect(img, AutoCropSettings(mode="booklet", margin_mm=0), dpi=200)
    assert len(boxes) == 1
    bx0, by0, bx1, by1 = boxes[0].bounding_rect()
    assert abs(bx1 - x1) < 0.4 * mm and abs(bx0 - x0) < 0.4 * mm, (boxes[0].bounding_rect(), (x0, x1))


def test_page_order_left_or_right_page_first():
    sc = synth_booklet(BookletOptions(fully_visible=True), seed=0)
    ltr = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0), dpi=200)
    rtl = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0, booklet_rtl=True), dpi=200)
    assert len(ltr) == len(rtl) == 2
    assert ltr[0].cx < ltr[1].cx  # left page is crop 1
    assert rtl[0].cx > rtl[1].cx  # right page is crop 1 (books read right to left)


def test_page_order_follows_the_turned_preview():
    # spine across the image; turned a quarter clockwise in the app, the lower page is on the left
    sc = synth_booklet(BookletOptions(fully_visible=True, spine="horizontal"), seed=0)
    st = AutoCropSettings(mode="booklet", margin_mm=0, rotate=1)
    ltr = detect(sc.image, st, dpi=200)
    assert len(ltr) == 2 and ltr[0].cy > ltr[1].cy
    st.booklet_rtl = True
    rtl = detect(sc.image, st, dpi=200)
    assert rtl[0].cy < rtl[1].cy


def test_crease_is_not_taken_at_the_image_edge():
    # The app scans only the crop area + 3 mm: the image ends inside the crease search.  Black
    # beyond the image looked like a valley and the fold moved 3 mm out (owner's test scan 0004).
    from hasami.core.geometry import CropBox
    from hasami.core.toppage import top_page

    mm = 300 / 25.4
    rng = np.random.default_rng(0)
    x_side = int(90 * mm)
    w = x_side + int(round(2.4 * mm))
    crease = x_side - int(round(0.6 * mm))
    shade = 40 * np.exp(-((np.arange(w) - crease) / (0.15 * mm)) ** 2)
    img = (np.full((int(110 * mm), w, 3), 228.0) - shade[None, :, None])
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    box = CropBox.from_xywh(int(10 * mm), int(10 * mm), x_side - int(10 * mm), int(90 * mm))
    out, _ = top_page(img, box, 300, keep=[(0, 1)])
    assert abs((out.bounding_rect()[2] - crease) / mm) < 0.2

