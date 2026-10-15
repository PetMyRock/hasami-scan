"""Sides moved by hand stay where they are at scan time; the others are still refined (generated scans)."""
from dataclasses import replace

import numpy as np
import pytest

from hasami.core.autocrop import AutoCropSettings, detect, refine
from hasami.core.geometry import SIDE_NAMES, CropBox
from hasami.demo import BookletOptions, synth_booklet


def _pos(box, ref, name):
    """Where ``box``'s side ``name`` lies along ``ref``'s normal, from ``ref``'s side (px, + = out)."""
    from hasami.core.geometry import side_frames

    (m, _), (mr, nr) = side_frames(box)[name], side_frames(ref)[name]
    return float((m - mr) @ nr)


def _moved(box, name, d):
    """``box`` with side ``name`` moved ``d`` px outwards (a hand edit)."""
    ux, uy = box.axes
    axis = {"top": -uy, "bottom": uy, "left": -ux, "right": ux}[name]
    wide = name in ("left", "right")
    c = np.array([box.cx, box.cy]) + axis * d / 2
    return replace(box, cx=c[0], cy=c[1], w=box.w + d * wide, h=box.h + d * (not wide), meta=dict(box.meta))


def _edited(box, name, d, pinned=None):
    b = _moved(box, name, d)
    return replace(b, manual=True, meta={**box.meta, "auto_id": "a1", "pinned_sides": pinned or [name]})


def test_pinned_sides_stay_and_the_others_follow_the_refined_box():
    from hasami.core.geometry import pin_sides

    given = CropBox(100, 80, 60, 40, 0.0, manual=True, meta={"auto_id": "a1"})
    refined = CropBox(101, 79, 64, 40, 0.0, meta={"top_page": {}})
    out = pin_sides(given, refined, ["left", "top"])
    x0, y0, x1, y1 = out.bounding_rect()
    assert (x0, y0) == (70.0, 60.0)  # pinned: exactly where they were
    assert x1 == pytest.approx(133.0) and y1 == pytest.approx(99.0)  # refined
    assert out.angle == 0.0 and out.manual and out.meta == {"auto_id": "a1"}


def test_a_turned_box_keeps_its_angle_and_its_pinned_side():
    from hasami.core.geometry import pin_sides, side_frames

    given = CropBox(200, 150, 80, 50, 2.0)
    refined = CropBox(201, 150.5, 82, 51, 2.3)
    out = pin_sides(given, refined, ["left"])
    assert out.angle == pytest.approx(2.0)
    assert _pos(out, given, "left") == pytest.approx(0.0, abs=1e-9)
    for name in ("top", "right", "bottom"):
        m_r = side_frames(refined)[name][0]
        m_o, n_o = side_frames(out)[name]
        assert float((m_r - m_o) @ n_o) == pytest.approx(0.0, abs=1e-9), name


def test_all_four_pinned_is_the_given_box():
    from hasami.core.geometry import pin_sides

    given = CropBox(200, 150, 80, 50, 2.0)
    out = pin_sides(given, CropBox(203, 149, 90, 44, -1.0), SIDE_NAMES)
    assert (out.cx, out.cy, out.w, out.h, out.angle) == pytest.approx((200, 150, 80, 50, 2.0))


def _card(dpi=150):
    mm = dpi / 25.4
    img = np.full((int(150 * mm), int(200 * mm), 3), 235, np.uint8)
    x0, y0, x1, y1 = int(50 * mm), int(40 * mm), int(140 * mm), int(100 * mm)
    img[y0:y1, x0:x1] = (90, 60, 140)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(x0 - 0.5, y0 - 0.5, x1 - x0, y1 - y0), mm


def test_a_pinned_side_stays_while_the_others_are_refined():
    img, card, mm = _card()
    st = AutoCropSettings(mode="multi", margin_mm=0)
    auto = card.translated(2.0, 1.5)  # auto-crop's box, a little off
    [ref] = refine(img, [auto], st, dpi=150)
    edited = _edited(auto, "left", 15.0)  # the left side dragged 15 px out
    [out] = refine(img, [edited], st, dpi=150)
    assert _pos(out, edited, "left") == pytest.approx(0.0, abs=1e-6)
    for name in ("top", "right", "bottom"):
        assert abs(_pos(out, ref, name)) <= 0.05 * mm, name
    assert out.angle == edited.angle and out.manual and out.meta["pinned_sides"] == ["left"]


def test_a_pinned_booklet_side_stays_while_the_others_are_refined():
    from hasami.core.sidelines import spine_names

    sc = synth_booklet(BookletOptions(), seed=0)
    st = AutoCropSettings(mode="booklet", margin_mm=0)
    [page] = detect(sc.image, st, dpi=200)
    name = next(n for n in SIDE_NAMES if n not in spine_names(page))
    mm = 200 / 25.4
    [ref] = refine(sc.image, [page], st, dpi=200)
    edited = _edited(page, name, 1.0 * mm)
    [out] = refine(sc.image, [edited], st, dpi=200)
    assert _pos(out, edited, name) == pytest.approx(0.0, abs=1e-6)
    for other in SIDE_NAMES:
        if other != name:
            # top_page refits every side from the box it starts on: a side moved 1 mm moves the others'
            # strips and its slope bins, so they vary by up to ~0.15 mm (synthetic truth: no worse)
            assert abs(_pos(out, ref, other)) <= 0.2 * mm, other
    assert out.angle == edited.angle


def test_a_pinned_spread_side_stays():
    sc = synth_booklet(BookletOptions(fully_visible=True), seed=1)
    st = AutoCropSettings(mode="booklet", margin_mm=0, booklet_output="spread")
    [spread] = detect(sc.image, st, dpi=200)
    mm = 200 / 25.4
    [ref] = refine(sc.image, [spread], st, dpi=200)
    edited = _edited(spread, "left", 1.0 * mm)
    [out] = refine(sc.image, [edited], st, dpi=200)
    assert "spread_pages" in out.meta
    assert _pos(out, edited, "left") == pytest.approx(0.0, abs=1e-6)
    assert abs(_pos(out, ref, "right")) <= 0.05 * mm


def test_drawn_unpinned_and_fully_pinned_boxes_are_untouched():
    img, card, mm = _card()
    st = AutoCropSettings(mode="multi", margin_mm=0)
    off = card.translated(2.0, 1.5)
    drawn = replace(off, manual=True, meta={"drawn": True, "pinned_sides": ["left"]})
    unpinned = replace(off, manual=True, meta={"auto_id": "a1"})  # edited before pins existed
    whole = replace(off, manual=True, meta={"auto_id": "a2", "pinned_sides": list(SIDE_NAMES)})
    out = refine(img, [drawn, unpinned, whole], st, dpi=150)
    assert [(b.cx, b.cy, b.w, b.h, b.angle) for b in out] == [(off.cx, off.cy, off.w, off.h, off.angle)] * 3
