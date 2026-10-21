"""Booklet pages: the fold side cropped clear of the crease (``crop.fold_clear``, opt-in; generated scans)."""
import math

import numpy as np

from hasami.core import foldclear
from hasami.core.autocrop import AutoCropSettings, detect, refine
from hasami.core.geometry import CropBox

DPI = 200.0
MM = DPI / 25.4
HALF = math.sqrt(math.log(2))  # exp(-(x/w)^2) is at half depth at x = w * HALF


def _page(lean_mm=1.2, depth=70.0, width_mm=0.5, side_out_mm=0.0, seed=0):
    """A page on white paper, 20 mm to the fold at 100 mm across, 20-180 mm down.  The fold's crease is a dark
    valley (``depth`` grey levels, exp(-(x/w)^2)) on 100 mm at the top, ``lean_mm`` further in at the bottom.
    The page box's right side (the fold, kept) lies ``side_out_mm`` outside the crease's top end.  Returns the
    image, the box and where the crease's page side reaches half depth at the 95 % point down the side (px)."""
    rng = np.random.default_rng(seed)
    W, H = int(130 * MM), int(200 * MM)
    y0, y1 = 20 * MM, 180 * MM
    xs, ys = np.arange(W)[None, :], np.arange(H)[:, None]
    centre = 100 * MM - lean_mm * MM * (ys - y0) / (y1 - y0)
    g = 235.0 - depth * np.exp(-((xs - centre) / (width_mm * MM)) ** 2) + rng.normal(0, 1.5, (H, W))
    img = np.clip(np.repeat(g[..., None], 3, 2), 0, 255).astype(np.uint8)
    box = CropBox.from_xywh(20 * MM, y0, (80 + side_out_mm) * MM, y1 - y0)
    box.meta = {"keep_sides": [(0, 1)], "spine": "crease"}
    clear = (100 - 0.95 * lean_mm - width_mm * HALF) * MM
    return img, box, clear


def _right(b: CropBox) -> float:
    return b.bounding_rect()[2]


def test_the_fold_side_moves_in_clear_of_the_crease():
    img, box, clear = _page()
    out = foldclear.clear_of_crease(img, box, DPI, margin_mm=0.0)
    assert abs(_right(out) - clear) <= 0.1 * MM, ((_right(out) - clear) / MM)
    x0, y0, _, y1 = out.bounding_rect()
    bx0, by0, _, by1 = box.bounding_rect()
    assert (x0, y0, y1) == (bx0, by0, by1)  # only the fold side moves
    assert abs(out.meta["fold_clear"]["0+1"] - (_right(box) - _right(out)) / MM) < 0.01


def test_the_margin_lies_inside_the_creases_edge():
    img, box, clear = _page()
    out = foldclear.clear_of_crease(img, box, DPI, margin_mm=0.25)
    assert abs(_right(out) - (clear - 0.25 * MM)) <= 0.1 * MM, ((_right(out) - clear) / MM)


def test_without_a_crease_the_page_is_left_alone():
    img, box, _ = _page(depth=0.0)
    out = foldclear.clear_of_crease(img, box, DPI, margin_mm=0.25)
    assert (out.cx, out.cy, out.w, out.h, out.angle) == (box.cx, box.cy, box.w, box.h, box.angle)
    assert "fold_clear" not in out.meta


def test_a_faint_shade_is_not_a_crease():
    img, box, _ = _page(depth=6.0)
    out = foldclear.clear_of_crease(img, box, DPI, margin_mm=0.25)
    assert "fold_clear" not in out.meta and _right(out) == _right(box)


def test_a_page_without_a_fold_side_is_left_alone():
    img, box, _ = _page()
    box.meta["keep_sides"] = []
    out = foldclear.clear_of_crease(img, box, DPI, margin_mm=0.25)
    assert "fold_clear" not in out.meta and _right(out) == _right(box)


def test_the_move_is_capped():
    img, box, _ = _page(lean_mm=0.0, side_out_mm=2.4)  # the crease's edge 2.8 mm inside the side
    out = foldclear.clear_of_crease(img, box, DPI, margin_mm=0.0)
    assert abs((_right(box) - _right(out)) / MM - foldclear.CAP_MM) < 0.01


def test_undo_puts_the_fold_side_back():
    img, box, _ = _page()
    back = foldclear.undo(foldclear.clear_of_crease(img, box, DPI, margin_mm=0.25), DPI)
    assert np.allclose(back.corners(), box.corners(), atol=1e-6)
    assert "fold_clear" not in back.meta


# ------------------------------------------------- in detect / refine
def _spread():
    from hasami.demo import BookletOptions, synth_booklet

    sc = synth_booklet(BookletOptions(fully_visible=True, fold_line=1.0, gutter_depth=(0.3, 0.3),
                                      gutter_width=(0.01, 0.01)), seed=1)
    return sc.image


def test_detect_moves_only_the_fold_sides_and_refine_keeps_them():
    img = _spread()
    off = detect(img, AutoCropSettings(mode="booklet"), dpi=DPI)
    st = AutoCropSettings(mode="booklet", fold_clear=True)
    on = detect(img, st, dpi=DPI)
    assert len(on) == len(off) == 2
    for a, b in zip(off, on):
        assert "fold_clear" in b.meta
        [(key, moved)] = b.meta["fold_clear"].items()
        assert 0.1 < moved <= foldclear.CAP_MM
        assert np.allclose(foldclear.undo(b, DPI).corners(), a.corners(), atol=1e-6)
    # refine starts from the crease (as with the setting off) and moves the fold side about as far again
    plain = refine(img, off, AutoCropSettings(mode="booklet"), dpi=DPI)
    for b, p, r in zip(on, plain, refine(img, on, st, dpi=DPI)):
        assert np.allclose(foldclear.undo(r, DPI).corners(), p.corners(), atol=1e-6)
        [(key, moved)] = b.meta["fold_clear"].items()
        assert abs(r.meta["fold_clear"][key] - moved) <= 0.1


def test_off_and_the_spread_output_are_unchanged():
    img = _spread()
    for st in (AutoCropSettings(mode="booklet"), AutoCropSettings(mode="booklet", booklet_output="spread")):
        base = detect(img, st, dpi=DPI)
        st.fold_clear = st.booklet_output == "spread"
        same = detect(img, st, dpi=DPI)
        assert [b.corners().tolist() for b in base] == [b.corners().tolist() for b in same]
        assert all("fold_clear" not in b.meta for b in same)


def test_thick_book_pages_move_at_detect_and_on_refine():
    from test_thickbook import DPI as TDPI, MM as TMM, _spread as thick_spread

    img, _ = thick_spread()
    xs = np.arange(img.shape[1])[None, :, None]
    crease = 1 - 0.45 * np.exp(-((xs - 140 * TMM) / (0.4 * TMM)) ** 2)  # the fold's dark crease
    img = np.clip(img * crease, 0, 255).astype(np.uint8)
    st = AutoCropSettings(mode="booklet", profile="epson-v600", fold_clear=True)
    on = detect(img, st, dpi=TDPI)
    assert len(on) == 2 and all(b.meta.get("spine") == "thick" and "fold_clear" in b.meta for b in on)
    again = refine(img, on, st, dpi=TDPI)
    for b, r in zip(on, again):
        assert np.abs(r.corners() - b.corners()).max() <= 0.1 * TMM


# ------------------------------------------------- the setting in the app
def test_cli_turns_the_setting_on():
    from hasami.cli import _apply, build_parser
    from hasami.settings import AppSettings

    args = build_parser().parse_args(["crop", "x.tif", "--crop", "booklet", "--fold-clear"])
    assert _apply(args, AppSettings()).crop.fold_clear
    assert not _apply(build_parser().parse_args(["crop", "x.tif"]), AppSettings()).crop.fold_clear


def test_crop_tab_offers_it_for_separate_pages_only():
    import pytest

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        pytest.skip(str(e))
    QApplication.instance() or QApplication([])
    from hasami.gui.settings_panel import SettingsPanel
    from hasami.settings import AppSettings

    st = AppSettings()
    panel = SettingsPanel(st)
    w = panel._widgets[("crop", "fold_clear")]
    st.crop.mode = "booklet"
    panel._update_enabled()
    assert w.isEnabled()
    st.crop.booklet_output = "spread"
    panel._update_enabled()
    assert not w.isEnabled()
    st.crop.mode, st.crop.booklet_output = "multi", "pages"
    panel._update_enabled()
    assert not w.isEnabled()


def test_flags_judge_the_fold_where_auto_crop_found_it(tmp_path, monkeypatch):
    from hasami import workflow
    from hasami.core import sidelines
    from hasami.settings import AppSettings

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.crop.mode = "booklet"
    st.output.folder = str(tmp_path / "scans")
    s = workflow.Session(st)
    s.run_preview()
    page = CropBox(400.0, 500.0, 300.0, 400.0, 0.0)
    page.meta = {"keep_sides": [(0, 1)], "spine": "crease", "top_page": {}, "fold_clear": {"0+1": 1.27}}
    monkeypatch.setattr(workflow.autocrop, "detect", lambda *a, **k: [page])
    seen = []
    monkeypatch.setattr(sidelines, "box_lines", lambda image, box, dpi: seen.append(box) or {})
    monkeypatch.setattr(workflow.Session, "_check_sides", lambda self, b, lines: {})
    s.detect_on_preview()
    assert np.allclose(seen[0].corners(), foldclear.undo(page, s.preview.dpi).corners())


def test_crop_cases_record_the_setting(tmp_path):
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.input.scan_dpi = 100
    st.crop.mode = "booklet"
    st.crop.fold_clear = True
    st.cases.enabled = True
    st.output.folder = str(tmp_path / "scans")
    s = Session(st)
    s.run_preview()
    s.run_scan()
    assert s.last_case is not None and s.last_case.crop["fold_clear"] is True


def test_a_float_image_works_too():
    img, box, clear = _page()
    out = foldclear.clear_of_crease(img.astype(np.float64) / 255.0, box, DPI, margin_mm=0.0)
    assert abs(_right(out) - clear) <= 0.1 * MM


def test_a_scan_ending_past_the_crease_still_moves():
    # The app scans the crop area + 3 mm: from a box already clear of the crease the image ends ~2 mm past it.
    img, box, clear = _page()
    img = np.ascontiguousarray(img[:, :int(102 * MM)])
    out = foldclear.clear_of_crease(img, box, DPI, margin_mm=0.0)
    assert abs(_right(out) - clear) <= 0.1 * MM, out.meta.get("fold_clear")
