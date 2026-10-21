import dataclasses

import numpy as np

from hasami.core import profiles
from hasami.core.autocrop import AutoCropSettings, detect
from hasami.demo import BookletOptions, synth_booklet


def test_v600_profile_differs_from_the_default_only_in_its_own_switches():
    # The default started as a copy of the V600's tuned values; behaviour added for the V600 later
    # (KLK artboard, Oct 2026) is switched on only in its profile, so other scanners are unchanged.
    ident = {"name", "label", "match", "tuned_on"}
    own = {"min_edge_step", "crease_step", "fine_stack", "stack_by_segment", "stack_jump", "depth_mm",
           "layer_depth_mm", "fine_strong_lines", "shrink_lid_hue", "walk_after_passes",
           "cover_rim", "paper_colour_edges", "lid_all_round_split_score", "stack_shadow_l",
           "dark_edge_l", "rim_print_inside", "crease_retry_mm", "thick_sheet_band"}
    v600, default = profiles.get("epson-v600"), profiles.DEFAULT
    for f in dataclasses.fields(profiles.Profile):
        if f.name not in ident | own:
            assert getattr(v600, f.name) == getattr(default, f.name), f.name
    assert default.min_edge_step == 0.0 and not default.crease_step and not default.fine_stack
    assert v600.min_edge_step > 0 and v600.crease_step and v600.fine_stack and v600.stack_by_segment
    assert not default.stack_by_segment and not default.stack_jump and v600.stack_jump


def test_profile_from_the_scanner_name():
    assert profiles.for_scanner("EPSON Perfection V600", "EPSON Perfection V600 - EPSON Flatbed Scanner, 32-bit").name == "epson-v600"
    assert profiles.for_scanner("EPSON GT-X820").name == "epson-v600"  # the V600's Japanese name
    assert profiles.for_scanner("Canon CanoScan LiDE 300").name == "default"
    assert profiles.get("auto") is profiles.DEFAULT and profiles.get("nonsense") is profiles.DEFAULT
    assert profiles.choices()[0] == ("Automatic (from the scanner)", "auto")


def test_detect_runs_with_the_chosen_profile(monkeypatch):
    # A profile's numbers reach the engine (here: the top page's layer lines need more support).
    seen = []
    from hasami.core import toppage
    real = toppage._layer_lines

    def spy(*a, **k):
        seen.append(profiles.current().name)
        return real(*a, **k)

    monkeypatch.setattr(toppage, "_layer_lines", spy)
    monkeypatch.setitem(profiles.PROFILES, "test-scanner",
                        dataclasses.replace(profiles.DEFAULT, name="test-scanner", support=0.95))
    img = synth_booklet(BookletOptions(), seed=1).image
    detect(img, AutoCropSettings(mode="booklet", profile="test-scanner"), dpi=300)
    assert seen and set(seen) == {"test-scanner"}
    assert profiles.current() is profiles.DEFAULT  # only while detect runs


def test_session_resolves_auto_from_the_open_scanner():
    from hasami.devices.base import DeviceInfo
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    s = Session(AppSettings())
    assert s.crop_settings().profile == "default"  # no scanner open
    s.device = type("Dev", (), {"info": DeviceInfo("twain:EPSON Perfection V600", "EPSON Perfection V600")})()
    assert s.crop_settings().profile == "epson-v600"
    s.settings.crop.profile = "default"  # chosen by hand
    assert s.crop_settings().profile == "default"
    s.device = None


def _fine_stack_scan(seed: int = 0):
    """Dark textured print up to the top page's edge, a stack of 7 very thin sheets (0.5 mm apart,
    faint edges) and a green cover beyond it, on a light lid (like the KLK artboard book)."""
    import cv2

    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(120 * mm), 3), 200, np.uint8)
    x0, y0, x1, y1 = int(30 * mm), int(10 * mm), int(110 * mm), int(110 * mm)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (y1 - y0, x1 - x0)).astype(np.float32), (0, 0), 4)
    img[y0:y1, x0:x1] = (80 + 50 * tex / tex.std()).clip(0, 255)[..., None].astype(np.uint8)
    edges = [x0 - int(round(k * 0.5 * mm)) for k in range(1, 8)]
    for xa in edges:
        img[y0:y1, xa:xa + int(round(0.5 * mm))] = 236
    for xa in edges:
        img[y0:y1, xa:xa + 2] = 215
    img[y0:y1, edges[-1] - int(mm):edges[-1]] = (60, 160, 130)
    box = CropBox.from_xywh(edges[-1] - int(mm), y0, x1 - edges[-1] + int(mm), y1 - y0)
    return img, box, x0, mm


def test_v600_profile_crops_a_thin_page_stack_off():
    from hasami.core.toppage import top_page

    img, box, x0, mm = _fine_stack_scan()
    found = {}
    for name in ("default", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["default"] < -2.0  # the faint sheet edges are missed: most of the stack stays
    assert abs(found["epson-v600"]) < 0.3, found


def _art_column_scan(dark_share: float = 0.7):
    """Plain top page; the page under it sticks out 1.7 mm and shows a column of its own (full
    bleed) dark art along ``dark_share`` of the side, plain paper elsewhere; 6 thin sheets beyond
    (like the KLK artboard crop from the app)."""
    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    img = np.full((int(120 * mm), int(120 * mm), 3), 200, np.uint8)
    x0, y0, x1, y1 = int(30 * mm), int(10 * mm), int(110 * mm), int(110 * mm)
    img[y0:y1, x0:x1] = 236
    lower = x0 - int(round(1.7 * mm))
    img[y0:y1, lower:x0] = 215
    img[y0:y0 + int(dark_share * (y1 - y0)), lower:x0] = 85
    edges = [lower - int(round(k * 0.5 * mm)) for k in range(1, 7)]
    for xa in edges:
        img[y0:y1, xa:xa + int(round(0.5 * mm))] = 222
    for xa in edges + [lower, x0]:
        img[y0:y1, xa:xa + 2] = 200
    box = CropBox.from_xywh(edges[-1], y0, x1 - edges[-1], y1 - y0)
    return img, box, x0, mm


def test_v600_profile_leaves_a_lower_pages_art_column_out():
    # The strip between the lower page's edge and the top page's is half art, half paper: its
    # median colour alone decided by a pixel. The top page's edge runs along the whole side.
    from hasami.core.toppage import top_page

    img, box, x0, mm = _art_column_scan()
    found = {}
    for name in ("default", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["default"] < -1.4  # stops at the lower page's edge: its art column stays
    assert abs(found["epson-v600"]) < 0.3, found
    # Dark along the whole side: nothing tells it from print along the top page's edge.
    img, box, x0, mm = _art_column_scan(1.0)
    with profiles.use("epson-v600"):
        out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
    assert (out.bounding_rect()[0] - x0) / mm < -1.4


def _thick_stack_preview(seed: int = 0):
    """200 dpi preview of a thick thin-paper stack: green cover, sheets blurred into one band,
    lower pages' art columns at different heights, a dark column next to the top page along half
    of it (elsewhere the top page's edge is only a faint step; KLK artboard preview)."""
    import cv2

    from hasami.core.geometry import CropBox

    mm = 200 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(120 * mm), 3), 200, np.uint8)
    x0, y0, x1, y1 = int(30 * mm), int(10 * mm), int(110 * mm), int(110 * mm)
    h = y1 - y0
    img[y0:y1, x0:x1] = 236
    st = x0 - int(round(6.5 * mm))
    img[y0:y1, st:x0] = 222
    for k in range(int(6.5 / 0.45)):
        img[y0:y1, st + int(round(k * 0.45 * mm))] = 212
    for a, b, r0, r1, v in ((4.6, 3.8, 0.1, 0.5, 90), (3.6, 2.4, 0.3, 0.8, 140), (2.2, 1.3, 0.55, 0.95, 70)):
        img[y0 + int(r0 * h):y0 + int(r1 * h), x0 - int(round(a * mm)):x0 - int(round(b * mm))] = v
    img[y0:y0 + int(0.55 * h), x0 - int(round(1.2 * mm)):x0] = 70
    img[y0:y1, st - int(1.5 * mm):st] = (60, 160, 130)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    box = CropBox.from_xywh(st - int(1.5 * mm), y0, x1 - st + int(1.5 * mm), y1 - y0)
    return img, box, x0, mm


def test_v600_profile_crops_a_thick_stack_off_in_a_preview(monkeypatch):
    from hasami.core.toppage import top_page

    img, box, x0, mm = _thick_stack_preview()
    keep = [(0, 1), (1, -1), (1, 1)]
    v600 = profiles.get("epson-v600")
    monkeypatch.setitem(profiles.PROFILES, "no-jump", dataclasses.replace(v600, name="no-jump", stack_jump=False,
                                                                                     cover_rim=False))
    # (cover_rim off: once the chain starts inside the cover band, any chain crosses a paper-coloured
    # gap <= gap_mm, here to the printed line; this test is about the jump)
    monkeypatch.setitem(profiles.PROFILES, "far", dataclasses.replace(v600, name="far", stack_jump_mm=12.0,
                                                                      cover_rim=False))
    found = {}
    for name in ("no-jump", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 200, keep=keep)
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["no-jump"] < -6.0  # the whole stack and the cover stay
    assert abs(found["epson-v600"]) < 0.4, found
    # A line printed on the top page inside its edge (in reach): its outside is the page's own
    # margin, not lower pages, so the crop does not jump to it.
    img[:, x0 + int(1.5 * mm):x0 + int(1.5 * mm) + 2] = 120
    img[:int(10 * mm)] = 200
    img[int(110 * mm):] = 200
    with profiles.use("far"):
        out, _ = top_page(img, box, 200, keep=keep)
    assert abs((out.bounding_rect()[0] - x0) / mm) < 0.4


def _thick_thin_stack(seed: int = 0):
    """A stack of thin sheets ~12 mm deep (owner's test scans, KLK artboard): a thin green cover,
    sheets 0.5 mm apart with lower pages' art columns, then one plain white sheet 2.2 mm wide
    before the top page; the box side lies on the cover (a 200 dpi preview cut nothing)."""
    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(130 * mm), 3), 200, np.uint8)
    x0, y0, x1, y1 = int(40 * mm), int(10 * mm), int(120 * mm), int(110 * mm)
    h = y1 - y0
    img[y0:y1, x0:x1] = 240
    last = x0 - int(round(2.2 * mm))
    img[y0:y1, last:x0] = 236
    st = last - int(round(9.5 * mm))
    img[y0:y1, st:last] = 234
    for k in range(1, 19):
        xa = last - int(round(k * 0.5 * mm))
        img[y0:y1, xa:xa + 2] = 205
    for a, b, r0, r1, v in ((1.0, 0.0, 0.1, 0.6, 70), (4.0, 2.6, 0.3, 0.9, 110), (7.5, 6.5, 0.0, 0.5, 80)):
        img[y0 + int(r0 * h):y0 + int(r1 * h), last - int(round(a * mm)):last - int(round(b * mm))] = v
    img[y0:y1, last:last + 2] = 200
    img[y0:y1, x0:x0 + 2] = 196
    img[y0:y1, st - int(0.6 * mm):st] = (60, 160, 130)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    box = CropBox.from_xywh(st - int(0.6 * mm), y0, x1 - st + int(0.6 * mm), y1 - y0)
    return img, box, x0, mm


def test_v600_profile_reaches_the_top_page_past_a_thick_thin_stack(monkeypatch):
    # Deeper than the 10 mm strip, and the last sheet's edge is missed by the usual thresholds
    # (they rise with all the stack lines): the strong-line pass on a 14 mm strip finds it.
    from hasami.core.toppage import top_page

    img, box, x0, mm = _thick_thin_stack()
    keep = [(0, 1), (1, -1), (1, 1)]
    v600 = profiles.get("epson-v600")
    monkeypatch.setitem(profiles.PROFILES, "before", dataclasses.replace(
        v600, name="before", fine_strong_lines=False, depth_mm=10.0, layer_depth_mm=0.0))
    found = {}
    for name in ("before", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=keep)
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["before"] < -2.0, found
    assert abs(found["epson-v600"]) < 0.4, found


def test_v600_profile_finds_a_soft_fold():
    # Dark print running into the fold, gutter shading on the page, a light facing page rising
    # slowly: no crease line.  The fold is the step up to the facing page, not the shading.
    from hasami.core.geometry import CropBox
    from hasami.core.toppage import top_page

    mm = 300 / 25.4
    W, f, up = int(160 * mm), int(100 * mm), int(1.2 * mm)
    prof = np.full(W, 225.0)
    prof[:f] = 75
    prof[f - int(1.2 * mm):f] = np.linspace(75, 58, int(1.2 * mm))
    prof[f:f + up] = np.linspace(58, 225, up)
    img = np.repeat(prof[None, :, None], int(120 * mm), 0).repeat(3, 2).astype(np.uint8)
    box = CropBox.from_xywh(int(20 * mm), int(20 * mm), f - int(mm) - int(20 * mm), int(80 * mm))
    fold = f + up / 2
    found = {}
    for name in ("default", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1), (0, -1)], snap=[(0, 1)])
        found[name] = (out.bounding_rect()[2] - fold) / mm
    assert found["default"] < -0.4  # snapped to the shading
    assert abs(found["epson-v600"]) < 0.2, found


def test_paper_edge_needs_a_real_step_with_min_step():
    # A propped-up lid's soft brightness hump passed as a paper edge (tiny noise on a smooth lid).
    from types import SimpleNamespace

    from hasami.core.booklet import _edge_reach

    h, w = 300, 400
    y = np.arange(h, dtype=np.float32)[:, None]
    for step, strong in ((0.2, False), (8.0, True)):
        L = np.where(y < 150, 85.0, 85.0 - step * np.clip(y - 150, 0, 10)).astype(np.float32) * np.ones((1, w), np.float32)
        a = SimpleNamespace(L=L, diff=np.zeros_like(L), threshold=1.0)
        p0, p1, n = np.array([50.0, 150.0]), np.array([350.0, 150.0]), np.array([0.0, 1.0])
        assert _edge_reach(a, p0, p1, n, w, h)[0]  # any step above the (tiny) noise
        assert _edge_reach(a, p0, p1, n, w, h, min_step=0.6)[0] == strong


# ---- owner's second batch of test scans (Oct 13 2026): a KLK artboard page on its green cover,
# covers on the lid, thick stacks of thin sheets ----

def _with(monkeypatch, **changes):
    """The V600 profile with some switches changed, as profile "before"."""
    monkeypatch.setitem(profiles.PROFILES, "before", dataclasses.replace(profiles.get("epson-v600"), name="before",
                                                                         **changes))


def _sides_mm(box, truth, mm):
    return [(v - t) / mm for v, t in zip(box.bounding_rect(), truth)]


def _page_on_green_cover(seed: int = 0):
    """A booklet page above its facing hard cover, plain green, that runs off the bed, with a lighter
    sheen line across it."""
    mm = 200 / 25.4
    rng = np.random.default_rng(seed)
    img = np.empty((int(297 * mm), int(216 * mm), 3), np.float32)
    img[:] = (170, 170, 165)
    x0, x1, y0, fold = int(30 * mm), int(170 * mm), int(25 * mm), int(205 * mm)
    yy, xx = np.mgrid[y0:fold, x0:x1]
    art = 120 + 60 * np.sin(xx / 37.0) * np.cos(yy / 23.0)
    img[y0:fold, x0:x1] = np.stack([art + 40, art, art - 20], -1)
    img[fold:, x0 - 3:x1 + 3] = (80, 170, 140)
    sheen = fold + int(25 * mm)
    img[sheen:sheen + int(3 * mm), x0 - 3:x1 + 3] = (110, 195, 168)
    img[fold - 2:fold + 1, x0:x1] = 60
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, (x0, y0, x1, fold), mm


def _whole_bed_shrink(img, dpi, names):
    """``_shrink_to_paper_edges`` from a box over the whole bed (a propped lid made the lid model
    see everything as foreground), per profile."""
    from hasami.core.autocrop import _coarse_detect, _working_image
    from hasami.core.booklet import _shrink_to_paper_edges
    from hasami.core.geometry import CropBox

    st = AutoCropSettings(mode="booklet")
    small, scale = _working_image(img, st.work_size)
    a = _coarse_detect(small, scale, st, dpi, "single", band_frac=0.006, require_uniform=False, surface="border",
                       drop_edge_slivers=True).analysis
    whole = CropBox.from_xywh(0, 0, small.shape[1], small.shape[0])
    out = {}
    for name in names:
        with profiles.use(name):
            out[name] = [v / scale for v in _shrink_to_paper_edges(a, whole).bounding_rect()]
    return out


def test_v600_profile_keeps_a_coloured_cover_out_of_the_lid_shrink(monkeypatch):
    # The lid-shadow shrink took the uniform green cover for shadow and its sheen line for the
    # booklet's end: no spine, the box ran into the cover.  Only lid-coloured strips are given up.
    img, (x0, y0, x1, fold), mm = _page_on_green_cover()
    _with(monkeypatch, shrink_lid_hue=0.0)
    bottom = {k: v[3] / mm for k, v in _whole_bed_shrink(img, 200, ("before", "epson-v600")).items()}
    assert bottom["before"] < img.shape[0] / mm - 30, bottom  # at the sheen line
    assert bottom["epson-v600"] > img.shape[0] / mm - 1, bottom  # the cover runs off the bed


def _orange_cover(seed: int = 0):
    """An orange cover on the grey lid: ~9 L* brighter than the lid but far more colourful; white
    printed panels and a drawn line 1.1 mm inside the right edge along part of it."""
    mm = 200 / 25.4
    rng = np.random.default_rng(seed)
    img = np.empty((int(297 * mm), int(216 * mm), 3), np.float32)
    img[:] = (170, 170, 165)
    x0, y0, x1, y1 = int(30 * mm), int(40 * mm), int(178 * mm), int(250 * mm)
    img[y0:y1, x0:x1] = (250, 172, 112)
    for a, b, c, d in ((40, 60, 100, 120), (110, 60, 160, 115), (60, 130, 120, 190)):
        img[int(b * mm):int(d * mm), int(a * mm):int(c * mm)] = (238, 232, 222)
    xl = x1 - int(round(1.1 * mm))
    img[int(100 * mm):int(190 * mm), xl - 2:xl + 1] = (150, 105, 80)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, (x0, y0, x1, y1), mm


def test_v600_profile_finds_a_coloured_covers_faint_edge(monkeypatch):
    # Orange on grey is a weak step in luminance: the edge refinement took the drawn line inside.
    # Orange paper on both sides of that step: the paper goes on, the edge is further out.
    img, truth, mm = _orange_cover()
    _with(monkeypatch, paper_colour_edges=0.0)
    found = {}
    for name in ("before", "epson-v600"):
        out = detect(img, AutoCropSettings(mode="booklet", profile=name, margin_mm=0.0), dpi=200)
        assert len(out) == 1
        found[name] = _sides_mm(out[0], truth, mm)
    assert found["before"][2] < -0.8, found
    assert max(abs(v) for v in found["epson-v600"]) < 0.4, found


def _cover_rim(seed: int = 0):
    """A hard cover's green rim 1.2 mm wide between the lid and the top page."""
    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(130 * mm), 3), 200, np.float32)
    x0, y0, x1, y1 = int(30 * mm), int(10 * mm), int(120 * mm), int(110 * mm)
    rim = x0 - int(round(1.2 * mm))
    img[y0 - 8:y1 + 8, rim:x1] = (55, 150, 120)
    img[y0:y1, x0:x1] = 238
    img[y0 + int(20 * mm):y1 - int(20 * mm), x0 + int(6 * mm):x1 - int(10 * mm)] = (120, 80, 70)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(rim, y0, x1 - rim, y1 - y0), x0, mm


def test_v600_profile_crops_a_cover_rim_off(monkeypatch):
    from hasami.core.toppage import top_page

    img, box, x0, mm = _cover_rim()
    _with(monkeypatch, cover_rim=False)
    found = {}
    for name in ("before", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["before"] < -1.0, found
    assert abs(found["epson-v600"]) < 0.3, found


def _stack_then_column(seed: int = 0):
    """~8 mm of thin white sheets 0.3 mm apart with two lower pages' art columns, then a lower page's
    dark art column 0.8 mm wide right before the top page's art, which runs to its edge."""
    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(130 * mm), 3), 200, np.float32)
    x0, y0, x1, y1 = int(40 * mm), int(10 * mm), int(120 * mm), int(110 * mm)
    h = y1 - y0
    img[y0:y1, x0:x1] = (150, 110, 90)
    col = x0 - int(round(0.8 * mm))
    img[y0:y1, col:x0] = (75, 45, 40)
    st = col - int(round(8.0 * mm))
    img[y0:y1, st:col] = 232
    for k in np.arange(0.0, 8.0, 0.3):
        xa = col - int(round(k * mm))
        img[y0:y1, xa - 1:xa + 1] = 205
    for a, b, r0, r1, v in ((2.7, 2.2, 0.0, 0.7, 80), (5.0, 4.4, 0.3, 1.0, 95)):
        img[y0 + int(r0 * h):y0 + int(r1 * h), col - int(round(a * mm)):col - int(round(b * mm))] = v
    img[y0:y1, st - int(0.6 * mm):st] = (60, 160, 130)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(st - int(0.6 * mm), y0, x1 - st + int(0.6 * mm), y1 - y0), x0, mm


def test_v600_profile_walks_on_after_the_faint_line_passes(monkeypatch):
    # The faint-line passes carried the chain through the white sheets; the last 0.8 mm (a lower
    # page's art column) is one more page edge <= near_mm away.
    from hasami.core.toppage import top_page

    img, box, x0, mm = _stack_then_column()
    _with(monkeypatch, walk_after_passes=False)
    found = {}
    for name in ("before", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["before"] < -0.7, found
    assert abs(found["epson-v600"]) < 0.3, found


def _shadowed_sheet(seed: int = 0):
    """A white top page with a lower sheet 1.3 mm wide beyond its edge, neutral but ~17 L* darker
    (further from the glass under a propped lid), then four thin sheets 0.4 mm apart."""
    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(130 * mm), 3), 170, np.float32)
    x0, y0, x1, y1 = int(40 * mm), int(10 * mm), int(120 * mm), int(110 * mm)
    img[y0:y1, x0:x1] = 242
    sheet = x0 - int(round(1.3 * mm))
    img[y0:y1, sheet:x0] = 196
    out = sheet
    for k in range(1, 5):
        out = sheet - int(round(k * 0.4 * mm))
        img[y0:y1, out:out + int(round(0.4 * mm))] = 196 - 4 * k
        img[y0:y1, out:out + 1] = 166
    img[y0:y1, x0 - 1:x0 + 1] = 171
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(out, y0, x1 - out, y1 - y0), x0, mm


def test_v600_profile_crosses_a_lower_sheet_in_the_lids_shadow(monkeypatch):
    # V600 stacks under a propped lid: the sheet just below the top page sticks out 1-2 mm and is
    # 16-19 L* darker, neutral; the 15 L* limit took it for print and the crop kept it.
    from hasami.core.toppage import top_page

    img, box, x0, mm = _shadowed_sheet()
    _with(monkeypatch, stack_shadow_l=0.0)
    found = {}
    for name in ("before", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["before"] < -1.0, found
    assert abs(found["epson-v600"]) < 0.3, found


def _dark_cover_in_shadow(seed: int = 0, margin: bool = False):
    """A dark cover with the propped lid's grey shadow, 1.3 mm wide, between it and the lid (or,
    ``margin``: a top page with its own white margin 1.3 mm wide before dark art)."""
    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(130 * mm), 3), 205, np.float32)
    x0, y0, x1, y1 = int(40 * mm), int(10 * mm), int(120 * mm), int(110 * mm)
    out = x0 - int(round(1.3 * mm))
    img[y0:y1, out:x1] = 245 if margin else 165
    img[y0:y1, x0:x1] = 70
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(out, y0, x1 - out, y1 - y0), x0, out, mm


def test_v600_profile_crosses_the_shadow_beside_a_dark_cover(monkeypatch):
    # A dark cover (or a top page printed dark to its edge) with a neutral band 1-1.5 mm wide
    # beside it, darker than the lid: the lid's shadow or the sheets below.  The band is lighter
    # than the print inside, so it never passed as stack paper and the crop kept it.
    from hasami.core.toppage import top_page

    img, box, x0, _out, mm = _dark_cover_in_shadow()
    _with(monkeypatch, dark_edge_l=0.0)
    found = {}
    for name in ("before", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["before"] < -1.0, found
    assert abs(found["epson-v600"]) < 0.3, found


def test_v600_profile_keeps_a_white_margin_before_dark_art():
    # The band must be darker than the lid: a top page's own white margin lies on the glass and is
    # brighter, so the crop stays on the paper's edge.
    from hasami.core.toppage import top_page

    img, box, _x0, edge, mm = _dark_cover_in_shadow(margin=True)
    with profiles.use("epson-v600"):
        out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
    assert abs((out.bounding_rect()[0] - edge) / mm) < 0.3


def test_v600_profile_crops_a_cover_rim_off_before_dark_print(monkeypatch):
    # A navy hard cover's rim beyond a top page printed dark to its edge (or a lower page's art
    # column just inside the rim): no light paper inside, so the rim was kept.
    from hasami.core.toppage import top_page

    img, box, x0, mm = _cover_rim()
    y0, y1 = int(10 * mm), int(110 * mm)
    rim = x0 - int(round(1.2 * mm))
    img[y0 - 8:y1 + 8, rim:x0] = (75, 85, 120)
    img[y0:y1, x0:int(120 * mm)] = (60, 55, 55)
    _with(monkeypatch, rim_print_inside=False)
    found = {}
    for name in ("before", "epson-v600"):
        with profiles.use(name):
            out, _ = top_page(img, box, 300, keep=[(0, 1), (1, -1), (1, 1)])
        found[name] = (out.bounding_rect()[0] - x0) / mm
    assert found["before"] < -1.0, found
    assert abs(found["epson-v600"]) < 0.3, found


def _fold_with_a_printed_line(seed: int = 0):
    """A page whose spine side lies 0.8 mm from the fold's crease, with a dark printed line 2.5 mm
    inside the page along half the side (the 3 mm crease search's segments split between them)."""
    from hasami.core.geometry import CropBox

    mm = 300 / 25.4
    rng = np.random.default_rng(seed)
    img = np.full((int(120 * mm), int(130 * mm), 3), 236, np.float32)
    x0, y0, x1, y1 = int(20 * mm), int(10 * mm), int(100 * mm), int(110 * mm)
    fold = x1 + int(round(0.8 * mm))
    img[:, fold - 2:fold + 3] = 150
    line = x1 - int(round(2.5 * mm))
    img[y0:(y0 + y1) // 2, line - 2:line + 3] = 60
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(x0, y0, x1 - x0, y1 - y0), (fold - x1) / mm


def test_v600_profile_finds_the_crease_nearer_the_spine_when_the_full_search_splits(monkeypatch):
    # Within 3 mm of the spine, half the segments' darkest valley was a printed line: no crease
    # agreed and the spine stayed 0.8 mm off the fold (V600 0002, 0003, 0031).
    from hasami.core.toppage import SIDES, top_page

    img, box, fold_mm = _fold_with_a_printed_line()
    _with(monkeypatch, crease_retry_mm=())
    found = {}
    for name in ("before", "epson-v600"):
        with profiles.use(name):
            _, shifts = top_page(img, box, 300, keep=SIDES, snap=[(0, 1)], crease_mm=3.0)
        found[name] = shifts[(0, 1)]
    assert found["before"] == 0.0, found
    assert abs(found["epson-v600"] - fold_mm) < 0.2, found
