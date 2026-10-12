import dataclasses

import numpy as np

from hasami.core import profiles
from hasami.core.autocrop import AutoCropSettings, detect
from hasami.demo import BookletOptions, synth_booklet


def test_v600_profile_differs_from_the_default_only_in_its_own_switches():
    # The default started as a copy of the V600's tuned values; behaviour added for the V600 later
    # (KLK artboard, Oct 2026) is switched on only in its profile, so other scanners are unchanged.
    ident = {"name", "label", "match", "tuned_on"}
    own = {"min_edge_step", "crease_step", "fine_stack", "stack_by_segment"}
    v600, default = profiles.get("epson-v600"), profiles.DEFAULT
    for f in dataclasses.fields(profiles.Profile):
        if f.name not in ident | own:
            assert getattr(v600, f.name) == getattr(default, f.name), f.name
    assert default.min_edge_step == 0.0 and not default.crease_step and not default.fine_stack
    assert v600.min_edge_step > 0 and v600.crease_step and v600.fine_stack and v600.stack_by_segment
    assert not default.stack_by_segment


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
