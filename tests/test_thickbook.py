"""Thick books: the top page's edge where a lower sheet shows as a plain band beyond it (generated scans)."""
import numpy as np
import pytest

from hasami.core import profiles
from hasami.core.thickbook import fit_spread

DPI = 150.0
MM = DPI / 25.4
PAGE = (246, 226, 205)  # cream paper
STACK = (215, 200, 182)  # the sheets' shaded edges
LINE = (185, 172, 158)  # a page step's shadow


def _spread(band_mm=0.0, alternate=False, seed=0):
    """A thick spread on a grey lid, fold across the middle (landscape), pages 20-138 / 142-260 mm wide and
    25-175 mm tall, an 8 mm striped page stack beyond each outer edge.  ``band_mm``: the right top page ends
    that far inside its stack's start (a dark step line), the lower sheet showing as plain paper between
    (``alternate``: the band is 1.5 / 2.5 mm wide every 10 mm along the edge, as 0036's ~1.9 mm band).
    Returns the image and the right top page's outer edge (x, px)."""
    rng = np.random.default_rng(seed)
    W, H = int(280 * MM), int(200 * MM)
    img = np.full((H, W, 3), 200, np.float32)  # lid
    y0, y1 = int(25 * MM), int(175 * MM)
    img[y0:y1, int(20 * MM):int(260 * MM)] = PAGE
    xs = np.arange(W)
    gutter = 1 - 0.18 * np.exp(-((xs - 140 * MM) / (2.5 * MM)) ** 2)  # the fold's shading
    img[y0:y1] *= gutter[None, :, None]
    for a, b in ((12, 20), (260, 268)):  # page stacks: shaded sheet edges every 0.4 mm
        img[y0:y1, int(a * MM):int(b * MM)] = STACK
        for x in np.arange(a, b, 0.4):
            img[y0:y1, int(x * MM)] = LINE
    edge = 260 * MM
    if band_mm or alternate:
        for y in range(y0, y1):
            w = (1.5 if ((y - y0) // int(10 * MM)) % 2 else 2.5) if alternate else band_mm
            e = int(round((260 - w) * MM))
            img[y, e] = LINE  # the top page's edge
            img[y, e + 1:int(260 * MM)] = PAGE  # the lower sheet, plain
        edge = (260 - (2.0 if alternate else band_mm)) * MM
    img += rng.normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), edge


def _right_outer(pages):
    return max(p.bounding_rect()[2] for p in pages)


def test_a_lower_sheet_band_beyond_the_top_page_is_left_out():
    img, edge = _spread(band_mm=2.7)
    with profiles.use("epson-v600"):
        pages = fit_spread(img, DPI)
    assert pages is not None and len(pages) == 2
    assert abs(_right_outer(pages) - edge) <= 0.3 * MM, (_right_outer(pages) / MM, edge / MM)


def test_a_band_on_the_edge_of_2_mm_still_gives_both_pages():
    img, _ = _spread(alternate=True)
    dbg = {}
    with profiles.use("epson-v600"):
        pages = fit_spread(img, DPI, dbg)
    assert pages is not None, dbg.get("fail")
    # the top page's edge: the step line, 1.5 mm (where the band is narrow) to 2.5 mm inside the stack
    assert 257.2 * MM <= _right_outer(pages) <= 258.8 * MM, _right_outer(pages) / MM


def test_without_a_band_the_page_ends_at_the_stack_as_before():
    img, edge = _spread()
    with profiles.use("epson-v600"):
        new = fit_spread(img, DPI)
    with profiles.use("default"):
        old = fit_spread(img, DPI)
    assert new is not None and old is not None
    assert abs(_right_outer(new) - edge) <= 0.3 * MM
    assert [(p.cx, p.cy, p.w, p.h, p.angle) for p in new] == [(p.cx, p.cy, p.w, p.h, p.angle) for p in old]


@pytest.mark.parametrize("name", ["default"])
def test_other_profiles_keep_the_old_fit(name):
    img, _ = _spread(band_mm=2.7)
    with profiles.use(name):
        pages = fit_spread(img, DPI)
    assert pages is not None and abs(_right_outer(pages) - 260 * MM) <= 0.3 * MM  # the band's outer end, as before


def test_a_triple_step_ends_the_page_at_its_innermost_line():
    img, edge = _spread(band_mm=3.5)  # the top page's edge 3.5 mm inside the stack
    for d in (0.55, 1.1):  # two more sheets stepping out between it and the band (as 0038)
        x = int(round((260 - 3.5 + d) * MM))
        img[int(25 * MM):int(175 * MM), x] = LINE
    with profiles.use("epson-v600"):
        pages = fit_spread(img, DPI)
    assert pages is not None
    assert abs(_right_outer(pages) - edge) <= 0.3 * MM, (_right_outer(pages) / MM, edge / MM)
