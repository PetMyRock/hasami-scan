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
    (``alternate``: the top page ends at 258 mm and the band's outer end wavers about 2 mm out, as 0036).
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
            e = int(round((258.0 if alternate else 260 - band_mm) * MM))
            # 0036: the band's outer end wavers about the fit's 2 mm paper run (1.7 / 2.4 mm every 10 mm)
            s = (259.7 if ((y - y0) // int(10 * MM)) % 2 else 260.4) if alternate else 260.0
            img[y, int(round(s * MM)):int(268 * MM)] = STACK
            for x in np.arange(260.0, 268, 0.4):
                if x >= s:
                    img[y, int(x * MM)] = LINE
            img[y, e] = LINE  # the top page's edge
            img[y, e + 1:int(round(s * MM))] = PAGE  # the lower sheet, plain
        edge = (258.0 if alternate else 260 - band_mm) * MM
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
    assert abs(_right_outer(pages) - 258.0 * MM) <= 0.3 * MM, _right_outer(pages) / MM  # the step line


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


def test_a_ragged_edge_that_is_not_two_lines_still_fails_the_fit():
    """V600 0019: the outer edge's points scatter over ~4 mm (not a band): the fit must still give up, as before,
    and leave the page to the general booklet logic."""
    img, _ = _spread()
    rng = np.random.default_rng(1)
    y0, y1 = int(25 * MM), int(175 * MM)
    for ya in range(y0, y1, int(2 * MM)):  # the paper ends anywhere between 257 and 260 mm, every 2 mm
        e = int(round(rng.uniform(257, 260) * MM))
        img[ya:min(ya + int(2 * MM), y1), e:int(260 * MM)] = STACK
    with profiles.use("default"):
        assert fit_spread(img, DPI) is None
    dbg = {}
    with profiles.use("epson-v600"):
        assert fit_spread(img, DPI, dbg) is None, "the V600 fit answered on a ragged edge"
    assert "ragged" in dbg["fail"]
