"""Booklet pages: the fold side cropped clear of the crease (``crop.fold_clear``, opt-in; owner, Oct 2026).

Auto-crop puts a page's fold side on the crease (its centre line, the truth convention).  Some books
(KLK Vol.7 Key Art Collection) show a dark crease the owner would rather leave out of each page, and
there the fold also leans ~0.45 deg against the page box (square to the paper edges), so the side
must move in to the crease's page-side edge at its innermost point.

Per point along a fold side (``meta["keep_sides"]``): the darkest point within ``REACH_MM`` of the side,
then inwards to where the brightness is back above half way between that and the page's own paper
(``PAPER_MM`` inside).  A straight line through those points (the fold leans) gives the innermost
end; the side moves there plus the crop margin, by at most ``CAP_MM``.  Only when a crease at least
``MIN_DEPTH_L`` deep runs along ``MIN_SHARE`` of the side.  The move is kept in ``meta["fold_clear"]``
(mm per side) so that a refinement can start from the crease again (``undo``).
"""
from __future__ import annotations

import cv2
import numpy as np

from .autocrop import MM_PER_INCH, luminance
from .geometry import CropBox

REACH_MM = 2.5  # the crease's darkest point: this far either side of the fold side
PAPER_MM = (2.5, 4.0)  # the page's own paper: this far inside the side
OUT_MM = 3.0  # the profile runs this far out (beyond REACH_MM: room for the blur)
CAP_MM = 2.5  # the fold side moves in by at most this
MIN_DEPTH_L = 8.0  # a crease: at least this much darker than the page's paper (luminance 0..100)
MIN_SHARE = 0.75  # ... at this share of the points along the side
FIT_TOL_MM = 0.3  # points further than this from the line through them do not count
POINTS = 19  # points along the side (5 .. 95 %)
_SIGMA = 0.1  # profiles are smoothed this much (mm)
ACROSS_MM = 1.0  # each point: the median of a strip this far either way along the side


def _side(box: CropBox, key: tuple[int, int]):
    """Endpoints (a, e) of a side, its unit direction d (a -> e) and outward normal n (as ``toppage``)."""
    from .toppage import _side as side

    return side(box, key)


def _key(key) -> str:
    return f"{key[0]}{key[1]:+d}"


def _move(box: CropBox, key: tuple[int, int], px: float) -> CropBox:
    """``box`` with one side moved inwards by ``px`` (negative: outwards)."""
    *_, n = _side(box, key)
    w, h = (box.w - px, box.h) if key[0] == 0 else (box.w, box.h - px)
    out = CropBox(float(box.cx - n[0] * px / 2), float(box.cy - n[1] * px / 2), float(w), float(h), box.angle,
                  confidence=box.confidence, manual=box.manual)
    out.meta = dict(box.meta)
    return out


def _clear_point(prof: np.ndarray, xs: np.ndarray) -> float | None:
    """The crease's page-side edge (mm, + = out) on one profile (``xs`` in mm, evenly spaced; -1 outside the
    image), or None without a clear crease."""
    if (prof[xs <= 0] < 0).any():  # the page itself runs off the image
        return None
    off = np.flatnonzero(prof < 0)  # the crop-area scan ends 3 mm past the box: the profile stops there
    end = int(off[0]) if len(off) else len(prof)
    step = float(xs[1] - xs[0])
    xs, prof = xs[:end], cv2.GaussianBlur(prof[None, :end].astype(np.float32), (0, 0), _SIGMA / step)[0]
    i = int(np.argmin(np.where(np.abs(xs) <= REACH_MM, prof, np.inf)))
    if abs(xs[i]) > REACH_MM - 0.1 or xs[i] > xs[-1] - 0.2:
        return None  # still falling at the window's (or the image's) end: not a valley here
    paper = float(np.median(prof[(xs >= -PAPER_MM[1]) & (xs <= -PAPER_MM[0])]))
    if paper - prof[i] < MIN_DEPTH_L:
        return None
    half = (paper + prof[i]) / 2
    j = i
    while j > 0 and prof[j] < half:  # into the page until it is back above half depth
        j -= 1
    return None if j == 0 else float(xs[j])


def _fold_move(image: np.ndarray, box: CropBox, key: tuple[int, int], dpi: float, margin_mm: float) -> float:
    """How far (mm) the fold side ``key`` moves in, 0 when it stays."""
    mm = dpi / MM_PER_INCH
    a, e, d, n = _side(box, key)
    length = float(np.linalg.norm(e - a))
    xs = np.arange(-PAPER_MM[1], OUT_MM, max(0.05, 0.25 / mm))  # mm
    ys = np.arange(-ACROSS_MM * mm, ACROSS_MM * mm + 1e-6, max(0.25 * mm, 1.0))
    ts = np.linspace(0.05, 0.95, POINTS) * length
    X, Y = np.meshgrid(xs * mm, ys)
    T = ts[:, None, None] + Y[None]
    mx = (a[0] + n[0] * X[None] + d[0] * T).reshape(-1, len(xs)).astype(np.float32)
    my = (a[1] + n[1] * X[None] + d[1] * T).reshape(-1, len(xs)).astype(np.float32)
    H, W = image.shape[:2]
    inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    strip = luminance(cv2.remap(np.ascontiguousarray(image), mx, my, cv2.INTER_LINEAR))
    strip[~inside] = -1.0
    pts = []
    for k, rows in enumerate(strip.reshape(POINTS, len(ys), len(xs))):
        c = _clear_point(np.median(rows, 0), xs)
        if c is not None:
            pts.append((ts[k] / mm, c))
    if len(pts) < MIN_SHARE * POINTS:
        return 0.0
    t, c = np.array(pts).T
    fit = np.polyfit(t, c, 1)
    good = np.abs(np.polyval(fit, t) - c) <= FIT_TOL_MM
    if good.sum() < MIN_SHARE * POINTS:
        return 0.0
    fit = np.polyfit(t[good], c[good], 1)
    inner = float(min(np.polyval(fit, ts[0] / mm), np.polyval(fit, ts[-1] / mm)))  # mm, + = out
    return float(min(max(0.0, margin_mm - inner), CAP_MM))


def clear_of_crease(image: np.ndarray, page: CropBox, dpi: float, margin_mm: float) -> CropBox:
    """``page`` with each fold side (``meta["keep_sides"]``) moved in clear of the fold's crease, ``margin_mm``
    inside its page-side edge; sides without a clear crease stay.  The moves (mm) go to ``meta["fold_clear"]``."""
    out, moved = page, {}
    for key in page.meta.get("keep_sides", []):
        key = tuple(key)
        m = round(_fold_move(image, page, key, dpi, margin_mm), 3)
        if m > 0:
            out = _move(out, key, m * dpi / MM_PER_INCH)
            moved[_key(key)] = m
    if moved:
        out.meta = {**out.meta, "fold_clear": moved}
    return out


def undo(page: CropBox, dpi: float) -> CropBox:
    """``page`` with its fold sides back where auto-crop found the fold (before ``clear_of_crease``)."""
    moved = page.meta.get("fold_clear")
    if not moved:
        return page
    out = page
    for key in page.meta.get("keep_sides", []):
        m = moved.get(_key(tuple(key)))
        if m:
            out = _move(out, tuple(key), -m * dpi / MM_PER_INCH)
    out.meta = {k: v for k, v in out.meta.items() if k != "fold_clear"}
    return out
