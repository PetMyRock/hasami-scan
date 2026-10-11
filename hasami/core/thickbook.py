"""Thick books: both pages of an open spread, each cut at the top page's edge.

A thick book lifts the scanner's lid.  The lid then darkens strongly towards
the edges of the glass (and leaves a black strip of uncovered glass), so the
general lid model - learnt from a ring at the image border - sees the whole
bed as foreground.  Beyond the top page lie the side edges of the other pages
(a striped band, the "page stack") and the cover board; the maintainer wants the
crop to end at the top page.

What holds on such scans: the top page is the brightest plain surface on the
bed.  The lid (even where it is lightest, near the middle), the page stack
(each page's edge is shaded) and the cover board are all darker.  So, coming
in from the image border, a page starts where bright plain paper begins and
lasts a few millimetres.  Each page gets its own top, bottom and outer edge -
the pages of a thick book do not line up (the right page sits lower near the
start of a book) - fitted as straight lines; the spine is the gutter's
valley.  Near the spine the paper curls down into the fold, so the top and
bottom edges are fitted only over the flat part of each page.

:func:`fit_spread` returns the two pages, or None when the scan does not look
like a whole spread with clean edges (the caller then uses the general
booklet logic).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import CropBox

MM_PER_INCH = 25.4
WORK_DPI = 150.0  # enough for sub-pixel straight-line fits, and fast


def _lab(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lab L (0-100) and b* of an RGB image (8/16-bit or float)."""
    img = image
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, 2)
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[:, :, 0] * (100.0 / 255.0), lab[:, :, 2] - 128.0


def _line_fit(pts, vertical: bool, tol: float):
    """Robust x = a*y + b (vertical) or y = a*x + b.  Returns (a, b, inlier share, rms) or None."""
    pts = np.asarray(pts, float)
    if len(pts) < 8:
        return None
    u, v = (pts[:, 1], pts[:, 0]) if vertical else (pts[:, 0], pts[:, 1])
    keep = np.ones(len(pts), bool)
    for _ in range(5):
        a, b = np.polyfit(u[keep], v[keep], 1)
        r = np.abs(v - (a * u + b))
        keep = r < max(tol, 2.5 * float(np.median(r[keep])))
        if keep.sum() < 8:
            return None
    a, b = np.polyfit(u[keep], v[keep], 1)
    r = v[keep] - (a * u[keep] + b)
    return float(a), float(b), float(keep.mean()), float(np.sqrt(np.mean(r ** 2)))


def _first_paper(prof: np.ndarray, level: float, run: int) -> float | None:
    """Coming from index 0, where the profile first stays above ``level`` for ``run`` samples;
    sub-pixel position of the crossing of ``level - 2`` just before it."""
    good = prof > level
    if len(good) < run:
        return None
    ok = np.convolve(good.astype(np.int32), np.ones(run, np.int32), "valid") >= run
    if not ok.any():
        return None
    k = int(np.argmax(ok))
    half = level - 2.0
    j = k
    while j > 0 and prof[j - 1] >= half:
        j -= 1
    if j == 0:
        return None  # paper right at the border: not an edge we can see
    p0, p1 = prof[j - 1], prof[j]
    t = (half - p0) / max(p1 - p0, 1e-6)
    return (j - 1) + float(np.clip(t, 0.0, 1.0))


def _angle(name: str, a: float) -> float:
    return -math.degrees(math.atan(a)) if name in ("outer", "spine") else math.degrees(math.atan(a))


def _page_box(top, bottom, outer, spine) -> CropBox:
    """Rotated box from a page's four lines: angle = median of its own edges, sides = mean positions."""
    t = math.radians(float(np.median([_angle("top", top[0]), _angle("bottom", bottom[0]),
                                      _angle("outer", outer[0])])))
    ux, uy = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])

    def x_at(ln, y):
        return ln[0] * y + ln[1]

    def y_at(ln, x):
        return ln[0] * x + ln[1]

    # corners from line intersections (approximately: lines are nearly axis-aligned)
    xa = x_at(outer, (y_at(top, x_at(outer, 0)) + y_at(bottom, x_at(outer, 0))) / 2)
    xb = x_at(spine, (y_at(top, x_at(spine, 0)) + y_at(bottom, x_at(spine, 0))) / 2)
    x0, x1 = sorted((xa, xb))
    yt, yb = y_at(top, (x0 + x1) / 2), y_at(bottom, (x0 + x1) / 2)

    def along_v(ln, a, b):
        ys = np.linspace(a, b, 16)
        return np.stack([ln[0] * ys + ln[1], ys], 1)

    def along_h(ln, a, b):
        xs = np.linspace(a, b, 16)
        return np.stack([xs, ln[0] * xs + ln[1]], 1)

    u_o = float(np.mean(along_v(outer, yt, yb) @ ux))
    u_s = float(np.mean(along_v(spine, yt, yb) @ ux))
    v_t = float(np.mean(along_h(top, x0, x1) @ uy))
    v_b = float(np.mean(along_h(bottom, x0, x1) @ uy))
    u0, u1 = sorted((u_o, u_s))
    c = ux * (u0 + u1) / 2 + uy * (v_t + v_b) / 2
    return CropBox(float(c[0]), float(c[1]), u1 - u0, v_b - v_t, math.degrees(t))


def fit_spread(image: np.ndarray, dpi: float, debug: dict | None = None) -> list[CropBox] | None:
    """The two pages of a whole spread, each to its top page's edges, or None."""
    H, W = image.shape[:2]
    s = min(1.0, WORK_DPI / float(dpi))
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
    L, B = _lab(small)
    h, w = L.shape
    mm = dpi * s / MM_PER_INCH  # working pixels per mm

    def fail(why: str):
        if debug is not None:
            debug["fail"] = why
        return None

    # Only for light lids (what this is about): on a black bed the general logic works, and the
    # bright-paper rule cut thin V39 booklets 3-4 mm short.
    ring = np.concatenate([L[:max(2, h // 50)].ravel(), L[-max(2, h // 50):].ravel(),
                           L[:, :max(2, w // 50)].ravel(), L[:, -max(2, w // 50):].ravel()])
    if float(np.median(ring)) < 30:
        return fail("black bed")
    # Paper level: blank paper dominates the middle of an open book.
    mid = L[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
    P = float(np.percentile(mid, 75))
    if P < 60:
        return fail(f"no bright paper in the middle (L {P:.0f})")
    level = P - 5.0
    # Cream paper: the lid can be nearly as light as the page where it is lightest (Key Art
    # Collection 10-13: lid L 80-82, paper 82-86), but the lid is neutral and the paper is not.
    Pb = float(np.percentile(B[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)], 50))
    if Pb > 5.0:
        L = np.where(B > Pb / 2, L, np.minimum(L, level - 3.0))
    bright = cv2.blur((L > level).astype(np.float32), (max(3, int(mm)), max(3, int(mm)))) > 0.5
    cols = np.where(bright[int(h * 0.3):int(h * 0.7)].mean(0) > 0.5)[0]
    rows = np.where(bright[:, int(w * 0.3):int(w * 0.7)].mean(1) > 0.5)[0]
    if len(cols) < w * 0.4 or len(rows) < h * 0.3:
        return fail("paper area too small")
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])
    if (x1 - x0) < 1.15 * (y1 - y0):
        return fail("not a landscape spread")
    # Spine: the darkest column of the gutter, near the middle.
    sm = cv2.GaussianBlur(L, (0, 0), max(1.0, 0.7 * mm))
    band = sm[int(y0 + 0.25 * (y1 - y0)):int(y1 - 0.25 * (y1 - y0))]
    c0, c1 = int(x0 + 0.4 * (x1 - x0)), int(x0 + 0.6 * (x1 - x0))
    xs0 = c0 + int(np.argmin(band.mean(0)[c0:c1]))
    half_w = int(12 * mm)
    pts = []
    for y in np.arange(int(y0 + 0.15 * (y1 - y0)), int(y1 - 0.15 * (y1 - y0)), max(1, int(mm))):
        seg = sm[y, max(xs0 - half_w, 0):xs0 + half_w]
        i = int(np.argmin(seg))
        if 1 <= i < len(seg) - 1:
            a, b, c = seg[i - 1], seg[i], seg[i + 1]
            d = (a - c) / (2 * (a - 2 * b + c) + 1e-6)
            pts.append((max(xs0 - half_w, 0) + i + float(np.clip(d, -1, 1)), y))
    spine = _line_fit(pts, True, tol=1.0 * mm)
    if spine is None or spine[2] < 0.6:
        return fail("no straight spine")
    if debug is not None:
        debug.update(P=P, extent=(x0, x1, y0, y1), spine=spine)

    run_out = max(4, int(round(2.0 * mm)))  # plain paper for >= 2 mm: the page has begun
    run_tb = max(4, int(round(2.0 * mm)))
    pages = []
    for side in ("L", "R"):
        xsp = spine[0] * (h / 2) + spine[1]
        lo, hi = (x0, xsp) if side == "L" else (xsp, x1)
        span = hi - lo
        if span < 0.3 * (x1 - x0):
            return fail(f"page {side} too narrow")
        lines = {}
        # top / bottom over the flat part of the page (the paper curls into the fold)
        if side == "L":
            cs = np.arange(int(lo + 0.06 * span), int(hi - 0.18 * span), max(1, int(mm)))
        else:
            cs = np.arange(int(lo + 0.18 * span), int(hi - 0.06 * span), max(1, int(mm)))
        for name in ("top", "bottom"):
            pts = []
            lim = int(h * 0.45)
            for x in cs:
                col = L[:, max(x - 1, 0):x + 2].mean(1)
                prof = col[:lim] if name == "top" else col[::-1][:lim]
                r = _first_paper(prof, level, run_tb)
                if r is not None:
                    pts.append((x, r if name == "top" else (h - 1) - r))
            lines[name] = _line_fit(pts, False, tol=0.5 * mm)
        if lines["top"] is None or lines["bottom"] is None:
            return fail(f"page {side}: no straight top/bottom edge")
        # outer edge over the middle of the page's height
        xm = (lo + hi) / 2
        yt = lines["top"][0] * xm + lines["top"][1]
        yb = lines["bottom"][0] * xm + lines["bottom"][1]
        pts = []
        lim = int(0.5 * w)
        for y in np.arange(int(yt + 0.12 * (yb - yt)), int(yb - 0.12 * (yb - yt)), max(1, int(mm))):
            row = L[max(y - 1, 0):y + 2].mean(0)
            prof = row[:lim] if side == "L" else row[::-1][:lim]
            r = _first_paper(prof, level, run_out)
            if r is not None:
                pts.append((r if side == "L" else (w - 1) - r, y))
        lines["outer"] = _line_fit(pts, True, tol=0.4 * mm)
        if lines["outer"] is None:
            return fail(f"page {side}: no straight outer edge")
        for k, ln in lines.items():
            if ln[2] < 0.5 or ln[3] > 0.6 * mm:
                return fail(f"page {side} {k}: ragged (inliers {ln[2]:.2f}, rms {ln[3] / mm:.2f} mm)")
        box = _page_box(lines["top"], lines["bottom"], lines["outer"], spine)
        # Outside the page there must be something clearly darker (lid, stack, board).
        pages.append((box, lines))
        if debug is not None:
            debug[side] = {k: v for k, v in lines.items()}

    (bl, _), (br, _) = pages
    if abs(bl.h - br.h) > 0.03 * max(bl.h, br.h):
        return fail(f"page heights differ ({bl.h:.0f} vs {br.h:.0f})")
    if abs(bl.w - br.w) > 0.15 * max(bl.w, br.w):
        return fail(f"page widths differ ({bl.w:.0f} vs {br.w:.0f})")
    for b in (bl, br):
        if not 1.0 < b.h / b.w < 2.0:
            return fail(f"page shape {b.w:.0f}x{b.h:.0f}")
        x_a, y_a, x_b, y_b = b.bounding_rect()
        if min(x_a, y_a, w - x_b, h - y_b) < 2.0 * mm:  # V600 V02: a page "ending" 0.6 mm from the border
            return fail("page at the image border")
    out = []
    for b, side in ((bl, "L"), (br, "R")):
        full = (b.scaled(1.0 / s) if s < 1 else b).clamp_to(W, H)
        # the spine side is not a paper edge: re-refinement must not move it
        full.meta = {"spine": "thick", "keep_sides": [(0, 1)] if side == "L" else [(0, -1)]}
        full.confidence = 0.9
        out.append(full)
    return out
