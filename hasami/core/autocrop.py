"""Auto-crop: find every photo or document on a flatbed scan.

Most scanning software finds one bounding box by thresholding against the lid
colour.  That approach struggles with several items on the glass, with prints
that have white borders on a white lid, with dust and hairs, and with crooked
photos.  This module goes further:

1. **Background model** - the lid colour is learnt from smooth pixels near the
   image border and a low-order surface is fitted so that uneven lighting and
   vignetting do not show up as foreground.
2. **Two cues** - a colour-difference map (in CIE Lab, relative to the fitted
   background) and a Canny edge map are combined, so faint paper edges and lid
   shadows count even when the paper is the same colour as the lid.
3. **Clean-up** - hole filling, dust/hair removal by morphological opening, and
   rejection of thin strips where the scan area meets the bed frame.
4. **Splitting** - blobs that are not rectangular (two photos that touch or
   nearly touch) are split with a distance-transform + watershed search, and a
   split is kept only if every part is clearly rectangular.
5. **Robust rectangle fitting** - orientation is found by minimising the area
   of a projection-profile box, which ignores small protrusions such as dust
   on an edge (``minAreaRect`` would grow to include them).
6. **Border search** - each side is pushed outwards to the strongest straight
   edge within a search band, so white borders of prints and the margins of
   text documents are kept.
7. **Sub-pixel edge refinement** at full resolution: each side is re-located
   from many short gradient profiles and a robust line fit, which corrects the
   rotation angle as well as the position.  Lid shadows outside the paper edge
   are excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from . import profiles
from .geometry import CropBox, reading_order

MM_PER_INCH = 25.4
#: Lab lightness below which tones are treated as black (see ``_analyze``).


@dataclass
class AutoCropSettings:
    #: "multi" finds every item, "single" returns one box around everything,
    #: "booklet" returns the page(s) of an open booklet that are fully on the
    #: glass (the facing page may run off the bed), "off" returns the whole
    #: image.
    mode: str = "multi"
    #: 0..1, higher finds fainter objects (and more false positives).
    sensitivity: float = 0.5
    #: Smallest object to keep, as a fraction of the image area.
    min_area: float = 0.004
    #: Positive values crop this many millimetres inside the detected edge,
    #: negative values keep a strip of background around the item.
    margin_mm: float = 0.25
    #: Straighten crooked items.  When off, crops are axis-aligned boxes
    #: around each item.
    deskew: bool = True
    #: Turn every saved item by this many quarter turns clockwise (0-3), as
    #: the GUI's Rotate buttons show the preview.
    rotate: int = 0
    #: Angles smaller than this (degrees) are snapped to zero to avoid a
    #: needless resample.
    snap_angle: float = 0.15
    #: Try to separate items that touch.
    split_touching: bool = True
    #: Locate edges precisely on the full-resolution image.
    edge_refine: bool = True
    #: Look this far (mm) beyond the image content for a paper edge, e.g. the
    #: white border of a print.  Documents use a larger distance automatically.
    border_search_mm: float = 10.0
    #: Long side of the working image used for detection.
    work_size: int = 1000
    #: Assumed resolution when the caller does not know it.
    default_dpi: float = 300.0
    #: Scanner profile (tuned numbers per scanner model, see ``profiles``): a profile name,
    #: or "auto" = the profile matching the scanner (the app resolves it; default otherwise).
    profile: str = "auto"
    #: Booklet mode: end each page at the top page's edge (page-stack bands
    #: and cover boards left out), see ``toppage``.
    top_page: bool = True
    #: Booklet mode: "pages" returns each complete page, "spread" one image
    #: over both pages when the whole spread is on the glass.
    booklet_output: str = "pages"
    #: Booklet mode, both pages as separate images: number the right page
    #: (as seen in the preview) 1 instead of the left (books read right to
    #: left, e.g. Japanese booklets).
    booklet_rtl: bool = False


@dataclass
class _Analysis:
    """Everything the detector derives from the working-resolution image."""

    rgb: np.ndarray  # float32 RGB in [0, 1]
    lab: np.ndarray  # float32 Lab
    L: np.ndarray  # float32 lightness, Gaussian smoothed
    diff: np.ndarray  # colour distance from the background model
    edges: np.ndarray  # uint8 Canny edges
    noise: float  # robust noise level of ``diff`` on background
    threshold: float  # foreground threshold on ``diff``
    bg_lab: np.ndarray  # median background colour
    uniform_bg: bool  # False when no plain background is visible
    strips: tuple = ()  # image edges (0 top, 1 bottom, 2 left, 3 right) with a black band painted over


# ------------------------------------------------------------------ helpers
def to_float_rgb(image: np.ndarray) -> np.ndarray:
    """Convert 8/16-bit or float, grey or colour, to float32 RGB in [0, 1]."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype == np.uint8:
        f = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        f = img.astype(np.float32) / 65535.0
    elif img.dtype == bool:
        f = img.astype(np.float32)
    else:
        f = img.astype(np.float32)
        if f.max(initial=0) > 1.5:
            f /= 255.0
    if f.ndim == 2:
        f = np.repeat(f[:, :, None], 3, axis=2)
    elif f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)
    return np.ascontiguousarray(f)


def luminance(image: np.ndarray) -> np.ndarray:
    """Float32 luminance in 0..100 for any input image.

    Avoids building a float RGB copy, which matters for 1200 dpi scans.
    """
    img = np.asarray(image)
    if img.dtype == np.uint8:
        scale = 100.0 / 255.0
    elif img.dtype == np.uint16:
        scale = 100.0 / 65535.0
    else:
        scale = 100.0 if float(np.max(img, initial=0)) <= 1.5 else 100.0 / 255.0
    if img.ndim == 2 or img.shape[2] == 1:
        y = img.reshape(img.shape[0], img.shape[1]).astype(np.float32)
    else:
        y = img[:, :, 0].astype(np.float32) * 0.2126
        y += img[:, :, 1].astype(np.float32) * 0.7152
        y += img[:, :, 2].astype(np.float32) * 0.0722
    y *= scale
    return y


def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def _robust_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


# --------------------------------------------------------------- analysis
def _fill_edge_strips(lab: np.ndarray, max_frac: float = 0.04) -> tuple[int, ...]:
    """Paint a black band along an image edge over with the lid next to it.

    A thick booklet props the lid up, and along the front of the bed the
    glass is then not covered: the scan shows a black band (1-4 mm) with
    lid right beside it.  It is not an item and must not be mistaken for
    the lid colour either.  Paper on the glass there is lit by the lamp and
    stays bright, so only dark pixels are replaced, column by column, by the
    first lid pixel beyond the band.  Works in place on ``lab``; returns the
    edges painted over (0 top, 1 bottom, 2 left, 3 right).
    """
    L = lab[:, :, 0]
    ref = float(np.median(L))
    if ref < 50.0:
        return ()  # dark lid or no lid
    thr = 0.5 * ref
    views = []
    for edge in range(4):  # top, bottom, left, right
        V = lab if edge < 2 else lab.transpose(1, 0, 2)
        views.append(V[::-1] if edge in (1, 3) else V)
    # The rest of the border must show a light lid; on a dark lid a thin
    # band of lid between a large booklet and the image edge looks the same.
    light = [float(np.median(V[:3, int(0.2 * V.shape[1]) : int(0.8 * V.shape[1]), 0])) > thr for V in views]
    if sum(light) < 2:
        return ()
    filled = []
    for edge, V in enumerate(views):
        Ls = V[:, :, 0]
        depth_dim, n = Ls.shape
        kmax = max(2, int(max_frac * depth_dim))
        k = np.cumprod(Ls[: kmax + 1] < thr, axis=0).sum(axis=0)  # dark run from the edge
        mid = k[int(0.2 * n) : int(0.8 * n)]
        # Along most of the edge, and thin: a dark item at the edge is wider.
        if mid.size == 0 or np.mean(mid >= 1) < 0.5 or np.percentile(mid, 90) >= kmax:
            continue
        k = cv2.medianBlur(np.minimum(k, 255).astype(np.uint8).reshape(1, -1), 5).ravel().astype(int)
        k = np.where(k > 0, np.minimum(k + 1, kmax), 0)  # and the blurred row next to it
        cols = np.arange(n)
        lid = V[np.minimum(k + 1, depth_dim - 1), cols]
        rows = np.arange(kmax + 1)[:, None]
        fill = (rows < k[None, :]) & (Ls[: kmax + 1] < np.maximum(thr, lid[None, :, 0] - 3.0))
        band = V[: kmax + 1]
        band[fill] = np.broadcast_to(lid[None], band.shape)[fill]
        filled.append(edge)
    return tuple(filled)


def _edge_falloff(lab: np.ndarray) -> np.ndarray:
    """Darkening of a light lid towards the edges of the glass, as a map.

    Many lids get darker within a few centimetres of the bed edge (by up to
    10-15 L*), more steeply than a smooth lighting surface can follow, so the
    strip would pass for an item.  For each image edge the lightness profile
    going inwards is measured along the stretches of the edge where the lid
    is plain all the way in (no item, no gradient, neutral colour); the
    darkening relative to the inner end, made monotone, is the falloff for
    that edge.  The four profiles are added up.
    """
    h, w = lab.shape[:2]
    F = np.zeros((h, w), np.float32)
    L = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 2.0)
    if float(np.median(L)) < 50.0:
        return F  # dark lid or no lid: nothing to correct
    grad = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    for edge in range(4):
        # Strips with depth along axis 0: top, bottom, left, right.
        flip = edge in (1, 3)
        Ls, Gs, As, Bs = (x if edge < 2 else x.T for x in (L, grad, lab[:, :, 1], lab[:, :, 2]))
        if flip:
            Ls, Gs, As, Bs = (x[::-1] for x in (Ls, Gs, As, Bs))
        depth_dim, n = Ls.shape
        dmax = int(0.25 * depth_dim)
        if dmax < 8:
            continue
        Ls, Gs, As, Bs = (x[:dmax] for x in (Ls, Gs, As, Bs))
        plain = (Gs < 6.0) & (Ls > 50.0)
        a0, b0 = float(np.median(As[plain])) if plain.any() else 0.0, float(np.median(Bs[plain])) if plain.any() else 0.0
        plain &= np.hypot(As - a0, Bs - b0) < 6.0
        plain[: max(1, int(0.01 * depth_dim))] = True  # scanner edge artefacts
        run = np.cumprod(plain, axis=0).sum(axis=0)
        for d in (dmax, dmax // 2, dmax // 4):
            sel = run >= d
            if d >= 8 and sel.sum() >= max(15, 0.05 * n):
                break
        else:
            continue
        prof = np.median(Ls[:d, sel], axis=1)
        f = float(np.median(prof[int(0.8 * d):])) - prof
        f = cv2.GaussianBlur(f.reshape(-1, 1).astype(np.float32), (0, 0), max(1.0, 0.005 * depth_dim)).ravel()
        f = np.clip(f - 0.5, 0.0, 30.0)
        f = np.maximum.accumulate(f[::-1])[::-1]  # monotone: darkest at the edge
        f[f < 1.0] = 0.0
        if not f.any():
            continue
        prof_full = np.zeros(depth_dim, np.float32)
        prof_full[:d] = f
        if flip:
            prof_full = prof_full[::-1]
        F += prof_full[:, None] if edge < 2 else prof_full[None, :]
    return F


def _analyze(rgb: np.ndarray, sensitivity: float, band_frac: float = 0.03,
             surface: str = "all") -> _Analysis:
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    # Lightness exaggerates differences between very dark tones, so on a
    # black background fabric texture or a faint light leak would look like
    # an item.  Everything darker than the profile's dark_floor_l counts as equally black, and
    # colour differences fade out in deep shadow.
    dark_floor = profiles.current().dark_floor_l
    shadow = np.clip(lab[:, :, 0] / dark_floor, 0.0, 1.0)
    lab[:, :, 0] = np.maximum(lab[:, :, 0], dark_floor)
    lab[:, :, 1] *= shadow
    lab[:, :, 2] *= shadow
    # Flatten the lid's darkening towards the edges of the glass first, so
    # that neither the lighting fit nor the threshold has to cope with it.
    strips = _fill_edge_strips(lab)
    lab[:, :, 0] += _edge_falloff(lab)
    L = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 1.0)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # Border band, skipping the outermost pixels (scanner edge artefacts).
    m = min(h, w)
    skip = max(1, int(round(min(0.004, band_frac / 4) * m)))
    band = max(3, int(round(band_frac * m)))
    border = np.zeros((h, w), bool)
    border[skip : skip + band, skip : w - skip] = True
    border[h - skip - band : h - skip, skip : w - skip] = True
    border[skip : h - skip, skip : skip + band] = True
    border[skip : h - skip, w - skip - band : w - skip] = True

    # Prefer smooth border pixels: the lid is untextured, photos are not.
    bgrad = grad[border]
    smooth = border & (grad <= np.percentile(bgrad, 50))
    samples = lab[smooth]
    bg_lab = np.median(samples, axis=0)

    # How uniform is the border?  A scan that is all photo has no lid visible.
    d_border = np.linalg.norm(lab[border] - bg_lab, axis=1)
    frac_close = float(np.mean(d_border < 6.0))
    uniform_bg = frac_close > 0.45

    # Fit a smooth lighting surface to pixels that look like the lid:
    # a quadratic over the whole image ("all"), or a plane fitted to the
    # border ring only ("border", for scans where an item covers most of the
    # image and could pass for the lid), or none.
    d0 = np.linalg.norm(lab - bg_lab, axis=2)
    cand = (d0 < 8.0) & (grad < max(np.percentile(bgrad, 60), 1.0))
    if surface == "border":
        cand &= border
    bg_model = np.empty_like(lab)
    bg_model[:] = bg_lab
    ys, xs = np.nonzero(cand)
    enough = ys.size > (0.05 * h * w if surface == "all" else 0.3 * border.sum())
    if surface != "none" and enough:
        step = max(1, ys.size // 20000)
        ys, xs = ys[::step], xs[::step]
        xn = xs / w - 0.5
        yn = ys / h - 0.5
        terms = [np.ones_like(xn), xn, yn]
        if surface == "all":
            terms += [xn * xn, xn * yn, yn * yn]
        A = np.stack(terms, 1)
        vals = lab[ys, xs]
        keep = np.ones(len(xn), bool)
        coef = None
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
            res = np.linalg.norm(A @ coef - vals, axis=1)
            s = max(_robust_sigma(res[keep]), 0.3)
            keep = res < 3 * s + 0.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx = xx / w - 0.5
        yy = yy / h - 0.5
        basis = [np.ones_like(xx), xx, yy]
        if surface == "all":
            basis += [xx * xx, xx * yy, yy * yy]
        for c in range(3):
            bg_model[:, :, c] = sum(k * bvec for k, bvec in zip(coef[:, c], basis))

    dl = lab - bg_model
    diff = np.sqrt(dl[:, :, 0] ** 2 + 1.5 * (dl[:, :, 1] ** 2 + dl[:, :, 2] ** 2))
    diff = cv2.medianBlur(diff.astype(np.float32), 3)
    noise = max(_robust_sigma(diff[cand]) if cand.any() else 1.0, 0.25)
    s = float(np.clip(sensitivity, 0.0, 1.0))
    k = 10.0 - 6.0 * s  # 10 sigma (strict) .. 4 sigma (lenient)
    floor = 7.0 - 4.5 * s  # in Lab units
    threshold = max(k * noise, floor)

    # Canny with thresholds tied to the measured gradient noise on the lid.
    gnoise = float(np.median(grad[cand])) if cand.any() else 1.0
    hi = max((14.0 - 8.0 * s) * gnoise, 6.0 - 3.0 * s)
    lo = 0.45 * hi
    dx = np.clip(gx * 8, -32767, 32767).astype(np.int16)
    dy = np.clip(gy * 8, -32767, 32767).astype(np.int16)
    edges = cv2.Canny(dx, dy, lo * 8, hi * 8, L2gradient=True)

    return _Analysis(
        rgb=rgb,
        lab=lab,
        L=L,
        diff=diff,
        edges=edges,
        noise=noise,
        threshold=threshold,
        bg_lab=bg_lab,
        uniform_bg=uniform_bg,
        strips=strips,
    )


def _foreground_mask(a: _Analysis) -> np.ndarray:
    h, w = a.diff.shape
    m = min(h, w)
    fg = (a.diff > a.threshold).astype(np.uint8)
    # Edges only help when they form long structures; short ones are noise.
    edges = a.edges.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    min_len = max(8, int(0.015 * m))
    big = np.zeros(n, bool)
    big[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= min_len
    edges = (big[lab]).astype(np.uint8)
    fg |= edges
    k_close = _odd(0.006 * m)
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    )
    fg = _fill_holes(fg)
    k_open = _odd(max(3, 0.005 * m))
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    )
    return fg


def _remove_frame_strips(fg: np.ndarray, max_frac: float = 0.025, coverage: float = 0.85) -> None:
    """Clear dark bands along the image edges (the scanner bed frame).

    Rows/columns at the very edge that are foreground across most of the
    image are a scan area larger than the glass, not an item.  Clearing them
    keeps a photo that touches the frame from merging with it.
    """
    h, w = fg.shape
    lim_r = max(1, int(max_frac * h))
    lim_c = max(1, int(max_frac * w))
    for rows in (range(0, lim_r), range(h - 1, h - 1 - lim_r, -1)):
        hit = []
        for r in rows:
            if fg[r].mean() > coverage:
                hit.append(r)
            else:
                break
        if hit and len(hit) < lim_r:
            fg[min(hit) : max(hit) + 1] = 0
    for cols in (range(0, lim_c), range(w - 1, w - 1 - lim_c, -1)):
        hit = []
        for c in cols:
            if fg[:, c].mean() > coverage:
                hit.append(c)
            else:
                break
        if hit and len(hit) < lim_c:
            fg[:, min(hit) : max(hit) + 1] = 0


def _drop_edge_slivers(fg: np.ndarray, max_frac: float = 0.03, elongation: float = 4.0) -> None:
    """Clear thin foreground pieces that lie along an image edge they touch.

    (Lines of text on a page that runs off the bed touch the edge too, but
    run across it.)
    """
    h, w = fg.shape
    m = min(h, w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    for i in range(1, n):
        x, y, cw, ch, _area = stats[i]
        along_x = (y <= 1 or y + ch >= h - 1) and ch < max_frac * m and cw > elongation * ch
        along_y = (x <= 1 or x + cw >= w - 1) and cw < max_frac * m and ch > elongation * cw
        if along_x or along_y:
            fg[labels == i] = 0


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out


# ------------------------------------------------------------ rect fitting
def _profile_extent(coords: np.ndarray, frac: float = 0.1) -> tuple[float, float]:
    """Extent of a projected point cloud, ignoring sparse tails.

    ``coords`` are the projections of the object's pixels on one axis.  The
    count of pixels per unit bin is the object's thickness at that position;
    bins thinner than ``frac`` of the typical thickness (dust, hairs, thin
    shadows) are not part of the rectangle.
    """
    lo = float(coords.min())
    idx = np.floor(coords - lo).astype(np.int64)
    counts = np.bincount(idx)
    nz = counts[counts > 0]
    ref = float(np.percentile(nz, 75)) if nz.size else 1.0
    valid = np.nonzero(counts >= frac * ref)[0]
    if valid.size == 0:
        return lo, float(coords.max())
    return lo + valid[0], lo + valid[-1] + 1.0


def _box_for_angle(pts: np.ndarray, angle: float, frac: float = 0.1):
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    u = pts[:, 0] * c + pts[:, 1] * s
    v = -pts[:, 0] * s + pts[:, 1] * c
    u0, u1 = _profile_extent(u, frac)
    v0, v1 = _profile_extent(v, frac)
    return u0, u1, v0, v1


def fit_rectangle(mask: np.ndarray, max_points: int = 60000) -> CropBox | None:
    """Fit a rotated rectangle to a binary mask, robust to small protrusions."""
    ys, xs = np.nonzero(mask)
    n = ys.size
    if n < 20:
        return None
    pts_all = np.stack([xs, ys], 1).astype(np.float64)
    step = max(1, n // max_points)
    # Sub-sampling keeps the profile shape; scale so thickness stays in pixels.
    pts = pts_all[::step]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack(contours))
    (_, _), (_, _), a0 = cv2.minAreaRect(hull)
    a0 = CropBox(0, 0, 1, 2, a0).angle

    def area(angle: float) -> float:
        u0, u1, v0, v1 = _box_for_angle(pts, angle, 0.1)
        return (u1 - u0) * (v1 - v0)

    cands = [a0, 0.0]
    best = min(cands, key=area)
    for span, stp in ((3.0, 0.25), (0.3, 0.05)):
        grid = best + np.arange(-span, span + 1e-9, stp)
        best = float(min(grid, key=area))
    frac = 0.1
    u0, u1, v0, v1 = _box_for_angle(pts_all, best, frac)
    t = math.radians(best)
    c, s = math.cos(t), math.sin(t)
    uc, vc = (u0 + u1) / 2.0 - 0.5, (v0 + v1) / 2.0 - 0.5
    # Projections used pixel centres; a pixel spans +-0.5, hence the extents
    # (``valid[-1] + 1``) are already edge-to-edge in the rotated frame.
    cx = uc * c - vc * s
    cy = uc * s + vc * c
    box = CropBox(cx, cy, u1 - u0, v1 - v0, best)
    box.meta["fill"] = float(n / max(box.area, 1.0))
    return box


# --------------------------------------------------------------- splitting
def _split_component(comp: np.ndarray, rgb8: np.ndarray, min_px: int, depth: int = 0) -> list[np.ndarray]:
    """Split a blob made of several touching rectangles, if it looks like one."""
    box = fit_rectangle(comp)
    if box is None:
        return [comp]
    fill = box.meta["fill"]
    if fill > 0.9 or depth > 2:
        return [comp]
    area = int(comp.sum())
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    maxd = float(dist.max())
    if maxd < 3:
        return [comp]
    for frac in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        cores = (dist > frac * maxd).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
        if n < 3:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, a_ in enumerate(areas) if a_ >= max(20, 0.04 * areas.sum())]
        if len(keep) < 2:
            continue
        markers = np.zeros(comp.shape, np.int32)
        for j, i in enumerate(keep):
            markers[lab == i] = j + 1
        outside = len(keep) + 1
        markers[comp == 0] = outside
        ws = cv2.watershed(rgb8, markers.copy())
        parts = []
        ok = True
        for j in range(1, len(keep) + 1):
            part = ((ws == j) & (comp > 0)).astype(np.uint8)
            part = cv2.morphologyEx(part, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            pa = int(part.sum())
            if pa < min_px:
                ok = False
                break
            pb = fit_rectangle(part)
            if pb is None or pb.meta["fill"] < max(0.85, fill + 0.04):
                ok = False
                break
            parts.append(part)
        if ok and sum(int(p.sum()) for p in parts) > 0.85 * area:
            out = []
            for p in parts:
                out.extend(_split_component(p, rgb8, min_px, depth + 1))
            return out
    return [comp]


# ----------------------------------------------------- border/edge search
def _sample_side(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, n: np.ndarray,
                 d_lo: float, d_hi: float, t_step: float = 1.0, d_step: float = 1.0,
                 trim: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``img`` on a grid aligned with a box side.

    Returns ``(S, ts, ds)`` where ``S[i, j]`` is the value at offset ``ds[i]``
    along the outward normal ``n`` from the point at ``ts[j]`` along the side.
    The ends of the side (``trim`` fraction) are skipped: photo corners are
    often rounded or bent.
    """
    side = p1 - p0
    length = float(np.linalg.norm(side))
    tdir = side / max(length, 1e-9)
    ts = np.arange(trim * length, (1 - trim) * length, t_step)
    ds = np.arange(d_lo, d_hi + 1e-9, d_step)
    if ts.size == 0 or ds.size == 0:
        return np.zeros((0, 0), np.float32), ts, ds
    T, D = np.meshgrid(ts, ds)
    X = (p0[0] + T * tdir[0] + D * n[0]).astype(np.float32)
    Y = (p0[1] + T * tdir[1] + D * n[1]).astype(np.float32)
    S = cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return S, ts, ds


def _box_sides(box: CropBox):
    """Yield ``(p0, p1, outward_normal, axis, sign)`` for the four sides.

    ``axis`` 0 means the side moves along the box's x axis (left/right sides),
    ``sign`` is +1 for the right/bottom side and -1 for the left/top side.
    """
    tl, tr, br, bl = box.corners()
    ux, uy = box.axes
    return [
        (tl, tr, -uy, 1, -1),  # top
        (tr, br, ux, 0, +1),  # right
        (br, bl, uy, 1, +1),  # bottom
        (bl, tl, -ux, 0, -1),  # left
    ]


def _apply_side_shifts(box: CropBox, shifts: dict[tuple[int, int], float]) -> CropBox:
    """Move sides outward by the given amounts (negative = inward)."""
    ux, uy = box.axes
    lo_u, hi_u = -box.w / 2 - shifts.get((0, -1), 0.0), box.w / 2 + shifts.get((0, 1), 0.0)
    lo_v, hi_v = -box.h / 2 - shifts.get((1, -1), 0.0), box.h / 2 + shifts.get((1, 1), 0.0)
    c = np.array([box.cx, box.cy]) + ux * (lo_u + hi_u) / 2 + uy * (lo_v + hi_v) / 2
    return replace(box, cx=c[0], cy=c[1], w=max(1.0, hi_u - lo_u), h=max(1.0, hi_v - lo_v),
                   meta=dict(box.meta))


@dataclass
class _BorderProfile:
    prof: np.ndarray  # |mean gradient| along the normal, per offset
    texture: np.ndarray  # mean |gradient| per offset
    ds: np.ndarray  # offsets (pixels, outward)
    diff: np.ndarray  # colour difference to the lid, (offset, position)
    reached: np.ndarray  # fraction of positions that met another item
    start: int  # first offset index considered (min border width)
    base: float
    spread: float
    grad: np.ndarray  # signed gradient along the normal, (offset, position)


def _border_profile(a: _Analysis, p0, p1, n, max_dist: float, min_dist: float,
                    other_f: np.ndarray) -> _BorderProfile | None:
    S, ts, ds = _sample_side(a.L, p0, p1, n, -2.0, max_dist)
    if S.size == 0:
        return None
    valid = ~np.isnan(S)
    # Only use offsets that are inside the image along most of the side.
    row_ok = valid.mean(axis=1) >= 0.6
    last = len(ds) if row_ok.all() else int(np.argmin(row_ok))
    start = int(np.searchsorted(ds, min_dist))
    if last < start + 4:
        return None
    cols = valid[:last].all(axis=0)
    if cols.sum() < 10:
        return None
    S = S[:last, cols]
    ds = ds[:last]
    D = _sample_side(a.diff, p0, p1, n, -2.0, max_dist)[0][:last, cols]
    O = _sample_side(other_f, p0, p1, n, -2.0, max_dist)[0][:last, cols] > 0.5
    g = np.gradient(S, axis=0)
    prof = np.abs(g.mean(axis=1))  # signed mean: texture cancels out
    tail = prof[start:]
    return _BorderProfile(
        prof=prof,
        texture=np.abs(g).mean(axis=1),
        ds=ds,
        diff=D,
        reached=np.maximum.accumulate(O, axis=0).mean(axis=1),
        start=start,
        base=float(np.median(tail)),
        spread=max(_robust_sigma(tail), 0.02),
        grad=g,
    )


def _edge_coverage(bp: _BorderProfile, i: int, j0: int, j1: int) -> float:
    """Fraction of positions along the side where the edge at offset ``i`` is
    visible above the lid's gradient noise.

    A paper edge runs the full length of a side; a hair or a speck of dust
    that crosses the search band only shows up at a few positions, even when
    it makes the side-averaged profile look significant.
    """
    g = bp.grad
    pol = 1.0 if g[i].mean() >= 0 else -1.0
    near = (g[max(0, i - 1) : i + 2] * pol).max(axis=0)
    sigma = max(_robust_sigma(g[j0:j1].ravel()), 1e-3)
    return float(np.mean(near > 3.0 * sigma))


def _border_search(bp: _BorderProfile, threshold: float, lid_threshold: float, k: int,
                   lo_i: int | None = None, hi_i: int | None = None,
                   min_coverage: float = 0.5) -> int | None:
    """Walk outward and return the index of the first acceptable paper edge."""
    prof, texture, ds = bp.prof, bp.texture, bp.ds
    lo_i = bp.start + 1 if lo_i is None else max(lo_i, bp.start + 1)
    hi_i = len(ds) - 1 if hi_i is None else min(hi_i, len(ds) - 1)
    for i in range(bp.start + 1, hi_i):
        if bp.reached[min(i + 1, len(ds) - 1)] > 0.01:
            return None  # the band would swallow part of another item
        j0 = i + 3
        j1 = min(len(ds), j0 + k)
        # The band's outer part, minus the blur of the edge itself.  Its inner
        # part may still hold the soft edge of the image content.
        lo = bp.start + int(0.4 * (i - bp.start))
        hi = i - 2
        if hi > lo and j1 - j0 >= 2:
            band_tex = float(np.mean(texture[lo:hi]))
            if band_tex > max(1.2, 3.0 * float(np.median(texture[j0:j1]))):
                return None  # walking into image content
        if i < lo_i:
            continue
        if prof[i] < threshold or prof[i] < prof[i - 1] or prof[i] < prof[i + 1]:
            continue
        if j1 - j0 < 2:
            return None
        if np.mean(bp.diff[j0:j1] < lid_threshold) < 0.6:
            continue  # beyond this edge is not the lid
        if _edge_coverage(bp, i, j0, j1) < min_coverage:
            continue  # a hair or speck, not a paper edge
        return i
    return None


def _find_borders(a: _Analysis, box: CropBox, max_dist: float, others: np.ndarray,
                  min_dist: float = 3.0) -> CropBox:
    """Push each side outward to a paper edge, e.g. a print's white border.

    For every side the gradient along the normal is averaged along the whole
    side.  Averaging makes faint but straight edges (paper on a lid of the same
    colour) stand out from noise.  A side moves to the first strong edge
    beyond the content when

    * the band it would add is smooth (a border or page margin, not image
      content),
    * the band does not reach into another detected item, and
    * what lies beyond the edge looks like the lid.

    Edges closer than ``min_dist`` are ignored: those are lid shadows.  Prints
    have borders of about the same width all round, so once two sides have
    found theirs, the remaining sides are searched again near that width with
    a lower threshold (a side without a lid shadow has a much fainter edge).
    """
    shifts: dict[tuple[int, int], float] = {}
    other_f = others.astype(np.float32)
    k = int(max(3, min_dist))
    profiles: dict[tuple[int, int], _BorderProfile] = {}
    for p0, p1, n, axis, sign in _box_sides(box):
        bp = _border_profile(a, p0, p1, n, max_dist, min_dist, other_f)
        if bp is None:
            continue
        profiles[(axis, sign)] = bp
        thr = max(bp.base + 8 * bp.spread, 0.35)
        i = _border_search(bp, thr, a.threshold, k)
        if i is not None:
            shifts[(axis, sign)] = float(bp.ds[i])
    if len(shifts) >= 2:
        widths = np.array(list(shifts.values()))
        med = float(np.median(widths))
        if widths.max() <= 1.4 * widths.min() + 2:
            for key, bp in profiles.items():
                if key in shifts:
                    continue
                lo_i = int(np.searchsorted(bp.ds, 0.6 * med - 1))
                hi_i = int(np.searchsorted(bp.ds, 1.4 * med + 1)) + 1
                thr = max(bp.base + 3 * bp.spread, 0.15)
                i = _border_search(bp, thr, a.threshold, k, lo_i, hi_i, min_coverage=0.35)
                if i is not None:
                    shifts[key] = float(bp.ds[i])
    if shifts:
        box = _apply_side_shifts(box, shifts)
        box.meta["border_shift"] = shifts
    return box


@dataclass
class _SideFit:
    shift: float = 0.0  # outward shift of the side's midpoint
    dtheta: float = 0.0  # rotation of the side line (degrees)
    weight: float = 0.0  # number of inlier samples
    reason: str = ""  # why the side was rejected (empty when accepted)


def _pick_edge(profile: np.ndarray, ds: np.ndarray, shadow_px: float) -> int | None:
    """Choose the item edge in a side-averaged gradient profile.

    Takes the strongest peak, mildly favouring peaks close to the current
    estimate (a neighbouring photo's edge may be in the window too).  If
    another strong peak lies just inside the chosen one, within
    ``shadow_px``, that one wins: the chosen peak is then the far side of a
    lid shadow and the inner one is the real paper edge.
    """
    mx = float(profile.max())
    if mx <= 0:
        return None
    peaks = [
        i
        for i in range(1, len(profile) - 1)
        if profile[i] >= 0.3 * mx and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]
    ]
    if not peaks:
        return int(np.argmax(profile))
    r = max(abs(ds[0]), abs(ds[-1]), 1e-6)
    best = max(peaks, key=lambda i: profile[i] * (1.0 - 0.5 * (ds[i] / r) ** 2))
    for i in peaks:
        if i < best and ds[best] - ds[i] <= shadow_px and profile[i] >= 0.5 * profile[best]:
            return i
    return best


def chroma(image: np.ndarray) -> np.ndarray:
    """Float32 a*b* chroma (distance from neutral grey) of an RGB image (8 or 16 bit, or float)."""
    img = np.asarray(image)
    if img.ndim == 2 or img.shape[2] == 1:
        return np.zeros(img.shape[:2], np.float32)
    rgb = img[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(to_float_rgb(rgb), 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2Lab)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    return np.sqrt(a * a + b * b)


def _past_printed_line(C: np.ndarray, p0: np.ndarray, p1: np.ndarray, n: np.ndarray, r: float,
                       ds: np.ndarray, centre: int, shadow_px: float) -> int:
    """Coloured paper on both sides of the chosen step: the paper goes on beyond it, so it is a
    printed line along the edge; the paper edge is the steepest drop to neutral lid further out
    (``Profile.paper_colour_edges``)."""
    lim = profiles.current().paper_colour_edges
    Cs, _, cds = _sample_side(C, p0, p1, n, -r, r, t_step=2.0, d_step=0.5, trim=0.08)
    if Cs.size == 0 or len(cds) != len(ds):
        return centre
    with np.errstate(all="ignore"):
        prof = np.nanmedian(Cs, axis=1)
    k = max(2, int(round(0.4 * shadow_px / max(ds[1] - ds[0], 1e-6))))
    at = lambda i: prof[min(max(i, 0), len(prof) - 1)]
    if not (at(centre - k) > lim and at(centre + k) > lim):
        return centre
    neutral = [j for j in range(centre + k, len(prof)) if prof[j] <= 0.5 * lim]
    if not neutral:
        return centre
    seg = prof[centre + 1 : neutral[0] + 1]
    return centre + 1 + int(np.argmin(np.gradient(seg))) if seg.size > 1 else neutral[0]


def _fit_side(L: np.ndarray, p0: np.ndarray, p1: np.ndarray, n: np.ndarray, r: float,
              exclude_shadow: bool = True, shadow_px: float = 6.0, C: np.ndarray | None = None) -> _SideFit:
    """Locate one side of a box precisely.

    The edge is first chosen on the gradient profile averaged along the whole
    side, then re-located on many short profiles across the side with
    sub-sample accuracy, and a line is fitted through those points robustly.
    Returns the shift of the side and its rotation.
    """
    length = float(np.linalg.norm(p1 - p0))
    if length < 20:
        return _SideFit(reason="short")
    S, ts, ds = _sample_side(L, p0, p1, n, -r, r, t_step=1.0, d_step=0.5, trim=0.08)
    if S.size == 0:
        return _SideFit(reason="empty")
    # Average along the side in short runs (anisotropic smoothing): noise
    # drops while the edge, which runs along the side, stays sharp.
    run = max(3, int(length / 60))
    ncols = S.shape[1] // run
    if ncols < 6:
        return _SideFit(reason="few columns")
    S = S[:, : ncols * run].reshape(S.shape[0], ncols, run)
    # Near the image border part of the search window falls outside the
    # image: keep the offsets that are inside for most of the side.
    row_ok = (~np.isnan(S)).all(axis=2).mean(axis=1) >= 0.8
    centre_row = len(ds) // 2
    if not row_ok[centre_row]:
        return _SideFit(reason="outside image")
    r0 = centre_row
    while r0 > 0 and row_ok[r0 - 1]:
        r0 -= 1
    r1 = centre_row
    while r1 < len(ds) - 1 and row_ok[r1 + 1]:
        r1 += 1
    if r1 - r0 < 6:
        return _SideFit(reason="outside image")
    S = S[r0 : r1 + 1]
    ds = ds[r0 : r1 + 1]
    good = ~np.isnan(S).any(axis=(0, 2))
    if good.sum() < 6:
        return _SideFit(reason="outside image")
    S = S[:, good].mean(axis=2)
    tcent = ts[: ncols * run].reshape(ncols, run).mean(axis=1)[good]
    S = cv2.GaussianBlur(S.astype(np.float32), (1, 5), sigmaX=0, sigmaY=1.0)
    g = np.gradient(S, axis=0)
    # Dominant polarity: the item/lid step has a consistent sign along a side.
    gm = g.mean(axis=1)
    pol = 1.0 if gm[np.argmax(np.abs(gm))] >= 0 else -1.0
    gp = np.maximum(g * pol, 0)
    mean_prof = np.maximum(gm * pol, 0)
    step = ds[1] - ds[0]
    centre = _pick_edge(mean_prof, ds, shadow_px if exclude_shadow else 0.0)
    if centre is None:
        return _SideFit(reason="flat")
    moved = 0.0
    if C is not None:
        c0, centre = centre, _past_printed_line(C, p0, p1, n, r, ds, centre, shadow_px)
        moved = abs(float(ds[centre] - ds[c0]))
    # Per-column search window around the side-averaged edge; wide enough to
    # follow a residual tilt of ~0.35 degrees along the side.
    half = 3.0 + 0.5 * length * math.tan(math.radians(0.35))
    if moved:
        half = min(half, max(1.5, 0.45 * moved))  # stay clear of the printed line passed over
    lo = max(0, int(centre - half / step))
    hi = min(len(ds), int(centre + half / step) + 1)
    offs = np.full(gp.shape[1], np.nan)
    strengths = np.zeros(gp.shape[1])
    for j in range(gp.shape[1]):
        col = gp[lo:hi, j]
        mx = col.max()
        if mx <= 0:
            continue
        i = int(np.argmax(col))
        di = 0.0
        if 0 < i < len(col) - 1:
            den = col[i - 1] - 2 * col[i] + col[i + 1]
            if abs(den) > 1e-9:
                di = float(np.clip(0.5 * (col[i - 1] - col[i + 1]) / den, -1, 1))
        offs[j] = ds[lo] + (i + di) * step
        strengths[j] = mx
    # Peaks must stand out from the gradient noise of this strip.
    noise = float(np.median(np.abs(g))) + 1e-6
    ok = ~np.isnan(offs) & (strengths > 3 * noise)
    if ok.sum() < max(6, 0.3 * len(offs)):
        return _SideFit(reason=f"weak edge ({int(ok.sum())}/{len(offs)})")
    t_ok, d_ok = tcent[ok], offs[ok]
    # Robust line fit d = a + b * t (iteratively trimmed least squares).
    inl = np.ones(len(t_ok), bool)
    a_, b_ = float(np.median(d_ok)), 0.0
    for _ in range(5):
        if inl.sum() < 4:
            break
        A = np.stack([np.ones(inl.sum()), t_ok[inl]], 1)
        (a_, b_), *_ = np.linalg.lstsq(A, d_ok[inl], rcond=None)
        res = np.abs(d_ok - (a_ + b_ * t_ok))
        tol = max(1.0, 2.5 * _robust_sigma(res[inl]))
        inl = res < tol
    if inl.sum() < max(5, 0.35 * len(offs)):
        return _SideFit(reason=f"no line ({int(inl.sum())}/{len(offs)})")
    # Sides run clockwise on screen with the outward normal on their left, so
    # an outward slope b turns the side by -atan(b) in the box angle sense.
    return _SideFit(
        shift=float(a_ + b_ * length / 2.0),
        dtheta=-math.degrees(math.atan(b_)),
        weight=float(inl.sum()),
    )


def _refine_edges(L: np.ndarray, box: CropBox, radius: float, exclude_shadow: bool = True,
                  iterations: int = 2, shadow_px: float = 6.0,
                  skip: set[tuple[int, int]] | None = None, C: np.ndarray | None = None) -> CropBox:
    """Locate each side precisely and re-estimate the angle.

    Works directly on a (large) full-resolution luminance image: only thin
    strips along the four sides are sampled.  Sides listed in ``skip`` (as
    ``(axis, sign)`` keys, see ``_box_sides``) are left where they are, e.g.
    the spine side of a booklet page, which is not a paper edge.
    """
    reasons: dict[str, str] = {}
    for it in range(iterations):
        r = radius if it == 0 else max(3.0, radius / 3.0)
        shifts: dict[tuple[int, int], float] = {}
        turns: list[tuple[float, float]] = []
        for (p0, p1, n, axis, sign), name in zip(_box_sides(box), ("top", "right", "bottom", "left")):
            if skip and (axis, sign) in skip:
                continue
            fit = _fit_side(L, p0, p1, n, r, exclude_shadow, shadow_px, C)
            if fit.reason:
                reasons[name] = fit.reason
                continue
            reasons.pop(name, None)
            shifts[(axis, sign)] = fit.shift
            turns.append((fit.dtheta, fit.weight))
        if not shifts:
            break
        new = _apply_side_shifts(box, shifts)
        if turns:
            wsum = sum(wt for _, wt in turns)
            dth = float(np.clip(sum(d * wt for d, wt in turns) / wsum, -1.5, 1.5))
            new = replace(new, angle=new.angle + dth, meta=dict(new.meta))
        box = new
        box.meta["refined_sides"] = len(shifts)
    box.meta["refine_failures"] = reasons
    return box


# --------------------------------------------------------------- main API
def _working_image(image: np.ndarray, work_size: int) -> tuple[np.ndarray, float]:
    """Downscale (without a full-size float copy) to the detection size."""
    H, W = image.shape[:2]
    src = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
    if src.dtype not in (np.uint8, np.uint16, np.float32):
        src = to_float_rgb(src)
    scale = min(1.0, work_size / max(H, W))
    if scale < 1:
        src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return to_float_rgb(src), scale


@dataclass
class _Coarse:
    """Result of detection at working resolution."""

    analysis: _Analysis
    fg: np.ndarray
    boxes: list[CropBox]  # in working-image pixels
    scale: float  # working / full resolution


def _coarse_detect(small: np.ndarray, scale: float, st: AutoCropSettings, dpi: float,
                   mode: str, band_frac: float = 0.03,
                   require_uniform: bool = True, surface: str = "all",
                   drop_edge_slivers: bool = False) -> _Coarse | None:
    """Find item boxes on the working image.

    Returns ``None`` when there is no plain lid to compare against (the item
    covers the whole bed).  ``drop_edge_slivers`` removes thin foreground
    pieces along the image edges before single-mode grouping (lid shading at
    the edge of the glass that would otherwise join a large item).
    """
    a = _analyze(small, st.sensitivity, band_frac, surface)
    if require_uniform and not a.uniform_bg:
        return None
    sh, sw = small.shape[:2]
    fg = _foreground_mask(a)
    _remove_frame_strips(fg)
    if drop_edge_slivers:
        _drop_edge_slivers(fg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_px = max(30, int(st.min_area * sh * sw))
    rgb8 = (np.clip(small, 0, 1) * 255).astype(np.uint8)

    comps: list[np.ndarray] = []
    m = min(sh, sw)
    if mode == "single":
        # Text is many small blobs; group them into blocks first so that
        # short words are not mistaken for dust.
        k = _odd(0.025 * m)
        blocks = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(blocks, connectivity=8)
        min_px = max(30, int(st.min_area * 0.25 * sh * sw))
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_px:
            continue
        comp = (labels == i).astype(np.uint8)
        if mode == "single":
            comp &= fg
        # Thin strips touching the image edge are the scanner bed frame.
        touches = x <= 1 or y <= 1 or x + cw >= sw - 1 or y + ch >= sh - 1
        if touches and min(cw, ch) < 0.03 * m and max(cw, ch) > 6 * min(cw, ch):
            continue
        comps.append(comp)

    if mode == "single":
        if not comps:
            return _Coarse(a, fg, [], scale)
        union = np.zeros_like(fg)
        for c in comps:
            union |= c
        comps = [union]
    elif st.split_touching:
        split: list[np.ndarray] = []
        for c in comps:
            split.extend(_split_component(c, rgb8, min_px))
        comps = split

    all_fg = np.zeros_like(fg)
    for c in comps:
        all_fg |= c
    boxes: list[CropBox] = []
    px_per_mm_small = dpi * scale / MM_PER_INCH
    min_border = max(3.0, 1.5 * px_per_mm_small)
    for c in comps:
        box = fit_rectangle(c)
        if box is None:
            continue
        others = (all_fg & (1 - c)).astype(np.uint8)
        if mode == "single":
            max_dist = 0.35 * min(sw, sh)
        else:
            max_dist = max(4.0, st.border_search_mm * px_per_mm_small)
            max_dist = min(max_dist, 0.25 * min(box.w, box.h) + 4)
        if max_dist > min_border + 2:
            box = _find_borders(a, box, max_dist, others, min_border)
        box.confidence = float(np.clip(box.meta.get("fill", 0.5), 0, 1))
        boxes.append(box)
    return _Coarse(a, fg, boxes, scale)


def _finish(image: np.ndarray, boxes: list[CropBox], scale: float, st: AutoCropSettings,
            dpi: float, skip_sides: list[set[tuple[int, int]] | None] | None = None) -> list[CropBox]:
    """Scale working-resolution boxes to the full image, refine and tidy them."""
    H, W = image.shape[:2]
    out: list[CropBox] = []
    Lfull = luminance(image) if st.edge_refine and boxes else None
    Cfull = chroma(image) if st.edge_refine and boxes and profiles.current().paper_colour_edges > 0 else None
    margin_px = st.margin_mm * dpi / MM_PER_INCH
    for k, box in enumerate(boxes):
        b = box.scaled(1.0 / scale) if scale < 1 else box
        b.meta = dict(box.meta)
        if st.edge_refine:
            radius = max(4.0, 6.0 / scale)
            skip = skip_sides[k] if skip_sides else None
            b = _refine_edges(Lfull, b, radius, shadow_px=1.0 * dpi / MM_PER_INCH, skip=skip, C=Cfull)
        if not st.deskew:
            x0, y0, x1, y1 = b.bounding_rect()
            b = replace(b, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0, angle=0.0,
                        meta=dict(b.meta))
        elif abs(b.angle) < st.snap_angle:
            b = replace(b, angle=0.0, meta=dict(b.meta))
        b = b.inset(margin_px)
        b = b.clamp_to(W, H)
        out.append(b)
    return out


def detect(image: np.ndarray, settings: AutoCropSettings | None = None, dpi: float | None = None) -> list[CropBox]:
    """Detect crop boxes on a scan.

    ``image`` may be 8/16-bit or float, grey or RGB.  Returned boxes are in
    the image's pixel coordinates, in reading order.  The scanner profile
    ``settings.profile`` is active while it runs.
    """
    st = settings or AutoCropSettings()
    with profiles.use(st.profile):
        return _detect(image, st, dpi)


def _detect(image: np.ndarray, st: AutoCropSettings, dpi: float | None) -> list[CropBox]:
    H, W = image.shape[:2]
    full = CropBox.from_xywh(0, 0, W, H, confidence=0.0)
    if st.mode == "off":
        return [full]
    dpi = dpi or st.default_dpi
    if st.mode == "booklet":
        from .booklet import detect_booklet

        return detect_booklet(image, st, dpi)
    small, scale = _working_image(image, st.work_size)
    coarse = _coarse_detect(small, scale, st, dpi, st.mode)
    if coarse is None:
        # The item fills the bed (or there is no plain lid to compare with).
        return [full]
    out = _finish(image, coarse.boxes, scale, st, dpi)
    if not out:
        return [] if st.mode == "multi" else [full]
    return reading_order(out)


def refine(image: np.ndarray, boxes: list[CropBox], settings: AutoCropSettings | None = None,
           dpi: float | None = None, radius: float | None = None) -> list[CropBox]:
    """Re-run edge refinement on boxes found on a preview.

    Typical use: detect on a low-resolution preview, scale the boxes to the
    final scan, then call ``refine`` so edges land exactly on the high-res
    image (and small shifts between the two passes are corrected).  Manual
    boxes are returned untouched, and sides listed in a box's
    ``meta["keep_sides"]`` (a booklet page's spine) are not moved.
    """
    st = settings or AutoCropSettings()
    with profiles.use(st.profile):
        return _refine(image, boxes, st, dpi, radius)


def _refine(image: np.ndarray, boxes: list[CropBox], st: AutoCropSettings, dpi: float | None,
            radius: float | None) -> list[CropBox]:
    dpi = dpi or st.default_dpi
    H, W = image.shape[:2]
    L = luminance(image)
    C = chroma(image) if profiles.current().paper_colour_edges > 0 else None
    out = []
    margin_px = st.margin_mm * dpi / MM_PER_INCH
    r = radius if radius is not None else max(4.0, 1.5 * dpi / 100.0)
    for b in boxes:
        if b.manual:
            out.append(b)
            continue
        if st.mode == "booklet" and ("top_page" in b.meta or "spread_pages" in b.meta):
            # Booklet pages end at the top page's edge: find it again (the plain edge snap
            # would jump to a page-stack line next to it).
            from .booklet import refine_booklet

            out.append(refine_booklet(image, b, st, dpi))
            continue
        # Undo the margin so that refinement starts on the edge itself.
        start = b.inset(-margin_px)
        keep = {tuple(k) for k in b.meta.get("keep_sides", [])}
        nb = _refine_edges(L, start, r, shadow_px=1.0 * dpi / MM_PER_INCH, skip=keep or None, C=C)
        if not st.deskew:
            x0, y0, x1, y1 = nb.bounding_rect()
            nb = replace(nb, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0, angle=0.0)
        elif abs(nb.angle) < st.snap_angle:
            nb = replace(nb, angle=0.0)
        out.append(nb.inset(margin_px).clamp_to(W, H))
    return out


def debug_overlay(image: np.ndarray, boxes: list[CropBox], truth: list[CropBox] | None = None) -> np.ndarray:
    """Draw boxes on an 8-bit RGB copy of the image (for --debug output)."""
    rgb8 = (np.clip(to_float_rgb(image), 0, 1) * 255).astype(np.uint8).copy()
    th = max(2, int(max(image.shape[:2]) / 600))
    for b in truth or []:
        cv2.polylines(rgb8, [np.round(b.corners()).astype(np.int32)], True, (0, 160, 255), th)
    for i, b in enumerate(boxes):
        pts = np.round(b.corners()).astype(np.int32)
        cv2.polylines(rgb8, [pts], True, (255, 40, 40), th)
        cv2.putText(rgb8, str(i + 1), tuple(int(v) for v in pts[0] + 6 * th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * th, (255, 40, 40), th)
    return rgb8
