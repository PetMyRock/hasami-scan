"""Booklet mode: crop the page of an open booklet that is fully on the glass.

A booklet or magazine opened flat is often wider than the scanner bed, so a
scan shows one complete page and part of the facing page, which runs off the
edge of the scan.  The general auto-crop would happily return everything on
the bed; this mode returns only the complete page, from its outer edge to the
spine.

Steps:

1. Find the visible part of the booklet with the single-item detector
   (content, then outward to the paper edges).  The lid colour is learnt from
   a thin band at the image border because the booklet covers most of it.
2. Decide which side runs off the bed: either the booklet reaches the image
   border there, or its top and bottom paper edges continue to the border.
3. Find the spine, parallel to that side, from a brightness profile across
   the booklet.  The gutter shadow and the fold line make a valley in the
   paper brightness; when the booklet lies perfectly flat the centre of the
   blank gutter between the two text blocks is used instead.
4. Return the part on the far side of the spine from the cut, refining its
   three paper edges at full resolution.  The spine side is not a paper edge
   and is left alone.  If no side is cut off, both pages are returned.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from .autocrop import (
    MM_PER_INCH,
    AutoCropSettings,
    _Analysis,
    _apply_side_shifts,
    _box_sides,
    _coarse_detect,
    _finish,
    _robust_sigma,
    _sample_side,
    _working_image,
    to_float_rgb,
)
from . import profiles
from .geometry import CropBox, reading_order

Side = tuple[int, int]  # (axis, sign) as produced by ``_box_sides``
DEBUG: dict | None = None  # set to a dict to collect the spine profiles (benchmarks)


@contextmanager
def _quiet():
    """Silence numpy's warnings about all-NaN slices (outside-image samples)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            yield


@dataclass
class _Spine:
    axis: int  # 0: spine runs along the box's y axis, 1: along its x axis
    offset: float  # position along ``axis`` measured from the box centre
    score: float
    method: str  # "shade", "seam" or "gap"


def _distance_to_border(p: np.ndarray, n: np.ndarray, w: int, h: int) -> float:
    """Distance from ``p`` along direction ``n`` to the image border."""
    ts = []
    for c, d, lim in ((p[0], n[0], w - 1), (p[1], n[1], h - 1)):
        if d > 1e-9:
            ts.append((lim - c) / d)
        elif d < -1e-9:
            ts.append(-c / d)
    return max(0.0, min(ts)) if ts else float("inf")


def _edge_reach(a: _Analysis, p0: np.ndarray, p1: np.ndarray, n: np.ndarray, w: int, h: int,
                window: float = 6.0, min_step: float = 0.0) -> tuple[bool, bool, bool]:
    """Follow the paper edge along box side ``p0 -> p1`` beyond both corners.

    Returns ``(is_edge, reaches_border_before_p0, reaches_border_after_p1)``.
    ``is_edge`` says whether the side is a paper edge along its own length
    (where the booklet is cut off by the scan it is not), a step of at least
    ``min_step`` L* per pixel.  The other two say
    whether the edge line carries on to the image border: a booklet that runs
    off the bed has top and bottom edges that reach the border on that side,
    while at a real page corner the edge stops.
    """
    length = float(np.linalg.norm(p1 - p0))
    if length < 20:
        return False, False, False
    tdir = (p1 - p0) / length
    back = _distance_to_border(p0, -tdir, w, h)
    ahead = _distance_to_border(p1, tdir, w, h)
    start = p0 - back * tdir
    total = back + length + ahead
    ts = np.arange(0.0, total, 1.0)
    ds = np.arange(-window, window + 1e-9, 1.0)
    T, D = np.meshgrid(ts, ds)
    X = (start[0] + T * tdir[0] + D * n[0]).astype(np.float32)
    Y = (start[1] + T * tdir[1] + D * n[1]).astype(np.float32)
    S = cv2.remap(a.L, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    g = np.gradient(S, axis=0)
    inside = (ts >= back + 0.2 * length) & (ts <= back + 0.8 * length)
    with _quiet():
        mean = np.nanmean(g[:, inside], axis=1)
    if np.all(np.isnan(mean)):
        return False, False, False
    k = int(np.nanargmax(np.abs(mean)))
    if abs(mean[k]) < min_step:
        return False, False, False  # a soft hump in the lid's brightness, not a paper edge
    # A paper edge has lid beyond it.  At the image border
    # there is nothing beyond (a step there is the scanner's edge, or content
    # of a page that runs off the bed), so look for lid-coloured pixels past
    # the step.  A slightly rotated booklet against the edge of the glass
    # leaves only a tapering sliver of lid, so part of the side is enough.
    Dm = cv2.remap(a.diff, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    with _quiet():
        lid = np.nanmean(np.where(np.isnan(Dm[:, inside]), 0.0, Dm[:, inside] < a.threshold), axis=1)
    lid_beyond = int(np.sum(lid[k + 2 :] > 0.3)) >= 2
    if DEBUG is not None:
        DEBUG.setdefault("edges", []).append(dict(step=float(mean[k]), lid=lid_beyond, p0=p0.tolist(), n=n.tolist()))
    pol = 1.0 if mean[k] >= 0 else -1.0
    rows = slice(max(0, k - 2), k + 3)
    with _quiet():
        near = np.nanmax(g[rows] * pol, axis=0)
    # Noise level from rows away from the edge, inside the box portion.
    far_rows = [r for r in range(len(ds)) if abs(r - k) > 3]
    if far_rows:
        sigma = max(_robust_sigma(np.nan_to_num(g[far_rows][:, inside]).ravel()), 0.05)
    else:
        sigma = 0.05
    present = np.nan_to_num(near, nan=0.0) > 3.0 * sigma
    # Majority filter to bridge text that touches the edge, dust, etc.
    win = 7
    frac = np.convolve(present.astype(np.float32), np.ones(win) / win, mode="same")
    present = frac >= 0.4
    if present[inside].mean() < 0.6:
        return False, False, False  # not a paper edge

    def reaches(seg: np.ndarray) -> bool:
        # ``seg`` runs from near the box corner to the image border.  The edge
        # must be there along most of it and right up to the border; at a
        # real page corner it stops short of the border.
        if seg.size < 3:
            return False
        if seg.size < 12:  # the booklet ends just short of the border
            return bool(seg.mean() >= 0.8 and seg[-1])
        trailing = 0
        for p in seg[::-1]:
            if p:
                break
            trailing += 1
        longest_gap, gap = 0, 0
        for p in seg:
            gap = 0 if p else gap + 1
            longest_gap = max(longest_gap, gap)
        return bool(seg.mean() >= 0.6 and trailing <= 4 and longest_gap <= 8)

    tail = 0.15 * length
    i_lo = int(back + tail)
    i_hi = int(back + length - tail)
    return lid_beyond, reaches(present[:i_lo][::-1]), reaches(present[i_hi:])


def _cut_sides(a: _Analysis, box: CropBox, w: int, h: int
               ) -> tuple[dict[Side, bool], dict[Side, bool], dict[Side, bool], dict[Side, bool]]:
    """For each side of the visible booklet: does the booklet run off the bed?

    Returns ``(cut, reach, is_edge, touch)``.  ``is_edge`` marks sides seen as
    a paper edge with lid beyond it, ``touch`` sides at the image border.  A side is cut when the booklet touches the
    image border there, or when the paper edges of the two neighbouring sides
    run on to the border.  ``reach`` marks the second, stronger kind of
    evidence: a booklet that merely touches the border may just be placed
    against the edge of the glass.
    """
    tl, tr, br, bl = box.corners()
    ux, uy = box.axes
    touch_px = max(3.0, 0.01 * min(w, h))
    touch: dict[Side, bool] = {}
    for p0, p1, n, axis, sign in _box_sides(box):
        # Along most of the side: a slightly rotated booklet running off the
        # bed meets the border at an angle.
        touch[(axis, sign)] = min(_distance_to_border(p0 + f * (p1 - p0), n, w, h) for f in (0.15, 0.5, 0.85)) <= touch_px
    # A box side at the border may just include the lid's shadow beside a
    # page edge a few millimetres in (a thick booklet props the lid up), so
    # the paper edge is looked for further inwards there.
    wide = max(6.0, 0.02 * min(w, h))
    win = {k: (wide if v else 6.0) for k, v in touch.items()}
    lines = {
        (1, -1): _edge_reach(a, tl, tr, -uy, w, h, win[(1, -1)]),  # top
        (1, 1): _edge_reach(a, bl, br, uy, w, h, win[(1, 1)]),  # bottom
        (0, -1): _edge_reach(a, tl, bl, -ux, w, h, win[(0, -1)]),  # left
        (0, 1): _edge_reach(a, tr, br, ux, w, h, win[(0, 1)]),  # right
    }
    # A side that is a paper edge along its length is a page edge, however
    # close to the border it lies; only the other sides can be cut off.
    is_edge = {k: v[0] for k, v in lines.items()}
    # Top and bottom edges decide about the left and right sides, and vice
    # versa.
    reach = {
        (0, -1): lines[(1, -1)][1] or lines[(1, 1)][1],
        (0, 1): lines[(1, -1)][2] or lines[(1, 1)][2],
        (1, -1): lines[(0, -1)][1] or lines[(0, 1)][1],
        (1, 1): lines[(0, -1)][2] or lines[(0, 1)][2],
    }
    cut: dict[Side, bool] = {}
    for key in touch:
        reach[key] = bool(reach[key] and not is_edge[key])
        cut[key] = bool((touch[key] or reach[key]) and not is_edge[key])
    return cut, reach, is_edge, touch


def _plain_strip(a: _Analysis, box: CropBox, key: Side, shift: float, max_edges: float = 0.01,
                 outer: float = 0.0, lines_ok: bool = False) -> bool:
    """Is the strip a side would give up (``shift`` < 0) free of sharp edges?

    Lid shading is soft; the visible part of a page that runs off the bed
    has print in it.  Only the strip from ``shift`` out to ``outer`` (<= 0)
    is looked at.  With ``lines_ok``, long straight lines along the side
    (the edge of the band a propped-up lid leaves, another page's edge) are
    not counted: print scatters its edges over the whole strip; nor are the
    soft outlines of shadows (print is sharp).
    """
    if shift >= outer:
        return True
    edges = (a.edges > 0).astype(np.float32)
    if lines_ok:
        grad = cv2.magnitude(cv2.Sobel(a.L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a.L, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
        edges *= grad > 3.0  # L* per pixel
    for p0, p1, n, axis, sign in _box_sides(box):
        if (axis, sign) == key:
            E, _, _ = _sample_side(edges, p0, p1, n, shift, outer, trim=0.05)
            if E.size == 0:
                return True
            with _quiet():
                if lines_ok:
                    line = np.nan_to_num(np.nanmean(E, axis=1)) >= 0.25
                    line = cv2.dilate(line.astype(np.uint8).reshape(-1, 1), np.ones((3, 1), np.uint8)).ravel() > 0
                    E = E[~line]
                    if E.size == 0:
                        return True
                return bool(np.nanmean(E) <= max_edges)
    return True


def _lid_coloured(a: _Analysis, p0: np.ndarray, p1: np.ndarray, n: np.ndarray, shift: float, outer: float) -> bool:
    """Is the strip from ``shift`` out to ``outer`` the lid's colour (a*b* within ``shrink_lid_hue``)?"""
    d = np.hypot(a.lab[:, :, 1] - a.bg_lab[1], a.lab[:, :, 2] - a.bg_lab[2]).astype(np.float32)
    D, _, _ = _sample_side(d, p0, p1, n, shift, outer, trim=0.05)
    with _quiet():
        med = np.nanmedian(D) if D.size else np.nan
    return bool(np.isnan(med) or med <= profiles.current().shrink_lid_hue)


def _trim_protrusions(fg: np.ndarray, box: CropBox, min_cover: float = 0.6,
                      keep: set[Side] | frozenset = frozenset(), a: _Analysis | None = None) -> CropBox:
    """Pull in sides that the foreground covers along only part of their length.

    On a black bed, dark fabric (a sleeve, a cloth) lying against the booklet
    is just bright enough to join it.  A paper edge is covered along its whole
    length; fabric beside the booklet only along part of it.
    """
    ux, uy = box.axes
    us = np.arange(-box.w / 2 + 0.5, box.w / 2)
    vs = np.arange(-box.h / 2 + 0.5, box.h / 2)
    if us.size < 20 or vs.size < 20:
        return box
    U, V = np.meshgrid(us, vs)
    X = (box.cx + U * ux[0] + V * uy[0]).astype(np.float32)
    Y = (box.cy + U * ux[1] + V * uy[1]).astype(np.float32)
    S = cv2.remap(fg.astype(np.float32), X, Y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    h, w = S.shape
    cols = S[int(0.1 * h) : int(0.9 * h)].mean(axis=0)
    rows = S[:, int(0.1 * w) : int(0.9 * w)].mean(axis=1)
    good_c, good_r = np.flatnonzero(cols >= min_cover), np.flatnonzero(rows >= min_cover)
    if good_c.size < 0.5 * w or good_r.size < 0.5 * h:
        return box
    min_c, min_r = max(3, int(0.01 * w)), max(3, int(0.01 * h))
    shifts = {}
    if good_c[0] >= min_c:
        shifts[(0, -1)] = -float(good_c[0])
    if w - 1 - good_c[-1] >= min_c:
        shifts[(0, 1)] = -float(w - 1 - good_c[-1])
    if good_r[0] >= min_r:
        shifts[(1, -1)] = -float(good_r[0])
    if h - 1 - good_r[-1] >= min_r:
        shifts[(1, 1)] = -float(h - 1 - good_r[-1])
    shifts = {k: v for k, v in shifts.items() if k not in keep and (a is None or _plain_strip(a, box, k, v))}
    return _apply_side_shifts(box, shifts) if shifts else box


def _shrink_to_paper_edges(a: _Analysis, box: CropBox, keep: set[Side] | frozenset = frozenset(),
                           depth: float = 0.2, cover: float = 0.7) -> CropBox:
    """Pull each side in to the outermost straight edge near it.

    A lid propped up by a thick booklet casts large soft shadows beside it,
    which join the booklet in the foreground.  A paper edge is a step in
    brightness along nearly the whole side; shading is soft and irregular.
    Each side that is not already a paper edge with lid beyond it is moved
    in to the first such step from the outside, if there is one within
    ``depth`` of the box size.  (A pale page on a pale lid has a faint edge,
    and the first strong step inside it may be a printed rule.)
    """
    shifts = {}
    h, w = a.L.shape
    for p0, p1, n, axis, sign in _box_sides(box):
        # A side on the image border may sit on the scanner's own edge (a
        # dark line along the glass) or on the band a propped-up lid leaves:
        # a step that close to the border is not a paper edge, even when
        # ``_edge_reach`` takes it for one.
        # (Measured away from the corners: on a page that runs off the bed the
        # corners of the neighbouring sides are at the border.)
        border = min(_distance_to_border(p0 + f * (p1 - p0), n, w, h) for f in (0.15, 0.5, 0.85))
        margin = max(4.0, profiles.current().shrink_margin * min(w, h))
        if (axis, sign) in keep:
            continue  # cut off
        # Already a paper edge with lid beyond it?  Right on the border that
        # may be the band's edge instead; near it (a pale page on a pale lid
        # close to the edge of the glass) only a step to darker shadow
        # outside may replace it, not the start of the print inside the page.
        verified = border > 2.0 and _edge_reach(a, p0, p1, n, w, h, min_step=profiles.current().min_edge_step)[0]
        if verified and border > 2.0 * margin:
            continue
        perp = box.w if axis == 0 else box.h
        # A side on the image border may have a wide band of shaded lid
        # inside it (the lid gets darker towards one end of the bed).
        reach_in = depth * perp * (1.5 if border < margin else 1.0)
        S, _ts, ds = _sample_side(a.L, p0, p1, n, -reach_in, 3.0, trim=0.15)
        run = 5
        nc = S.shape[1] // run if S.size else 0
        if nc < 6:
            continue
        with _quiet():
            S = np.nanmean(S[:, : nc * run].reshape(S.shape[0], nc, run), axis=2)
        # Brightness step across 4 px: a pale page on a pale lid may differ
        # by only 4-5 L*, spread over 2-3 px.
        g = np.full_like(S, np.nan)
        g[2:-2] = S[4:] - S[:-4]
        valid = ~np.isnan(g)
        if not valid.any():
            continue
        # Noise from the typical step (most of the strip is plain paper or
        # lid; a printed band would inflate a standard deviation).
        thr = max(6.0 * float(np.median(np.abs(g[valid]))), 2.5)
        gz = np.nan_to_num(g)
        # A step one row either way still counts (the side may lean a little).
        k = np.ones((3, 1), np.uint8)
        up = cv2.dilate((gz > thr).astype(np.uint8), k)
        dn = cv2.dilate((gz < -thr).astype(np.uint8), k)
        nv = np.maximum(valid.sum(axis=1), 1)
        cov = np.maximum((up * valid).sum(axis=1), (dn * valid).sum(axis=1)) / nv
        cov[valid.mean(axis=1) <= 0.5] = 0
        usable = border - ds >= margin
        hits = np.flatnonzero((cov >= cover) & usable)
        if hits.size:
            shift = float(ds[hits.max()])
        else:
            # The two pages of a spread may end a little apart (one page
            # sticks out, or curls): no single line covers the side, but the
            # outermost step of nearly every column lies in a narrow band.
            steps = (np.abs(gz) > thr) & valid & usable[:, None]
            has = steps.any(axis=0)
            if has.mean() < 0.5:
                if DEBUG is not None:
                    DEBUG.setdefault("shrink", []).append(dict(side=(axis, sign), shift=0.0, plain=f"has {has.mean():.2f}", straight=False))
                continue
            outer_pos = np.array([ds[np.flatnonzero(steps[:, j]).max()] for j in np.flatnonzero(has)])
            band = max(3.0, 0.025 * perp)
            starts = np.sort(outer_pos)
            inside = np.array([np.sum((outer_pos >= s0) & (outer_pos <= s0 + band)) for s0 in starts])
            best = int(np.argmax(inside))
            if inside[best] < max(0.8 * has.sum(), 0.5 * has.size):
                if DEBUG is not None:
                    DEBUG.setdefault("shrink", []).append(dict(side=(axis, sign), shift=float(starts[best]), plain=f"band {inside[best] / has.size:.2f}", straight=False))
                continue
            sel = outer_pos[(outer_pos >= starts[best]) & (outer_pos <= starts[best] + band)]
            shift = float(np.percentile(sel, 90))
        if verified:
            # Paper to shadow is a step to darker outwards, at or just inside
            # the outer edge of the shadow; the end of the print inside a
            # page is a step to brighter (paper) outwards.
            cov_dn = (dn * valid).sum(axis=1) / nv
            near = (ds <= shift + 1.0) & (ds >= shift - max(6.0, 0.03 * perp))
            if not np.any(cov_dn[near] >= cover):
                continue
        # The strip given up must be plain lid or shadow: not counting the
        # new edge itself (a few px) or the scanner's edge.
        # (A strip that lies all within the border margin cannot be checked:
        # the side stays.)
        outer = min(0.0, border - margin)
        # The soft outlines of the lid's shadows and the edge of the band a
        # propped-up lid leaves are not print.
        plain = shift < -3.0 and shift + 4.0 < outer and _plain_strip(a, box, (axis, sign), shift + 4.0, outer=outer,
                                                                       lines_ok=True)
        if plain and profiles.current().shrink_lid_hue > 0:
            plain = _lid_coloured(a, p0, p1, n, shift + 4.0, outer)
        if DEBUG is not None:
            DEBUG.setdefault("shrink", []).append(dict(side=(axis, sign), shift=shift, plain=plain, straight=bool(hits.size),
                                                       border=border))
        if plain:
            shifts[(axis, sign)] = shift
    return _apply_side_shifts(box, shifts) if shifts else box


def _extend_to_border(box: CropBox, cut: dict[Side, bool], w: int, h: int) -> CropBox:
    """Extend the sides that run off the bed to the image border."""
    shifts = {}
    for p0, p1, n, axis, sign in _box_sides(box):
        if cut.get((axis, sign)):
            d = max(_distance_to_border(p0, n, w, h), _distance_to_border(p1, n, w, h))
            if d > 0.5:
                shifts[(axis, sign)] = d
    return _apply_side_shifts(box, shifts) if shifts else box


def _axis_frame(box: CropBox, axis: int):
    ux, uy = box.axes
    if axis == 0:
        return ux, uy, box.w, box.h
    return uy, ux, box.h, box.w


def _valley_depth(G: np.ndarray, i0: int, i1: int, W: int) -> np.ndarray:
    """How far ``G`` dips below the higher of its two neighbourhoods."""
    depth = np.zeros(len(G), np.float32)
    for i in range(i0, i1):
        left = G[max(0, i - W) : i]
        right = G[i + 1 : i + W + 1]
        if left.size and right.size:
            depth[i] = min(left.max(), right.max()) - G[i]
    return depth


def _blur1d(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x.reshape(1, -1).astype(np.float32), (0, 0), sigma).ravel()


def _outline_breaks(image: np.ndarray, scale: float, dpi: float, c: np.ndarray, u_dir: np.ndarray,
                    v_dir: np.ndarray, len_v: float, us: np.ndarray) -> np.ndarray:
    """Where the two outer paper edges across the spine break, in mm.

    The pages of an open booklet meet at the spine, and the paper edges
    crossing it (top and bottom of a spread) show that: a small notch where
    both pages curve into the gutter, or a jump where one page ends a little
    further out than the other.  Print does not move the outline, so this
    finds the spine even among photo columns.  Both edges must break at the
    same place.  Measured at full resolution (the breaks are 1-2 mm).
    """
    D = max(6, int(round(0.018 * len_v / scale)))
    ds = np.arange(-D, D + 1, 1.0)
    subs = np.array([-1 / 3, 0.0, 1 / 3])
    U = (us[:, None] + subs[None, :]).ravel()
    profiles = []
    for sgn in (-1.0, 1.0):
        Ug, Dg = np.meshgrid(U, ds)
        P0 = c / scale
        X = (P0[0] + Ug / scale * u_dir[0] + (sgn * len_v / 2 / scale + sgn * Dg) * v_dir[0]).astype(np.float32)
        Y = (P0[1] + Ug / scale * u_dir[1] + (sgn * len_v / 2 / scale + sgn * Dg) * v_dir[1]).astype(np.float32)
        patch = to_float_rgb(cv2.remap(image, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        Lp = 100.0 * (0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2])
        Lp = Lp.reshape(len(ds), len(us), len(subs)).mean(axis=2)
        g = np.zeros_like(Lp)
        g[2:-2] = np.abs(Lp[4:] - Lp[:-4])
        # The outermost clear step: the paper edge (print lies further in).
        strong = g >= np.minimum(0.5 * g.max(axis=0), 3.0)[None]
        idx = len(ds) - 1 - np.argmax(strong[::-1], axis=0)
        e = cv2.medianBlur(ds[idx].astype(np.float32).reshape(1, -1), 5).ravel()
        n = len(e)
        W, w, k = max(4, int(0.04 * n)), max(1, int(0.006 * n)), 3
        brk = np.zeros(n, np.float32)
        for i in range(n):
            l, r = e[max(0, i - W) : max(0, i - w)], e[i + w + 1 : i + W + 1]
            notch = min(np.median(l), np.median(r)) - e[i] if l.size >= 3 and r.size >= 3 else 0.0
            jump = abs(np.median(e[i + 1 : i + k + 1]) - np.median(e[i - k : i])) if k <= i < n - k else 0.0
            # A jump counts half: where the page corners leave a gap at the
            # fold, the jump is at the gap's edges and the notch at its centre.
            brk[i] = max(notch, 0.5 * jump)
        brk[np.abs(e) >= D - 3] = 0  # no edge found near the side
        profiles.append(brk)
    a, b = profiles
    r = 3
    dil = lambda x: cv2.dilate(x.reshape(1, -1), np.ones((1, 2 * r + 1), np.uint8)).ravel()
    both = np.minimum(a, dil(b)) + np.minimum(dil(a), b)
    return both * 25.4 / dpi


def _find_spine(a: _Analysis, box: CropBox, axis: int, cut: dict[Side, bool],
                window: tuple[float, float] | None = None,
                full: tuple[np.ndarray, float, float] | None = None) -> _Spine | None:
    """Locate the spine as a line across ``axis`` of the booklet box.

    Three kinds of evidence are combined, all measured on a grid aligned with
    the booklet so that the spine is a column:

    * **Gutter shading.**  The shadow where the pages curve into the spine
      darkens every row by the same factor.  The per-row median of the
      horizontal derivative of log brightness cancels page content (even
      full-bleed artwork) and keeps that common factor; integrated, it gives
      a shading profile whose valley is the spine.
    * **Full-height seam.**  Where two different pages meet, or along a fold
      line, there is a step in almost every row from top to bottom.  Content
      edges (an illustration's frame, a character's outline) do not run the
      full height of the booklet.
    * **Text gap.**  Text pages leave a blank gutter either side of the spine.
    """
    u_dir, v_dir, len_u, len_v = _axis_frame(box, axis)
    c = np.array([box.cx, box.cy])
    us = np.arange(-len_u / 2 + 1, len_u / 2 - 1, 1.0)
    vs = np.arange(-0.46 * len_v, 0.46 * len_v, 1.0)
    if us.size < 30 or vs.size < 30:
        return None
    U, V = np.meshgrid(us, vs)
    X = (c[0] + U * u_dir[0] + V * v_dir[0]).astype(np.float32)
    Y = (c[1] + U * u_dir[1] + V * v_dir[1]).astype(np.float32)
    S = cv2.remap(a.L, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    valid = ~np.isnan(S)
    col_ok = valid.mean(axis=0) > 0.5
    if col_ok.sum() < 30:
        return None
    n = len(us)
    idx = np.arange(n)

    lo_cut = cut.get((axis, -1), False)
    hi_cut = cut.get((axis, 1), False)
    if window is not None:
        rng = window
    elif hi_cut and not lo_cut:
        rng = (0.5, 0.9)  # full page on the low side: spine nearer the cut
    elif lo_cut and not hi_cut:
        rng = (0.1, 0.5)
    else:
        rng = (0.12, 0.88)
    i0, i1 = max(1, int(rng[0] * n)), min(n - 1, int(rng[1] * n))
    if i1 - i0 < 5:
        return None
    W = max(4, int(0.06 * n))

    with _quiet():
        # Gutter shading from the per-row median log-brightness derivative.
        d = np.gradient(np.log(np.maximum(S, 0.0) + 2.0), axis=1)
        m = np.nan_to_num(np.nanmedian(d, axis=0))
        shade = _blur1d(np.cumsum(m), 1.0)
        shade_depth = _valley_depth(shade, i0, i1, W)
        shade_sigma = max(_robust_sigma((shade - _blur1d(shade, 4.0))[i0:i1]), 0.002)
        # Where page content lines up in every row (a column of photos), the
        # common profile has a big step; the slope at its foot is not a
        # gutter.  A gutter rises on both sides.
        rise_l = np.array([shade[max(0, i - W) : i + 1].max() for i in range(n)]) - shade
        rise_r = np.array([shade[i : i + W + 1].max() for i in range(n)]) - shade
        symmetry = np.clip(2.0 * np.minimum(rise_l, rise_r) / np.maximum(np.maximum(rise_l, rise_r), 1e-6), 0.15, 1.0)

        # Crease: the fold is a narrow dark line across the paper in nearly
        # every row.  The median dip over the rows of each third ignores
        # content that only some rows have; the weakest third counts.
        k = max(2, int(round(0.006 * n)))
        Sf = np.where(valid, S, -1e6).astype(np.float32)
        lmax = cv2.dilate(Sf, np.ones((1, k + 1), np.uint8), anchor=(k, 0), borderType=cv2.BORDER_REPLICATE)
        rmax = cv2.dilate(Sf, np.ones((1, k + 1), np.uint8), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
        # Only rows with paper on both sides can show it (where a photo runs
        # up to the fold there is nothing to see); in those rows most must
        # dip, by less than ink would.
        paper_ref = float(np.nanpercentile(S[valid], 95))
        both = np.minimum(lmax, rmax)
        elig = valid & (both > paper_ref - 15.0)
        depth = both - Sf
        is_dip = elig & (depth > 1.5) & (depth < 15.0)
        thirds = np.array_split(np.arange(len(vs)), 3)
        tol = max(2, int(round(0.004 * n)))  # the fold may lean a little relative to the box

        def spread(x: np.ndarray) -> np.ndarray:
            return cv2.dilate(np.nan_to_num(x).astype(np.float32).reshape(1, -1), np.ones((1, 2 * tol + 1), np.uint8)).ravel()

        strengths = []
        for t in thirds:
            e = elig[t].mean(axis=0)
            f = is_dip[t].sum(axis=0) / np.maximum(elig[t].sum(axis=0), 1)
            med = np.nanmedian(np.where(is_dip[t], depth[t], np.nan), axis=0)
            ok = (spread(e) >= 0.2) & (spread(np.where(e >= 0.2, f, 0.0)) >= 0.45)
            strengths.append(np.where(ok, spread(np.where(e >= 0.2, med, 0.0)), np.inf))
        strengths = np.array(strengths)
        n_ok = np.isfinite(strengths).sum(axis=0)
        crease = np.where(n_ok >= 2, np.min(strengths, axis=0), 0.0)

        # Full-height steps: rows with a clear step, in the top, middle and
        # bottom thirds separately; the weakest third counts.
        noise = _robust_sigma(d[valid])
        strong = np.abs(np.nan_to_num(d)) > max(4.0 * noise, 0.03)
        cov = np.min([strong[t].mean(axis=0) for t in thirds], axis=0)
        cov = _blur1d(np.maximum.reduce([cov, np.roll(cov, 1), np.roll(cov, -1)]), 0.7)
        local = np.array([np.median(cov[max(0, i - W) : i + W + 1]) for i in range(n)])
        # Text lines that all start at the same margin give steps in about
        # half the rows; a page seam or fold shows in most rows.
        seam = np.where(cov >= 0.7, np.maximum(cov - local, 0.0), 0.0)

        # Text gap.
        paper = np.nanpercentile(np.where(valid, S, np.nan), 80, axis=0)
        ink = np.nanmean(np.where(valid, S < (paper[None, :] - 15.0), np.nan), axis=0)
    ink = np.interp(idx, idx[col_ok], np.nan_to_num(ink[col_ok]))
    g = max(3, int(0.015 * n))
    ink_s = np.convolve(ink, np.ones(g) / g, mode="same")
    text_level = np.percentile(ink_s[ink_s > 0.01], 60) if np.any(ink_s > 0.01) else 0.05
    gapness = np.clip(1.0 - ink_s / max(text_level, 0.02), 0.0, 1.0)

    s_crease = np.where(crease > 1.5, crease / 3.0, 0.0)
    s_shade = np.where(shade_depth > 4.0 * shade_sigma, shade_depth / 0.015, 0.0)
    if s_crease[i0:i1].max() >= 1.0:
        # A paper fold is visible, so lopsided "valleys" at the foot of a
        # content step are not the gutter.  (In full-bleed artbooks the
        # gutter often is lopsided: two different pictures meet there.)
        s_shade = s_shade * symmetry
    blank = (gapness > 0.6) & col_ok
    run, longest = 0, 0
    for i in range(i0, i1):
        run = run + 1 if blank[i] else 0
        longest = max(longest, run)
    if longest >= max(g, int(0.01 * n)):
        # The pages have blank margins at the gutter: the spine lies in blank
        # paper, not where a column of photos or text starts (those line up
        # in every row and fake a shading valley).
        r = max(2, g // 3)
        near_blank = cv2.dilate(blank.astype(np.uint8).reshape(1, -1), np.ones((1, 2 * r + 1), np.uint8)).ravel() > 0
        s_shade = np.where(near_blank, s_shade, 0.3 * s_shade)
    s_seam = seam / 0.35
    s_outline = np.zeros(n, np.float32)
    if full is not None:
        # A clear break (~1 mm) outweighs a column of photos or a rule.
        s_outline = profiles.current().outline_weight * _outline_breaks(full[0], full[1], full[2], c, u_dir, v_dir, len_v, us)
    cues = s_shade + s_seam + s_crease + s_outline
    score = cues + 0.5 * gapness * (cues > 0.3)
    score[~col_ok] = 0
    score[:i0] = 0
    score[i1:] = 0
    best = int(np.argmax(score))
    if DEBUG is not None:
        DEBUG.setdefault("spine", []).append(dict(axis=axis, us=us, shade=shade, shade_depth=shade_depth,
                                                  shade_sigma=shade_sigma, cov=cov, seam=seam, gapness=gapness,
                                                  crease=crease, symmetry=symmetry, outline=s_outline,
                                                  s_shade=s_shade, s_seam=s_seam, s_crease=s_crease,
                                                  score=score, i0=i0, i1=i1, S=S))
    # A spine is one line; text lines and repeated motifs give many similar
    # peaks.  Discount a best peak that is not clearly above the runner-up.
    others = score.copy()
    others[max(0, best - W) : best + W + 1] = 0
    uniqueness = float(score[best] / max(others.max(), 1e-6))
    if score[best] >= 0.5:
        # Sub-pixel position from whichever cue dominates.
        main = int(np.argmax([s_shade[best], s_seam[best], s_crease[best], s_outline[best]]))
        prof = (-shade, cov, crease, s_outline)[main]
        off = 0.0
        if 0 < best < n - 1:
            den = prof[best - 1] - 2 * prof[best] + prof[best + 1]
            if abs(den) > 1e-9:
                off = float(np.clip(0.5 * (prof[best - 1] - prof[best + 1]) / den, -1, 1))
        method = ("shade", "seam", "crease", "outline")[main]
        quality = float(score[best]) * min(uniqueness, 3.0) / 3.0
        return _Spine(axis, float(us[best] + off), quality, method)

    # Perfectly flat text booklet: use the centre of the widest blank band.
    blank = (gapness > 0.75) & col_ok
    best_run, run_start = (0, 0), None
    for i in range(i0, i1 + 1):
        if i < i1 and blank[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, i)
            run_start = None
    if best_run[1] - best_run[0] >= g:
        centre = (best_run[0] + best_run[1] - 1) / 2.0
        score_gap = 0.05 * (best_run[1] - best_run[0]) / max(n, 1)
        return _Spine(axis, float(np.interp(centre, idx, us)), score_gap, "gap")
    return None


def _split(box: CropBox, spine: _Spine) -> tuple[CropBox, CropBox]:
    """Split a booklet box at the spine into its low and high parts."""
    _, _, len_u, _ = _axis_frame(box, spine.axis)
    lo = _apply_side_shifts(box, {(spine.axis, 1): -(len_u / 2 - spine.offset)})
    hi = _apply_side_shifts(box, {(spine.axis, -1): -(len_u / 2 + spine.offset)})
    return lo, hi


def _resquare(a: _Analysis, box: CropBox, min_diff: float = 1.0) -> CropBox:
    """Re-align a box whose angle the lid's shadows have pulled askew.

    The shadows of a propped-up lid join the booklet in the foreground and
    can tilt the fitted rectangle by degrees.  The paper edges, text lines
    and picture frames inside give the booklet's own angle: the peak of the
    gradient orientations.  When that clearly differs, the box is turned to
    it (still enclosing the old one; the sides are pulled in to the paper
    edges afterwards).
    """
    gx = cv2.Sobel(a.L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a.L, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mask = np.zeros(a.L.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(box.corners()).astype(np.int32)], 1)
    inside = mask > 0
    if inside.sum() < 1000:
        return box
    sel = inside & (mag > np.percentile(mag[inside], 90))
    th = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 45.0) % 90.0 - 45.0
    bins = np.arange(-6.0, 6.01, 0.1)
    hist, _ = np.histogram(th, bins=bins, weights=mag[sel])
    hist = np.convolve(hist, [1, 2, 3, 2, 1], "same")
    i = int(np.argmax(hist))
    if hist[i] < 3.0 * max(float(np.median(hist)), 1e-6):
        return box  # no clear direction
    angle = float((bins[i] + bins[i + 1]) / 2)
    if abs(angle - box.angle) <= min_diff:
        return box
    t = np.radians(angle)
    ux, uy = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    cs = box.corners() - [box.cx, box.cy]
    u, v = cs @ ux, cs @ uy
    c = np.array([box.cx, box.cy]) + ux * (u.max() + u.min()) / 2 + uy * (v.max() + v.min()) / 2
    return CropBox(float(c[0]), float(c[1]), float(u.max() - u.min()), float(v.max() - v.min()), angle,
                   confidence=box.confidence)


def _visible_booklet(coarse, sw: int, sh: int) -> CropBox:
    """The visible part of the booklet from the coarse single-item box."""
    a = coarse.analysis
    booklet = coarse.boxes[0].clamp_to(sw, sh)
    if a.bg_lab[0] < 30.0:
        # Dark fabric stuck to the booklet on a black bed.
        return _trim_protrusions(coarse.fg, booklet)
    # Light lid: pale paper is not all foreground, so only sides that are
    # hardly covered are pulled in (not the sides whose neighbouring paper
    # edges run on to the border: where the booklet leaves the bed), and
    # the soft shadows of a propped-up lid are cut off at the paper edge;
    # never past print.
    booklet = _resquare(a, booklet).clamp_to(sw, sh)
    _, reach, _, _ = _cut_sides(a, booklet, sw, sh)
    keep = {k for k, v in reach.items() if v}
    booklet = _trim_protrusions(coarse.fg, booklet, 0.3, keep, a)
    # Every side may move in to a paper edge: next to a box side at the
    # border "reach" is unreliable (the lid's shadows reach the border too),
    # and a page that really runs off the bed has print in the strip, which
    # ``_plain_strip`` refuses to give up.
    if not profiles.current().shrink_to_paper_edges:
        return booklet
    return _shrink_to_paper_edges(a, booklet)


def detect_booklet(image: np.ndarray, settings: AutoCropSettings | None = None,
                   dpi: float | None = None) -> list[CropBox]:
    """Return the booklet page(s) that are completely on the scan."""
    st = settings or AutoCropSettings(mode="booklet")
    dpi = dpi or st.default_dpi
    # Booklet logic is tuned with the spine running up and down (as the maintainer's scans are saved:
    # turned in the app); on the glass the spine often runs across the bed.
    across = _spine_across(image, st, dpi)
    rot = lambda im: np.ascontiguousarray(np.rot90(im, 1))
    if across is None:
        # Nothing cut off: a whole spread (landscape across its spine) or a cover (portrait),
        # either of them possibly lying turned.  A spread shows a clear spine (V600 whole spreads
        # score >= 13; confidence = score / 5); otherwise it is a cover, looked at upright
        # (portrait), where it is not split.
        tall = _booklet_is_tall(image, st, dpi)
        land, port = (rot(image), image) if tall else (image, rot(image))
        spread = _detect_pages(land, st, dpi)
        if len(spread) == 2 and min(p.confidence for p in spread) >= profiles.current().spread_confidence:
            across, pages = tall, spread
        else:
            across, pages = not tall, _detect_pages(port, st, dpi)
        work = rot(image) if across else image
    else:
        work = rot(image) if across else image
        pages = _detect_pages(work, st, dpi)
    turned = across
    # A thick book props the lid up so far that the general lid model sees the whole bed as
    # foreground (one box, no spine) or splits it badly.  Its top pages are still the brightest
    # plain surface: see ``thickbook``, which answers only for a whole spread with clean edges.
    # When it answers, its pages win (also over two general pages: on thick books the general
    # ones are often split badly, and their printed panels mislead the top-page search).
    thick = _fit_thick(work, dpi)
    if thick is not None:
        margin = st.margin_mm * dpi / MM_PER_INCH
        pages = []
        from .toppage import SIDES, _rgb8, top_page

        rgb8 = _rgb8(work)
        for p in thick:
            # only the spine moves (to the fold's crease); the fitted paper edges stay
            spine = p.meta["keep_sides"]
            q, shifts = top_page(work, p, dpi, keep=SIDES, rgb8=rgb8, snap=spine)
            b = q.inset(margin).clamp_to(work.shape[1], work.shape[0])
            b.meta = dict(p.meta)
            b.meta["top_page"] = {f"{k[0]}{k[1]:+d}": round(v, 2) for k, v in shifts.items()}
            pages.append(b)
    if turned:
        pages = _turn_back(pages, image.shape[1])
    pages = reading_order(pages)
    if st.booklet_output == "spread" and len(pages) == 2:
        return [spread_box(pages)]
    if len(pages) == 2:
        pages = _page_order(pages, st)
    return pages


def _page_order(pages: list[CropBox], st: AutoCropSettings) -> list[CropBox]:
    """The two pages of a spread numbered as seen in the preview (turned by ``crop.rotate``,
    quarter turns clockwise): left page first, or right page first for books read right to left
    (Japanese booklets; ``crop.booklet_rtl``)."""
    turns = int(st.rotate) % 4
    def seen_x(b: CropBox) -> float:  # x after turning the image clockwise (only the order counts)
        return (b.cx, -b.cy, -b.cx, b.cy)[turns]
    out = sorted(pages, key=seen_x)
    return out[::-1] if st.booklet_rtl else out


def _spine_across(image: np.ndarray, st: AutoCropSettings, dpi: float) -> bool | None:
    """Does the spine (if any) run across the image?  True when the visible booklet is cut off at
    (or touches) the top or bottom border (the spine is parallel to the cut), False when cut at the left or
    right, None when nothing is cut (decided by the caller)."""
    small, scale = _working_image(image, st.work_size)
    sh, sw = small.shape[:2]
    coarse = _coarse_detect(small, scale, st, dpi, "single", band_frac=0.006,
                            require_uniform=False, surface="border", drop_edge_slivers=True)
    if coarse is None or not coarse.boxes:
        return False
    booklet = _visible_booklet(coarse, sw, sh)
    cut, reach, _is_edge, touch = _cut_sides(coarse.analysis, booklet, sw, sh)
    off = {ax for (ax, _sg), v in list(cut.items()) + list(reach.items()) + list(touch.items()) if v}
    if off:
        return off == {1}
    return None


def _booklet_is_tall(image: np.ndarray, st: AutoCropSettings, dpi: float) -> bool:
    small, scale = _working_image(image, st.work_size)
    coarse = _coarse_detect(small, scale, st, dpi, "single", band_frac=0.006,
                            require_uniform=False, surface="border", drop_edge_slivers=True)
    if coarse is None or not coarse.boxes:
        return image.shape[0] > image.shape[1]
    b = _visible_booklet(coarse, small.shape[1], small.shape[0])
    return b.h > b.w


def _fit_thick(image: np.ndarray, dpi: float) -> list[CropBox] | None:
    """``thickbook.fit_spread`` with the spine either way: the maintainer turns scans in the app, so on
    the glass a book often lies with its spine across the bed; then the image is turned a quarter
    for the fit and the pages are turned back."""
    from .thickbook import fit_spread

    if not profiles.current().thick_books:
        return None
    dbg: dict = {}
    pages = fit_spread(image, dpi, dbg)
    if pages is not None or "landscape" not in dbg.get("fail", ""):
        return pages
    turned = fit_spread(np.ascontiguousarray(np.rot90(image, 1)), dpi)
    return None if turned is None else _turn_back(turned, image.shape[1])


def _turn_back(pages: list[CropBox], W: int) -> list[CropBox]:
    """Pages found on ``np.rot90(image, 1)`` (a quarter turn anticlockwise) in the coordinates of
    ``image`` (width W), with their spine sides (``keep_sides``) carried over."""
    back = lambda q: np.stack([W - 1 - q[:, 1], q[:, 0]], axis=1)
    out = []
    for p in pages:
        b = CropBox.from_points(back(p.corners()))
        b.confidence = p.confidence
        b.meta = dict(p.meta)
        pc = p.corners()
        mids = {(0, 1): (pc[1] + pc[2]) / 2, (0, -1): (pc[0] + pc[3]) / 2,
                (1, 1): (pc[2] + pc[3]) / 2, (1, -1): (pc[0] + pc[1]) / 2}
        keep = []
        for k in p.meta.get("keep_sides", []):
            m = back(mids[tuple(k)][None])[0] - np.array([b.cx, b.cy])
            ux, uy = b.axes
            u, v = m @ ux, m @ uy
            keep.append((0, 1 if u > 0 else -1) if abs(u) / b.w > abs(v) / b.h else (1, 1 if v > 0 else -1))
        b.meta["keep_sides"] = keep
        b.meta.pop("top_page", None)
        b.meta["top_page"] = p.meta.get("top_page", {})
        out.append(b)
    return out


def _detect_pages(image: np.ndarray, st: AutoCropSettings, dpi: float) -> list[CropBox]:
    """The general booklet logic: the complete page(s), each ending at the top page's edge."""
    H, W = image.shape[:2]
    full = CropBox.from_xywh(0, 0, W, H, confidence=0.0)
    small, scale = _working_image(image, st.work_size)
    sh, sw = small.shape[:2]
    # Paper covers most of the image and is often close to the lid colour,
    # so the lid's lighting is learnt from a thin ring at the image border
    # only (plus the darkening towards the edges of the glass, see
    # ``autocrop._edge_falloff``).
    coarse = _coarse_detect(small, scale, st, dpi, "single", band_frac=0.006,
                            require_uniform=False, surface="border", drop_edge_slivers=True)
    if coarse is None or not coarse.boxes:
        return [full]
    a = coarse.analysis
    booklet = _visible_booklet(coarse, sw, sh)
    cut, reach, is_edge, touch = _cut_sides(a, booklet, sw, sh)
    booklet = _extend_to_border(booklet, reach, sw, sh)

    cut_axes = sorted({ax for (ax, _sg), v in cut.items() if v})
    reach_axes = sorted({ax for (ax, _sg), v in reach.items() if v})
    # A booklet at the border at both ends of one axis but cut off on the
    # other just fills the bed that way: no page would be complete if it ran
    # off at both ends, so the spine is across the other axis.
    full = {ax for ax in (0, 1) if all(touch[(ax, sg)] or cut[(ax, sg)] for sg in (-1, 1))}
    if len(full) == 1 and any(cut[(1 - ax, sg)] for ax in full for sg in (-1, 1)):
        other = 1 - next(iter(full))
        cut_axes = [ax for ax in cut_axes if ax == other]
        reach_axes = [ax for ax in reach_axes if ax == other]
    if len(reach_axes) == 1:
        axes = reach_axes  # the spine is parallel to the side that runs off
    elif len(cut_axes) == 1:
        axes = cut_axes
    elif not cut_axes and max(booklet.w, booklet.h) > 1.15 * min(booklet.w, booklet.h):
        # Whole spread visible: two pages side by side are longer across the
        # spine than along it for any ordinary page shape.
        axes = [0 if booklet.w > booklet.h else 1]
    else:
        axes = [0, 1]
    full_res = (image, scale, dpi)
    spine = None
    whole = False  # the spine was found in the middle of an uncut booklet
    for ax in axes:
        sp = None
        if not any(cut.get((ax, sg)) or touch.get((ax, sg)) for sg in (-1, 1)):
            # Lid visible beyond both ends along this axis: if this is a whole
            # spread, its two pages are the same size and the spine is in the
            # middle.  Look there first; content edges elsewhere (a column of
            # photos) must not split it unevenly.  (An end at the image border
            # may be a page running off the bed that looks like a paper edge,
            # e.g. white paper on a white lid.)
            sp = _find_spine(a, booklet, ax, cut, window=profiles.current().middle_window, full=full_res)
            if sp is not None and (sp.method == "gap" or sp.score < 0.5):
                sp = None
            if sp is not None and (spine is None or sp.score > spine.score):
                spine, whole = sp, True
                continue
            if all(is_edge.values()):
                # The whole booklet is on the glass with lid all round: a
                # spread would have its spine in the middle, so this is one
                # page (a cover), whatever lines run across it.
                continue
        sp = _find_spine(a, booklet, ax, cut, full=full_res)
        if sp is not None and (spine is None or sp.score > spine.score):
            spine, whole = sp, False
    if spine is not None and all(is_edge.values()) and booklet.w < 1.15 * booklet.h:
        # All four sides are paper edges with lid beyond them, so the whole
        # booklet is on the glass, and it is not wider than tall: most likely
        # a single page (a cover).  Booklet pages are portrait, so a spread on
        # the glass is clearly landscape.  A clearly portrait booklet is never
        # split (cover art has strong horizontal bands); a square-ish one only
        # by a very clear spine.
        if booklet.h > 1.15 * booklet.w or spine.score < profiles.current().cover_split_score or spine.method == "gap":
            spine = None
    elif (spine is not None and spine.score < profiles.current().lid_all_round_split_score
          and not any(cut.values()) and not any(touch.values())):
        # The whole booklet is on the glass with lid all round: a whole spread has a clear spine
        # (V600 whole spreads score >= 13 at 200 dpi), a cover only faint lines (V43 landscape:
        # 0.8; V17 at 200 dpi, whose bottom edge is not verified: 1.6).
        spine = None
    if spine is None:
        # No spine found: treat the visible booklet as one page.
        return _top_pages(image, _finish(image, [booklet], scale, st, dpi), st, dpi)

    lo, hi = _split(booklet, spine)
    lo_cut = cut.get((spine.axis, -1), False)
    hi_cut = cut.get((spine.axis, 1), False)
    lo_skip = {(spine.axis, 1)}  # the low part's high side is the spine
    hi_skip = {(spine.axis, -1)}
    lo_len = lo.w if spine.axis == 0 else lo.h
    hi_len = hi.w if spine.axis == 0 else hi.h
    ratio = min(lo_len, hi_len) / max(lo_len, hi_len, 1e-9)
    if not (lo_cut or hi_cut) and (whole or ratio >= 0.85):
        # Whole spread on the glass: both pages are complete.
        pages = [(lo, lo_skip), (hi, hi_skip)]
    elif hi_cut and not lo_cut and lo_len >= hi_len:
        pages = [(lo, lo_skip)]
    elif lo_cut and not hi_cut and hi_len >= lo_len:
        pages = [(hi, hi_skip)]
    else:
        # A partially visible page is always the smaller part.
        pages = [(lo, lo_skip)] if lo_len >= hi_len else [(hi, hi_skip)]
    for p, skip in pages:
        p.meta["spine"] = spine.method
        # Re-refinement on a later high-resolution scan must not move the
        # spine side either (see ``autocrop.refine``).
        p.meta["keep_sides"] = sorted(skip)
        p.confidence = float(min(1.0, spine.score / 5.0)) if spine.method != "gap" else 0.4
    return reading_order(_top_pages(image, _finish(image, [p for p, _ in pages], scale, st, dpi,
                                                    [s for _, s in pages]), st, dpi))


def spread_box(pages: list[CropBox]) -> CropBox:
    """One box over both pages of a spread (maintainer, Oct 2026: from the left page's outer edge to the
    right page's, top from the higher page, bottom from the lower: no print is ever cut).  The
    pages are kept in ``meta["spread_pages"]`` (corners in the box's own frame, 0..1) so that a
    later refinement can redo each page and the union."""
    t = np.radians(np.mean([p.angle for p in pages]))
    ux, uy = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    pts = np.vstack([p.corners() for p in pages])
    u, v = pts @ ux, pts @ uy
    c = ux * (u.min() + u.max()) / 2 + uy * (v.min() + v.max()) / 2
    box = CropBox(float(c[0]), float(c[1]), float(u.max() - u.min()), float(v.max() - v.min()),
                  float(np.degrees(t)), confidence=min(p.confidence for p in pages))
    box.meta = {k: v for k, v in pages[0].meta.items() if k not in ("keep_sides", "top_page")}
    box.meta["spread_pages"] = [_to_frame(box, p) for p in pages]
    box.meta["spread_keep"] = [p.meta.get("keep_sides", []) for p in pages]
    return box


def _to_frame(box: CropBox, page: CropBox) -> list[list[float]]:
    ux, uy = box.axes
    c0 = box.corners()[0]
    return [[round(float((q - c0) @ ux / box.w), 6), round(float((q - c0) @ uy / box.h), 6)]
            for q in page.corners()]


def _from_frame(box: CropBox, corners) -> CropBox:
    ux, uy = box.axes
    c0 = box.corners()[0]
    pts = np.array([c0 + ux * u * box.w + uy * v * box.h for u, v in corners])
    return CropBox.from_points(pts)


def refine_booklet(image: np.ndarray, box: CropBox, st: AutoCropSettings, dpi: float) -> CropBox:
    """Re-find a booklet page's top-page edges on a (higher resolution) scan; the spine stays.  A
    spread box is redone page by page."""
    H, W = image.shape[:2]
    if "spread_pages" in box.meta:
        pages = []
        for i, corners in enumerate(box.meta["spread_pages"]):
            p = _from_frame(box, corners)
            p.meta = {"keep_sides": box.meta.get("spread_keep", [[], []])[i], "top_page": {},
                      "spine": box.meta.get("spine", "")}
            pages.append(refine_booklet(image, p, st, dpi))
        out = spread_box(pages)
        out.meta = {**box.meta, "spread_pages": out.meta["spread_pages"]}
        return out
    if box.meta.get("spine") == "thick":
        # Found by ``thickbook`` (already at the top page's edges); its pages carry printed panels
        # whose lines the layer chain would follow inwards, and the final scan may cover only the
        # crop area, which ``fit_spread`` cannot work with.
        return box
    from .toppage import top_page

    margin = st.margin_mm * dpi / MM_PER_INCH
    b, shifts = top_page(image, box.inset(-margin), dpi, box.meta.get("keep_sides", ()), start_out=1.0,
                         crease_mm=_crease_reach(box))
    b = b.inset(margin).clamp_to(W, H)
    b.meta = dict(box.meta)
    b.meta["top_page"] = {f"{k[0]}{k[1]:+d}": round(v, 2) for k, v in shifts.items()}
    return b


def _crease_reach(page: CropBox) -> float:
    """How far (mm) the spine may move to the fold's crease: a spine found as the gutter's shading
    valley is already there (V39 scans: other dark lines 2-3 mm away), others may be off by more
    (V600 V16/V40: 1.5-2.7 mm)."""
    return 1.0 if page.meta.get("spine") == "shade" else 3.0


def _top_pages(image: np.ndarray, pages: list[CropBox], st: AutoCropSettings, dpi: float) -> list[CropBox]:
    """End every page at the top page's edge (see ``toppage``); the spine side stays."""
    if not st.top_page:
        return pages
    from .toppage import _rgb8, top_page

    H, W = image.shape[:2]
    margin = st.margin_mm * dpi / MM_PER_INCH
    rgb8 = _rgb8(image)
    out = []
    for p in pages:
        b, shifts = top_page(image, p.inset(-margin), dpi, p.meta.get("keep_sides", ()), rgb8,
                             crease_mm=_crease_reach(p))
        b = b.inset(margin).clamp_to(W, H)
        b.meta = dict(p.meta)
        b.meta["top_page"] = {f"{k[0]}{k[1]:+d}": round(v, 2) for k, v in shifts.items()}
        out.append(b)
    return out
