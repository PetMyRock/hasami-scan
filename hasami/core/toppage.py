"""Booklet pages: end the crop at the TOP page's edge.

Thick booklets show the edges of the pages underneath as thin bands beyond the top page (the
page stack: fine lines parallel to the side, often greyer), and covers can show their board or a
shadow band along the edge.  The maintainer wants the crop to end at the top page's printed edge
(Oct 2026).  This works on the full-resolution image, one page side at a time:

1. a strip across the side (from ``depth_mm`` inside to ``out_mm`` outside the current side) is
   cut into segments along the side; each segment's median profile gives its edge positions
   (peaks of the Lab gradient);
2. straight lines that most segments agree on are the "layer lines" (the stack's page edges,
   the top page's edge, a cover's board edge) - print inside the page rarely forms a line
   along the whole side;
3. starting at the layer line at the current side, the chain of layer lines going inwards
   (each at most ``gap_mm`` from the next) is the page stack; its innermost line is the top
   page's edge.  Each side gets its own line (own angle: the pages of a book need not line up).

The page is then rebuilt as the rectangle through its four lines (the spine side unchanged).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import CropBox
from .profiles import current as _profile  # tuned numbers: the scanner's profile

MM_PER_INCH = 25.4
SIDES = ((1, -1), (0, 1), (1, 1), (0, -1))  # top, right, bottom, left (axis, sign)

#: Diagnostics of the last call per side (filled when not None).
DEBUG: dict | None = None


def _rgb8(image: np.ndarray) -> np.ndarray:
    """8-bit RGB (no copy when the image already is one)."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    return img[:, :, :3]


def _to_lab(rgb8: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _side(box: CropBox, key: tuple[int, int]):
    """Endpoints (a, e) of a side, its unit direction d (a -> e) and outward normal n."""
    c = box.corners()  # tl tr br bl
    idx = {(1, -1): (0, 1), (0, 1): (1, 2), (1, 1): (2, 3), (0, -1): (3, 0)}[key]
    a, e = c[idx[0]], c[idx[1]]
    d = (e - a) / max(np.linalg.norm(e - a), 1e-9)
    return a, e, d, np.array([d[1], -d[0]])


def _strip(rgb8: np.ndarray, a, e, d, n, px_mm: float, step_px: float, out_mm: float | None = None):
    """Lab strip in the side's frame: rows along the side (6..94 %), columns from -depth_mm to
    +out_mm; L is -1 outside the image.  Only the strip is converted (scans are big)."""
    out_mm = _profile().out_mm if out_mm is None else out_mm
    L = float(np.linalg.norm(e - a))
    xs = np.arange(-_profile().depth_mm * px_mm, out_mm * px_mm, step_px)
    ys = np.arange(0.06 * L, 0.94 * L, step_px)
    X, Y = np.meshgrid(xs, ys)
    mx = (a[0] + n[0] * X + d[0] * Y).astype(np.float32)
    my = (a[1] + n[1] * X + d[1] * Y).astype(np.float32)
    H, W = rgb8.shape[:2]
    valid = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    s = _to_lab(cv2.remap(rgb8, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    s[~valid, 0] = -1.0
    return xs, ys, s, valid.astype(np.float32)


def _segment_peaks(s: np.ndarray, xs: np.ndarray, ys: np.ndarray, step_mm: float):
    """Edge positions per segment: (x_px, y_px, strength) of gradient peaks of the median profile."""
    pts = []
    rows = np.array_split(np.arange(len(ys)), _profile().segments)
    sig = max(0.05 / step_mm, 0.7)
    for k, r in enumerate(rows):
        if len(r) < 3:
            continue
        prof = np.median(s[r], axis=0)  # (nx, 3)
        sm = cv2.GaussianBlur(prof[None], (0, 0), sig)[0]
        g = np.sqrt((np.gradient(sm, axis=0) ** 2).sum(1)) / step_mm  # Lab units per mm
        noise = np.median(g) * 1.5 + 1e-6
        thr = max(6.0, 3.0 * noise)
        w = max(1, int(round(0.08 / step_mm)))
        for i in range(w, len(g) - w):
            if g[i] >= thr and g[i] == g[i - w:i + w + 1].max():
                pts.append((xs[i], float(ys[r].mean()), float(g[i]), k))
    return pts


def _layer_lines(pts, px_mm: float, side_len: float):
    """Straight lines (x at the side's middle, slope) supported by >= support of the segments."""
    if not pts:
        return []
    P = np.array(pts)
    x, y, k = P[:, 0], P[:, 1] - side_len / 2, P[:, 3].astype(int)
    tol = 0.12 * px_mm
    binw = 0.05 * px_mm
    nb = max(1, int(_profile().max_tilt * side_len / 2 / binw))
    slopes = np.linspace(-_profile().max_tilt, _profile().max_tilt, 2 * nb + 1)
    lo, hi = x.min() - tol, x.max() + tol
    edges = np.arange(lo, hi + binw, binw)
    best = []
    for b in slopes:
        x0 = x - b * y
        # segments supporting a line at each bin centre (each segment once)
        sup = np.zeros(len(edges), int)
        for seg in np.unique(k):
            m = k == seg
            hit = np.zeros(len(edges), bool)
            for v in x0[m]:
                hit |= np.abs(edges - v) <= tol
            sup += hit
        best.append(sup)
    S = np.array(best)  # (slopes, bins)
    need = _profile().support * _profile().segments
    lines = []
    used = np.zeros_like(S, bool)
    order = np.argsort(-S, axis=None)
    for idx in order:
        si, bi = np.unravel_index(idx, S.shape)
        if S[si, bi] < need:
            break
        if used[si, bi]:
            continue
        lines.append((float(edges[bi]), float(slopes[si]), int(S[si, bi])))
        # suppress everything near this line (any slope within the tolerance band at the middle)
        near = np.abs(edges - edges[bi]) <= 2.5 * tol
        used[:, near] = True
    return sorted(lines, key=lambda t: t[0])


def _between(S: np.ndarray, X: np.ndarray, Y: np.ndarray, inner, outer, pad: float):
    """Median Lab colour of the strip between two lines (x0, slope at the side's middle), or None."""
    xi = inner[0] + inner[1] * Y + pad
    xo = outer[0] + outer[1] * Y - pad
    m = (X > xi) & (X < xo) & (S[:, :, 0] >= 0)
    return np.median(S[m], axis=0) if m.sum() > 50 else None


def _stack_gap(S, X, Y, lines, ln, cur, px_mm: float, wide: bool) -> bool:
    """Is the strip between lines ``ln`` (inner) and ``cur`` (outer) part of the page stack?

    The pages under the top one show their edges or a few mm of plain paper: about the colour of
    the top page's paper, or a little darker (further from the glass, less light).  Print along
    the page edge (a coloured band, a frame) differs in hue or is much darker.  The paper is
    sampled just inside ``ln``, or inside a line right next to it (``ln`` can be the far side of a
    thin dark stripe: the printed edge of a lower page)."""
    band = _between(S, X, Y, ln, cur, 0.3 * px_mm)
    if band is None:
        return False
    near = [ln] + [o for o in lines if 0 < ln[0] - o[0] <= _profile().gap_mm * px_mm]
    for c in near:
        inside = _between(S, X, Y, (c[0] - 1.3 * px_mm, c[1]), (c[0], c[1]), 0.3 * px_mm)
        if inside is None:
            continue
        dL = inside[0] - band[0]  # > 0: the band is darker
        dab = float(np.hypot(*(inside[1:] - band[1:])))
        if dab > _profile().max_hue_diff or dL > _profile().max_darker_l:
            continue
        if not wide and dL >= -_profile().lighter_l:
            return True
        if wide and inside[0] >= _profile().light_paper_l and dL >= _profile().darker_l:
            return True
    return False


def _stack_goes_on(lines, ln, px_mm: float, n: int = 2) -> bool:
    """More page edges follow inside ``ln`` at stack spacing (lower pages with coloured edges:
    print along the top page's edge is not followed by a row of lines)."""
    x = ln[0]
    for _ in range(n):
        nxt = [o for o in lines if 0 < x - o[0] <= _profile().near_mm * px_mm]
        if not nxt:
            return False
        x = max(o[0] for o in nxt)
    return True


def _chain(lines, px_mm: float, S=None, X=None, Y=None, start_out: float | None = None):
    """Innermost line of the layer chain starting near the current side (x = 0), or None.

    Neighbouring page edges of a stack are close together (<= near_mm: always chained).  A gap up
    to gap_mm (pages stepped out a little) is chained when its strip looks like paper or more
    page edges follow at stack spacing, and up to
    wide_gap_mm (a lower page sticking out a few mm) when it is also a little darker than light
    paper; see ``_stack_gap``."""
    start_out = _profile().start_out_mm if start_out is None else start_out
    start = [ln for ln in lines if -_profile().start_in_mm * px_mm <= ln[0] <= start_out * px_mm]
    far = not start
    if far:
        # The side may sit on soft lid shadow far beyond the paper (a low-resolution preview):
        # start at the outermost line deeper in, and keep the result only when the strip outside
        # that line is plain shadow, clearly darker than the top page's paper.
        start = [ln for ln in lines if ln[0] < -_profile().start_in_mm * px_mm]
        if S is None or not start:
            return None
    first = cur = max(start, key=lambda ln: ln[0])
    for ln in sorted((ln for ln in lines if ln[0] < cur[0]), key=lambda t: -t[0]):
        gap = cur[0] - ln[0]
        if gap > _profile().near_mm * px_mm:
            if S is None or gap > _profile().wide_gap_mm * px_mm:
                break
            if not _stack_gap(S, X, Y, lines, ln, cur, px_mm, wide=gap > _profile().gap_mm * px_mm) and not (
                    gap <= _profile().gap_mm * px_mm and _stack_goes_on(lines, ln, px_mm)):
                break
        cur = ln
    if far:
        shade = _between(S, X, Y, first, (0.0, first[1]), 0.3 * px_mm)
        paper = _between(S, X, Y, (cur[0] - 1.3 * px_mm, cur[1]), cur, 0.3 * px_mm)
        if (shade is None or paper is None or paper[0] < _profile().light_paper_l
                or paper[0] - shade[0] < _profile().shadow_l):
            return None
    return cur


def _crease(rgb8: np.ndarray, a, e, d, n, px_mm: float, step: float, reach_mm: float = None):
    """The fold's crease near a spine side: a narrow dark valley that most segments along the side
    agree on, within crease_mm of the side.  (x0 at the side's middle in px, slope) or None."""
    side_len = float(np.linalg.norm(e - a))
    reach_mm = _profile().crease_mm if reach_mm is None else reach_mm
    xs = np.arange(-reach_mm * px_mm, reach_mm * px_mm, step)
    ys = np.arange(0.15 * side_len, 0.85 * side_len, step)
    X, Y = np.meshgrid(xs, ys)
    g = cv2.cvtColor(np.ascontiguousarray(rgb8), cv2.COLOR_RGB2GRAY)
    s = cv2.remap(g, (a[0] + n[0] * X + d[0] * Y).astype(np.float32),
                  (a[1] + n[1] * X + d[1] * Y).astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
    step_mm = step / px_mm
    pts = []
    for rows in np.array_split(np.arange(len(ys)), _profile().segments):
        if len(rows) < 3:
            continue
        pr = np.median(s[rows], axis=0)
        hp = pr - cv2.GaussianBlur(pr[None], (0, 0), 0.8 / step_mm)[0]
        hp = cv2.GaussianBlur(hp[None], (0, 0), max(0.05 / step_mm, 0.5))[0]
        i = int(np.argmin(hp))
        pts.append((xs[i], float(ys[rows].mean()) - side_len / 2))
    if not pts:
        return None
    P = np.array(pts)
    med = np.median(P[:, 0])
    ok = np.abs(P[:, 0] - med) < 0.3 * px_mm
    if ok.mean() < _profile().crease_support or ok.sum() < 4:
        return None
    b, x0 = np.polyfit(P[ok, 1], P[ok, 0], 1)
    if abs(b) > _profile().max_tilt:
        return None
    return float(x0), float(b)


def _line_of(box: CropBox, key):
    a, e, d, _n = _side(box, key)
    return (a + e) / 2, d


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    t = np.linalg.solve(A, q - p)
    return p + d * t[0]


def box_from_lines(lines: dict, confidence: float = 1.0) -> CropBox:
    """Rectangle for the quadrilateral of four side lines {(axis, sign): (point, direction)}: the
    mean of the side angles, each side through the middle of its two corners."""
    T, R, B, Lf = (lines[k] for k in SIDES)
    c = np.array([_intersect(Lf, T), _intersect(T, R), _intersect(R, B), _intersect(B, Lf)])
    angs = []
    for key in SIDES:
        d = lines[key][1]
        if key[0] == 1:  # top / bottom: x-direction
            d = d if d[0] > 0 else -d
            angs.append(math.atan2(d[1], d[0]))
        else:
            d = d if d[1] > 0 else -d
            angs.append(math.atan2(-d[0], d[1]))
    t = float(np.mean(angs))
    ux, uy = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    u, v = c @ ux, c @ uy
    u0, u1 = (u[0] + u[3]) / 2, (u[1] + u[2]) / 2
    v0, v1 = (v[0] + v[1]) / 2, (v[2] + v[3]) / 2
    cc = ux * (u0 + u1) / 2 + uy * (v0 + v1) / 2
    return CropBox(float(cc[0]), float(cc[1]), float(u1 - u0), float(v1 - v0), math.degrees(t),
                   confidence=confidence)


def top_page(image: np.ndarray, box: CropBox, dpi: float, keep=(), rgb8: np.ndarray | None = None,
             start_out: float | None = None, snap_spine: bool = True, snap=None,
             crease_mm: float | None = None) -> tuple[CropBox, dict]:
    """Move every side of ``box`` not in ``keep`` in to the top page's edge.

    ``box`` must lie on the paper's outer edge (no margin).  Returns the new box (meta copied) and
    the per-side shift in mm at the side's middle (0 for sides left alone).  ``start_out``: how
    far outside the side (mm) the outermost layer may be (larger when the box comes from a
    low-resolution preview).  Kept sides (spines) snap to the fold's crease when one is clear
    (``snap``: only these of them)."""
    prof = _profile()
    start_out = prof.start_out_mm if start_out is None else start_out
    crease_mm = prof.crease_mm if crease_mm is None else crease_mm
    rgb8 = _rgb8(image) if rgb8 is None else rgb8
    px_mm = dpi / MM_PER_INCH
    step = max(1.0, px_mm / 16.0)  # sample at <= 16 px/mm
    lines, shifts = {}, {}
    keep = {tuple(k) for k in keep}
    for key in SIDES:
        a, e, d, n = _side(box, key)
        lines[key] = ((a + e) / 2, d)
        shifts[key] = 0.0
        if key in keep:
            # A spine side is not a paper edge; when the fold shows a clear crease, the spine is
            # its centre line (V600 truth: the crease within about 0.2 mm).
            snap_this = snap_spine and (snap is None or key in {tuple(k) for k in snap})
            cr = _crease(rgb8, a, e, d, n, px_mm, step, crease_mm) if snap_this else None
            if cr is not None:
                side_len = float(np.linalg.norm(e - a))
                dd = d + n * cr[1]
                lines[key] = (a + d * (side_len / 2) + n * cr[0], dd / np.linalg.norm(dd))
                shifts[key] = cr[0] / px_mm
            continue
        xs, ys, s, valid = _strip(rgb8, a, e, d, n, px_mm, step, max(_profile().out_mm, start_out + 0.4))
        if valid.mean() < 0.9:
            continue  # the side runs off the image
        side_len = float(np.linalg.norm(e - a))
        pts = _segment_peaks(s, xs, ys, step / px_mm)
        layer = _layer_lines(pts, px_mm, side_len)
        X, Y = np.meshgrid(xs, ys - side_len / 2)
        top = _chain(layer, px_mm, s, X, Y, start_out)
        if DEBUG is not None:
            DEBUG[key] = {"lines": [(round(x0 / px_mm, 2), round(b, 4), sup) for x0, b, sup in layer],
                          "top": None if top is None else (round(top[0] / px_mm, 2), round(top[1], 4))}
        if top is None:
            continue
        x0, b = top[0], top[1]
        mid = a + d * (side_len / 2) + n * x0
        dd = d + n * b  # x = x0 + b * y (x outwards along n)
        lines[key] = (mid, dd / np.linalg.norm(dd))
        shifts[key] = x0 / px_mm
    if not any(shifts.values()):
        return box, shifts
    out = box_from_lines(lines, box.confidence)
    out.meta = dict(box.meta)
    return out, shifts
