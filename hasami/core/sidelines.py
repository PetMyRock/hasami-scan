"""Candidate edge lines around a final crop box: what a dragged box side snaps to, and what
"check this side" flags are measured on.

Read-only: computed on the preview after ``detect`` (``Session.detect_on_preview``, booklet mode),
never inside the engine, so no crop changes.  Uses toppage's own line finding (same segments,
thresholds and support) on a wider strip than the rules see: 14 mm inside to 8 mm outside a paper
side, and dark valleys within 4 mm of a spine.  Spec: docs/specs/2026-10-15-edge-snapping-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import SIDE_NAMES, CropBox
from .profiles import current as _profile
from .toppage import MM_PER_INCH, SIDES, _layer_lines, _rgb8, _segment_peaks, _side, _to_lab

IN_MM, OUT_MM = 14.0, 8.0  # strip across a paper side
SPINE_MM = 4.0  # valleys this far either side of a spine
MERGE_MM = 0.15  # lines closer than this are one
KINDS = ("layer", "strong", "faint", "crease")  # preferred in this order when support ties
KEY = dict(zip(SIDE_NAMES, SIDES))  # name -> toppage's (axis, sign)


@dataclass(frozen=True)
class SideLine:
    offset_mm: float  # at the side's middle, + = outwards from the box's side
    slope: float  # tilt against the side: x = offset + slope * y (y along the side)
    support: float  # share of the segments that show it, 0..1
    kind: str  # "layer" | "faint" | "strong" | "crease"


def side_lines(image: np.ndarray, box: CropBox, dpi: float) -> dict[str, list[SideLine]]:
    """Every candidate line near each side of a final page box, keyed "top" / "right" / "bottom" /
    "left" (the box's own frame), each list sorted by offset.  Spine sides (``box.meta["keep_sides"]``)
    get the fold's dark valleys, the other sides the edges toppage's rules choose from."""
    rgb8 = _rgb8(image)
    px_mm = dpi / MM_PER_INCH
    step = max(1.0, px_mm / 16.0)  # as top_page: sample at <= 16 px/mm
    keep = {tuple(k) for k in box.meta.get("keep_sides", ())}
    out = {}
    for name, key in KEY.items():
        a, e, d, n = _side(box, key)
        find = _valleys if key in keep else _edges
        out[name] = find(rgb8, a, e, d, n, px_mm, step)
    return out


def _grid(a, d, n, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    return (a[0] + n[0] * X + d[0] * Y).astype(np.float32), (a[1] + n[1] * X + d[1] * Y).astype(np.float32)


def _usable(mx, my, shape, xs, px_mm):
    """The run of columns on the image (in >= 90% of the rows) around the side itself, or None."""
    H, W = shape[:2]
    ok = ((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)).mean(axis=0) >= 0.9
    i = int(np.argmin(np.abs(xs)))
    if not ok[i]:
        return None
    lo, hi = i, i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return slice(lo, hi + 1) if xs[hi] - xs[lo] >= 2.0 * px_mm else None


def _strip(rgb8, a, d, n, xs, ys, px_mm):
    """RGB samples of the usable columns and their positions, or (None, None)."""
    mx, my = _grid(a, d, n, xs, ys)
    cols = _usable(mx, my, rgb8.shape, xs, px_mm)
    if cols is None:
        return None, None
    mx, my = np.ascontiguousarray(mx[:, cols]), np.ascontiguousarray(my[:, cols])
    return cv2.remap(rgb8, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), xs[cols]


def _edges(rgb8, a, e, d, n, px_mm, step):
    """Layer lines across a paper side at toppage's three thresholds."""
    side_len = float(np.linalg.norm(e - a))
    ys = np.arange(0.06 * side_len, 0.94 * side_len, step)  # as toppage._strip
    rgb, xs = _strip(rgb8, a, d, n, np.arange(-IN_MM * px_mm, OUT_MM * px_mm, step), ys, px_mm)
    if rgb is None:
        return []
    s = _to_lab(rgb)
    prof = _profile()
    lines = []
    for kind, pct in (("layer", 50.0), ("faint", prof.fine_stack_percentile),
                      ("strong", prof.fine_strong_percentile)):
        for x0, b, sup in _layer_lines(_segment_peaks(s, xs, ys, step / px_mm, pct), px_mm, side_len):
            lines.append(SideLine(x0 / px_mm, b, sup / prof.segments, kind))
    return _merge(lines)


def _valleys(rgb8, a, e, d, n, px_mm, step):
    """Dark valleys along a spine side (each segment's local minima, where toppage._crease takes only
    its deepest), pooled into lines."""
    side_len = float(np.linalg.norm(e - a))
    ys = np.arange(0.15 * side_len, 0.85 * side_len, step)  # as toppage._crease
    rgb, xs = _strip(rgb8, a, d, n, np.arange(-SPINE_MM * px_mm, SPINE_MM * px_mm, step), ys, px_mm)
    if rgb is None:
        return []
    s = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    step_mm = step / px_mm
    w = max(1, int(round(0.08 / step_mm)))
    pts = []
    for k, rows in enumerate(np.array_split(np.arange(len(ys)), _profile().segments)):
        if len(rows) < 3:
            continue
        pr = np.median(s[rows], axis=0)
        hp = pr - cv2.GaussianBlur(pr[None], (0, 0), 0.8 / step_mm)[0]
        hp = cv2.GaussianBlur(hp[None], (0, 0), max(0.05 / step_mm, 0.5))[0]
        thr = -max(2.0, 4.5 * float(np.median(np.abs(hp - np.median(hp)))))  # deeper than the paper's texture
        for i in range(w, len(hp) - w):
            if hp[i] <= thr and hp[i] == hp[i - w:i + w + 1].min():
                pts.append((xs[i], float(ys[rows].mean()), float(-hp[i]), k))
    return _merge([SideLine(x0 / px_mm, b, sup / _profile().segments, "crease")
                   for x0, b, sup in _layer_lines(pts, px_mm, side_len)])


def _merge(lines: list[SideLine]) -> list[SideLine]:
    """One line per place (within MERGE_MM): the best supported, then by KINDS; sorted by offset."""
    kept: list[SideLine] = []
    for ln in sorted(lines, key=lambda l: (-l.support, KINDS.index(l.kind))):
        if all(abs(ln.offset_mm - k.offset_mm) > MERGE_MM for k in kept):
            kept.append(ln)
    return sorted(kept, key=lambda l: l.offset_mm)
