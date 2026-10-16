"""Candidate edge lines around a final crop box: what a dragged box side snaps to, and what
"check this side" flags are measured on.

Read-only: computed on the preview after ``detect`` (``Session.detect_on_preview``, booklet mode),
never inside the engine, so no crop changes.  Uses toppage's own line finding (same segments,
thresholds and support) on a wider strip than the rules see: 14 mm inside to 8 mm outside a paper
side, and dark valleys within 4 mm of a spine.  Spec: docs/specs/2026-10-15-edge-snapping-design.md.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class SnapLine:
    """A candidate line in absolute coordinates (bed mm in the Session, preview px in the GUI), already
    moved in by the crop margin: a side snapped to it lands where auto-crop would have put it."""

    p: tuple[float, float]  # a point on the line
    d: tuple[float, float]  # unit direction
    support: float
    kind: str


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


FAR_MM = 99.0  # signals: no such line
OWN_MM = 0.3  # a line this close to the paper edge under the side is that edge
CREASE_MM = 0.5
CROWD_MM = 3.0
STRONG = 0.85  # a page edge runs along (nearly) the whole side

#: The "check this side" rule: an OR of (signal, threshold) conditions, fitted by the snap check in the
#: private data repo (hasami-data benchmarks/learned/snapcheck_2026-10-15.md: on held-out books 11.2% of the
#: sides flagged, 12 of the 18 sides > 0.5 mm off caught).  Empty: no flags.
CHECK_RULE: tuple[tuple[str, float], ...] = (("no_crease", 0.5), ("crowd", 8))


def normal_hit(p, dd, m, nrm) -> float | None:
    """Where the line through ``p`` along ``dd`` crosses the line from ``m`` along ``nrm``: the distance
    from ``m`` in ``nrm``'s units, or None when the two run parallel."""
    den = nrm[0] * dd[1] - nrm[1] * dd[0]
    if abs(den) < 1e-9:
        return None
    return float(((p[0] - m[0]) * dd[1] - (p[1] - m[1]) * dd[0]) / den)


def to_image(line: SideLine, box: CropBox, name: str, px_mm: float, inset_mm: float = 0.0):
    """A side's line as a point (at the side's middle) and a unit direction in the image (px);
    ``inset_mm`` moves it inwards."""
    a, e, d, n = _side(box, KEY[name])
    p = (a + e) / 2 + n * (line.offset_mm - inset_mm) * px_mm
    dd = d + n * line.slope
    return p, dd / np.linalg.norm(dd)


def from_image(p, dd, box: CropBox, name: str, px_mm: float) -> tuple[float, float] | None:
    """(offset_mm, slope) of an image line against side ``name`` of ``box``; None when it runs across it."""
    a, e, d, n = _side(box, KEY[name])
    along = float(np.dot(dd, d))
    t = normal_hit(p, dd, (a + e) / 2, n)
    if t is None or abs(along) < 1e-9:
        return None
    return t / px_mm, float(np.dot(dd, n)) / along


def box_lines(image: np.ndarray, box: CropBox, dpi: float) -> dict[str, list[SideLine]]:
    """``side_lines`` of a final box as ``detect`` returns it.  A spread box (``meta["spread_pages"]``)
    gets its pages' lines on the spread sides they lie along (outward normal within 10 deg, side's
    middle within 1 mm of the spread side); the pages' spines are no spread side and give none."""
    if "spread_pages" not in box.meta:
        return side_lines(image, box, dpi)
    from .booklet import _from_frame

    px_mm = dpi / MM_PER_INCH
    out: dict[str, list[SideLine]] = {name: [] for name in SIDE_NAMES}
    for i, corners in enumerate(box.meta["spread_pages"]):
        page = _from_frame(box, corners)
        page.meta = {"keep_sides": box.meta.get("spread_keep", [[], []])[i]}
        for pname, lines in side_lines(image, page, dpi).items():
            pa, pe, _pd, pn = _side(page, KEY[pname])
            for sname in SIDE_NAMES:
                sa, se, _sd, sn = _side(box, KEY[sname])
                if float(pn @ sn) < math.cos(math.radians(10)) or abs(float(((pa + pe) / 2 - (sa + se) / 2) @ sn)) > px_mm:
                    continue
                for ln in lines:
                    got = from_image(*to_image(ln, page, pname, px_mm), box, sname, px_mm)
                    if got is not None:
                        out[sname].append(SideLine(got[0], got[1], ln.support, ln.kind))
    return {name: _merge(lines) for name, lines in out.items()}


def spine_names(box: CropBox) -> set[str]:
    """The names of a box's spine sides (``meta["keep_sides"]``)."""
    keep = {tuple(k) for k in box.meta.get("keep_sides", ())}
    return {name for name, key in KEY.items() if key in keep}


def signals(lines: list[SideLine], margin_mm: float, spine: bool) -> dict[str, float]:
    """What a flag rule looks at on one side, from the paper edge under it (a box side lies
    ``margin_mm`` inside the edge auto-crop found): ``near_strong`` the distance to the nearest other
    line with support >= STRONG (FAR_MM: none) and ``near_strong_at`` its signed offset (+ = out);
    ``own_support`` the support of the line at the edge (0: none within OWN_MM); ``no_crease`` 1 for a
    spine without a valley line within CREASE_MM; ``crowd`` the lines within CROWD_MM."""
    rel = [(ln.offset_mm - margin_mm, ln) for ln in lines]
    strong = [x for x, ln in rel if ln.support >= STRONG and abs(x) > OWN_MM]
    at = min(strong, key=abs) if strong else 0.0
    return {"near_strong": abs(at) if strong else FAR_MM, "near_strong_at": at,
            "own_support": max((ln.support for x, ln in rel if abs(x) <= OWN_MM), default=0.0),
            "no_crease": float(spine and not any(abs(x) <= CREASE_MM for x, ln in rel if ln.kind == "crease")),
            "crowd": float(sum(abs(x) <= CROWD_MM for x, _ in rel))}


_FIRES = {"near_strong": lambda v, t: v <= t, "own_support": lambda v, t: v < t,
          "no_crease": lambda v, t: v >= t, "crowd": lambda v, t: v >= t}


def check_reason(sig: dict, rule) -> str | None:
    """Why a side should be checked under ``rule`` (its first condition that fires), or None."""
    for name, t in rule:
        if not _FIRES[name](sig[name], t):
            continue
        if name == "near_strong":
            at = sig["near_strong_at"]
            return f"another edge {abs(at):.1f} mm further {'out' if at > 0 else 'in'}"
        if name == "own_support":
            return "the edge here is faint"
        if name == "no_crease":
            return "no clear crease at the fold"
        return f"{int(sig['crowd'])} edges within 3 mm"
    return None
