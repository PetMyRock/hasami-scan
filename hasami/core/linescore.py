"""Strip patches and features around candidate edge lines (learned line classifier).

The same code builds the training data (in the private data repo) and, later, the classifier's
input, so the two cannot drift apart.  Everything is in mm: a 200 dpi preview and a 400 dpi scan
give the same patch for the same paper.  A candidate line is ``x = x0 + slope * y`` in its side's
frame (as in ``toppage``: x outwards along the normal, y along the side from its middle, in px).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .toppage import _to_lab

MM_PER_INCH = 25.4
SEGMENTS = 16  # patch rows: medians of this many segments along the side (as toppage)
ALONG_MM = 0.25  # sampling step along the side
FINE_COLS = -4.0 + 0.1 * np.arange(64)  # mm across the line, + = outwards
COARSE_COLS = -8.0 + 0.5 * np.arange(32)
MERGE_MM = 0.15  # lines of different pools this close are one candidate
POOLS = ("layer", "faint", "strong", "edge")


@dataclass(frozen=True)
class Side:
    """A side's frame in an image: its middle, unit direction along it, outward unit normal,
    length (px) and the image's dpi."""

    mid: np.ndarray
    d: np.ndarray
    n: np.ndarray
    length: float
    dpi: float

    @property
    def px_mm(self) -> float:
        return self.dpi / MM_PER_INCH

    @classmethod
    def from_record(cls, rec: dict) -> "Side":
        """The frame of a ``candidates`` record ("edge" records need ``dpi`` added by the caller)."""
        if rec["kind"] == "edge":
            p0, p1 = np.asarray(rec["p0"], float), np.asarray(rec["p1"], float)
            length = float(np.linalg.norm(p1 - p0))
            return cls((p0 + p1) / 2, (p1 - p0) / length, np.asarray(rec["n"], float), length, float(rec["dpi"]))
        return cls(np.asarray(rec["mid"], float), np.asarray(rec["d"], float), np.asarray(rec["n"], float),
                   float(rec["side_len"]), float(rec["dpi"]))


def sample(rgb8: np.ndarray, side: Side, x0: float, slope: float, cols_mm: np.ndarray) -> np.ndarray:
    """(SEGMENTS, len(cols_mm), 4) float32 across the line: median L*, a*, b* per segment and
    column, and the share of the samples inside the image (outside: L* -1, a* b* 0, as toppage)."""
    px = side.px_mm
    t = np.arange(-0.44 * side.length, 0.44 * side.length, ALONG_MM * px)
    if len(t) < SEGMENTS:
        t = np.linspace(-0.44 * side.length, 0.44 * side.length, SEGMENTS)
    X = x0 + slope * t[:, None] + np.asarray(cols_mm, float)[None, :] * px
    T = np.broadcast_to(t[:, None], X.shape)
    mx = (side.mid[0] + side.n[0] * X + side.d[0] * T).astype(np.float32)
    my = (side.mid[1] + side.n[1] * X + side.d[1] * T).astype(np.float32)
    H, W = rgb8.shape[:2]
    valid = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    lab = _to_lab(cv2.remap(rgb8, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    lab[~valid] = (-1.0, 0.0, 0.0)
    out = np.empty((SEGMENTS, X.shape[1], 4), np.float32)
    for k, rows in enumerate(np.array_split(np.arange(len(t)), SEGMENTS)):
        out[k, :, :3] = np.median(lab[rows], axis=0)
        out[k, :, 3] = valid[rows].mean(axis=0)
    return out


def patches(rgb8: np.ndarray, side: Side, x0: float, slope: float) -> tuple[np.ndarray, np.ndarray]:
    """The fine patch (0.1 mm columns, -4.0..+2.3 mm) and the coarse band (0.5 mm, -8.0..+7.5 mm)."""
    return sample(rgb8, side, x0, slope, FINE_COLS), sample(rgb8, side, x0, slope, COARSE_COLS)


def merge_pools(pools: dict, px_mm: float) -> list[dict]:
    """One candidate per line: lines of the pools within MERGE_MM are one (the first pool's
    position, in POOLS order; the highest support).  Outermost first."""
    out: list[dict] = []
    for name in POOLS:
        for x0, slope, sup in pools.get(name, ()):
            same = next((c for c in out if abs(c["x0"] - x0) <= MERGE_MM * px_mm), None)
            if same is not None:
                same["pools"].add(name)
                same["support"] = max(same["support"], sup)
            else:
                out.append({"x0": float(x0), "slope": float(slope), "support": sup, "pools": {name}})
    return sorted(out, key=lambda c: -c["x0"])
