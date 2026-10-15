"""Rotated crop rectangles and the helpers that move them between images.

Coordinates follow the OpenCV convention: the centre of pixel ``(i, j)`` sits at
``x = j, y = i`` and the y axis points down.  A box's ``angle`` (degrees) is the
direction of its local x axis, so a positive angle looks like a clockwise turn
on screen.  Boxes are kept normalised so that ``-45 < angle <= 45``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Sequence

import cv2
import numpy as np

#: A box's sides by name, in its own frame (top = towards -y at angle 0), in the order of ``corners()``' pairs.
SIDE_NAMES = ("top", "right", "bottom", "left")


def normalize_angle(w: float, h: float, angle: float) -> tuple[float, float, float]:
    """Return an equivalent ``(w, h, angle)`` with the angle in ``(-45, 45]``."""
    angle = math.fmod(angle, 180.0)
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    # Now in (-90, 90].  Fold into (-45, 45] by swapping axes.
    if angle > 45.0:
        angle -= 90.0
        w, h = h, w
    elif angle <= -45.0:
        angle += 90.0
        w, h = h, w
    return w, h, angle


@dataclass
class CropBox:
    """A (possibly rotated) rectangle in image pixel coordinates."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0
    confidence: float = 1.0
    #: Set when the user drew or edited the box by hand.  Manual boxes are
    #: never moved by edge refinement.
    manual: bool = False
    #: Extra information the detector wants to pass along (diagnostics).
    meta: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.w, self.h, self.angle = normalize_angle(
            float(self.w), float(self.h), float(self.angle)
        )
        self.cx = float(self.cx)
        self.cy = float(self.cy)

    # ------------------------------------------------------------------ basics
    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors of the box's local x (width) and y (height) axes."""
        t = math.radians(self.angle)
        ux = np.array([math.cos(t), math.sin(t)])
        uy = np.array([-math.sin(t), math.cos(t)])
        return ux, uy

    def corners(self) -> np.ndarray:
        """Corner points ``(4, 2)`` ordered top-left, top-right, bottom-right,
        bottom-left in the box's own frame."""
        ux, uy = self.axes
        c = np.array([self.cx, self.cy])
        hw, hh = self.w / 2.0, self.h / 2.0
        return np.array(
            [
                c - hw * ux - hh * uy,
                c + hw * ux - hh * uy,
                c + hw * ux + hh * uy,
                c - hw * ux + hh * uy,
            ]
        )

    def bounding_rect(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds ``(x0, y0, x1, y1)`` in continuous coordinates."""
        pts = self.corners()
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    # ------------------------------------------------------------- transforms
    def scaled(self, factor: float, factor_y: float | None = None) -> "CropBox":
        """Scale about the image origin (use when resizing the image).

        Pixel centres sit at integer coordinates, so the pixel *edge* origin is
        at -0.5; scaling is done about that edge so that boxes stay aligned with
        the content when an image is resampled.
        """
        fy = factor if factor_y is None else factor_y
        if abs(factor - fy) > 1e-9 and abs(self.angle) > 1e-9:
            # Anisotropic scaling of a rotated box: go through the corners.
            pts = (self.corners() + 0.5) * np.array([factor, fy]) - 0.5
            return replace(self, **_rect_from_points(pts, self.angle))
        return replace(
            self,
            cx=(self.cx + 0.5) * factor - 0.5,
            cy=(self.cy + 0.5) * fy - 0.5,
            w=self.w * factor,
            h=self.h * fy,
            meta=dict(self.meta),
        )

    def translated(self, dx: float, dy: float) -> "CropBox":
        return replace(self, cx=self.cx + dx, cy=self.cy + dy, meta=dict(self.meta))

    def inset(self, amount: float) -> "CropBox":
        """Shrink every side by ``amount`` pixels (negative grows the box)."""
        w = max(1.0, self.w - 2 * amount)
        h = max(1.0, self.h - 2 * amount)
        return replace(self, w=w, h=h, meta=dict(self.meta))

    def rotated90(self, turns: int = 1) -> "CropBox":
        """Swap width/height ``turns`` times (output orientation hint only)."""
        if turns % 2:
            return replace(self, w=self.h, h=self.w, meta=dict(self.meta))
        return replace(self, meta=dict(self.meta))

    def clamp_to(self, width: int, height: int) -> "CropBox":
        """Shrink the box so that it lies within an image of the given size.

        The angle is kept; each side that sticks out of the image is pulled in.
        Because angles are normalised to +-45 degrees, an x overflow is always
        fixed by moving the left/right side and a y overflow by moving the
        top/bottom side.
        """
        x0, y0, x1, y1 = -0.5, -0.5, width - 0.5, height - 0.5
        t = math.radians(self.angle)
        cos, sin = math.cos(t), math.sin(t)
        lo_u, hi_u = -self.w / 2.0, self.w / 2.0
        lo_v, hi_v = -self.h / 2.0, self.h / 2.0
        eps = 1e-6
        for _ in range(8):
            d_lo_u = d_hi_u = d_lo_v = d_hi_v = 0.0
            for u in (lo_u, hi_u):
                for v in (lo_v, hi_v):
                    x = self.cx + u * cos - v * sin
                    y = self.cy + u * sin + v * cos
                    d_lo_u = max(d_lo_u, (x0 - x) / cos)
                    d_hi_u = max(d_hi_u, (x - x1) / cos)
                    d_lo_v = max(d_lo_v, (y0 - y) / cos)
                    d_hi_v = max(d_hi_v, (y - y1) / cos)
            if max(d_lo_u, d_hi_u, d_lo_v, d_hi_v) <= eps:
                break
            lo_u += d_lo_u
            hi_u -= d_hi_u
            lo_v += d_lo_v
            hi_v -= d_hi_v
            if hi_u - lo_u < 1:
                mid = (hi_u + lo_u) / 2
                lo_u, hi_u = mid - 0.5, mid + 0.5
            if hi_v - lo_v < 1:
                mid = (hi_v + lo_v) / 2
                lo_v, hi_v = mid - 0.5, mid + 0.5
        mu, mv = (lo_u + hi_u) / 2.0, (lo_v + hi_v) / 2.0
        return replace(
            self,
            cx=self.cx + mu * cos - mv * sin,
            cy=self.cy + mu * sin + mv * cos,
            w=hi_u - lo_u,
            h=hi_v - lo_v,
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------ conversions
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("meta", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CropBox":
        keys = {"cx", "cy", "w", "h", "angle", "confidence", "manual"}
        return cls(**{k: v for k, v in d.items() if k in keys})

    @classmethod
    def from_points(cls, pts: np.ndarray, **kw) -> "CropBox":
        """Minimum-area rectangle around a point set."""
        (cx, cy), (w, h), angle = cv2.minAreaRect(np.asarray(pts, np.float32))
        return cls(cx, cy, w, h, angle, **kw)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float, **kw) -> "CropBox":
        """Axis-aligned box from pixel-edge origin ``(x, y)`` and size."""
        return cls(x + w / 2.0 - 0.5, y + h / 2.0 - 0.5, w, h, 0.0, **kw)


def side_frames(box: CropBox) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each side's middle and outward unit normal, by name (the box's own frame)."""
    c = box.corners()
    ux, uy = box.axes
    return {"top": ((c[0] + c[1]) / 2, -uy), "right": ((c[1] + c[2]) / 2, ux),
            "bottom": ((c[2] + c[3]) / 2, uy), "left": ((c[3] + c[0]) / 2, -ux)}


def pin_sides(given: CropBox, refined: CropBox, pinned: Iterable[str]) -> CropBox:
    """``given`` with its angle kept, each pinned side where it is and each other side moved parallel to
    where ``refined`` put it (the refined side's middle projected on ``given``'s normal).  Confidence from
    ``refined``; ``manual`` and ``meta`` from ``given`` (a hand-edited box keeps its ids and pins)."""
    pinned = set(pinned)
    mine, theirs = side_frames(given), side_frames(refined)
    move = {}
    for name, (m, n) in mine.items():
        if name in pinned:
            move[name] = 0.0
            continue
        rm = max(theirs.values(), key=lambda t: float(t[1] @ n))[0]  # the refined side facing the same way
        move[name] = float((rm - m) @ n)
    ux, uy = given.axes
    c = (np.array([given.cx, given.cy]) + ux * (move["right"] - move["left"]) / 2
         + uy * (move["bottom"] - move["top"]) / 2)
    return CropBox(c[0], c[1], given.w + move["left"] + move["right"], given.h + move["top"] + move["bottom"],
                   given.angle, refined.confidence, manual=given.manual, meta=dict(given.meta))


def _rect_from_points(pts: np.ndarray, angle_hint: float) -> dict:
    box = CropBox.from_points(pts)
    return {"cx": box.cx, "cy": box.cy, "w": box.w, "h": box.h, "angle": box.angle}


# ---------------------------------------------------------------- extraction
def box_to_affine(box: CropBox) -> tuple[np.ndarray, tuple[int, int]]:
    """Affine matrix mapping *output* pixel coords to *input* coords.

    Returns ``(M, (out_w, out_h))`` suitable for ``cv2.warpAffine`` with the
    ``WARP_INVERSE_MAP`` flag.
    """
    out_w = max(1, int(round(box.w)))
    out_h = max(1, int(round(box.h)))
    ux, uy = box.axes
    # x_in = cx + ux * (u + 0.5 - out_w/2) + uy * (v + 0.5 - out_h/2)
    ox = 0.5 - out_w / 2.0
    oy = 0.5 - out_h / 2.0
    M = np.array(
        [
            [ux[0], uy[0], box.cx + ux[0] * ox + uy[0] * oy],
            [ux[1], uy[1], box.cy + ux[1] * ox + uy[1] * oy],
        ],
        dtype=np.float64,
    )
    return M, (out_w, out_h)


def extract(image: np.ndarray, box: CropBox, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Cut a box out of an image, rotating it upright.

    Works for 8- and 16-bit, grey or colour images.  Axis-aligned boxes whose
    edges fall on pixel boundaries are sliced directly with no resampling.
    """
    out_w = max(1, int(round(box.w)))
    out_h = max(1, int(round(box.h)))
    x0 = box.cx - out_w / 2.0 + 0.5
    y0 = box.cy - out_h / 2.0 + 0.5
    H, W = image.shape[:2]
    if (
        abs(box.angle) < 1e-6
        and abs(x0 - round(x0)) < 1e-6
        and abs(y0 - round(y0)) < 1e-6
    ):
        xi, yi = int(round(x0)), int(round(y0))
        if xi >= 0 and yi >= 0 and xi + out_w <= W and yi + out_h <= H:
            return image[yi : yi + out_h, xi : xi + out_w].copy()
    M, size = box_to_affine(box)
    src = image
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        src = image.astype(np.float32)
    out = cv2.warpAffine(
        src,
        M,
        size,
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if out.ndim == 2 and image.ndim == 3:
        out = out[:, :, None]
    return out


# ------------------------------------------------------------------- metrics
def polygon_iou(a: CropBox, b: CropBox) -> float:
    """Intersection-over-union of two rotated rectangles."""
    pa = a.corners().astype(np.float32)
    pb = b.corners().astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    union = a.area + b.area - inter
    return float(inter / union) if union > 0 else 0.0


def angle_difference(a: float, b: float) -> float:
    """Smallest difference between two box angles (mod 90 degrees)."""
    d = math.fmod(a - b, 90.0)
    if d > 45:
        d -= 90
    elif d < -45:
        d += 90
    return abs(d)


def match_boxes(
    found: Sequence[CropBox], truth: Sequence[CropBox]
) -> list[tuple[int, int, float]]:
    """Greedy best-IoU matching; returns ``(found_idx, truth_idx, iou)``."""
    pairs = []
    for i, f in enumerate(found):
        for j, t in enumerate(truth):
            iou = polygon_iou(f, t)
            if iou > 0:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_f: set[int] = set()
    used_t: set[int] = set()
    out = []
    for iou, i, j in pairs:
        if i in used_f or j in used_t:
            continue
        used_f.add(i)
        used_t.add(j)
        out.append((i, j, iou))
    return out


def reading_order(boxes: Iterable[CropBox]) -> list[CropBox]:
    """Sort boxes top-to-bottom in rows, then left-to-right within a row."""
    items = sorted(boxes, key=lambda b: b.bounding_rect()[1])
    rows: list[list[CropBox]] = []
    for b in items:
        x0, y0, x1, y1 = b.bounding_rect()
        placed = False
        for row in rows:
            ry0 = min(r.bounding_rect()[1] for r in row)
            ry1 = max(r.bounding_rect()[3] for r in row)
            overlap = min(y1, ry1) - max(y0, ry0)
            if overlap > 0.5 * min(y1 - y0, ry1 - ry0):
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    out: list[CropBox] = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: b.cx))
    return out
