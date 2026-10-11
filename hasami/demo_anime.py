"""Procedural anime-style content for synthetic scans.

Everything here is drawn from simple shapes: original, stylised characters
(big eyes, spiky hair, bold outlines, cel shading) in illustrations, artbook
pages (full-bleed art, or art with a caption; one illustration may also
spread across two pages) and trading cards with rounded corners.

These matter for auto-crop because they behave differently from ordinary
photos: full-bleed art puts strong content right on the paper edge, vivid
flat colours and bold outlines create many strong straight edges inside the
item, pastel art can be nearly as light as the lid, and cards have rounded
corners.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

INK = (28.0, 22.0, 30.0)

HAIR = [
    (255, 140, 190), (95, 150, 255), (212, 215, 232), (255, 150, 60), (170, 110, 230),
    (45, 48, 85), (250, 222, 120), (220, 55, 65), (60, 200, 180), (120, 70, 50),
]
EYES = [(60, 120, 255), (230, 60, 120), (60, 190, 120), (250, 170, 40), (150, 80, 220), (220, 40, 40)]
SKIN = (255, 230, 214)
SKIN_SHADE = (242, 192, 178)
SKIES = [
    [(110, 180, 255), (205, 235, 255)],  # day
    [(255, 140, 120), (255, 205, 150), (150, 120, 205)],  # sunset
    [(18, 28, 78), (70, 60, 150)],  # night
    [(255, 205, 225), (255, 245, 250)],  # pastel pink
    [(150, 230, 210), (240, 255, 245)],  # mint
]


def _shade(c, f):
    return tuple(float(np.clip(v * f, 0, 255)) for v in c)


def _lerp(c0, c1, t):
    return tuple(a + (b - a) * t for a, b in zip(c0, c1))


def _gradient(h: int, w: int, stops, rng) -> np.ndarray:
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    n = len(stops) - 1
    img = np.zeros((h, 1, 3), np.float32)
    for k in range(n):
        lo, hi = k / n, (k + 1) / n
        seg = np.clip((t - lo) / (hi - lo), 0, 1)
        m = (t >= lo) & (t <= hi)
        col = np.array(stops[k], np.float32) * (1 - seg) + np.array(stops[k + 1], np.float32) * seg
        img = np.where(m, col, img)
    return np.repeat(img, w, axis=1)


def _star(img, cx, cy, r, color, thickness=-1):
    pts = []
    for k in range(8):
        a = k * math.pi / 4 - math.pi / 2
        rr = r if k % 2 == 0 else r * 0.28
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    cv2.fillPoly(img, [np.round(np.array(pts)).astype(np.int32)], color, cv2.LINE_AA)


def _poly(img, pts, fill, outline=INK, thick=2):
    p = np.round(np.array(pts)).astype(np.int32)
    if fill is not None:
        cv2.fillPoly(img, [p], fill, cv2.LINE_AA)
    if outline is not None and thick > 0:
        cv2.polylines(img, [p], True, outline, thick, cv2.LINE_AA)


# ------------------------------------------------------------- characters
def draw_character(img: np.ndarray, cx: float, cy: float, s: float, rng: np.random.Generator) -> None:
    """Draw a stylised anime character bust with its head centred at (cx, cy).

    ``s`` is the head radius in pixels.
    """
    hair = HAIR[rng.integers(len(HAIR))]
    eye = EYES[rng.integers(len(EYES))]
    skin = SKIN
    lw = max(1, int(round(s / 28)))
    ink = INK

    # Body / uniform.
    shirt = [(70, 80, 140), (240, 240, 245), (40, 40, 50), (200, 70, 90)][rng.integers(4)]
    body = [(cx - 1.6 * s, cy + 3.2 * s), (cx - 1.1 * s, cy + 1.25 * s), (cx + 1.1 * s, cy + 1.25 * s),
            (cx + 1.6 * s, cy + 3.2 * s)]
    _poly(img, body, shirt, ink, lw)
    collar = [(cx - 1.05 * s, cy + 1.3 * s), (cx, cy + 2.0 * s), (cx + 1.05 * s, cy + 1.3 * s),
              (cx + 0.6 * s, cy + 1.25 * s), (cx, cy + 1.6 * s), (cx - 0.6 * s, cy + 1.25 * s)]
    _poly(img, collar, (30, 40, 90), ink, lw)
    # Neck.
    _poly(img, [(cx - 0.28 * s, cy + 0.7 * s), (cx + 0.28 * s, cy + 0.7 * s), (cx + 0.3 * s, cy + 1.35 * s),
                (cx - 0.3 * s, cy + 1.35 * s)], _shade(skin, 0.95), ink, lw)

    # Back hair: a big rounded mass with spiky ends.
    back = []
    for k in range(25):
        a = math.pi * (0.95 + 1.1 * k / 24)
        back.append((cx + 1.25 * s * math.cos(a), cy - 0.1 * s + 1.2 * s * math.sin(a)))
    length = rng.uniform(1.0, 2.6)
    for k in range(7):
        x = cx + s * (1.25 - 2.5 * k / 6)
        back.append((x + rng.uniform(-0.1, 0.1) * s, cy + length * s + (0.35 * s if k % 2 else 0)))
    _poly(img, back, _shade(hair, 0.85), ink, lw)

    # Face: round top, pointed chin.
    face = []
    for k in range(21):
        a = math.pi * (1 + k / 20)
        face.append((cx + 0.95 * s * math.cos(a), cy + 0.95 * s * math.sin(a)))
    face += [(cx + 0.9 * s, cy + 0.25 * s), (cx + 0.45 * s, cy + 0.85 * s), (cx, cy + 1.02 * s),
             (cx - 0.45 * s, cy + 0.85 * s), (cx - 0.9 * s, cy + 0.25 * s)]
    _poly(img, face, skin, ink, lw)
    # Shadow under the bangs (cel shading).
    _poly(img, [(cx - 0.9 * s, cy - 0.2 * s), (cx + 0.9 * s, cy - 0.2 * s), (cx + 0.85 * s, cy),
                (cx - 0.85 * s, cy)], SKIN_SHADE, None, 0)

    # Eyes.
    for side in (-1, 1):
        ex, ey = cx + side * 0.42 * s, cy + 0.22 * s
        ew, eh = 0.26 * s, 0.36 * s
        cv2.ellipse(img, (int(ex), int(ey)), (int(ew), int(eh)), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(ex), int(ey + 0.04 * s)), (int(ew * 0.8), int(eh * 0.85)), 0, 0, 360, eye, -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(ex), int(ey - 0.08 * s)), (int(ew * 0.8), int(eh * 0.45)), 0, 180, 360,
                    _shade(eye, 0.55), -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(ex), int(ey + 0.06 * s)), (int(ew * 0.35), int(eh * 0.45)), 0, 0, 360, (20, 15, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (int(ex - 0.08 * s * side), int(ey - 0.1 * s)), max(1, int(0.08 * s)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(ex + 0.08 * s * side), int(ey + 0.15 * s)), max(1, int(0.04 * s)), (255, 255, 255), -1, cv2.LINE_AA)
        # Upper lash line.
        cv2.ellipse(img, (int(ex), int(ey)), (int(ew * 1.15), int(eh * 1.02)), 0, 200, 340, ink, max(2, lw * 2), cv2.LINE_AA)
    # Mouth and blush.
    cv2.ellipse(img, (int(cx), int(cy + 0.68 * s)), (int(0.12 * s), int(0.06 * s)), 0, 10, 170, ink, lw, cv2.LINE_AA)
    for side in (-1, 1):
        cv2.ellipse(img, (int(cx + side * 0.55 * s), int(cy + 0.55 * s)), (int(0.16 * s), int(0.07 * s)), 0, 0, 360,
                    (255, 170, 185), -1, cv2.LINE_AA)

    # Bangs: spiky locks over the forehead.
    bangs = []
    for k in range(13):
        a = math.pi * (1.0 + k / 12)
        bangs.append((cx + 1.08 * s * math.cos(a), cy - 0.05 * s + 1.08 * s * math.sin(a)))
    n_locks = int(rng.integers(5, 8))
    for k in range(n_locks + 1):
        x = cx + 1.0 * s - 2.0 * s * k / n_locks
        y = cy + (0.12 + rng.uniform(-0.06, 0.12)) * s if k % 2 == 0 else cy - 0.35 * s
        bangs.append((x, y))
    _poly(img, bangs, hair, ink, lw)
    # Highlight band on the hair.
    cv2.ellipse(img, (int(cx), int(cy - 0.55 * s)), (int(0.7 * s), int(0.18 * s)), 0, 200, 340,
                _shade(hair, 1.25), max(2, int(0.08 * s)), cv2.LINE_AA)
    # Ahoge.
    cv2.ellipse(img, (int(cx + 0.15 * s), int(cy - 1.15 * s)), (int(0.25 * s), int(0.3 * s)), 0, 120, 300, ink, lw, cv2.LINE_AA)


def anime_illustration(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """A full-colour, cel-shaded anime illustration (float32 RGB 0..255)."""
    img = _gradient(h, w, SKIES[rng.integers(len(SKIES))], rng)
    night = img[0, 0].mean() < 80
    if night:
        for _ in range(int(w * h / 3000)):
            x, y = rng.uniform(0, w), rng.uniform(0, h * 0.7)
            cv2.circle(img, (int(x), int(y)), 1, (255, 255, 230), -1, cv2.LINE_AA)
    # Clouds.
    for _ in range(rng.integers(2, 6)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h * 0.6)
        base = rng.uniform(0.05, 0.12) * w
        col = (255, 255, 255) if not night else (90, 90, 150)
        for _k in range(5):
            cv2.circle(img, (int(cx + rng.uniform(-2, 2) * base), int(cy + rng.uniform(-0.4, 0.4) * base)),
                       int(base * rng.uniform(0.6, 1.1)), col, -1, cv2.LINE_AA)
    # Ground / city silhouette.
    gy = int(h * rng.uniform(0.62, 0.8))
    ground = [(0, h), (0, gy)]
    x = 0
    while x < w:
        bw = rng.uniform(0.05, 0.15) * w
        bh = rng.uniform(0.0, 0.18) * h
        ground += [(x, gy - bh), (x + bw, gy - bh)]
        x += bw
    ground += [(w, gy), (w, h)]
    _poly(img, ground, _shade(img[-1, 0], 0.55) if not night else (25, 25, 50), INK, max(1, w // 300))
    # Bokeh and sparkles.
    for _ in range(rng.integers(5, 15)):
        r = rng.uniform(0.01, 0.04) * w
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        over = img.copy()
        cv2.circle(over, (int(cx), int(cy)), int(r), (255, 250, 220), -1, cv2.LINE_AA)
        img = 0.7 * img + 0.3 * over
    # Character(s).
    n = 1 if rng.random() < 0.75 else 2
    for k in range(n):
        s = min(w, h) * rng.uniform(0.13, 0.22) / (1 + 0.3 * (n - 1))
        cx = w * ((k + 1) / (n + 1)) + rng.uniform(-0.05, 0.05) * w
        cy = h * rng.uniform(0.35, 0.55)
        draw_character(img, cx, cy, s, rng)
    for _ in range(rng.integers(4, 12)):
        _star(img, rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.008, 0.03) * w, (255, 255, 240))
    # Sakura petals.
    for _ in range(rng.integers(0, 25)):
        cv2.ellipse(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))),
                    (max(1, int(0.008 * w)), max(1, int(0.005 * w))), rng.uniform(0, 180), 0, 360,
                    (255, 185, 205), -1, cv2.LINE_AA)
    img += rng.normal(0, 2.0, img.shape)  # print grain
    return np.clip(img, 0, 255).astype(np.float32)


def artbook_page(w: int, h: int, rng: np.random.Generator, paper: float = 248.0) -> np.ndarray:
    """An artbook page: a full-bleed illustration, or art with a caption."""
    if rng.random() < 0.6:
        return anime_illustration(w, h, rng)
    img = np.full((h, w, 3), paper, np.float32)
    m = int(w * rng.uniform(0.08, 0.12))
    ah = int((h - 2 * m) * rng.uniform(0.6, 0.8))
    img[m : m + ah, m : w - m] = anime_illustration(w - 2 * m, ah, rng)
    y = m + ah + int(0.04 * h)
    while y < h - m:
        x1 = int(m + rng.uniform(0.3, 1.0) * (w - 2 * m))
        cv2.rectangle(img, (m, y), (x1, y + max(2, h // 160)), (90, 90, 100), -1)
        y += max(6, h // 60)
    return img


def trading_card(w: int, h: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A trading card: framed illustration, name bar, text box, rounded corners.

    Returns ``(rgb, alpha)``; alpha is 0 outside the rounded corners.
    """
    frame_col = [(230, 190, 60), (90, 160, 255), (220, 70, 90), (160, 90, 220), (60, 190, 120)][rng.integers(5)]
    img = _gradient(h, w, [_shade(frame_col, 1.15), frame_col, _shade(frame_col, 0.7)], rng)
    # Foil shimmer.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img += (12 * np.sin((xx + yy) / (w * 0.05)))[..., None]
    b = int(w * 0.06)
    name_h = int(h * 0.08)
    cv2.rectangle(img, (b, b), (w - b, b + name_h), (30, 30, 40), -1)
    cv2.rectangle(img, (b + name_h // 3, b + name_h // 3), (int(w * 0.6), b + 2 * name_h // 3), (240, 240, 240), -1)
    art_y0 = b + name_h + b // 2
    art_y1 = int(h * 0.68)
    img[art_y0:art_y1, b : w - b] = anime_illustration(w - 2 * b, art_y1 - art_y0, rng)
    cv2.rectangle(img, (b, art_y0), (w - b, art_y1), INK, max(1, w // 150))
    cv2.rectangle(img, (b, art_y1 + b // 2), (w - b, h - b), (245, 240, 225), -1)
    y = art_y1 + b
    while y < h - 1.5 * b:
        cv2.rectangle(img, (int(1.5 * b), y), (int(w - 1.5 * b - rng.uniform(0, 0.3) * w), y + max(2, h // 150)),
                      (60, 60, 70), -1)
        y += max(6, h // 45)
    alpha = np.ones((h, w), np.float32)
    r = max(3, int(w * 0.045))
    corner = np.zeros((r, r), np.float32)
    cv2.circle(corner, (r, r), r, 1.0, -1, cv2.LINE_AA)
    alpha[:r, :r] = corner
    alpha[:r, -r:] = corner[:, ::-1]
    alpha[-r:, :r] = corner[::-1, :]
    alpha[-r:, -r:] = corner[::-1, ::-1]
    return np.clip(img, 0, 255).astype(np.float32), alpha
