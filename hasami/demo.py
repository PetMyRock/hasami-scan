"""Synthetic flatbed scans with known ground truth.

Used by the test-suite, the benchmark, and the built-in "Demo scanner" device so
the app can be tried without hardware.  The generator models the things that
trip up auto-crop in real scans: lids that are not perfectly white, uneven
lighting, photos with white borders, lid shadows along one or two edges, dust,
hairs, a dark strip where the scan area meets the bed frame, and photos that
almost touch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import demo_anime
from .core.geometry import CropBox


@dataclass
class SynthOptions:
    width: int = 1700  # ~A4 at 200 dpi
    height: int = 2340
    n_photos: int = 3
    background: str = "white"  # "white", "black", "grey"
    max_angle: float = 8.0
    min_gap: int = 25  # pixels between photos; can be 0 for touching photos
    photo_size: tuple[float, float] = (0.28, 0.45)  # fraction of page width
    bordered: float = 0.35  # probability that a photo has a white border
    light_photos: float = 0.15  # probability of a washed-out, mostly white photo
    shadow: bool = True
    vignette: bool = True
    dust: bool = True
    hair: bool = True
    frame_strip: bool = False  # dark band along one image edge
    noise: float = 2.0  # sigma in 8-bit levels
    corner_photo: bool = False  # push the first photo into the top-left corner
    document: bool = False  # one white page with text instead of photos
    jpeg_quality: int | None = None  # simulate a JPEG-compressed source
    #: "generic" photos/documents, or "anime": illustration prints, trading
    #: cards with rounded corners and shikishi boards; a document becomes an
    #: artbook page / art print.
    theme: str = "generic"


@dataclass
class SynthScan:
    image: np.ndarray  # uint8 RGB
    truth: list[CropBox] = field(default_factory=list)
    options: SynthOptions | None = None


# ------------------------------------------------------------------ content
def _photo_content(w: int, h: int, rng: np.random.Generator, light: bool) -> np.ndarray:
    """A plausible-looking photograph: sky, ground, blobs and grain."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    horizon = h * rng.uniform(0.35, 0.65) + (xx - w / 2) * rng.uniform(-0.15, 0.15)
    sky_top = rng.uniform(80, 200, 3)
    sky_bot = np.clip(sky_top + rng.uniform(20, 60, 3), 0, 255)
    ground = rng.uniform(30, 160, 3)
    t = np.clip(yy / max(horizon.max(), 1), 0, 1)[..., None]
    img = sky_top * (1 - t) + sky_bot * t
    below = (yy > horizon)[..., None]
    gtex = (np.sin(xx / rng.uniform(5, 30)) * np.cos(yy / rng.uniform(5, 30)))[..., None]
    img = np.where(below, ground + 18 * gtex, img)
    for _ in range(rng.integers(3, 9)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax, ay = rng.uniform(0.04, 0.25) * w, rng.uniform(0.04, 0.25) * h
        col = rng.uniform(0, 255, 3)
        m = (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2) < 1
        img[m] = 0.35 * img[m] + 0.65 * col
    img += rng.normal(0, 6, img.shape)  # film grain
    if light:
        # A washed-out photo (snow scene / overexposure): mostly near-white.
        img = 255 - (255 - img) * rng.uniform(0.08, 0.2)
    return np.clip(img, 0, 255).astype(np.float32)


def _document_content(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    paper = rng.uniform(242, 250)
    img = np.full((h, w, 3), paper, np.float32)
    margin_x = int(w * rng.uniform(0.09, 0.14))
    margin_y = int(h * rng.uniform(0.07, 0.1))
    line_h = max(6, int(h * 0.018))
    y = margin_y
    while y < h - margin_y - line_h:
        if rng.random() < 0.12:  # paragraph gap
            y += line_h
            continue
        x = margin_x
        end = w - margin_x - (rng.uniform(0, 0.4) * w if rng.random() < 0.15 else 0)
        while x < end:
            word = int(rng.uniform(0.02, 0.08) * w)
            x1 = int(min(x + word, end))
            img[y + line_h // 4 : y + line_h - 1, x:x1] = rng.uniform(10, 50)
            x = x1 + int(0.012 * w)
        y += int(line_h * 1.6)
    return img


def _add_border(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    b = int(min(w, h) * rng.uniform(0.04, 0.08))
    paper = np.full((h + 2 * b, w + 2 * b, 3), rng.uniform(238, 250), np.float32)
    paper[b : b + h, b : b + w] = img
    return paper


# -------------------------------------------------------------- composition
def _place(rng, W, H, sizes, opts) -> list[tuple[float, float, float, float, float]]:
    """Choose non-overlapping rotated placements; returns (cx, cy, w, h, angle)."""
    placed: list[CropBox] = []
    out = []
    for i, (pw, ph) in enumerate(sizes):
        for attempt in range(400):
            ang = rng.uniform(-opts.max_angle, opts.max_angle)
            if rng.random() < 0.25:
                ang = rng.uniform(-0.4, 0.4)
            box = CropBox(0, 0, pw, ph, ang)
            x0, y0, x1, y1 = box.bounding_rect()
            bw, bh = x1 - x0, y1 - y0
            if bw >= W - 10 or bh >= H - 10:
                break
            if i == 0 and opts.corner_photo:
                cx, cy = bw / 2 + 1, bh / 2 + 1
            else:
                cx = rng.uniform(bw / 2 + 8, W - bw / 2 - 8)
                cy = rng.uniform(bh / 2 + 8, H - bh / 2 - 8)
            cand = CropBox(cx, cy, pw, ph, ang)
            grown = cand.inset(-opts.min_gap / 2.0)
            ok = True
            for other in placed:
                g2 = other.inset(-opts.min_gap / 2.0)
                inter, _ = cv2.intersectConvexConvex(
                    grown.corners().astype(np.float32), g2.corners().astype(np.float32)
                )
                if inter > 0.5:
                    ok = False
                    break
            if ok:
                placed.append(cand)
                out.append((cx, cy, pw, ph, ang))
                break
    return out


def synth_scan(opts: SynthOptions | None = None, seed: int | None = None) -> SynthScan:
    opts = opts or SynthOptions()
    rng = np.random.default_rng(seed)
    W, H = opts.width, opts.height
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Background (scanner lid).
    if opts.background == "white":
        base = rng.uniform(232, 250) + rng.uniform(-4, 4, 3)
    elif opts.background == "black":
        base = rng.uniform(8, 30) + rng.uniform(-3, 3, 3)
    else:
        base = rng.uniform(110, 160) + rng.uniform(-5, 5, 3)
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = base
    if opts.vignette:
        r2 = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
        grad = 1 - 0.06 * r2 - 0.015 * (yy / H)
        canvas *= grad[..., None]

    # Photos (each piece is an image plus an optional alpha mask).
    pieces: list[tuple[np.ndarray, np.ndarray | None]] = []
    anime = opts.theme == "anime"
    if opts.document:
        pw = int(W * rng.uniform(0.7, 0.85))
        ph = int(pw * 1.414)
        if ph > H * 0.9:
            ph = int(H * 0.88)
            pw = int(ph / 1.414)
        if anime:
            pieces.append((demo_anime.artbook_page(pw, ph, rng, paper=rng.uniform(240, 250)), None))
        else:
            pieces.append((_document_content(pw, ph, rng), None))
    else:
        for _ in range(opts.n_photos):
            s = rng.uniform(*opts.photo_size) * W
            aspect = rng.choice([1.0, 1.25, 1.33, 1.5])
            if rng.random() < 0.5:
                pw, ph = int(s), int(s / aspect)
            else:
                pw, ph = int(s / aspect), int(s)
            light = rng.random() < opts.light_photos
            if anime:
                kind = rng.choice(["card", "print", "print", "shikishi"])
                if kind == "card":
                    cw = int(rng.uniform(0.75, 1.0) * s * 0.72)
                    img, alpha = demo_anime.trading_card(cw, int(cw * 88 / 63), rng)
                    pieces.append((img, alpha))
                    continue
                if kind == "shikishi":
                    side = int(s)
                    inner = demo_anime.anime_illustration(int(side * 0.86), int(side * 0.86), rng)
                    img = np.full((side, side, 3), rng.uniform(240, 250), np.float32)
                    o = (side - inner.shape[0]) // 2
                    img[o : o + inner.shape[0], o : o + inner.shape[1]] = inner
                    gold = (215.0, 175.0, 80.0)
                    img[:3], img[-3:], img[:, :3], img[:, -3:] = gold, gold, gold, gold
                    pieces.append((img, None))
                    continue
                img = demo_anime.anime_illustration(pw, ph, rng)
                if light:
                    img = 255 - (255 - img) * rng.uniform(0.1, 0.25)
            else:
                img = _photo_content(pw, ph, rng, light)
            if rng.random() < opts.bordered:
                img = _add_border(img, rng)
            pieces.append((img, None))
    placements = _place(rng, W, H, [(p.shape[1], p.shape[0]) for p, _ in pieces], opts)

    truth: list[CropBox] = []
    shadow_layer = np.zeros((H, W), np.float32)
    for (img, alpha), (cx, cy, pw, ph, ang) in zip(pieces, placements):
        box = CropBox(cx, cy, pw, ph, ang)
        truth.append(box)
        # Map photo pixels to the canvas.  M maps photo coords -> canvas coords.
        t = math.radians(box.angle)
        ux = np.array([math.cos(t), math.sin(t)])
        uy = np.array([-math.sin(t), math.cos(t)])
        pw_, ph_ = img.shape[1], img.shape[0]
        # Use the box's normalised axes; the photo may need a transpose when
        # normalisation swapped width and height.
        if abs(box.w - pw_) > 0.5:
            img = np.ascontiguousarray(np.rot90(img))
            if alpha is not None:
                alpha = np.ascontiguousarray(np.rot90(alpha))
            pw_, ph_ = img.shape[1], img.shape[0]
        o = np.array([cx, cy]) - ux * (pw_ / 2 - 0.5) - uy * (ph_ / 2 - 0.5)
        M = np.array([[ux[0], uy[0], o[0]], [ux[1], uy[1], o[1]]])
        # Coverage mask: warping a zero-padded block of ones with bilinear
        # interpolation gives anti-aliased edges exactly on the pixel border.
        shift = M[:, 2] - M[:, :2] @ np.array([1.0, 1.0])
        cov = cv2.warpAffine(
            np.pad(np.ones((ph_, pw_), np.float32) if alpha is None else alpha, 1),
            np.hstack([M[:, :2], shift[:, None]]),
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped = cv2.warpAffine(
            img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        # Paper edge darkening (cut paper catches less light).
        edge = cv2.erode((cov > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        rim = (cov > 0.5) & (edge == 0)
        warped[rim] *= 0.9
        if opts.shadow:
            # Lid shadow on the two sides facing away from the lamp.
            sh = cv2.warpAffine(cov, np.float32([[1, 0, 2.5], [0, 1, 2.5]]), (W, H))
            sh = cv2.GaussianBlur(sh, (0, 0), 1.6)
            shadow_layer = np.maximum(shadow_layer, sh * rng.uniform(0.12, 0.3))
        canvas = canvas * (1 - cov[..., None]) + warped * cov[..., None]
        # Shadow is cast onto the lid only.
        shadow_layer *= 1 - (cov > 0.01)

    if opts.shadow:
        canvas *= (1 - shadow_layer)[..., None]

    if opts.dust:
        for _ in range(rng.integers(10, 40)):
            x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
            r = int(rng.integers(1, 4))
            val = rng.choice([20.0, 90.0, 255.0]) if opts.background != "black" else 230.0
            cv2.circle(canvas, (x, y), r, (val, val, val), -1, lineType=cv2.LINE_AA)
    if opts.hair:
        for _ in range(rng.integers(0, 3)):
            pts = []
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            a = rng.uniform(0, 2 * math.pi)
            for _ in range(30):
                a += rng.normal(0, 0.15)
                x += 8 * math.cos(a)
                y += 8 * math.sin(a)
                pts.append((int(x), int(y)))
            val = 40.0 if opts.background != "black" else 200.0
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (val, val, val), 1, cv2.LINE_AA)
    if opts.frame_strip:
        band = int(H * rng.uniform(0.006, 0.015))
        canvas[:band] *= rng.uniform(0.15, 0.4)

    canvas += rng.normal(0, opts.noise, canvas.shape)
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    if opts.jpeg_quality:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, opts.jpeg_quality])
        out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return SynthScan(image=out, truth=truth, options=opts)


# ------------------------------------------------------------------ booklets
@dataclass
class BookletOptions:
    """An open booklet whose spread is wider than the scanner bed.

    The full page lies on the glass; the facing page runs off the edge of the
    scan, so only ``visible`` (a fraction of the whole spread) is captured.
    """

    width: int = 1700  # ~A4 bed at 200 dpi
    height: int = 2340
    visible: tuple[float, float] = (0.65, 0.75)
    spine: str = "vertical"  # "vertical" (pages side by side), "horizontal", "random"
    full_side: str = "random"  # "first" (left/top), "second" (right/bottom), "random"
    background: str = "white"  # "white" (lid closed) or "dark" (lid propped open)
    gutter_depth: tuple[float, float] = (0.04, 0.45)  # darkening at the spine
    gutter_width: tuple[float, float] = (0.01, 0.05)  # fraction of page width
    fold_line: float = 0.5  # probability of a visible fold line
    staples: float = 0.4  # probability of saddle-stitch staples
    pictures: float = 0.5  # probability of a picture block per page
    max_angle: float = 2.5
    margin: tuple[int, int] = (15, 90)  # lid visible beyond the full page's outer edge
    fully_visible: bool = False  # whole spread on the bed (both pages complete)
    noise: float = 2.0
    dust: bool = True
    jpeg_quality: int | None = None
    #: Page content: "text", "artbook" (full-bleed illustrations or art with
    #: a caption) or "mixed" (anime magazine: text pages and art pages).
    content: str = "text"
    #: Artbooks: probability that one illustration spans both pages.
    spread_art: float = 0.3


def _page_content(w: int, h: int, rng: np.random.Generator, paper: float,
                  inner_on_right: bool, pictures: float) -> np.ndarray:
    """A booklet page: text with a wider inner (gutter) margin and maybe a picture."""
    img = np.full((h, w, 3), paper, np.float32)
    inner = int(w * rng.uniform(0.09, 0.15))
    outer = int(w * rng.uniform(0.07, 0.12))
    left, right = (outer, inner) if inner_on_right else (inner, outer)
    top = int(h * rng.uniform(0.07, 0.1))
    bottom = int(h * rng.uniform(0.08, 0.11))
    line_h = max(6, int(h * rng.uniform(0.014, 0.02)))
    pic = None
    if rng.random() < pictures:
        ph = int(h * rng.uniform(0.15, 0.35))
        py = int(rng.uniform(top, h - bottom - ph))
        pic = (py, py + ph)
        content = _photo_content(w - left - right, ph, rng, light=False)
        img[py : py + ph, left : w - right] = content
    y = top
    while y < h - bottom - line_h:
        if pic and pic[0] - line_h <= y < pic[1] + line_h // 2:
            y = pic[1] + line_h
            continue
        if rng.random() < 0.1:
            y += line_h
            continue
        x = left
        end = w - right - (rng.uniform(0, 0.4) * w if rng.random() < 0.15 else 0)
        while x < end:
            x1 = int(min(x + rng.uniform(0.02, 0.08) * w, end))
            img[y + line_h // 4 : y + line_h - 1, x:x1] = rng.uniform(10, 60)
            x = x1 + int(0.012 * w)
        y += int(line_h * 1.6)
    # Page number.
    nx = w // 2 + int(rng.uniform(-0.02, 0.02) * w)
    img[h - bottom // 2 : h - bottom // 2 + line_h // 2, nx : nx + max(4, w // 60)] = 40
    return img


def synth_booklet(opts: BookletOptions | None = None, seed: int | None = None) -> SynthScan:
    """Render an open booklet that does not fit on the bed.

    Ground truth is the one page that is completely on the glass (two pages
    when ``fully_visible`` is set), from its outer edge to the spine.
    """
    opts = opts or BookletOptions()
    rng = np.random.default_rng(seed)
    spine = opts.spine if opts.spine != "random" else rng.choice(["vertical", "horizontal"])
    second = opts.full_side == "second" or (opts.full_side == "random" and rng.random() < 0.5)
    # Work in a canonical frame: spine vertical, full page on the left.
    if spine == "vertical":
        across, along = opts.width, opts.height
    else:
        across, along = opts.height, opts.width

    for _attempt in range(50):
        m = int(rng.integers(*opts.margin))
        if opts.fully_visible:
            pw = int((across - 2 * m) / 2 * rng.uniform(0.85, 0.97))
        else:
            f = rng.uniform(*opts.visible)
            pw = int((across - m) / (2 * f))
        ph = int(pw * rng.uniform(1.25, 1.45))
        if ph > along - 30:
            ph = int(along - rng.uniform(30, 120))
        top = int(rng.uniform(10, max(11, along - ph - 10)))
        ang = rng.uniform(-opts.max_angle, opts.max_angle)
        # Rotate the spread about the centre of the full page.
        pc = np.array([m + pw / 2.0 - 0.5, top + ph / 2.0 - 0.5])
        page = CropBox(pc[0], pc[1], pw, ph, ang)
        x0, y0, x1, y1 = page.bounding_rect()
        if x0 >= 3 and y0 >= 3 and x1 <= across - 4 and y1 <= along - 4:
            if opts.fully_visible:
                other = page.translated(pw * math.cos(math.radians(ang)), pw * math.sin(math.radians(ang)))
                ox0, oy0, ox1, oy1 = other.bounding_rect()
                if not (ox1 <= across - 12 and oy0 >= 12 and oy1 <= along - 12 and y0 >= 12 and x0 >= 12):
                    continue
            break
    else:
        raise RuntimeError("could not place booklet")

    paper = rng.uniform(236, 250)

    def make_page(inner_on_right: bool) -> np.ndarray:
        kind = opts.content
        if kind == "mixed":
            kind = rng.choice(["artbook", "text"])
        if kind == "artbook":
            return demo_anime.artbook_page(pw, ph, rng, paper=paper)
        return _page_content(pw, ph, rng, paper, inner_on_right=inner_on_right, pictures=opts.pictures)

    if opts.content == "artbook" and rng.random() < opts.spread_art:
        spread = demo_anime.anime_illustration(2 * pw, ph, rng)
    else:
        spread = np.concatenate([make_page(True), make_page(False)], axis=1)
    sx = np.arange(2 * pw, dtype=np.float32) - (pw - 0.5)
    depth = rng.uniform(*opts.gutter_depth)
    gw = max(2.0, rng.uniform(*opts.gutter_width) * pw)
    asym = rng.uniform(0.7, 1.3)
    width_profile = np.where(sx < 0, gw, gw * asym)
    shade = 1 - depth * np.exp(-((sx / width_profile) ** 2))
    spread *= shade[None, :, None]
    if rng.random() < opts.fold_line:
        c = pw
        spread[:, c - 1 : c + 1] *= rng.uniform(0.75, 0.9)
    if rng.random() < opts.staples:
        for fy in (0.25, 0.75):
            y = int(ph * fy)
            L_ = int(ph * 0.04)
            spread[y - L_ // 2 : y + L_ // 2, pw - 2 : pw + 2] = rng.uniform(120, 170)
    # Paper edges of the booklet are slightly darker.
    spread[:, :2] *= 0.9
    spread[:2] *= 0.9
    spread[-2:] *= 0.9

    if spine == "vertical":
        W, H = across, along
    else:
        W, H = across, along  # canonical; rotated at the end
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    canvas = np.empty((H, W, 3), np.float32)
    if opts.background == "white":
        canvas[:] = rng.uniform(232, 250) + rng.uniform(-4, 4, 3)
        r2 = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
        canvas *= (1 - 0.06 * r2)[..., None]
    else:
        base = rng.uniform(10, 45)
        canvas[:] = base + 25 * (yy / H)[..., None] * rng.uniform(0, 1)

    t = math.radians(ang)
    ux = np.array([math.cos(t), math.sin(t)])
    uy = np.array([-math.sin(t), math.cos(t)])
    # Spread pixel (0, 0) centre sits at the full page's top-left pixel centre.
    o = pc - ux * (pw / 2 - 0.5) - uy * (ph / 2 - 0.5)
    M = np.array([[ux[0], uy[0], o[0]], [ux[1], uy[1], o[1]]])
    shift = M[:, 2] - M[:, :2] @ np.array([1.0, 1.0])
    cov = cv2.warpAffine(
        np.pad(np.ones((ph, 2 * pw), np.float32), 1),
        np.hstack([M[:, :2], shift[:, None]]),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped = cv2.warpAffine(spread, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if opts.background == "white":
        sh = cv2.warpAffine(cov, np.float32([[1, 0, 3], [0, 1, 3]]), (W, H))
        sh = cv2.GaussianBlur(sh, (0, 0), 2.0) * rng.uniform(0.15, 0.3)
        canvas *= (1 - sh * (cov < 0.01))[..., None]
    canvas = canvas * (1 - cov[..., None]) + warped * cov[..., None]
    if opts.dust:
        for _ in range(rng.integers(10, 30)):
            x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
            r = int(rng.integers(1, 3))
            val = rng.choice([20.0, 90.0])
            cv2.circle(canvas, (x, y), r, (val, val, val), -1, lineType=cv2.LINE_AA)
    canvas += rng.normal(0, opts.noise, canvas.shape)
    img = np.clip(canvas, 0, 255).astype(np.uint8)

    truth = [page]
    if opts.fully_visible:
        truth.append(page.translated(pw * ux[0], pw * ux[1]))

    # Mirror so that the full page is on the right / bottom.
    if second:
        img = np.ascontiguousarray(img[:, ::-1])
        truth = [CropBox(W - 1 - b.cx, b.cy, b.w, b.h, -b.angle) for b in truth]
    if spine == "horizontal":
        # Rotate the canonical image clockwise: (x, y) -> (H-1-y, x).
        img = np.ascontiguousarray(np.rot90(img, k=-1))
        truth = [CropBox(H - 1 - b.cy, b.cx, b.w, b.h, b.angle + 90.0) for b in truth]
    if opts.jpeg_quality:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, opts.jpeg_quality])
        img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return SynthScan(image=img, truth=truth, options=opts)
