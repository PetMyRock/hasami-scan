"""Colour and filter pipeline applied to each cropped item.

All processing happens in float32 RGB in [0, 1] so 16-bit scans keep their
precision until the final conversion.  Order of operations:

1. media conversion (colour / B&W negative inversion)
2. descreen (remove the halftone pattern of printed material)
3. dust removal
4. white balance
5. levels (auto levels or restore fading)
6. brightness, contrast, saturation
7. sharpening
8. output mode (colour, grey or black-and-white line art)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .autocrop import to_float_rgb


@dataclass
class ProcessSettings:
    #: "photo", "document", "color_negative", "bw_negative" or "slide".
    media: str = "photo"
    #: Stretch so that the darkest/lightest ``*_clip`` percent become black/white.
    auto_levels: bool = True
    black_clip: float = 0.05
    white_clip: float = 0.05
    #: Treat channels independently in auto levels (removes casts, may shift
    #: deliberate colour moods).
    levels_per_channel: bool = False
    #: Per-channel stretch plus a saturation lift for faded colour prints.
    restore_fading: bool = False
    #: "none", "auto" (neutral highlights) or "gray_world".
    white_balance: str = "none"
    #: Gamma-style brightness: >1 brighter, <1 darker.
    brightness: float = 1.0
    #: -1..1 S-curve contrast.
    contrast: float = 0.0
    #: Saturation multiplier.
    saturation: float = 1.0
    #: Unsharp mask amount (0 = off) and radius in pixels at 300 dpi.
    sharpen: float = 0.0
    sharpen_radius: float = 1.0
    #: "off", "auto" (only if a halftone screen is detected) or "on".
    descreen: str = "off"
    #: Remove small specks of dust.  Off by default: tiny highlights in
    #: illustrations (sparkles, eye highlights) can look like dust.
    dust_removal: bool = False
    #: "color", "gray" or "bw" (line art).
    output_mode: str = "color"
    #: Output bit depth per channel: 8 or 16.
    bits: int = 8


# ------------------------------------------------------------------ helpers
def _luma(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def _percentile_sample(x: np.ndarray, q, max_n: int = 400_000):
    flat = x.reshape(-1) if x.ndim < 3 else x.reshape(-1, x.shape[-1])
    if flat.shape[0] > max_n:
        idx = np.linspace(0, flat.shape[0] - 1, max_n).astype(np.int64)
        flat = flat[idx]
    return np.percentile(flat, q, axis=0)


def to_output(img: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantise float [0, 1] to uint8 or uint16."""
    img = np.clip(img, 0.0, 1.0)
    if bits == 16:
        return (img * 65535.0 + 0.5).astype(np.uint16)
    return (img * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- negatives
def invert_negative(img: np.ndarray, color: bool = True) -> np.ndarray:
    """Turn a scanned negative into a positive.

    The orange mask of colour film is removed by dividing by the film base
    colour (the brightest, unexposed part of the film).  Each channel is then
    converted to density and stretched independently, which also balances
    the colours.
    """
    if not color:
        img = np.repeat(_luma(img)[:, :, None], 3, axis=2)
    eps = 1e-4
    base = np.maximum(_percentile_sample(img, 99.8), eps)
    t = np.clip(img / base, eps, 1.0)
    density = -np.log10(t)
    lo = _percentile_sample(density, 0.3)
    hi = _percentile_sample(density, 99.7)
    v = np.clip((density - lo) / np.maximum(hi - lo, eps), 0.0, 1.0)
    # Density is logarithmic; a gentle power curve gives a natural midtone.
    return np.power(v, 1.25).astype(np.float32)


# ----------------------------------------------------------------- descreen
def detect_halftone_period(img: np.ndarray, tile: int = 256) -> float | None:
    """Estimate the period (pixels) of a halftone screen, if there is one.

    Looks for a strong isolated peak in the spectrum of a few textured tiles.
    Returns ``None`` for continuous-tone images (photos, digital prints).
    """
    gray = _luma(img) if img.ndim == 3 else img
    h, w = gray.shape
    if min(h, w) < tile:
        tile = 1 << int(math.log2(max(32, min(h, w))))
        if tile < 64:
            return None
    # Pick the most textured tiles (halftone shows best in midtones).
    cands = []
    step = tile
    for y in range(0, h - tile + 1, step):
        for x in range(0, w - tile + 1, step):
            t = gray[y : y + tile, x : x + tile]
            m = float(t.mean())
            if 0.15 < m < 0.85:
                cands.append((float(t.std()), y, x))
    if not cands:
        return None
    cands.sort(reverse=True)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    spec = np.zeros((tile, tile), np.float32)
    for _, y, x in cands[:6]:
        t = gray[y : y + tile, x : x + tile].astype(np.float32)
        t = (t - t.mean()) * win
        spec += np.abs(np.fft.fftshift(np.fft.fft2(t))).astype(np.float32)
    c = tile // 2
    yy, xx = np.mgrid[0:tile, 0:tile]
    r = np.hypot(yy - c, xx - c)
    # Ignore low frequencies (image content) and the very highest (noise).
    band = (r > tile / 16) & (r < tile * 0.48)
    vals = np.where(band, spec, 0)
    peak = float(vals.max())
    if peak <= 0:
        return None
    ring_med = float(np.median(spec[band]))
    if peak < 25.0 * ring_med:
        return None
    py, px = np.unravel_index(int(np.argmax(vals)), vals.shape)
    freq = math.hypot(py - c, px - c) / tile  # cycles per pixel
    return 1.0 / freq if freq > 0 else None


def descreen(img: np.ndarray, period: float) -> np.ndarray:
    """Suppress a halftone screen of the given period, then restore edges."""
    sigma = max(0.5, 0.45 * period)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    # Recover edge contrast lost to the blur (on a larger scale than the dots).
    soft = cv2.GaussianBlur(blurred, (0, 0), sigma * 2.0)
    return np.clip(blurred + 0.5 * (blurred - soft), 0, 1).astype(np.float32)


# --------------------------------------------------------------------- dust
def remove_dust(img: np.ndarray, dpi: float = 300.0) -> np.ndarray:
    """Inpaint small isolated specks that differ strongly from their surroundings."""
    gray = _luma(img).astype(np.float32)
    k = max(3, int(round(dpi / 100.0)) | 1)
    med = cv2.medianBlur((gray * 255).astype(np.uint8), k if k <= 255 else 255).astype(np.float32) / 255.0
    diff = np.abs(gray - med)
    local = cv2.GaussianBlur(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3)), (0, 0), 6)
    mask = (diff > np.maximum(0.12, 4.0 * local)).astype(np.uint8)
    # Keep only tiny blobs (dust is at most ~0.25 mm across).
    max_area = max(4, int((0.25 * dpi / 25.4) ** 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    mask = keep[lab].astype(np.uint8)
    if not mask.any():
        return img
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out8 = cv2.inpaint((np.clip(img, 0, 1) * 255).astype(np.uint8), mask, 3, cv2.INPAINT_TELEA)
    repaired = out8.astype(np.float32) / 255.0
    m = mask.astype(bool)
    result = img.copy()
    result[m] = repaired[m]
    return result


# ------------------------------------------------------------------- colour
def white_balance(img: np.ndarray, method: str) -> np.ndarray:
    if method == "gray_world":
        means = img.reshape(-1, 3).mean(axis=0)
        gain = means.mean() / np.maximum(means, 1e-4)
    elif method == "auto":
        # Average of the brightest near-neutral pixels should be neutral.
        lum = _luma(img)
        thr = _percentile_sample(lum, 98.0)
        sel = img[lum >= thr]
        if sel.size == 0:
            return img
        ref = sel.mean(axis=0)
        gain = ref.max() / np.maximum(ref, 1e-4)
    else:
        return img
    gain = np.clip(gain, 0.5, 2.0)
    return np.clip(img * gain[None, None, :], 0, 1).astype(np.float32)


def auto_levels(img: np.ndarray, black_clip: float, white_clip: float, per_channel: bool) -> np.ndarray:
    if per_channel:
        lo = _percentile_sample(img, black_clip)
        hi = _percentile_sample(img, 100.0 - white_clip)
    else:
        lum = _luma(img)
        lo = np.full(3, _percentile_sample(lum, black_clip))
        hi = np.full(3, _percentile_sample(lum, 100.0 - white_clip))
    scale = 1.0 / np.maximum(hi - lo, 1e-3)
    return np.clip((img - lo) * scale, 0, 1).astype(np.float32)


def adjust(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    out = img
    if abs(brightness - 1.0) > 1e-3:
        out = np.power(np.clip(out, 0, 1), 1.0 / max(brightness, 0.05))
    if contrast > 1e-3:
        # Sigmoid S-curve around mid grey, rescaled so 0 and 1 stay put.
        k = 1.0 + 8.0 * contrast
        lo, hi = 1.0 / (1.0 + math.exp(k / 2)), 1.0 / (1.0 + math.exp(-k / 2))
        out = (1.0 / (1.0 + np.exp(-k * (out - 0.5))) - lo) / (hi - lo)
    elif contrast < -1e-3:
        out = 0.5 + (out - 0.5) * (1.0 + contrast)
    if abs(saturation - 1.0) > 1e-3:
        lum = _luma(out)[:, :, None]
        out = lum + (out - lum) * saturation
    return np.clip(out, 0, 1).astype(np.float32)


def sharpen(img: np.ndarray, amount: float, radius_px: float) -> np.ndarray:
    """Unsharp mask on lightness only, which avoids colour fringes."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycc[:, :, 0]
    blur = cv2.GaussianBlur(y, (0, 0), max(0.3, radius_px))
    ycc[:, :, 0] = np.clip(y + amount * (y - blur), 0, 1)
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1).astype(np.float32)


def to_lineart(img: np.ndarray) -> np.ndarray:
    """Adaptive black/white conversion that copes with uneven paper."""
    gray = (_luma(img) * 255).astype(np.uint8)
    block = max(15, (min(gray.shape) // 20) | 1)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)
    return bw.astype(np.float32) / 255.0


# --------------------------------------------------------------------- main
def process(image: np.ndarray, settings: ProcessSettings | None = None, dpi: float = 300.0) -> np.ndarray:
    """Run the full pipeline; returns uint8/uint16 RGB, grey or line art."""
    st = settings or ProcessSettings()
    img = to_float_rgb(image)
    if st.media == "color_negative":
        img = invert_negative(img, color=True)
    elif st.media == "bw_negative":
        img = invert_negative(img, color=False)

    if st.descreen != "off":
        period = detect_halftone_period(img)
        if period is None and st.descreen == "on":
            period = max(2.0, dpi / 150.0)  # assume a ~150 lpi screen
        if period is not None:
            img = descreen(img, period)
    if st.dust_removal:
        img = remove_dust(img, dpi)
    img = white_balance(img, st.white_balance)
    if st.restore_fading:
        img = auto_levels(img, max(st.black_clip, 0.3), max(st.white_clip, 0.3), per_channel=True)
        img = adjust(img, 1.0, 0.1, 1.25)
    elif st.auto_levels and st.media not in ("color_negative", "bw_negative"):
        img = auto_levels(img, st.black_clip, st.white_clip, st.levels_per_channel)
    img = adjust(img, st.brightness, st.contrast, st.saturation)
    if st.sharpen > 0:
        img = sharpen(img, st.sharpen, st.sharpen_radius * dpi / 300.0)

    if st.output_mode == "bw":
        return to_output(to_lineart(img), 8)
    if st.output_mode == "gray" or st.media == "bw_negative":
        return to_output(_luma(img), st.bits)
    return to_output(img, st.bits)
