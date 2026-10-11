import numpy as np

from hasami.core.processing import (
    ProcessSettings,
    auto_levels,
    descreen,
    detect_halftone_period,
    invert_negative,
    process,
    to_output,
)


def test_halftone_period_detected_and_removed():
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w]
    period = 6.0
    dots = 0.5 + 0.5 * np.cos(2 * np.pi * xx / period) * np.cos(2 * np.pi * yy / period)
    img = np.repeat(dots[:, :, None], 3, axis=2).astype(np.float32) * 0.8 + 0.1  # float RGB in [0, 1]
    p = detect_halftone_period(img)
    # cos(x)cos(y) dots: the strongest frequency runs diagonally.
    assert p is not None and abs(p - period / np.sqrt(2)) < 0.5
    out = descreen(img, p)
    assert out.astype(float).std() < img.astype(float).std() * 0.3


def test_plain_image_has_no_halftone():
    rng = np.random.default_rng(0)
    img = rng.normal(0.5, 0.01, (512, 512, 3)).clip(0, 1).astype(np.float32)
    assert detect_halftone_period(img) is None


def test_negative_inversion_makes_dark_film_light():
    film = np.full((100, 100, 3), (200, 120, 70), np.uint8)  # orange base
    film[30:70, 30:70] = (60, 30, 20)  # dense area -> bright in the positive
    pos = invert_negative(film, color=True)
    pos8 = to_output(pos, 8) if pos.dtype != np.uint8 else pos
    assert pos8[50, 50].mean() > pos8[5, 5].mean()


def test_auto_levels_stretches():
    img = np.linspace(80, 160, 256 * 256, dtype=np.float32).reshape(256, 256)
    img = np.repeat(img[:, :, None], 3, axis=2) / 255.0
    out = auto_levels(img, 0.1, 0.1, per_channel=False)
    assert out.min() < 0.02 and out.max() > 0.98


def test_process_and_bit_depths():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    out8 = process(img, ProcessSettings())
    assert out8.dtype == np.uint8 and out8.shape == img.shape
    st = ProcessSettings()
    st.bits = 16
    out16 = process(img, st)
    assert out16.dtype == np.uint16
