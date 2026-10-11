"""Auto-crop accuracy on synthetic scans with known ground truth."""
import numpy as np
import pytest

from hasami.core.autocrop import AutoCropSettings, detect, refine
from hasami.core.geometry import match_boxes
from hasami.demo import SynthOptions, synth_scan


def _score(opts, seeds, mode="multi"):
    ious, wrong = [], 0
    for seed in seeds:
        sc = synth_scan(opts, seed=seed)
        boxes = detect(sc.image, AutoCropSettings(mode=mode, margin_mm=0), dpi=200)
        m = match_boxes(boxes, sc.truth)
        good = [iou for _, _, iou in m if iou > 0.5]
        wrong += len(boxes) != len(sc.truth) or len(good) != len(sc.truth)
        ious += good
    return float(np.mean(ious)), wrong


@pytest.mark.parametrize("name,opts", [
    ("basic", SynthOptions(bordered=0.0, light_photos=0.0)),
    ("bordered", SynthOptions(bordered=1.0, light_photos=0.0)),
    ("black", SynthOptions(background="black")),
    ("anime", SynthOptions(theme="anime")),
])
def test_multi_accuracy(name, opts):
    mean_iou, wrong = _score(opts, range(3))
    assert wrong == 0
    assert mean_iou > 0.985


def test_single_mode_one_box():
    sc = synth_scan(SynthOptions(n_photos=1), seed=4)
    boxes = detect(sc.image, AutoCropSettings(mode="single", margin_mm=0), dpi=200)
    assert len(boxes) == 1


def test_off_mode_returns_the_whole_scan():
    sc = synth_scan(SynthOptions(), seed=1)
    boxes = detect(sc.image, AutoCropSettings(mode="off"), dpi=200)
    h, w = sc.image.shape[:2]
    assert len(boxes) == 1 and round(boxes[0].w) == w and round(boxes[0].h) == h


def test_lid_falloff_does_not_merge_with_item():
    # A cover on a white lid that darkens steeply towards the left edge of the
    # glass (as on the maintainer's Epson scanners): only the cover is found.
    h, w = 1400, 1000
    xx = np.arange(w, dtype=np.float32)[None, :]
    lid = 0.97 - 0.12 * np.exp(-xx / 120.0)
    img = np.repeat(np.repeat(lid, h, axis=0)[:, :, None], 3, axis=2)
    img[300:1100, 250:800] = (0.85, 0.15, 0.2)
    img = (img * 255).astype(np.uint8)
    boxes = detect(img, AutoCropSettings(mode="single", margin_mm=0), dpi=150)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0].bounding_rect()
    assert abs(x0 - 250) < 6 and abs(x1 - 800) < 6 and abs(y0 - 300) < 6 and abs(y1 - 1100) < 6


def test_refine_on_higher_resolution():
    sc = synth_scan(SynthOptions(n_photos=2), seed=2)
    small = sc.image[::2, ::2]
    boxes = detect(small, AutoCropSettings(mode="multi", margin_mm=0), dpi=100)
    up = [b.scaled(2.0) for b in boxes]
    fine = refine(sc.image, up, AutoCropSettings(mode="multi", margin_mm=0), dpi=200)
    m = match_boxes(fine, sc.truth)
    assert len(m) == len(sc.truth) and min(iou for _, _, iou in m) > 0.98
