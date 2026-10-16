"""An edited box is scanned exactly as the user left it (owner, Oct 15 2026: refining its untouched sides moved
three real-scan sides inward by 0.55-0.8 mm and improved none; generated scans here)."""
from dataclasses import replace

import numpy as np

from hasami.core.autocrop import AutoCropSettings, refine
from hasami.core.geometry import CropBox


def _card(dpi=150):
    mm = dpi / 25.4
    img = np.full((int(150 * mm), int(200 * mm), 3), 235, np.uint8)
    x0, y0, x1, y1 = int(50 * mm), int(40 * mm), int(140 * mm), int(100 * mm)
    img[y0:y1, x0:x1] = (90, 60, 140)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(x0 - 0.5, y0 - 0.5, x1 - x0, y1 - y0)


def test_an_edited_auto_box_is_not_refined():
    img, card = _card()
    st = AutoCropSettings(mode="multi", margin_mm=0)
    off = card.translated(2.0, 1.5)  # auto-crop's box a little off: refine would move it
    assert refine(img, [off], st, dpi=150)[0].cx != off.cx
    edited = replace(off, w=off.w + 15, cx=off.cx - 7.5, manual=True,
                     meta={"auto_id": "a1", "snapped_sides": ["left"]})  # its left side dragged out
    drawn = replace(off, manual=True, meta={"drawn": True})
    out = refine(img, [edited, drawn], st, dpi=150)
    assert [(b.cx, b.cy, b.w, b.h, b.angle) for b in out] == [(b.cx, b.cy, b.w, b.h, b.angle) for b in (edited, drawn)]
