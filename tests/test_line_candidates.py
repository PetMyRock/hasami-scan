import numpy as np
import pytest

from hasami.core import candidates
from hasami.core.autocrop import AutoCropSettings, detect
from hasami.core.geometry import CropBox
from hasami.core.toppage import SIDES, top_page
from hasami.demo import BookletOptions, synth_booklet


def _stack_scan(dpi=300):
    """A white page on a grey lid with a page-stack band beyond its right edge: 6 page edges,
    0.4 mm apart (as tests/test_booklet.py)."""
    mm = dpi / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 205, np.uint8)
    x0, y0, x1, y1 = (int(20 * mm), int(20 * mm), int(130 * mm), int(180 * mm))
    img[y0:y1, x0:x1] = 246
    for i in range(6):
        a = x1 + int(i * 0.4 * mm)
        img[y0 + 2:y1 - 2, a:a + int(0.4 * mm)] = 238 - 3 * i
        img[y0 + 2:y1 - 2, a:a + 1] = 200
    img[y0:y1, x1:x1 + 1] = 215
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, (x0, y0, x1, y1), mm


def test_top_page_records_every_side_its_lines_and_its_choice():
    img, (x0, y0, x1, y1), mm = _stack_scan()
    outer = x1 + 6 * 0.4 * mm
    box = CropBox.from_xywh(x0, y0, outer - x0, y1 - y0)
    with candidates.collecting() as recs:
        _out, shifts = top_page(img, box, 300)
    sides = {r["key"]: r for r in recs if r["kind"] == "toppage"}
    assert set(sides) == set(SIDES)
    right = sides[(0, 1)]
    assert right["rule"] == "chain" and right["skipped"] is None and not right["kept"]
    assert right["chosen"][0] / mm == pytest.approx(shifts[(0, 1)])
    truth = (x1 - outer) / mm  # the top page's edge, mm from the side (inwards: negative)
    layer = [ln[0] / mm for ln in right["pools"]["layer"]]
    assert any(abs(x - truth) < 0.3 for x in layer)  # the top page's edge is a candidate
    assert sum(x > truth + 0.2 for x in layer) >= 2  # and stack lines outside it
    assert set(right["pools"]) == {"layer", "faint", "strong"}
    assert right["image"].shape == img.shape and right["dpi"] == 300.0


def test_a_kept_spine_side_records_its_crease():
    mm = 300 / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 245, np.uint8)
    crease = int(100 * mm)
    img[:, crease - 2:crease + 3] = 170
    box = CropBox.from_xywh(int(20 * mm), int(20 * mm), crease + 2 * mm - int(20 * mm), int(160 * mm))
    with candidates.collecting() as recs:
        top_page(img, box, 300, keep=[(0, 1)], snap=[(0, 1)])
    spine = next(r for r in recs if r["key"] == (0, 1))
    assert spine["kept"] and spine["spine"] and spine["chosen"] is None
    assert abs(spine["crease"][0] / mm + 2.0) < 0.3  # the crease 2 mm inside the side


def test_a_side_off_the_image_is_recorded_as_skipped():
    img, (x0, y0, x1, y1), mm = _stack_scan()
    W = img.shape[1]
    box = CropBox.from_xywh(x0, y0, W + 5 * mm - x0, y1 - y0)
    with candidates.collecting() as recs:
        top_page(img, box, 300)
    right = next(r for r in recs if r["key"] == (0, 1))
    assert right["skipped"] == "off_image" and right["chosen"] is None


@pytest.mark.parametrize("profile", ["default", "epson-v600"])
def test_collecting_changes_no_crop(profile):
    sc = synth_booklet(BookletOptions(fully_visible=True), seed=1)
    st = AutoCropSettings(mode="booklet", profile=profile)
    plain = detect(sc.image, st, dpi=200)
    with candidates.collecting() as recs:
        seen = detect(sc.image, st, dpi=200)
    assert any(r["kind"] == "toppage" for r in recs)
    assert [b.corners().tolist() for b in plain] == [b.corners().tolist() for b in seen]
