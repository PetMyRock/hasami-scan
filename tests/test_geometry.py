import numpy as np
import pytest

from hasami.core.geometry import CropBox, angle_difference, extract, match_boxes, normalize_angle, polygon_iou, reading_order


def test_normalize_angle_folds_into_range():
    for a in (-170, -91, -45, 0, 30, 45, 60, 135, 400):
        w, h, ang = normalize_angle(100, 50, a)
        assert -45 < ang <= 45
        assert {round(w), round(h)} == {100, 50}


def test_corners_and_area():
    b = CropBox(50, 40, 20, 10, 0)
    c = b.corners()
    assert np.allclose(c.min(axis=0), [40, 35]) and np.allclose(c.max(axis=0), [60, 45])
    assert b.area == pytest.approx(200)


def test_iou_identity_and_shift():
    a = CropBox(100, 100, 50, 80, 7)
    assert polygon_iou(a, a) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < polygon_iou(a, a.translated(10, 0)) < 1.0
    assert polygon_iou(a, a.translated(500, 0)) == 0.0


def test_angle_difference_is_symmetric_mod_90():
    assert angle_difference(44, -44) == pytest.approx(2)
    assert angle_difference(10, 10) == 0


def test_extract_axis_aligned_matches_slice():
    img = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    b = CropBox.from_xywh(10, 20, 30, 40)
    out = extract(img, b)
    assert out.shape[:2] == (40, 30)
    assert np.array_equal(out, img[20:60, 10:40])


def test_clamp_and_match_and_order():
    b = CropBox(5, 5, 40, 40, 0).clamp_to(100, 100)
    x0, y0, _, _ = b.bounding_rect()
    assert x0 >= -0.5 and y0 >= -0.5
    found = [CropBox(10, 10, 10, 10), CropBox(100, 10, 10, 10)]
    truth = [CropBox(101, 10, 10, 10), CropBox(11, 10, 10, 10)]
    m = {(i, j) for i, j, _ in match_boxes(found, truth)}
    assert m == {(0, 1), (1, 0)}
    order = reading_order([CropBox(100, 10, 10, 10), CropBox(10, 200, 10, 10), CropBox(10, 10, 10, 10)])
    assert [(round(o.cx), round(o.cy)) for o in order] == [(10, 10), (100, 10), (10, 200)]
