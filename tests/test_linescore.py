import numpy as np

from hasami.core import linescore as ls


def _step_scan(dpi, edge_mm=50.0):
    """Paper (L* ~95) left of a vertical edge at edge_mm, grey lid (L* ~51) right of it, and the
    right side of the paper as a candidate's frame."""
    mm = dpi / 25.4
    H, W = int(80 * mm), int(100 * mm)
    e = int(round(edge_mm * mm))
    img = np.full((H, W, 3), 120, np.uint8)
    img[:, :e] = 240
    side = ls.Side(mid=np.array([e - 0.5, H / 2]), d=np.array([0.0, 1.0]), n=np.array([1.0, 0.0]),
                   length=0.8 * H, dpi=float(dpi))
    return img, side


def test_patch_shapes_and_the_edge_at_the_centre():
    img, side = _step_scan(300)
    fine, coarse = ls.patches(img, side, 0.0, 0.0)
    assert fine.shape == (16, 64, 4) and coarse.shape == (16, 32, 4)
    L = fine[..., 0].mean(axis=0)
    assert L[ls.FINE_COLS < -0.3].min() > 90  # paper inside
    assert L[ls.FINE_COLS > 0.3].max() < 55  # lid outside
    assert np.all(fine[..., 3] == 1.0)


def test_patch_is_the_same_in_mm_at_200_and_400_dpi():
    a, sa = _step_scan(200)
    b, sb = _step_scan(400)
    fa, ca = ls.patches(a, sa, 0.0, 0.0)
    fb, cb = ls.patches(b, sb, 0.0, 0.0)
    far = np.abs(ls.FINE_COLS) > 0.2  # away from the edge itself (its blur differs)
    assert np.abs(fa[:, far, 0] - fb[:, far, 0]).max() < 1.0
    assert np.abs(ca[..., 0] - cb[..., 0])[:, np.abs(ls.COARSE_COLS) > 0.6].max() < 1.0


def test_columns_outside_the_image_are_flagged():
    img, side = _step_scan(300)
    W = img.shape[1]
    x0 = W - side.mid[0] - 2 * side.px_mm  # the line 2 mm before the image's right edge
    _fine, coarse = ls.patches(img, side, x0, 0.0)
    out = ls.COARSE_COLS > 2.5
    assert np.all(coarse[:, out, 3] == 0.0) and np.all(coarse[:, out, 0] == -1.0)
    assert np.all(coarse[:, ls.COARSE_COLS < 1.5, 3] == 1.0)


def test_lines_of_several_pools_merge_into_one_candidate():
    px = 300 / 25.4
    pools = {"layer": [(-10.0, 0.0, 12)], "faint": [(-10.0 + 0.1 * px, 0.001, 9), (5.0, 0.0, 16)], "strong": []}
    c = ls.merge_pools(pools, px)
    assert [round(x["x0"], 3) for x in c] == [5.0, -10.0]  # outermost first; the layer pool's position
    assert c[1]["pools"] == {"layer", "faint"} and c[1]["support"] == 12


def test_features_of_a_paper_edge_on_the_lid():
    img, side = _step_scan(300)
    cands = ls.merge_pools({"layer": [(0.0, 0.0, 16)]}, side.px_mm)
    fine, _coarse = ls.patches(img, side, 0.0, 0.0)
    v = ls.features(cands, 0, fine, side.dpi)
    assert v.shape == (len(ls.FEATURES),) and v.dtype == np.float32
    f = dict(zip(ls.FEATURES, v))
    assert f["dL"] > 40 and f["seg_darker_share"] == 1.0  # paper inside, much lighter than the lid
    assert f["strength"] > 6 and f["seg_strength_share"] == 1.0
    assert f["in_layer"] == 1.0 and f["in_faint"] == 0.0 and f["support"] == 1.0
    assert f["rank_out"] == 0 and f["n_lines"] == 1 and f["gap_out_mm"] == 20.0 and f["gap_in_mm"] == 20.0
    assert f["valid_out"] == 1.0 and f["dpi"] == 0.75


def test_features_count_lines_outside_and_gaps_in_mm():
    img, side = _step_scan(300)
    px = side.px_mm
    cands = ls.merge_pools({"layer": [(0.0, 0.0, 16), (1.0 * px, 0.0, 10), (-3.0 * px, 0.0, 10)]}, px)
    fine, _ = ls.patches(img, side, 0.0, 0.0)
    f = dict(zip(ls.FEATURES, ls.features(cands, 1, fine, side.dpi)))  # the middle line (x0 = 0)
    assert f["rank_out"] == 1 and f["n_lines"] == 3
    assert abs(f["gap_out_mm"] - 1.0) < 1e-4 and abs(f["gap_in_mm"] - 3.0) < 1e-4
