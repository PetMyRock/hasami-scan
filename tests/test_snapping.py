"""Snap lines in the Session (demo scanner, generated bed)."""
from dataclasses import replace

import numpy as np
import pytest

from hasami.core.geometry import SIDE_NAMES, side_frames


def _session(tmp_path, mode="booklet"):
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.crop.mode = mode
    st.output.folder = str(tmp_path / "scans")
    return Session(st)


def test_snap_lines_are_kept_per_auto_box_in_booklet_mode(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    assert s.boxes_mm and set(s.snap_lines) == {b.meta["auto_id"] for b in s.boxes_mm}
    for sides in s.snap_lines.values():
        assert set(sides) == set(SIDE_NAMES)
    assert sum(len(ls) for sides in s.snap_lines.values() for ls in sides.values()) > 0


def test_the_line_under_a_side_is_where_the_side_is(tmp_path):
    from hasami.core.sidelines import normal_hit

    s = _session(tmp_path)
    s.run_preview()
    b = s.boxes_mm[0]
    near = 0
    for name, (m, n) in side_frames(b).items():
        hits = [normal_hit(np.array(ln.p), np.array(ln.d), m, n) for ln in s.snap_lines[b.meta["auto_id"]][name]]
        near += any(h is not None and abs(h) <= 0.3 for h in hits)  # mm: the margin is already applied
    assert near >= 2


def test_no_snap_lines_outside_booklet_mode(tmp_path):
    s = _session(tmp_path, "multi")
    s.run_preview()
    assert s.snap_lines == {}


def test_a_new_preview_drops_the_old_lines(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    old = set(s.snap_lines)
    s.run_preview()
    assert s.snap_lines and not old & set(s.snap_lines)


def test_redetect_keeps_the_lines_of_edited_boxes(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    b = s.boxes_mm[0]
    aid = b.meta["auto_id"]
    s.boxes_mm[0] = replace(b, manual=True, meta={**b.meta, "pinned_sides": ["top"]})
    s.detect_on_preview()
    assert s.snap_lines[aid] and set(s.snap_lines) == {x.meta["auto_id"] for x in s.boxes_mm}


def test_snap_lines_px_are_in_the_preview_frame(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    p = s.preview
    f = p.dpi / 25.4
    px = s.snap_lines_px()
    for aid, sides in s.snap_lines.items():
        for name, ls in sides.items():
            for a, b in zip(ls, px[aid][name]):
                assert b.p == pytest.approx(((a.p[0] - p.area_mm[0]) * f - 0.5, (a.p[1] - p.area_mm[1]) * f - 0.5))
                assert b.d == a.d and b.support == a.support and b.kind == a.kind
