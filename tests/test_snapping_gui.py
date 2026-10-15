"""Snapping a dragged side to candidate lines, and which sides an edit pins (offscreen Qt)."""
import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402

from hasami.core.geometry import CropBox  # noqa: E402
from hasami.core.sidelines import SnapLine  # noqa: E402


class Ev:
    def __init__(self, x, y, mods=Qt.NoModifier):
        self._p, self._m = QPointF(x, y), mods

    def pos(self):
        return self._p

    def modifiers(self):
        return self._m


def _view(lines=None):
    from PySide6.QtWidgets import QApplication
    from hasami.gui.preview import PreviewView

    app = QApplication.instance() or QApplication([])
    v = PreviewView()
    v.set_image(np.full((200, 300, 3), 200, np.uint8))
    v.resetTransform()  # 1 scene px = 1 screen px: the 8 px reach is 8 scene px
    v.set_snap_lines({"a1": lines if lines is not None else {"right": [SnapLine((230.0, 100.0), (0.0, 1.0), 0.9, "layer")]}})
    v.set_boxes([CropBox(150, 100, 100, 60, meta={"auto_id": "a1"})])
    return app, v, v.items_[0]


def _drag(it, kind, start, end, mods=Qt.NoModifier):
    it._drag = (kind, QPointF(*start), it.box)
    it.mouseMoveEvent(Ev(*end, mods))
    it._drag = None  # (mouseReleaseEvent needs a real Qt event)


def test_a_dragged_side_snaps_to_a_line_within_reach():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (225, 100))  # the right side, 5 px short of the line at x = 230
    assert it.box.bounding_rect()[2] == pytest.approx(230.0)
    assert it.box.meta["pinned_sides"] == ["right"] and it.box.meta["snapped_sides"] == ["right"]
    assert it.box.manual and it.box.meta["auto_id"] == "a1"


def test_shift_drags_freely():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (225, 100), Qt.ShiftModifier)
    assert it.box.bounding_rect()[2] == pytest.approx(225.0)
    assert it.box.meta["pinned_sides"] == ["right"] and it.box.meta["snapped_sides"] == []


def test_a_line_out_of_reach_does_not_snap():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (215, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(215.0)


def test_a_line_tilted_against_the_side_is_ignored():
    t = math.radians(3.0)
    app, v, it = _view({"right": [SnapLine((230.0, 100.0), (math.sin(t), math.cos(t)), 0.9, "layer")]})
    _drag(it, "s1", (200, 100), (228, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(228.0)


def test_a_free_drag_after_a_snap_clears_snapped_but_stays_pinned():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (225, 100))
    _drag(it, "s1", (230, 100), (212, 100), Qt.ShiftModifier)
    assert it.box.meta["pinned_sides"] == ["right"] and it.box.meta["snapped_sides"] == []


def test_corners_moves_and_skew_pin_their_sides():
    app, v, it = _view()
    _drag(it, "c1", (200, 70), (205, 65))  # top-right corner
    assert it.box.meta["pinned_sides"] == ["top", "right"]
    app, v, it = _view()
    _drag(it, "move", (150, 100), (152, 101))
    assert it.box.meta["pinned_sides"] == ["top", "right", "bottom", "left"]
    app, v, it = _view()
    v.set_box_angle(0, 1.5)
    assert v.items_[0].box.meta["pinned_sides"] == ["top", "right", "bottom", "left"]


def test_pins_accumulate():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (212, 100))
    _drag(it, "s0", (150, 70), (150, 60))
    assert it.box.meta["pinned_sides"] == ["top", "right"]


def test_guide_lines_show_only_while_a_side_is_dragged():
    app, v, it = _view()
    assert it.guide_lines() == []
    it._drag = ("s1", QPointF(200, 100), it.box)
    it.mouseMoveEvent(Ev(212, 100))
    [(p0, p1, support)] = it.guide_lines()
    assert p0.x() == pytest.approx(230.0) and p1.x() == pytest.approx(230.0) and support == 0.9
    assert it.boundingRect().contains(p0) and it.boundingRect().contains(p1)
    it.mouseMoveEvent(Ev(212, 100, Qt.ShiftModifier))
    assert it.guide_lines() == []  # Shift: free drag, no guides
    it._drag, it._free = ("s0", QPointF(150, 70), it.box), False
    assert it.guide_lines() == []  # the top side has no lines


def test_snapping_works_in_a_turned_view():
    app, v, it = _view()
    v.set_rotation(1)
    v.resetTransform()
    v.rotate(90)  # turned, 1 scene px = 1 screen px
    _drag(it, "s1", (200, 100), (225, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(230.0)
    v.set_rotation(0)


def test_drawn_boxes_do_not_snap():
    app, v, it = _view()
    v.set_boxes([CropBox(150, 100, 100, 60, manual=True, meta={"drawn": True})])
    it = v.items_[0]
    _drag(it, "s1", (200, 100), (225, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(225.0)


def test_the_window_passes_the_sessions_lines_to_the_preview(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from hasami.gui.app import MainWindow
    from hasami.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    st = AppSettings()
    st.output.folder = str(tmp_path)
    win = MainWindow(st)
    lines = {"a1": {"right": [SnapLine((230.0, 100.0), (0.0, 1.0), 0.9, "layer")]}}
    monkeypatch.setattr(win.session, "snap_lines_px", lambda: lines)
    monkeypatch.setattr(win.session, "preview_boxes_px", lambda: [CropBox(150, 100, 100, 60, meta={"auto_id": "a1"})])
    win._show_boxes()
    assert win.preview.items_[0].snap == lines["a1"]
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    win.close()


def test_a_flagged_side_shows_its_reason_and_loses_the_flag_when_moved():
    app, v, it = _view()
    it.box.meta["check_sides"] = {"right": "another edge 1.6 mm further in", "top": "the edge here is faint"}

    it.hoverMoveEvent(Ev(200, 100))
    assert it.toolTip() == "Check this side: another edge 1.6 mm further in"
    it.hoverMoveEvent(Ev(150, 100))  # inside: no side
    assert it.toolTip() == ""
    _drag(it, "s1", (200, 100), (212, 100))
    assert it.box.meta["check_sides"] == {"top": "the edge here is faint"}


def test_the_status_counts_the_sides_to_check(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from hasami.gui.app import MainWindow
    from hasami.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    st = AppSettings()
    st.output.folder = str(tmp_path)
    st.crop.mode = "booklet"
    win = MainWindow(st)
    box = CropBox(150, 100, 100, 60, meta={"auto_id": "a1", "check_sides": {"right": "x", "top": "y"}})
    monkeypatch.setattr(win.session, "preview_boxes_px", lambda: [box])
    win._show_boxes()
    assert "2 sides to check" in win.status_label.text()
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    win.close()


def test_a_corner_dragged_past_the_opposite_corner_pins_the_sides_it_placed():
    app, v, it = _view()
    _drag(it, "c1", (200, 70), (60, 60))  # top-right corner past the left side: the box flips
    x0, y0, x1, y1 = it.box.bounding_rect()
    assert (x0, y0) == pytest.approx((60, 60)) and (x1, y1) == pytest.approx((100, 130))
    assert it.box.meta["pinned_sides"] == ["top", "left"]  # the sides at (60, 60): where the corner went


def test_the_flag_mark_is_drawn_upright_at_the_label_size():
    from unittest.mock import MagicMock

    from hasami.gui.preview import CropItem

    app, v, it = _view()
    it.box.meta["check_sides"] = {"right": "the edge here is faint"}
    for turns in (0, 1):
        CropItem.view_turns = turns
        painter = MagicMock()
        it.paint(painter, None)
        calls = painter.method_calls
        i = next(k for k, c in enumerate(calls) if c[0] == "drawText" and c[1][-1] == "?")
        font = next(c[1][0] for c in reversed(calls[:i]) if c[0] == "setFont")
        assert font.pointSizeF() == pytest.approx(max(1.0, 11 * it._scale()))
        assert any(c[0] == "rotate" and c[1][0] == -90 * turns for c in calls[:i])
    CropItem.view_turns = 0
