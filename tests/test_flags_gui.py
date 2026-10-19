"""Check-this-side flags on the preview's crop boxes (offscreen Qt)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402

from hasami.core.geometry import CropBox  # noqa: E402


class Ev:
    def __init__(self, x, y, mods=Qt.NoModifier):
        self._p, self._m = QPointF(x, y), mods

    def pos(self):
        return self._p

    def modifiers(self):
        return self._m


def _view():
    from PySide6.QtWidgets import QApplication
    from hasami.gui.preview import PreviewView

    app = QApplication.instance() or QApplication([])
    v = PreviewView()
    v.set_image(np.full((200, 300, 3), 200, np.uint8))
    v.resetTransform()
    v.set_boxes([CropBox(150, 100, 100, 60, meta={"auto_id": "a1"})])
    return app, v, v.items_[0]


def _drag(it, kind, start, end, mods=Qt.NoModifier):
    it._drag = (kind, QPointF(*start), it.box)
    it.mouseMoveEvent(Ev(*end, mods))
    it._drag = None  # (mouseReleaseEvent needs a real Qt event)


def test_corners_moves_and_skew_clear_the_flags_of_the_sides_they_move():
    app, v, it = _view()
    it.box.meta["check_sides"] = {"top": "a", "right": "b", "bottom": "c", "left": "d"}
    _drag(it, "c1", (200, 70), (205, 65))  # top-right corner
    assert it.box.meta["check_sides"] == {"bottom": "c", "left": "d"}
    app, v, it = _view()
    it.box.meta["check_sides"] = {"top": "a", "right": "b", "bottom": "c", "left": "d"}
    _drag(it, "move", (150, 100), (152, 101))
    assert "check_sides" not in it.box.meta
    app, v, it = _view()
    it.box.meta["check_sides"] = {"top": "a", "right": "b", "bottom": "c", "left": "d"}
    v.set_box_angle(0, 1.5)
    assert "check_sides" not in v.items_[0].box.meta


def test_flags_go_side_by_side():
    app, v, it = _view()
    it.box.meta["check_sides"] = {"top": "a", "right": "b", "bottom": "c", "left": "d"}
    _drag(it, "s1", (200, 100), (212, 100))
    _drag(it, "s0", (150, 70), (150, 60))
    assert it.box.meta["check_sides"] == {"bottom": "c", "left": "d"}


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


def test_a_corner_dragged_past_the_opposite_corner_clears_the_flags_of_the_sides_it_placed():
    app, v, it = _view()
    it.box.meta["check_sides"] = {"top": "a", "right": "b", "bottom": "c", "left": "d"}
    _drag(it, "c1", (200, 70), (60, 60))  # top-right corner past the left side: the box flips
    x0, y0, x1, y1 = it.box.bounding_rect()
    assert (x0, y0) == pytest.approx((60, 60)) and (x1, y1) == pytest.approx((100, 130))
    assert it.box.meta["check_sides"] == {"right": "b", "bottom": "c"}  # top and left: where the corner went


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
