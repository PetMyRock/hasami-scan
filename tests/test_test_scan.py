import os
import time

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _settings(tmp_path):
    from hasami.settings import AppSettings

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.input.scan_dpi = 150
    st.crop.mode = "booklet"
    st.output.folder = str(tmp_path / "scans")
    st.output.test_folder = str(tmp_path / "test data")
    st.output.test_prefix = "KLK test "
    st.output.test_dpi = 100
    return st


def test_test_scan_saves_the_whole_bed_uncropped_in_its_own_folder(tmp_path):
    from hasami.workflow import Session

    s = Session(_settings(tmp_path))
    s.run_preview()
    boxes = list(s.boxes_mm)
    assert boxes  # crops found: the test scan must ignore them
    result, path = s.run_test_scan()
    assert path == tmp_path / "test data" / "KLK test 0001.tif"
    bw, bh = s.open().capabilities().max_area_mm
    assert result.dpi == 100 and result.area_mm[2] == pytest.approx(bw, abs=0.5)
    assert result.area_mm[3] == pytest.approx(bh, abs=0.5)
    with Image.open(path) as im:
        assert im.size == (result.image.shape[1], result.image.shape[0])
        assert round(im.info["dpi"][0]) == 100
        saved = np.asarray(im)
    assert np.array_equal(saved, result.image)  # as the scanner delivered it: no crop, no colours
    assert s.boxes_mm == boxes  # preview crops kept for the next normal scan
    assert not (tmp_path / "scans").exists()  # the normal output is not touched
    _, second = s.run_test_scan()
    assert second.name == "KLK test 0002.tif"


def test_test_scan_button(tmp_path):
    pytest.importorskip("PySide6")
    try:
        from PySide6.QtWidgets import QApplication
        from hasami.gui.app import MainWindow
    except ImportError as e:
        pytest.skip(str(e))
    from PySide6.QtCore import QThreadPool

    app = QApplication.instance() or QApplication([])
    win = MainWindow(_settings(tmp_path))
    try:
        def wait():
            end = time.monotonic() + 30
            while win.task is not None and time.monotonic() < end:
                app.processEvents()
                time.sleep(0.01)
            assert win.task is None

        win.do_preview()
        wait()
        n = len(win.preview.boxes())
        assert n
        assert win.a_test.shortcut().toString() == "Ctrl+T"
        win.a_test.trigger()
        assert not win.a_scan.isEnabled()  # busy like a normal scan
        wait()
        saved = tmp_path / "test data" / "KLK test 0001.tif"
        assert saved.exists()
        assert win.results.count() == 1 and win.results.item(0).text() == saved.name
        assert len(win.preview.boxes()) == n  # the crops are still there
        assert "Saved test scan" in win.status_label.text()
    finally:
        QThreadPool.globalInstance().waitForDone(10000)
        app.processEvents()
        win.close()


def test_next_preview_replaces_edited_auto_boxes_but_keeps_drawn_ones(tmp_path):
    # Owner: a box fixed by hand (Skew / dragged) before a test scan came back on the next page.
    from dataclasses import replace

    from hasami.core.geometry import CropBox
    from hasami.workflow import Session

    s = Session(_settings(tmp_path))
    s.run_preview()
    edited = replace(s.boxes_mm[0], angle=-0.7, manual=True, meta=dict(s.boxes_mm[0].meta))
    drawn = CropBox.from_xywh(5, 5, 20, 20, manual=True)
    drawn.meta["drawn"] = True
    s.boxes_mm = [edited, drawn]
    s.run_test_scan()
    s.run_preview()
    assert drawn in s.boxes_mm  # drawn from scratch: stays (Off mode, the same spot every time)
    assert edited not in s.boxes_mm
    assert all(not b.manual for b in s.boxes_mm if b is not drawn)  # the rest found anew


def test_edited_box_is_gone_after_test_scan_and_preview_in_the_app(tmp_path):
    pytest.importorskip("PySide6")
    try:
        from PySide6.QtWidgets import QApplication
        from hasami.gui.app import MainWindow
    except ImportError as e:
        pytest.skip(str(e))
    from PySide6.QtCore import QThreadPool

    app = QApplication.instance() or QApplication([])
    win = MainWindow(_settings(tmp_path))
    try:
        def wait():
            end = time.monotonic() + 30
            while win.task is not None and time.monotonic() < end:
                app.processEvents()
                time.sleep(0.01)
            assert win.task is None

        win.do_preview()
        wait()
        n = len(win.preview.boxes())
        win.preview.set_box_angle(0, -0.7)  # the Skew field
        win.do_test_scan()
        wait()
        win.do_preview()
        wait()
        boxes = win.preview.boxes()
        assert len(boxes) == n and not any(b.manual for b in boxes)
    finally:
        QThreadPool.globalInstance().waitForDone(10000)
        app.processEvents()
        win.close()
