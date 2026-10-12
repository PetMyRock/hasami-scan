import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_builds(tmp_path):
    pytest.importorskip("PySide6")
    try:
        from PySide6.QtWidgets import QApplication
        from hasami.gui.app import MainWindow
    except ImportError as e:  # e.g. missing libEGL in a headless container
        pytest.skip(str(e))
    from hasami.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    st = AppSettings()
    st.output.folder = str(tmp_path)
    win = MainWindow(st)
    win.show()
    app.processEvents()
    from hasami import __version__
    assert win.windowTitle() == f"Hasami Scan {__version__}"
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)  # e.g. the device listing started on launch
    app.processEvents()
    win.close()


def _window(tmp_path):
    pytest.importorskip("PySide6")
    try:
        from PySide6.QtWidgets import QApplication
        from hasami.gui.app import MainWindow
    except ImportError as e:
        pytest.skip(str(e))
    from hasami.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    st = AppSettings()
    st.output.folder = str(tmp_path)
    win = MainWindow(st)
    return app, win


def _done(app, win):
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    win.close()


def test_rotate_buttons_turn_the_preview(tmp_path):
    import numpy as np
    from hasami.core.geometry import CropBox
    app, win = _window(tmp_path)
    win.resize(1000, 700)
    win.show()
    win.preview.set_image(np.full((200, 300, 3), 200, np.uint8))
    win.preview.set_boxes([CropBox(100, 100, 80, 60)])
    win.a_rot_right.trigger()
    assert win.settings.crop.rotate == 1
    t = win.preview.transform()
    assert abs(t.m11()) < 1e-6 and t.m12() > 0  # 90 degrees clockwise
    win.a_rot_left.trigger()
    win.a_rot_left.trigger()
    assert win.settings.crop.rotate == 3
    assert win.preview.boxes()[0].w == 80  # boxes stay in scan coordinates
    _done(app, win)


def test_zoom_survives_the_scroll_bars(tmp_path):
    import numpy as np
    app, win = _window(tmp_path)
    win.resize(1000, 700)
    win.show()
    app.processEvents()
    win.preview.set_image(np.full((800, 600, 3), 200, np.uint8))
    app.processEvents()
    s0 = win.preview.transform().m11()
    win.a_zoom_in.trigger()
    win.a_zoom_in.trigger()
    app.processEvents()  # scroll bars appear, the view is resized
    assert win.preview.transform().m11() == pytest.approx(s0 * 1.25 ** 2, rel=1e-3)
    win.a_zoom_out.trigger()
    win.a_fit.trigger()
    assert win.preview.transform().m11() == pytest.approx(s0, rel=0.05)
    _done(app, win)


def test_side_handle_moves_one_side(tmp_path):
    import numpy as np
    from PySide6.QtCore import QPointF
    from hasami.core.geometry import CropBox
    app, win = _window(tmp_path)
    win.preview.set_image(np.full((200, 300, 3), 200, np.uint8))
    win.preview.set_boxes([CropBox(150, 100, 100, 60)])
    it = win.preview.items_[0]
    assert set(it._handles()) >= {"s0", "s1", "s2", "s3"}

    class Ev:
        def __init__(self, x, y):
            self._p = QPointF(x, y)

        def pos(self):
            return self._p

    it._drag = ("s1", QPointF(200, 100), it.box)  # right side
    it.mouseMoveEvent(Ev(230, 100))
    x0, y0, x1, y1 = it.box.bounding_rect()
    assert (x0, x1) == pytest.approx((100, 230)) and (y0, y1) == pytest.approx((70, 130))
    it._drag = ("s0", QPointF(150, 70), it.box)  # top side
    it.mouseMoveEvent(Ev(150, 90))
    x0, y0, x1, y1 = it.box.bounding_rect()
    assert (y0, y1) == pytest.approx((90, 130)) and (x0, x1) == pytest.approx((100, 230))
    _done(app, win)


def test_browse_button_sets_the_folder(tmp_path, monkeypatch):
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QFileDialog, QPushButton
    app, win = _window(tmp_path)
    target = str(tmp_path / "chosen")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: target))
    btn = [b for b in win.panel.findChildren(QPushButton) if b.text() == "Browse..."][0]
    btn.click()
    assert win.settings.output.folder == target
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    win.close()


def test_dragging_anywhere_on_an_edge_resizes(tmp_path):
    # Maintainer: dragging an edge moved the whole box (only a small dot in the
    # middle of a side resized, and only once the box was selected).
    import numpy as np
    from PySide6.QtCore import QPointF
    from hasami.core.geometry import CropBox
    app, win = _window(tmp_path)
    win.preview.set_image(np.full((200, 300, 3), 200, np.uint8))
    win.preview.set_boxes([CropBox(150, 100, 100, 60)])  # x 100-200, y 70-130
    it = win.preview.items_[0]
    assert not it.isSelected()
    px = it._scale()  # scene units per screen pixel
    assert it._hit(QPointF(200, 80)) == "s1"  # right edge, away from its middle
    assert it._hit(QPointF(200 + 3 * px, 120)) == "s1"  # just outside it
    assert it._hit(QPointF(130, 70 + 2 * px)) == "s0"  # top edge, just inside
    assert it._hit(QPointF(100, 130)) == "c3"  # bottom-left corner
    assert it._hit(QPointF(150, 100)) == "move"
    assert it.shape().contains(QPointF(200 + 3 * px, 120))  # the outside band takes the click
    _done(app, win)


def test_preview_and_scan_are_shown_as_they_come_in(tmp_path, monkeypatch):
    # Like VueScan: the image appears strip by strip while the scanner moves.
    import time
    from hasami.devices import demo_backend
    monkeypatch.setattr(demo_backend.DemoScanner, "strip_delay", 0.05)
    app, win = _window(tmp_path)
    win.settings.input.device = "demo:anime"
    win.settings.input.preview_dpi = 50
    win.settings.input.scan_dpi = 100
    shown, areas = [], []
    real, real_area = win.preview.show_live, win.preview.live_area

    def spy(img, x, y, scale=1.0, full=None):
        shown.append((img.shape[:2], round(x), round(y), round(scale, 3)))
        real(img, x, y, scale, full)
        ph = win.preview._placeholder_item
        boxes_hidden = not any(it.isVisible() for it in win.preview.items_)
        areas.append((ph is not None and ph.isVisible(), boxes_hidden, win.preview._zoomed_live))

    monkeypatch.setattr(win.preview, "show_live", spy)

    def wait():
        end = time.monotonic() + 30
        while win.task is not None and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert win.task is None

    win.do_preview()
    wait()
    pv = win.session.preview
    h, w = pv.image.shape[:2]
    assert len(shown) >= 3  # several strips, not just the end result
    rows = [s[0][0] for s in shown]
    assert rows == sorted(rows) and rows[0] < h and rows[-1] == h
    assert all(s[1:4] == (0, 0, 1.0) for s in shown)
    # Like VueScan: a grey area where the preview comes in, filled strip by strip.
    assert all(a[0] and not a[2] for a in areas)
    assert win.preview._placeholder_item is None  # gone with the finished preview
    shown.clear()
    areas.clear()
    before = win.preview.pixmap_item.pixmap().toImage().pixelColor(w // 2, h // 2)
    win.do_scan()
    # The preview stays until the scan's first strip arrives ...
    assert win.preview.pixmap_item.pixmap().toImage().pixelColor(w // 2, h // 2) == before
    wait()
    # ... then the view shows just the area being scanned, grey, filling in, no crop boxes ...
    assert len(shown) >= 3 and win.results.count() > 0
    assert all(a == (True, True, True) for a in areas)
    assert all(abs(s[3] - 1.0) < 0.02 for s in shown)  # shrunk to the preview's scale
    assert len({s[3] for s in shown}) == 1  # the same scale every time: no shaking
    # ... and the preview, its boxes and the whole view come back when the scan is done.
    assert win.preview.pixmap_item.pixmap().toImage().pixelColor(w // 2, h // 2) == before
    assert all(it.isVisible() for it in win.preview.items_)
    assert win.preview._live is None and win.preview._placeholder_item is None
    assert win.preview._fitted and not win.preview._zoomed_live
    _done(app, win)


def test_live_view_can_be_switched_off(tmp_path):
    app, win = _window(tmp_path)
    win.settings.input.live_view = False
    assert win._live_fn("preview") is None
    _done(app, win)


def test_rotate_and_zoom_buttons_under_the_preview(tmp_path):
    # Maintainer: rotate buttons as icons at the bottom right, like VueScan, not in the toolbar.
    from PySide6.QtWidgets import QToolBar, QToolButton
    app, win = _window(tmp_path)
    buttons = [win.view_buttons.itemAt(i).widget() for i in range(win.view_buttons.count())]
    buttons = [b for b in buttons if isinstance(b, QToolButton)]
    assert [b.defaultAction() for b in buttons] == [win.a_rot_left, win.a_rot_right, win.a_zoom_out,
                                                    win.a_zoom_in, win.a_fit]
    assert all(not b.icon().isNull() for b in buttons)
    toolbar = win.findChild(QToolBar).actions()
    assert win.a_rot_left not in toolbar and win.a_zoom_in not in toolbar
    buttons[1].click()
    assert win.settings.crop.rotate == 1
    _done(app, win)


def test_cropped_area_tab_shows_the_selected_box(tmp_path):
    # Like VueScan's "Cropped area": only what is inside a crop box, straightened and turned.
    import numpy as np
    from hasami.core.geometry import CropBox
    from hasami.devices.base import ScanResult
    app, win = _window(tmp_path)
    win.resize(1000, 700)
    win.show()
    img = np.full((300, 400, 3), 230, np.uint8)
    img[50:110, 40:200] = (200, 30, 30)  # red, 160 x 60
    img[150:280, 250:330] = (30, 30, 200)  # blue, 80 x 130
    win.settings.color.auto_levels = False
    win.session.preview = ScanResult(img, 100.0, (0, 0, 101.6, 76.2))
    win.preview.set_image(img)
    win.preview.set_boxes([CropBox(120, 80, 160, 60), CropBox(290, 215, 80, 130)])
    win.tabs.setCurrentWidget(win.cropped)
    win.update_cropped()
    pm = win.cropped._pm
    assert (pm.width(), pm.height()) == (160, 60) and pm.toImage().pixelColor(80, 30).red() > 150
    win.preview.items_[1].setSelected(True)
    win.update_cropped()
    pm = win.cropped._pm
    assert (pm.width(), pm.height()) == (80, 130) and pm.toImage().pixelColor(40, 65).blue() > 150
    win.a_rot_right.trigger()
    win.update_cropped()
    assert (win.cropped._pm.width(), win.cropped._pm.height()) == (130, 80)  # turned like the saved scan
    win.preview.set_boxes([])
    win.update_cropped()
    assert win.cropped._pm is None and "No crop box" in win.cropped.text()
    _done(app, win)



def test_progress_bar_moves_until_the_scanner_reports(tmp_path):
    # Epson's TWAIN driver reports nothing until the scan is done: no stuck "0%".
    app, win = _window(tmp_path)
    win._busy(True, "Scanning...")
    assert win.progress.maximum() == 0  # busy indicator
    win._progress(0.5)
    assert (win.progress.maximum(), win.progress.value()) == (1000, 500)
    win._busy(False)
    _done(app, win)


def test_blank_preview_comes_back_when_a_scan_fails(tmp_path):
    import numpy as np
    app, win = _window(tmp_path)
    win.preview.set_image(np.full((40, 30, 3), 200, np.uint8))
    win.preview.begin_live()
    assert win.preview.pixmap_item.pixmap().toImage().pixelColor(5, 5).red() == 38
    win.preview.show_live(np.zeros((10, 30, 3), np.uint8), 0, 0)
    win._task_failed("Cancelled")
    assert win.preview.pixmap_item.pixmap().toImage().pixelColor(5, 5).red() == 200
    assert win.preview._live is None
    _done(app, win)


def test_preview_resolution_goes_up_to_300(tmp_path):
    from hasami.devices.base import Capabilities
    app, win = _window(tmp_path)
    win._caps_loaded(Capabilities(resolutions=[50, 75, 100, 150, 200, 300, 400, 600, 800, 1200]))
    combo = win.panel._widgets[("input", "preview_dpi")]
    assert [combo.itemData(i) for i in range(combo.count())] == [50, 75, 100, 150, 200, 300]
    _done(app, win)


def test_scanner_list_shows_the_connection_first(tmp_path):
    # Maintainer scanned through WIA for days without knowing: the box showed only
    # "EPSON Perfection V600 (" and the "(wia)" was cut off.
    from hasami.devices.base import DeviceInfo
    app, win = _window(tmp_path)
    devs = [DeviceInfo("wia:{X}\\0000", "EPSON Perfection V600", "", "EPSON Perfection V600", "wia"),
            DeviceInfo("twain:VueScan TWAIN", "VueScan TWAIN", "", "VueScan TWAIN", "twain"),
            DeviceInfo("twain:EPSON Perfection V600", "EPSON Perfection V600", "", "EPSON Perfection V600", "twain"),
            DeviceInfo("demo:anime", "Demo scanner", "", "Demo scanner", "demo")]
    win.settings.input.device = "wia:{X}\\0000"
    win._devices_listed(devs)
    combo = win.panel.device_combo
    # The WIA twin stays listed because it was chosen before; VueScan and the demo are hidden.
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "TWAIN: EPSON Perfection V600", "WIA: EPSON Perfection V600"]
    assert "Tip: choose 'TWAIN: EPSON Perfection V600'" in win.status_label.text()
    _done(app, win)


def _owner_devices():
    from hasami.devices.base import DeviceInfo
    return [
        DeviceInfo("twain:VueScan TWAIN", "VueScan TWAIN", "", "VueScan TWAIN - VueScan x64, 64-bit", "twain"),
        DeviceInfo("twain:EPSON Perfection V600", "EPSON Perfection V600", "",
                   "EPSON Perfection V600 - EPSON Flatbed Scanner, 32-bit", "twain"),
        DeviceInfo("twain:VueScan TWAIN [VueScan]", "VueScan TWAIN", "", "VueScan TWAIN - VueScan, 32-bit", "twain"),
        DeviceInfo("twain:WIA-EPSON Perfection V600", "WIA-EPSON Perfection V600", "",
                   "WIA-EPSON Perfection V600 - Twain Data Source On WIA, 32-bit", "twain"),
        DeviceInfo("wia:{6BDD}\\0000", "EPSON Perfection V600", "", "", "wia"),
        DeviceInfo("demo:anime", "Demo scanner", "", "Demo scanner", "demo"),
    ]


def test_first_start_picks_the_best_scanner_and_hides_the_rest(tmp_path):
    # Install plan step 6 (maintainer): best connection chosen by itself, duplicates and
    # demos hidden; "Show all scanner connections" brings the duplicates back.
    app, win = _window(tmp_path)
    win.settings.input.device = ""  # nothing chosen yet
    win._devices_listed(_owner_devices())
    combo = win.panel.device_combo
    assert [combo.itemData(i) for i in range(combo.count())] == ["twain:EPSON Perfection V600"]
    assert win.settings.input.device == "twain:EPSON Perfection V600"
    assert win.status_label.text() == "1 scanner(s) found"
    win.settings.input.show_all_devices = True
    win.on_setting_changed("input", "show_all_devices")
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "twain:EPSON Perfection V600", "wia:{6BDD}\\0000", "twain:WIA-EPSON Perfection V600",
        "twain:VueScan TWAIN", "twain:VueScan TWAIN [VueScan]"]
    assert combo.currentData() == "twain:EPSON Perfection V600"
    _done(app, win)


def test_no_scanner_says_what_to_install(tmp_path):
    from hasami.devices.base import DeviceInfo
    from hasami.devices.choose import NO_SCANNER
    app, win = _window(tmp_path)
    win.settings.input.device = ""
    win._devices_listed([DeviceInfo("demo:anime", "Demo scanner", "", "Demo scanner", "demo")])
    assert win.panel.device_combo.count() == 0
    assert win.status_label.text() == "No scanner found"
    texts = [it.toPlainText() for it in win.preview.scene().items() if hasattr(it, "toPlainText")]
    assert texts == [NO_SCANNER]
    win._devices_listed(_owner_devices()[1:2])  # plugged in, Refresh
    texts = [it.toPlainText() for it in win.preview.scene().items() if hasattr(it, "toPlainText")]
    assert texts and texts[0].startswith("1.  Put the item face down on the glass")
    _done(app, win)


def test_scan_of_one_tilted_crop_is_shown_straight(tmp_path, monkeypatch):
    # VueScan: while a crop comes in, only that crop is shown, turned upright.
    import math
    import time
    from PySide6.QtCore import QPointF
    from hasami.core.geometry import CropBox
    from hasami.devices import demo_backend
    from PySide6.QtCore import QThreadPool
    from hasami.gui import app as gui_app
    monkeypatch.setattr(demo_backend.DemoScanner, "strip_delay", 0.05)
    # Only the demo scanner: SANE must not be listed while another test's window opens it.
    monkeypatch.setattr(gui_app, "list_devices", lambda: [demo_backend.DemoBackend().list_devices()[0]])
    QThreadPool.globalInstance().waitForDone(10000)
    app, win = _window(tmp_path)
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    win.settings.input.device = "demo:anime"
    win.settings.input.preview_dpi = 50
    win.settings.input.scan_dpi = 100
    win.settings.crop.mode = "off"
    win.resize(1200, 800)
    win.show()

    def wait():
        end = time.monotonic() + 30
        while win.task is not None and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert win.task is None

    win.do_preview()
    wait()
    box = CropBox(200, 150, 160, 100, angle=6.0, manual=True)
    win.preview.set_boxes([box])
    seen, fronts = [], []
    real = win.preview.show_live

    def spy(img, x, y, scale=1.0, full=None):
        real(img, x, y, scale, full)
        v = win.preview
        t = v.transform()
        ph = v._placeholder_item
        seen.append((round(math.degrees(math.atan2(t.m12(), t.m11())), 1),
                     v._live.parentItem().parentItem() is ph,
                     bool(ph.flags() & ph.GraphicsItemFlag.ItemClipsChildrenToShape),
                     v.mapFromScene(QPointF(box.cx, box.cy)), v.viewport().rect().center()))
        fr = v._front.polygon()  # where the scan has come in: edge along the box, not the bed
        edge = fr[2] - fr[3]
        fronts.append((round(math.degrees(math.atan2(edge.y(), edge.x())), 1),
                       math.hypot(*(fr[3] - fr[0]).toTuple())))

    monkeypatch.setattr(win.preview, "show_live", spy)
    win.do_scan()
    wait()
    assert seen
    for angle, child, clips, centre, vc in seen:
        assert angle == -6.0 and child and clips  # turned upright, cut to the crop
        assert abs(centre.x() - vc.x()) <= 2 and abs(centre.y() - vc.y()) <= 2
    assert all(a == 6.0 for a, _ in fronts)  # a straight edge in the straightened view
    done = [d for _, d in fronts]
    assert done == sorted(done) and done[-1] == pytest.approx(box.h)
    t = win.preview.transform()
    assert abs(t.m12()) < 1e-9 and win.preview._fitted  # straight and whole again
    _done(app, win)


def test_skew_field_turns_the_selected_crop_box(tmp_path):
    # Maintainer: a Skew field like VueScan's Crop > Skew (angle of the crop box).
    import numpy as np
    from hasami.core.geometry import CropBox
    app, win = _window(tmp_path)
    win.show()
    skew = win.panel.skew
    assert not skew.isEnabled()  # no crop box yet
    win.preview.set_image(np.full((300, 400, 3), 200, np.uint8))
    win.session.preview = None
    win.preview.set_boxes([CropBox(100, 100, 80, 60, angle=2.5), CropBox(300, 200, 60, 40, angle=-1.0)])
    win._show_skew()
    assert skew.isEnabled() and skew.value() == 2.5  # crop 1 when none is selected
    win.preview.items_[1].setSelected(True)
    app.processEvents()
    assert skew.value() == -1.0
    skew.setValue(4.0)
    b = win.preview.boxes()[1]
    assert b.angle == 4.0 and b.manual and (b.cx, b.cy, b.w, b.h) == (300, 200, 60, 40)
    assert win.preview.boxes()[0].angle == 2.5
    # Left / Right arrow keys step it (maintainer), 0.1 degree each.
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.keyClick(skew, Qt.Key_Right)
    assert abs(win.preview.boxes()[1].angle - 4.1) < 1e-9
    QTest.keyClick(skew, Qt.Key_Left)
    QTest.keyClick(skew, Qt.Key_Left)
    assert abs(win.preview.boxes()[1].angle - 3.9) < 1e-9
    # Turning a box with its handle shows the new angle in the field.
    it = win.preview.items_[1]
    it.box = CropBox(300, 200, 60, 40, angle=-3.0)
    win.preview.boxes_changed.emit()
    assert skew.value() == -3.0
    win.clear_boxes()
    assert not skew.isEnabled()
    _done(app, win)


def test_turning_auto_crop_off_removes_every_box(tmp_path):
    # Owner: switching auto-crop off deletes all crop boxes (drawn ones too); Off adds none.
    import numpy as np
    from hasami.core.geometry import CropBox
    app, win = _window(tmp_path)
    win.settings.crop.mode = "multi"
    win.preview.set_image(np.full((300, 400, 3), 200, np.uint8))
    win.preview.set_boxes([CropBox(100, 100, 80, 60), CropBox(300, 200, 60, 40, manual=True)])
    win.on_boxes_edited()
    win.settings.crop.mode = "off"
    win.on_setting_changed("crop", "mode")
    assert win.preview.boxes() == [] and win.session.boxes_mm == []
    assert not win.panel.skew.isEnabled()
    _done(app, win)
