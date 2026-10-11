"""Main window."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, updates
from ..devices.base import list_devices
from ..settings import load_settings, save_settings
from ..workflow import Session, box_mm_to_px
from . import worker
from .preview import CroppedView, PreviewView, to_qimage, view_icon
from .settings_panel import SettingsPanel

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)"


BACKEND_NAMES = {"twain": "TWAIN", "wia": "WIA", "sane": "SANE", "escl": "Network", "file": "File"}


def device_label(d) -> str:
    """The connection first ("TWAIN: EPSON Perfection V600"), so a narrow box still shows it."""
    if d.backend == "demo":
        return d.name
    desc = " ".join(p for p in (d.vendor, d.model) if p) or d.name
    return f"{BACKEND_NAMES.get(d.backend, d.backend)}: {desc}"


class _LiveRelay(QObject):
    """Carries live scan strips from the scan thread to the GUI thread."""

    frame = Signal(object, object)  # image, (kind, area_mm, dpi, width, height, shrink)


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle(f"Hasami Scan {__version__}")  # the version shows which copy is running
        self.resize(1280, 860)
        self.settings = settings or load_settings()
        self.session = Session(self.settings)
        self.task: worker.Task | None = None
        self._live_started = False  # the current preview/scan has shown its first strip
        self._live_relay = _LiveRelay(self)
        self._live_relay.frame.connect(self._show_live)

        # Left: settings and the two big buttons.
        self.panel = SettingsPanel(self.settings)
        self.preview_btn = QPushButton("Preview")
        self.scan_btn = QPushButton("Scan")
        for b in (self.preview_btn, self.scan_btn):
            b.setMinimumHeight(36)
        self.scan_btn.setDefault(True)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.panel, 1)
        row = QHBoxLayout()
        row.addWidget(self.preview_btn)
        row.addWidget(self.scan_btn)
        lv.addLayout(row)

        # Right: preview and results.
        self.preview = PreviewView()
        self.preview.set_rotation(self.settings.crop.rotate)
        self.results = QListWidget()
        self.results.setViewMode(QListWidget.IconMode)
        self.results.setIconSize(QSize(220, 220))
        self.results.setResizeMode(QListWidget.Adjust)
        self.results.setSpacing(10)
        self.tabs = QTabWidget()
        self.cropped = CroppedView()
        self.tabs.addTab(self.preview, "Preview")
        self.tabs.addTab(self.cropped, "Cropped area")
        self.tabs.addTab(self.results, "Scanned")
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        # Bottom right, as in VueScan: rotate and zoom buttons (filled in by _build_actions).
        self.view_buttons = QHBoxLayout()
        self.view_buttons.addStretch(1)
        rv.addLayout(self.view_buttons)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 920])
        self.setCentralWidget(split)

        # Status bar with progress and cancel.
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.hide()
        sb = self.statusBar()
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cancel_btn)

        self._build_actions()
        self._redetect_timer = QTimer(self, singleShot=True, interval=350)
        self._cropped_timer = QTimer(self, singleShot=True, interval=120)
        self._save_timer = QTimer(self, singleShot=True, interval=1000)
        self._save_timer.timeout.connect(self._save_settings)

        # Signals.
        self.preview_btn.clicked.connect(self.do_preview)
        self.scan_btn.clicked.connect(self.do_scan)
        self.cancel_btn.clicked.connect(self.cancel)
        self.panel.changed.connect(self.on_setting_changed)
        self.panel.device_combo.currentIndexChanged.connect(self.on_device_selected)
        self.panel.device_refresh.clicked.connect(self.refresh_devices)
        self.preview.boxes_changed.connect(self.on_boxes_edited)
        self.results.itemDoubleClicked.connect(self.open_result)
        self._redetect_timer.timeout.connect(self.redetect)
        self._cropped_timer.timeout.connect(self.update_cropped)
        self.preview.boxes_changed.connect(self._cropped_timer.start)
        self.preview.scene().selectionChanged.connect(self._cropped_timer.start)
        self.tabs.currentChanged.connect(lambda *_: self._cropped_timer.start())
        self.panel.skew.valueChanged.connect(self.on_skew_edited)
        self.preview.boxes_changed.connect(self._show_skew)
        self.preview.scene().selectionChanged.connect(self._show_skew)

        self.refresh_devices()
        if getattr(sys, "frozen", False) and updates.due(self.settings):
            self.check_updates(manual=False)  # the installed app only; source checkouts use git

    # ----------------------------------------------------------------- setup
    def _build_actions(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=""):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            return a

        self.a_open = act("Open image...", self.open_image, "Ctrl+O", "Auto-crop an existing scan")
        self.a_preview = act("Preview", self.do_preview, "Ctrl+P")
        self.a_scan = act("Scan", self.do_scan, "Ctrl+S")
        self.a_autocrop = act("Auto-crop", self.redetect, "Ctrl+R", "Detect crop boxes again (keeps boxes you drew)")
        self.a_clear = act("Clear crops", self.clear_boxes)
        self.a_delete = act("Delete crop", self.preview.delete_selected, "Del")
        self.a_fit = act("Fit", self.preview.fit, "Ctrl+0", "Show the whole preview")
        self.a_zoom_in = act("Zoom in", self.preview.zoom_in, "Ctrl+=",
                             "Zoom in (or use the mouse wheel; drag with the right button to move around)")
        self.a_zoom_out = act("Zoom out", self.preview.zoom_out, "Ctrl+-")
        self.a_rot_left = act("Rotate left", lambda: self.rotate(-1), "Ctrl+[",
                              "Turn the preview and the saved scans 90° anticlockwise")
        self.a_rot_right = act("Rotate right", lambda: self.rotate(1), "Ctrl+]",
                               "Turn the preview and the saved scans 90° clockwise")
        self.a_folder = act("Open output folder", self.open_output_folder)
        for a in (self.a_open, self.a_preview, self.a_scan, self.a_autocrop, self.a_delete, self.a_clear):
            tb.addAction(a)
        colour = self.palette().color(QPalette.ButtonText)
        for a, kind in ((self.a_rot_left, "rotate_left"), (self.a_rot_right, "rotate_right"),
                        (self.a_zoom_out, "zoom_out"), (self.a_zoom_in, "zoom_in"), (self.a_fit, "fit")):
            a.setIcon(view_icon(kind, colour))
            b = QToolButton()
            b.setDefaultAction(a)
            b.setIconSize(QSize(28, 28))
            b.setAutoRaise(True)
            b.setToolTip(f"{a.text()} ({a.shortcut().toString(QKeySequence.NativeText)})"
                         + (f": {a.toolTip()}" if a.toolTip() and a.toolTip() != a.text() else ""))
            self.view_buttons.addWidget(b)

        m = self.menuBar().addMenu("&File")
        m.addAction(self.a_open)
        m.addAction(self.a_folder)
        m.addSeparator()
        m.addAction(act("Save settings", lambda: save_settings(self.settings)))
        m.addAction(act("Quit", self.close, "Ctrl+Q"))
        m = self.menuBar().addMenu("&Scan")
        for a in (self.a_preview, self.a_scan, self.a_autocrop, self.a_delete, self.a_clear):
            m.addAction(a)
        m = self.menuBar().addMenu("&View")
        for a in (self.a_rot_left, self.a_rot_right, self.a_zoom_in, self.a_zoom_out, self.a_fit):
            m.addAction(a)
        m = self.menuBar().addMenu("&Help")
        m.addAction(act("Check for updates", lambda: self.check_updates(manual=True)))
        self.a_auto_update = act("Check for updates on start", self._toggle_auto_update,
                                 tip="At most once a day, ask GitHub whether a newer version is out")
        self.a_auto_update.setCheckable(True)
        self.a_auto_update.setChecked(self.settings.updates.check_on_start)
        m.addAction(self.a_auto_update)
        m.addSeparator()
        m.addAction(act("About", self.about))

    # --------------------------------------------------------------- updates
    def _toggle_auto_update(self, *_) -> None:
        self.settings.updates.check_on_start = self.a_auto_update.isChecked()
        self._save_timer.start()

    def check_updates(self, manual: bool = False) -> None:
        """Ask GitHub for a newer release (in the background); ``manual``: from the Help menu."""
        t = worker.Task(lambda p, c, s: updates.fetch_latest())
        t.signals.finished.connect(lambda rel: self._update_checked(rel, manual))
        t.signals.failed.connect(lambda msg: manual and QMessageBox.information(
            self, "Hasami Scan", f"Could not check for updates:\n{msg}"))
        worker.start(t)
        self._keep(t)

    def _update_checked(self, rel, manual: bool) -> None:
        st = self.settings.updates
        st.last_check = time.time()
        self._save_timer.start()
        if not updates.should_offer(rel, __version__, st.skipped, manual):
            if manual:
                QMessageBox.information(self, "Hasami Scan", f"You have the latest version ({__version__}).")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Hasami Scan")
        box.setIcon(QMessageBox.Information)
        box.setTextFormat(Qt.RichText)
        box.setText(f"<b>Hasami Scan {rel.version}</b> is available (you have {__version__}).<p>"
                    f"<a href='{rel.page}'>What's new</a>.  Download the installer and run it: "
                    "your settings are kept.")
        download = box.addButton("Download", QMessageBox.AcceptRole)
        skip = box.addButton("Skip this version", QMessageBox.RejectRole) if not manual else None
        box.addButton("Later", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is download:
            QDesktopServices.openUrl(QUrl(rel.installer or rel.page))
        elif skip is not None and box.clickedButton() is skip:
            st.skipped = rel.version
            self._save_timer.start()

    # --------------------------------------------------------------- devices
    def refresh_devices(self) -> None:
        self.set_status("Looking for scanners...")
        t = worker.Task(lambda p, c, s: list_devices())
        t.signals.finished.connect(self._devices_listed)
        t.signals.failed.connect(lambda msg: self.set_status(f"Could not list scanners: {msg}"))
        worker.start(t)
        self._keep(t)

    def _devices_listed(self, devices) -> None:
        from ..devices.choose import NO_SCANNER, is_extra, offered

        self._devices = list(devices)
        combo = self.panel.device_combo
        current = self.settings.input.device
        shown = offered(devices, self.settings.input.show_all_devices, keep=current)
        combo.blockSignals(True)
        combo.clear()
        for d in shown:
            combo.addItem(device_label(d), d.id)
            if d.backend == "wia":
                combo.setItemData(combo.count() - 1, "Windows' basic scanner interface: no live view; the "
                                  "Epson V600 stops at 1200 dpi.  Use the TWAIN entry when there is one.",
                                  Qt.ToolTipRole)
            elif is_extra(d, devices):
                combo.setItemData(combo.count() - 1, "Another way to reach a scanner listed above (or "
                                  "another program): usually not the one to use.", Qt.ToolTipRole)
        if current.startswith("file:") and combo.findData(current) < 0:
            combo.addItem(f"File: {Path(current[5:]).name}", current)
        i = combo.findData(current)
        combo.setCurrentIndex(i if i >= 0 else 0)  # none chosen yet (or gone): the best one
        combo.blockSignals(False)
        real = [d for d in shown if d.backend not in ("demo", "file")]
        self.preview.set_hint("" if real else NO_SCANNER)
        self.set_status(f"{len(real)} scanner(s) found" if real else "No scanner found")
        chosen = next((d for d in devices if d.id == combo.currentData()), None)
        if chosen is not None and chosen.backend == "wia" and any(
                d.backend == "twain" and d.name.lower() in chosen.label().lower() for d in devices):
            self.set_status(f"Tip: choose 'TWAIN: {chosen.model or chosen.name}' under Scanner - "
                            "full resolution, live view and the scanner's normal speed.")
        self.on_device_selected()

    def on_device_selected(self, *_):
        dev_id = self.panel.device_combo.currentData()
        if not dev_id:
            return
        self.settings.input.device = dev_id
        t = worker.Task(lambda p, c, s: self.session.open(dev_id).capabilities())
        t.signals.finished.connect(self._caps_loaded)
        t.signals.failed.connect(lambda msg: self.set_status(f"Cannot open scanner: {msg}"))
        worker.start(t)
        self._keep(t)

    def _caps_loaded(self, caps) -> None:
        self.panel.set_choices("input", "source", [(s, s) for s in caps.sources])
        self.panel.set_choices("input", "scan_dpi", [(str(r), r) for r in caps.resolutions])
        prev = [r for r in caps.resolutions if r <= 300] or caps.resolutions[:1]
        self.panel.set_choices("input", "preview_dpi", [(str(r), r) for r in prev])
        names = {8: "8 (24-bit colour)", 16: "16 (48-bit colour)"}
        self.panel.set_choices("input", "bit_depth", [(names.get(b, str(b)), b) for b in caps.bit_depths])

    # ------------------------------------------------------------- scanning
    def _keep(self, t: worker.Task) -> None:
        # Keep a reference until the task reports back.
        self._tasks = [x for x in getattr(self, "_tasks", []) if x is not t] + [t]

    def _busy(self, busy: bool, label: str = "") -> None:
        for w in (self.preview_btn, self.scan_btn):
            w.setEnabled(not busy)
        for a in (self.a_preview, self.a_scan, self.a_open):
            a.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.cancel_btn.setVisible(busy)
        # A moving bar until the scanner reports progress: many drivers (e.g. Epson's
        # TWAIN driver) only report when the whole image is done.
        self.progress.setRange(0, 0)
        if label:
            self.set_status(label)

    def _run(self, label: str, fn, done) -> None:
        if self.task is not None:
            return
        self._busy(True, label)
        t = worker.Task(fn)
        t.signals.progress.connect(self._progress)
        t.signals.status.connect(self.set_status)
        t.signals.finished.connect(lambda r: (self._task_done(), done(r)))
        t.signals.failed.connect(self._task_failed)
        self.task = t
        self._keep(t)
        worker.start(t)

    def _progress(self, f: float) -> None:
        if f > 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(f * 1000))

    def _task_done(self) -> None:
        self.task = None
        self._busy(False)

    def _task_failed(self, msg: str) -> None:
        self._task_done()
        self.preview.end_live()
        self.set_status(msg)
        if msg != "Cancelled":
            QMessageBox.warning(self, "Hasami Scan", msg)

    def cancel(self) -> None:
        if self.task is not None:
            self.task.cancel.set()
            self.set_status("Cancelling...")

    def _live_fn(self, kind: str):
        """Callback for :attr:`Session.live`, or None.  It runs in the scan thread:
        it shrinks the strips received so far to preview size and hands them on."""
        prev = self.session.preview
        if not self.settings.input.live_view or (kind == "scan" and prev is None):
            return None
        import cv2

        relay, last = self._live_relay, [0.0]
        small = {"buf": None, "done": 0}  # the scan shrunk to preview size, row by row

        def shrunk(li, shrink: float):
            """New rows shrunk onto a buffer of the final size: the scale stays the same
            from update to update (shrinking all rows each time rounded differently,
            and the picture shook back and forth)."""
            h, w = li.image.shape[:2]
            ht, wt = max(1, round(h * shrink)), max(1, round(w * shrink))
            buf = small["buf"]
            if buf is None or buf.shape != (ht, wt) + li.image.shape[2:] or buf.dtype != li.image.dtype:
                buf = small["buf"] = np.empty((ht, wt) + li.image.shape[2:], li.image.dtype)
                small["done"] = 0
            sy = ht / h
            j0 = small["done"]
            j1 = ht if li.rows >= h else min(ht, int(li.rows * sy))
            if j1 > j0:
                a = min(h - 1, int(j0 / sy))
                b = min(h, max(a + 1, int(np.ceil(j1 / sy))))
                buf[j0:j1] = cv2.resize(li.image[a:b], (wt, j1 - j0),
                                        interpolation=cv2.INTER_AREA).reshape(buf[j0:j1].shape)
                small["done"] = j1
            return buf[:small["done"]].copy(), sy

        def live(li) -> None:
            now = time.monotonic()
            if (not li.done and now - last[0] < 0.2) or li.rows <= 0:
                return
            last[0] = now
            shrink = 1.0 if kind == "preview" else min(1.0, prev.dpi / li.dpi)
            if shrink < 1.0:
                part, shrink = shrunk(li, shrink)
                if not len(part):
                    return
            else:
                part = li.image[:li.rows].copy()  # the scanner keeps filling li.image
            h, w = li.image.shape[:2]
            relay.frame.emit(part, (kind, li.area_mm, li.dpi, w, h, shrink))

        return live

    def _show_live(self, part, info) -> None:
        kind, area, dpi, w, h, shrink = info
        if self.task is None:
            return  # arrived after the scan ended
        if self.tabs.currentWidget() is self.results:
            self.tabs.setCurrentWidget(self.preview)
        # Like VueScan: when the first strip arrives, a grey area shows where the
        # image will come in (for a scan: zoomed to just that area), and fills in.
        first = not self._live_started
        self._live_started = True
        if kind == "preview":
            if first:
                self.preview.ensure_size(w, h)
                self.preview.live_area(0, 0, w, h)
            self.preview.show_live(part, 0, 0, 1 / shrink)
            return
        p = self.session.preview
        if p is None:
            return
        x = (area[0] - p.area_mm[0]) / 25.4 * p.dpi
        y = (area[1] - p.area_mm[1]) / 25.4 * p.dpi
        k = p.dpi / dpi
        if first:
            # One crop box: only that box, turned straight (VueScan); several: the area.
            crops = self.session._crop_boxes()
            box = box_mm_to_px(crops[0], p.dpi, p.area_mm[:2]) if len(crops) == 1 else None
            self.preview.live_area(x, y, w * k, h * k, zoom=True, box=box)
        self.preview.show_live(part, x, y, k / shrink)

    def do_preview(self) -> None:
        self.session.live = self._live_fn("preview")
        self._live_started = False
        if self.session.live is not None and self.task is None:
            self.preview.begin_live()
        self._run("Previewing...", lambda p, c, s: self.session.run_preview(p, c), self._preview_done)

    def _preview_done(self, result) -> None:
        self.tabs.setCurrentWidget(self.preview)
        self.preview.set_image(result.image)
        self._show_boxes()

    def update_cropped(self) -> None:
        """Fill the "Cropped area" tab with the selected crop box (else the first)."""
        if self.tabs.currentWidget() is not self.cropped:
            return  # done when the tab is opened
        p, boxes = self.session.preview, self.preview.boxes()
        if p is None:
            self.cropped.show_image(None, "Press Preview, then click a crop box to see it here.")
            return
        if not boxes:
            self.cropped.show_image(None, "No crop box: draw one on the preview.")
            return
        i = self.preview.selected_index()
        i = 0 if i is None else i
        from ..core.geometry import extract
        from ..core.processing import process

        crop = extract(p.image, boxes[i])
        turns = int(self.settings.crop.rotate) % 4
        if turns:
            crop = np.ascontiguousarray(np.rot90(crop, -turns))
        try:
            crop = process(crop, self.settings.color, dpi=p.dpi)  # as it will be saved
        except Exception:
            pass
        b = boxes[i]
        mm = 25.4 / p.dpi
        self.cropped.show_image(crop, f"Crop {i + 1} of {len(boxes)}: {b.w * mm:.0f} x {b.h * mm:.0f} mm "
                                      "(click another box on the preview to see it)")

    def _show_boxes(self) -> None:
        boxes = self.session.preview_boxes_px()
        self.preview.set_boxes(boxes)
        self._cropped_timer.start()
        self._show_skew()
        mode = self.settings.crop.mode
        if mode == "off":
            self.set_status("Auto-crop is off: the whole area will be scanned, or draw a crop box")
        elif mode == "booklet":
            if not boxes:
                self.set_status("No page found")
            elif any("spread_pages" in b.meta for b in boxes):
                self.set_status("Booklet: the whole spread as one image")
            else:
                self.set_status(f"Booklet: cropping the fully visible page{'s' if len(boxes) > 1 else ''}")
        else:
            self.set_status(f"Found {len(boxes)} item(s). Drag to adjust, drag on empty space to add a crop.")
        if mode != "off":
            self.set_status(f"{self.status_label.text()}  [profile: {self.session.profile().label}]")

    def redetect(self) -> None:
        if self.session.preview is None or self.task is not None:
            return
        self.session.set_preview_boxes_px(self.preview.boxes())
        self._run("Detecting...", lambda p, c, s: self.session.detect_on_preview(), lambda r: self._show_boxes())

    def rotate(self, turns: int) -> None:
        """Turn the preview, and with it every saved scan, by quarter turns."""
        st = self.settings.crop
        st.rotate = (st.rotate + turns) % 4
        self.panel.load()
        self.preview.set_rotation(st.rotate)
        self._cropped_timer.start()
        self.set_status(f"Rotation: {('none', '90° right', '180°', '90° left')[st.rotate]}")

    def clear_boxes(self) -> None:
        self.preview.set_boxes([])
        self.session.boxes_mm = []
        self._show_skew()

    def on_boxes_edited(self) -> None:
        self.session.set_preview_boxes_px(self.preview.boxes())

    def _show_skew(self) -> None:
        """Crop tab "Skew": the angle of the selected crop box (else crop 1)."""
        i = self.preview.current_index()
        self.panel.set_skew(None if i is None else self.preview.boxes()[i].angle)

    def on_skew_edited(self, value: float) -> None:
        i = self.preview.current_index()
        if i is None:
            return
        if self.preview.selected_index() is None:
            self.preview.items_[i].setSelected(True)  # show which box turns
        self.preview.set_box_angle(i, value)

    def on_setting_changed(self, section: str, name: str) -> None:
        self._save_timer.start()  # keep the saved settings current too
        self._cropped_timer.start()
        if (section, name) == ("input", "show_all_devices"):
            self._devices_listed(getattr(self, "_devices", []))
            return
        if (section, name) == ("crop", "rotate"):
            self.preview.set_rotation(self.settings.crop.rotate)
            return
        if section == "crop" and self.session.preview is not None:
            # Re-detect with the new settings (boxes drawn by hand are kept).
            self._redetect_timer.start()

    def do_scan(self) -> None:
        if self.session.preview is not None:
            self.session.set_preview_boxes_px(self.preview.boxes())
        self.session.live = self._live_fn("scan")
        self._live_started = False  # the preview stays until the scan's first strip arrives
        self._run("Scanning...", lambda p, c, s: self.session.run_scan(p, c, s), self._scan_done)

    def _scan_done(self, outcomes) -> None:
        self.preview.end_live()  # the preview back, for the next crops
        n = 0
        for oc in outcomes:
            for img, path in zip(oc.items, oc.paths or [None] * len(oc.items)):
                pm = QPixmap.fromImage(to_qimage(img)).scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                it = QListWidgetItem(QIcon(pm), Path(path).name if path else "")
                if path:
                    it.setData(Qt.UserRole, str(path))
                    it.setToolTip(str(path))
                self.results.insertItem(0, it)
                n += 1
        self.tabs.setCurrentWidget(self.results)
        areas = "; ".join(f"{oc.scan.area_mm[2]:.0f} x {oc.scan.area_mm[3]:.0f} mm at {oc.scan.dpi:.0f} dpi"
                          for oc in outcomes)
        self.set_status(f"Saved {n} item(s) to {self.settings.output.path} (scanned {areas})")
        notes = sorted({oc.scan.note for oc in outcomes if oc.scan.note})
        if notes:
            QMessageBox.warning(self, "Hasami Scan", "\n".join(notes))
        # The next preview should look for new items.
        self.session.boxes_mm = [b for b in self.session.boxes_mm if b.manual]

    # ----------------------------------------------------------------- misc
    def open_image(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Open scan", str(Path.home()), IMAGE_FILTER)
        if not f:
            return
        dev_id = f"file:{f}"
        combo = self.panel.device_combo
        if combo.findData(dev_id) < 0:
            combo.addItem(f"File: {Path(f).name}", dev_id)
        combo.setCurrentIndex(combo.findData(dev_id))
        self.settings.input.device = dev_id
        self.session.boxes_mm = []
        self.do_preview()

    def open_result(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.UserRole)
        if p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def open_output_folder(self) -> None:
        path = self.settings.output.path
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    def about(self) -> None:
        QMessageBox.about(self, "About Hasami Scan",
                          f"<b>Hasami Scan {__version__}</b><p>Scanning with smarter auto-crop: every photo on the "
                          "glass, white borders kept, crooked items straightened, and booklet pages cropped "
                          "at the spine.</p>")

    def _save_settings(self) -> None:
        try:
            save_settings(self.settings)
        except OSError:
            pass

    def closeEvent(self, event) -> None:
        if self.task is not None:
            self.task.cancel.set()
        try:
            save_settings(self.settings)
        except OSError:
            pass
        self.session.close()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Hasami Scan")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
