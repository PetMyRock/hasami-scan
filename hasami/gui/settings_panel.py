"""Settings tabs (Input, Crop, Filter, Colour, Output) bound to AppSettings."""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QWidget,
)

from ..settings import AppSettings


class SkewBox(QDoubleSpinBox):
    """Spin box whose Left / Right keys step the value (maintainer), like Up / Down."""

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            self.stepBy(1 if event.key() == Qt.Key_Right else -1)
            event.accept()
            return
        super().keyPressEvent(event)


@dataclass
class Field:
    section: str  # "input", "crop", "color", "output"
    name: str
    label: str
    choices: list[tuple[str, Any]] | None = None  # (label, value)
    lo: float = 0.0
    hi: float = 100.0
    step: float = 1.0
    decimals: int = 2
    tip: str = ""
    kind: str = ""  # "folder" for a folder chooser
    scale: float = 1.0  # shown value = stored value * scale
    enabled_if: Callable[[Any], bool] | None = None  # greyed out unless true for the settings


DPI_CHOICES = [(str(d), d) for d in (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2400, 4800)]

TABS: list[tuple[str, list[Field]]] = [
    ("Input", [
        Field("input", "source", "Source", [("Flatbed", "Flatbed"), ("ADF", "ADF"), ("ADF Duplex", "ADF Duplex"),
                                            ("Transparency", "Transparency")]),
        Field("input", "mode", "Mode", [("Colour", "color"), ("Grey", "gray"), ("Black & white", "lineart")]),
        Field("color", "media", "Media", [("Photo / print", "photo"), ("Document", "document"),
                                          ("Colour negative", "color_negative"), ("B&W negative", "bw_negative"),
                                          ("Slide", "slide")]),
        Field("input", "preview_dpi", "Preview resolution", DPI_CHOICES[:6]),
        Field("input", "scan_dpi", "Scan resolution", DPI_CHOICES,
              tip="600 dpi or more is recommended for printed artbooks (needed for good descreening)"),
        Field("input", "bit_depth", "Bits per channel", [("8 (24-bit colour)", 8), ("16 (48-bit colour)", 16)]),
        Field("input", "scan_crop_area_only", "Scan only the crop area",
              tip="Faster: the final scan covers just the area around the crop boxes"),
        Field("input", "live_view", "Show the scan as it comes in",
              tip="Draw previews and scans in the preview strip by strip as the scanner driver hands "
                  "them over (TWAIN scanners and the demo scanner).  Epson's TWAIN driver hands over the "
                  "image only when the scanner has finished, so there it appears all at once."),
        Field("input", "twain_show_ui", "TWAIN: show the scanner's own window",
              tip="For TWAIN scanners: open the driver's window (e.g. Epson Scan) for each scan, "
                  "to use its own settings.  Needed by drivers that cannot scan without it.  "
                  "Settings only that window has (e.g. Epson's Scanning Quality) are remembered "
                  "and used for later scans without it."),
        Field("input", "show_all_devices", "Show all scanner connections",
              tip="Also list the other ways Windows offers to reach a scanner (its WIA driver, the "
                  "WIA-to-TWAIN bridge) and programs posing as scanners (VueScan).  Normally only the "
                  "best connection of each scanner is listed."),
        Field("input", "twain_python", "TWAIN: Python to use",
              tip="Leave empty to find one automatically.  Many TWAIN drivers are 32-bit only: "
                  "install 32-bit Python, run  py -3-32 -m pip install pytwain  and the scanner appears."),
    ]),
    ("Crop", [
        Field("crop", "mode", "Auto-crop", [("Every item (multi)", "multi"), ("One item (single)", "single"),
                                            ("Booklet page (fully visible page only)", "booklet"), ("Off", "off")],
              tip="Booklet: for an open booklet that does not fit on the glass, crop only the page that is "
                  "completely visible, from its outer edge to the spine"),
        Field("crop", "booklet_output", "Booklet: whole spread",
              [("Each page on its own", "pages"), ("Both pages as one image", "spread")],
              enabled_if=lambda st: st.crop.mode == "booklet",
              tip="Booklet mode, when both pages are on the glass: save each page as its own image, or "
                  "the whole spread as one image (spine inside).  Covers and single pages are one image "
                  "either way.  Crops end at the top page's edge: the edges of the pages underneath and "
                  "cover boards are left out."),
        Field("crop", "booklet_rtl", "Right page first (books read right to left)",
              enabled_if=lambda st: st.crop.mode == "booklet" and st.crop.booklet_output == "pages",
              tip="Booklet mode with both pages as separate images: which page is crop 1 (saved first). "
                  "Tick for books read right to left (Japanese booklets).  Left and right as "
                  "seen in the preview."),
        Field("crop", "sensitivity", "Sensitivity", lo=0, hi=1, step=0.05,
              tip="Higher finds fainter items, such as pale prints on a white lid"),
        Field("crop", "margin_mm", "Margin (mm)", lo=-10, hi=10, step=0.05,
              tip="Positive crops inside the edge, negative keeps a strip of background"),
        Field("crop", "min_area", "Smallest item (% of area)", lo=0.05, hi=50, step=0.1, scale=100.0),
        Field("crop", "rotate", "Rotation", [("None", 0), ("90° right", 1), ("180°", 2), ("90° left", 3)],
              tip="Turns the preview and every saved scan (the Rotate left / right buttons)."),
        Field("crop", "deskew", "Straighten crooked items"),
        Field("crop", "split_touching", "Separate items that touch"),
        Field("crop", "edge_refine", "Precise edges (full resolution)"),
        Field("crop", "border_search_mm", "White border search (mm)", lo=0, hi=40, step=1, decimals=0,
              tip="How far beyond the image to look for a print's white border"),
    ]),
    ("Filter", [
        Field("color", "descreen", "Descreen", [("Off", "off"), ("Auto (detect print screen)", "auto"), ("On", "on")],
              tip="Removes the dot pattern (moire) of printed artbooks and magazines"),
        Field("color", "dust_removal", "Remove dust",
              tip="Off by default: small highlights in illustrations can look like dust"),
        Field("color", "sharpen", "Sharpen", lo=0, hi=3, step=0.1),
        Field("color", "sharpen_radius", "Sharpen radius (px @300dpi)", lo=0.3, hi=5, step=0.1),
    ]),
    ("Colour", [
        Field("color", "auto_levels", "Auto levels"),
        Field("color", "black_clip", "Black clip (%)", lo=0, hi=5, step=0.01),
        Field("color", "white_clip", "White clip (%)", lo=0, hi=5, step=0.01),
        Field("color", "levels_per_channel", "Levels per channel (remove casts)"),
        Field("color", "restore_fading", "Restore faded colours"),
        Field("color", "white_balance", "White balance", [("None", "none"), ("Auto", "auto"), ("Grey world", "gray_world")]),
        Field("color", "brightness", "Brightness", lo=0.2, hi=3, step=0.05),
        Field("color", "contrast", "Contrast", lo=-1, hi=1, step=0.05),
        Field("color", "saturation", "Saturation", lo=0, hi=3, step=0.05),
        Field("color", "output_mode", "Output", [("Colour", "color"), ("Grey", "gray"), ("Black & white", "bw")]),
        Field("color", "bits", "Output bits per channel", [("8 (24-bit colour)", 8), ("16 (48-bit colour, TIFF/PNG)", 16)]),
    ]),
    ("Output", [
        Field("output", "folder", "Folder", kind="folder"),
        Field("output", "prefix", "File name prefix"),
        Field("output", "format", "Format", [("JPEG", "jpg"), ("PNG", "png"), ("TIFF", "tif"), ("WebP", "webp"), ("PDF", "pdf")]),
        Field("output", "jpeg_quality", "JPEG / WebP quality", lo=10, hi=100, step=1, decimals=0),
        Field("output", "tiff_compression", "TIFF compression", [("LZW", "lzw"), ("Deflate", "deflate"), ("None", "none")]),
        Field("output", "multipage_pdf", "All items of a scan in one PDF"),
        Field("output", "raw_file", "Raw file", [("Off", "off"), ("Instead of the processed file", "only"),
                                                 ("Also, next to it (*_raw.tif)", "also")],
              tip="The cropped scan exactly as the scanner delivered it: no levels, white balance or other "
                  "colour processing, saved as TIFF at the scanner's bit depth.  Reprocess it later with "
                  "the file: scanner or 'scanner crop FILE --crop off'."),
    ]),
]


class SettingsPanel(QTabWidget):
    """Tabs of form fields that read from and write to an AppSettings."""

    changed = Signal(str, str)  # section, field name

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._getters: dict[tuple[str, str], Callable[[], Any]] = {}
        self._setters: dict[tuple[str, str], Callable[[Any], None]] = {}
        self._widgets: dict[tuple[str, str], QWidget] = {}
        self._labels: dict[tuple[str, str], QLabel] = {}
        self._fields: list[Field] = []
        self._loading = False
        self.device_combo = QComboBox()
        self.device_refresh = QPushButton("Refresh")
        # Skew of the selected crop box (as VueScan's Crop > Skew); not a saved setting.
        self.skew = SkewBox()
        self.skew.setRange(-45.0, 45.0)
        self.skew.setDecimals(2)
        self.skew.setSingleStep(0.1)
        self.skew.setSuffix("°")
        self.skew.setKeyboardTracking(False)  # apply when typing is done, not per digit
        self.skew.setToolTip("Angle of the selected crop box (else crop 1), in degrees; positive turns it "
                             "clockwise.  Left / Right (or Up / Down) arrow keys turn it by 0.1°.  You can also "
                             "turn a box with the round handle above it.")
        self.skew_label = QLabel("Skew")
        self.set_skew(None)
        for title, fields in TABS:
            page = QWidget()
            form = QFormLayout(page)
            if title == "Input":
                row = QHBoxLayout()
                row.addWidget(self.device_combo, 1)
                row.addWidget(self.device_refresh)
                form.addRow("Scanner", row)
            for f in fields:
                w = self._make(f)
                self._fields.append(f)
                lbl = QLabel(f.label) if not isinstance(w, QCheckBox) else None
                if lbl is not None:
                    self._labels[(f.section, f.name)] = lbl
                if f.tip:
                    w.setToolTip(f.tip)
                if lbl is None:
                    form.addRow(w)
                else:
                    form.addRow(lbl, w)
                if (f.section, f.name) == ("crop", "rotate"):
                    form.addRow(self.skew_label, self.skew)
            self.addTab(page, title)
        self.load()

    def _section(self, name: str):
        return getattr(self.settings, name)

    def _make(self, f: Field) -> QWidget:
        key = (f.section, f.name)
        cur = getattr(self._section(f.section), f.name)
        if f.choices:
            w = QComboBox()
            for label, val in f.choices:
                w.addItem(label, val)

            def get(w=w):
                return w.currentData()

            def put(v, w=w):
                i = w.findData(v)
                if i < 0:
                    w.addItem(str(v), v)
                    i = w.count() - 1
                w.setCurrentIndex(i)

            w.currentIndexChanged.connect(lambda *_: self._emit(key))
        elif isinstance(cur, bool):
            w = QCheckBox(f.label)
            get, put = w.isChecked, w.setChecked
            w.toggled.connect(lambda *_: self._emit(key))
        elif isinstance(cur, int):
            w = QSpinBox()
            w.setRange(int(f.lo), int(f.hi))
            w.setSingleStep(int(f.step))
            get, put = w.value, w.setValue
            w.valueChanged.connect(lambda *_: self._emit(key))
        elif isinstance(cur, float):
            w = QDoubleSpinBox()
            w.setDecimals(f.decimals)
            w.setRange(f.lo, f.hi)
            w.setSingleStep(f.step)

            def get(w=w, s=f.scale):
                return w.value() / s

            def put(v, w=w, s=f.scale):
                w.setValue(float(v) * s)

            w.valueChanged.connect(lambda *_: self._emit(key))
        else:
            edit = QLineEdit()
            get, put = edit.text, edit.setText
            edit.editingFinished.connect(lambda: self._emit(key))
            w = edit
            if f.kind == "folder":
                box = QWidget()
                row = QHBoxLayout(box)
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(edit, 1)
                btn = QPushButton("Browse...")
                row.addWidget(btn)

                def browse(*_, edit=edit):
                    # (clicked() passes its "checked" flag: keep it out of ``edit``.)
                    start = os.path.expanduser(edit.text() or "~")
                    d = QFileDialog.getExistingDirectory(self, "Output folder", start)
                    if d:
                        edit.setText(d)
                        self._emit(key)

                btn.clicked.connect(browse)
                w = box
        self._getters[key] = get
        self._setters[key] = put
        self._widgets[key] = w
        return w

    def _emit(self, key: tuple[str, str]) -> None:
        if self._loading:
            return
        setattr(self._section(key[0]), key[1], self._getters[key]())
        self._update_enabled()
        self.changed.emit(*key)

    def _update_enabled(self) -> None:
        """Grey out fields that do not apply with the current settings (e.g. booklet options)."""
        for f in self._fields:
            if f.enabled_if is not None:
                on = bool(f.enabled_if(self.settings))
                self._widgets[(f.section, f.name)].setEnabled(on)
                lbl = self._labels.get((f.section, f.name))
                if lbl is not None:
                    lbl.setEnabled(on)

    def load(self) -> None:
        """Copy settings into the widgets."""
        self._loading = True
        try:
            for key, put in self._setters.items():
                put(getattr(self._section(key[0]), key[1]))
        finally:
            self._loading = False
        self._update_enabled()

    def set_skew(self, angle: float | None) -> None:
        """Show the crop box's angle (None: no box, greyed out) without emitting a change."""
        self.skew.blockSignals(True)
        try:
            self.skew.setValue(0.0 if angle is None else round(float(angle), 2) + 0.0)
        finally:
            self.skew.blockSignals(False)
        self.skew.setEnabled(angle is not None)
        self.skew_label.setEnabled(angle is not None)

    def set_choices(self, section: str, name: str, choices: list[tuple[str, Any]]) -> None:
        """Replace the options of a combo box (e.g. sources offered by a scanner)."""
        w = self._widgets.get((section, name))
        if not isinstance(w, QComboBox):
            return
        self._loading = True
        try:
            w.clear()
            for label, val in choices:
                w.addItem(label, val)
            cur = getattr(self._section(section), name)
            i = w.findData(cur)
            w.setCurrentIndex(i if i >= 0 else 0)
        finally:
            self._loading = False
        if w.findData(getattr(self._section(section), name)) < 0 and w.count():
            setattr(self._section(section), name, w.currentData())
