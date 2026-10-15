"""File > Share crop cases...: pick saved crop cases, zip them for the Discord server, or delete them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QVBoxLayout)

from .. import cases
from ..core.geometry import CropBox
from .preview import to_qimage

DISCORD = "https://discord.gg/TAAV8B9WWg"
NOTE = ("Each case contains scans of your item (the preview and the scanned area). Share only what you are happy "
        f"to share. Post the zips in the tuning channel on the Hasami Scan Discord: {DISCORD}")
THUMB = 160


class CasesDialog(QDialog):
    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Share crop cases")
        self.resize(640, 560)
        self.folder = Path(folder)
        self.list = QListWidget()
        self.list.setIconSize(QSize(THUMB, THUMB))
        note = QLabel(NOTE.replace(DISCORD, f'<a href="{DISCORD}">{DISCORD}</a>'))
        note.setWordWrap(True)
        note.setTextFormat(Qt.RichText)
        note.setOpenExternalLinks(True)
        zip_btn, del_btn, open_btn, close_btn = (QPushButton(t) for t in
                                                 ("Create zip", "Delete", "Open folder", "Close"))
        zip_btn.clicked.connect(lambda *_: self.create_zips())
        del_btn.clicked.connect(lambda *_: self.delete_checked())
        open_btn.clicked.connect(lambda *_: self._open(self.folder))
        close_btn.clicked.connect(lambda *_: self.accept())
        row = QHBoxLayout()
        for b in (zip_btn, del_btn, open_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close_btn)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addWidget(note)
        lay.addLayout(row)
        self.reload()

    def reload(self) -> None:
        self.list.clear()
        for path in cases.list_cases(self.folder):
            try:
                doc = cases.read_case(path)
                scanner = doc["scanner"].get("name") or doc["scanner"].get("id") or "?"
                text = f"{path.name}  ·  {scanner}  ·  {doc['reason']}\n{cases.summary(doc['edits'], doc.get('crop', {}).get('rotate', 0))}"
                icon = QIcon(self._thumb(path, doc))
            except (ValueError, OSError, KeyError) as e:
                text, icon = f"{path.name}  ·  unreadable ({e})", QIcon()
            it = QListWidgetItem(icon, text)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, str(path))
            self.list.addItem(it)

    @staticmethod
    def _thumb(path: Path, doc: dict) -> QPixmap:
        """The preview with the boxes scanned drawn on it."""
        with Image.open(path / doc["preview"]["file"]) as im:
            img = np.asarray(im.convert("RGB"))
        pm = QPixmap.fromImage(to_qimage(img))
        f = float(doc["preview"]["dpi"]) / 25.4
        x0, y0 = doc["preview"]["area_mm"][:2]
        p = QPainter(pm)
        p.setPen(QPen(QColor(255, 140, 0), max(2, pm.width() // 150)))
        for b in doc["final"]:
            pts = (CropBox(*b["box_mm"]).corners() - [x0, y0]) * f - 0.5
            p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in pts]))
        p.end()
        return pm.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def checked(self) -> list[Path]:
        return [Path(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def create_zips(self) -> list[Path]:
        sel = self.checked()
        if not sel:
            return []
        out = self.folder / cases.SHARE_DIR
        zips = cases.pack(sel, out)
        self._open(out)
        return zips

    def delete_checked(self) -> list[Path]:
        sel = self.checked()
        if not sel or not self._confirm(len(sel)):
            return []
        done, kept = [], []
        for p in sel:
            try:
                cases.delete_case(p)
                done.append(p)
            except OSError:
                kept.append(p.name)
        self.reload()
        if kept:
            QMessageBox.warning(self, "Delete crop cases",
                                "Not deleted (other files are in the folder, or a file is in use):\n"
                                + "\n".join(kept))
        return done

    def _confirm(self, n: int) -> bool:
        return QMessageBox.question(self, "Delete crop cases",
                                    f"Delete {n} crop case{'s' if n > 1 else ''}? This cannot be undone.") \
            == QMessageBox.Yes

    def _open(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
