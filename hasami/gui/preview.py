"""Preview area: the scanned bed with editable, rotatable crop boxes."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
)

from ..core.geometry import CropBox

ACCENT = QColor(255, 72, 120)
ACCENT_FILL = QColor(255, 72, 120, 40)
LOW_CONF = QColor(255, 170, 0)
HANDLE = 9.0  # handle size in screen pixels
BLANK = (38, 38, 44)  # an empty preview: the view's own background
PLACEHOLDER = (128, 128, 128)  # where a preview or scan is about to come in (VueScan: grey)
EDGE = 6.0  # how close (screen pixels) to a box's edge a click resizes instead of moving


def to_qimage(img: np.ndarray) -> QImage:
    """RGB/grey uint8 or uint16 numpy array -> QImage (deep copy)."""
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / 257.0).astype(np.uint8)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        return QImage(img.data, w, h, img.strides[0], QImage.Format_Grayscale8).copy()
    return QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888).copy()


def view_icon(kind: str, colour: QColor, size: int = 64) -> QIcon:
    """Line icons for the buttons under the preview: "rotate_left", "rotate_right",
    "zoom_in", "zoom_out" and "fit" (drawn, so they follow the theme's colour)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 32, size / 32)
    pen = QPen(colour, 2.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    if kind.startswith("rotate"):
        if kind == "rotate_right":
            p.translate(32, 0)
            p.scale(-1, 1)  # the mirror image of the anticlockwise arrow
        # An arc over the top, turning anticlockwise, the arrowhead at its lower-left end.
        r, cx, cy, start, span = 10.0, 16.0, 16.0, -30.0, 245.0
        p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), int(start * 16), int(span * 16))
        a = math.radians(start + span)
        end = QPointF(cx + r * math.cos(a), cy - r * math.sin(a))
        tang = QPointF(-math.sin(a), -math.cos(a))  # direction of travel on screen
        norm = QPointF(math.cos(a), -math.sin(a))
        p.setBrush(colour)
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([end + tang * 6, end + norm * 4.5, end - norm * 4.5]))
    elif kind.startswith("zoom"):
        p.drawEllipse(QPointF(13, 13), 8.5, 8.5)
        p.drawLine(QPointF(19.5, 19.5), QPointF(27, 27))
        p.drawLine(QPointF(9, 13), QPointF(17, 13))
        if kind == "zoom_in":
            p.drawLine(QPointF(13, 9), QPointF(13, 17))
    elif kind == "fit":
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x, y = 16 - 11 * sx, 16 - 9 * sy
            p.drawPolyline(QPolygonF([QPointF(x, y + 6 * sy), QPointF(x, y), QPointF(x + 6 * sx, y)]))
    p.end()
    return QIcon(pm)


class CropItem(QGraphicsObject):
    """A rotated crop rectangle with move, resize and rotate handles."""

    changed = Signal()

    #: Quarter turns (clockwise) the view is rotated by: labels stay upright.
    view_turns = 0

    def __init__(self, box: CropBox, index: int):
        super().__init__()
        self.box = box
        self.index = index
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self._drag = None

    # --------------------------------------------------------------- geometry
    def _scale(self) -> float:
        views = self.scene().views() if self.scene() else []
        if not views:
            return 1.0
        t = views[0].transform()
        return 1.0 / max(math.hypot(t.m11(), t.m12()), 1e-6)  # also when the view is rotated

    def _poly(self) -> QPolygonF:
        return QPolygonF([QPointF(x, y) for x, y in self.box.corners()])

    def _handles(self) -> dict[str, QPointF]:
        c = self.box.corners()
        ux, uy = self.box.axes
        top_mid = (c[0] + c[1]) / 2
        rot = top_mid - uy * (28 * self._scale())
        pts = {f"c{i}": QPointF(*c[i]) for i in range(4)}
        # Side handles: top, right, bottom, left (move that side only).
        for i, (a, b) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
            pts[f"s{i}"] = QPointF(*((c[a] + c[b]) / 2))
        pts["rot"] = QPointF(*rot)
        return pts

    def boundingRect(self) -> QRectF:
        x0, y0, x1, y1 = self.box.bounding_rect()
        pad = 40 * self._scale()
        return QRectF(x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addPolygon(self._poly())
        s = HANDLE * self._scale()
        for p in self._handles().values():
            path.addEllipse(p, s, s)
        # A band along the outline, so that the edges can be grabbed from outside too.
        band = QPainterPathStroker()
        band.setWidth(2 * EDGE * self._scale())
        return path.united(band.createStroke(path))

    def _hit(self, pos: QPointF) -> str:
        """What a click at ``pos`` grabs: a corner "c0".."c3", a side "s0".."s3"
        (top, right, bottom, left; anywhere along it), "rot" or "move"."""
        sc = self._scale()
        p = np.array([pos.x(), pos.y()])
        handles = self._handles()
        if self.isSelected():
            r = handles["rot"]
            if math.hypot(r.x() - p[0], r.y() - p[1]) <= HANDLE * 1.3 * sc:
                return "rot"
        c = self.box.corners()
        grab = max(HANDLE * 1.3, EDGE * 1.5) * sc
        dists = [float(np.hypot(*(p - c[i]))) for i in range(4)]
        i = int(np.argmin(dists))
        if dists[i] <= grab:
            return f"c{i}"
        best, side = EDGE * sc, None
        for i, (a, b) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
            seg = c[b] - c[a]
            t = float(np.clip((p - c[a]) @ seg / max(float(seg @ seg), 1e-9), 0, 1))
            d = float(np.hypot(*(p - (c[a] + t * seg))))
            if d <= best:
                best, side = d, f"s{i}"
        return side or "move"

    def hoverMoveEvent(self, event) -> None:
        kind = self._hit(event.pos())
        if kind == "move":
            self.setCursor(Qt.SizeAllCursor)
        elif kind == "rot":
            self.setCursor(Qt.CrossCursor)
        else:
            # Which way the side or corner runs on screen (the view may be turned).
            i = int(kind[1])
            turned = CropItem.view_turns % 2
            if kind[0] == "s":
                vertical = (i % 2 == 0) != bool(turned)
                self.setCursor(Qt.SizeVerCursor if vertical else Qt.SizeHorCursor)
            else:
                diag = (i % 2 == 0) != bool(turned)
                self.setCursor(Qt.SizeFDiagCursor if diag else Qt.SizeBDiagCursor)
        super().hoverMoveEvent(event)

    # ---------------------------------------------------------------- drawing
    def paint(self, painter: QPainter, option, widget=None) -> None:
        s = self._scale()
        colour = ACCENT if self.box.confidence >= 0.5 or self.box.manual else LOW_CONF
        pen = QPen(colour, 2.0 * s)
        if self.box.manual:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(ACCENT_FILL if self.isSelected() else Qt.NoBrush))
        painter.drawPolygon(self._poly())
        c = self.box.corners()
        font = QFont()
        font.setPointSizeF(max(1.0, 11 * s))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(colour))
        # The number goes in the corner that is top-left on screen, upright.
        turns = CropItem.view_turns % 4
        corner = c[(4 - turns) % 4]
        painter.save()
        painter.translate(QPointF(*corner))
        painter.rotate(-90 * turns)
        painter.drawText(QPointF(6 * s, 16 * s), str(self.index + 1))
        painter.restore()
        if self.isSelected():
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(colour, 1.5 * s))
            hs = HANDLE * s * 0.6
            h = self._handles()
            top_mid = QPointF(*((c[0] + c[1]) / 2))
            painter.drawLine(top_mid, h["rot"])
            for p in h.values():
                painter.drawEllipse(p, hs, hs)

    # ---------------------------------------------------------------- editing
    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        # Edges and corners resize even when the box is not selected yet.
        self._drag = (self._hit(pos), pos, self.box)
        super().mousePressEvent(event)
        self.setSelected(True)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            return
        kind, start, b0 = self._drag
        pos = event.pos()
        dx, dy = pos.x() - start.x(), pos.y() - start.y()
        self.prepareGeometryChange()
        if kind == "move":
            nb = b0.translated(dx, dy)
        elif kind == "rot":
            # The handle sits above the top edge, i.e. at -90 degrees in the box frame.
            ang = math.degrees(math.atan2(pos.y() - b0.cy, pos.x() - b0.cx)) + 90.0
            nb = CropBox(b0.cx, b0.cy, b0.w, b0.h, ang, b0.confidence)
            if abs(nb.angle) < 0.5:  # snap to straight
                nb = CropBox(b0.cx, b0.cy, nb.w, nb.h, 0.0, b0.confidence)
        elif kind[0] == "s":
            # One side: the opposite side stays where it is.
            i = int(kind[1])
            ux, uy = b0.axes
            axis, sign = ((uy, -1), (ux, 1), (uy, 1), (ux, -1))[i]
            d = float((np.array([pos.x(), pos.y()]) - np.array([b0.cx, b0.cy])) @ axis) * sign
            half = (b0.h if axis is uy else b0.w) / 2
            size = max(4.0, d + half)  # from the fixed side to the dragged one
            shift = (size - 2 * half) / 2 * sign
            c = np.array([b0.cx, b0.cy]) + axis * shift
            w, h = (b0.w, size) if axis is uy else (size, b0.h)
            nb = CropBox(c[0], c[1], w, h, b0.angle, b0.confidence)
        else:
            # Resize: the opposite corner stays fixed, in the box's own frame.
            i = int(kind[1])
            corners = b0.corners()
            fixed = corners[(i + 2) % 4]
            ux, uy = b0.axes
            d = np.array([pos.x(), pos.y()]) - fixed
            u, v = float(d @ ux), float(d @ uy)
            w, h = max(4.0, abs(u)), max(4.0, abs(v))
            c = fixed + ux * (math.copysign(w, u) / 2) + uy * (math.copysign(h, v) / 2)
            nb = CropBox(c[0], c[1], w, h, b0.angle, b0.confidence)
        nb.manual = True
        nb.meta = dict(b0.meta)  # keep e.g. the crop's own rotation
        self.box = nb
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        super().mouseReleaseEvent(event)


START_HINT = ("1.  Put the item face down on the glass and close the lid.\n"
              "2.  Press Preview.\n"
              "3.  Check the crop boxes (drag their edges to adjust), then press Scan.\n\n"
              "Or File > Open image to crop a scan you already have.")


class PreviewView(QGraphicsView):
    """Zoomable preview with crop boxes; drag on empty space to add a box."""

    boxes_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(38, 38, 44)))
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.items_: list[CropItem] = []
        self._new_box_start: QPointF | None = None
        self._fitted = True  # follow the window size until the user zooms
        self._pan: QPointF | None = None
        self._live: QGraphicsPixmapItem | None = None  # a scan coming in
        self._old_pixmap: QPixmap | None = None  # the picture before a scan blanked it
        self._placeholder_item = None  # grey: the area a preview or scan will fill
        self._zoomed_live = False  # the view shows just the area being scanned
        self._clip_live = False  # a scan of one crop box: only the box is shown
        self._live_box: CropBox | None = None
        self._front = None  # clips the strips to the rows of the box received so far
        self._hint = ""  # shown instead of START_HINT (e.g. no scanner found)
        self._placeholder()

    def set_hint(self, text: str) -> None:
        """Text for the empty preview ("" = how to start)."""
        if text != self._hint:
            self._hint = text
            if self.pixmap_item is None and self._placeholder_item is None:
                self._placeholder()

    def _placeholder(self) -> None:
        sc = self.scene()
        sc.clear()
        self._live = self._old_pixmap = self._placeholder_item = None
        self._zoomed_live = self._clip_live = False
        self._front = None
        self.pixmap_item = None
        self.items_ = []
        t = sc.addText(self._hint or START_HINT)
        t.setTextWidth(520)
        t.setDefaultTextColor(QColor(200, 200, 210))
        t.setTransformOriginPoint(t.boundingRect().center())
        t.setRotation(-90 * CropItem.view_turns)  # upright in a turned view

    # ------------------------------------------------------------------ data
    def set_image(self, img: np.ndarray) -> None:
        sc = self.scene()
        sc.clear()
        self._live = self._old_pixmap = self._placeholder_item = None
        self._zoomed_live = self._clip_live = False
        self._front = None
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.items_ = []
        self.pixmap_item = sc.addPixmap(QPixmap.fromImage(to_qimage(img)))
        self.pixmap_item.setZValue(0)
        sc.setSceneRect(QRectF(self.pixmap_item.boundingRect()))
        self.fit()

    def show_live(self, img: np.ndarray, x: float, y: float, scale: float = 1.0,
                  full: tuple[int, int] | None = None) -> None:
        """Draw the part of a scan received so far over the preview.

        ``img`` goes at ``(x, y)`` in preview pixels, ``scale`` times its size
        (below the crop boxes).  ``full`` = (width, height) of a preview being
        scanned: when the current preview has another size it is replaced by
        a blank one of that size first.
        """
        if full is not None:
            cur = self.pixmap_item.pixmap().size() if self.pixmap_item is not None else None
            if cur is None or (cur.width(), cur.height()) != tuple(full):
                self.set_image(np.full((full[1], full[0], 3), BLANK, np.uint8))
        if self.pixmap_item is None:
            return
        if self._live is None:
            if self._clip_live:  # inside the grey crop box, cut to its shape
                self._front = QGraphicsPolygonItem(self._placeholder_item)
                self._front.setPen(QPen(Qt.NoPen))
                self._front.setFlag(QGraphicsItem.ItemClipsChildrenToShape, True)
                self._live = QGraphicsPixmapItem(self._front)
            else:
                self._live = self.scene().addPixmap(QPixmap())
                self._live.setZValue(1)
            self._live.setTransformationMode(Qt.SmoothTransformation)
        self._live.setPixmap(QPixmap.fromImage(to_qimage(img)))
        self._live.setPos(x, y)
        self._live.setScale(scale)
        if self._clip_live and self._live_box is not None:
            self._front.setPolygon(self._received(self._live_box, y + img.shape[0] * scale))

    @staticmethod
    def _received(box: CropBox, y_end: float) -> QPolygonF:
        """The part of a turned crop box whose rows (in the box's own direction) the
        scanner has finished, up to bed row ``y_end``: the edge where the scan comes
        in stays straight in the straightened view, not slanted along the bed's rows."""
        t = math.radians(box.angle)
        ux, uy = box.axes
        hw, hh = box.w / 2, box.h / 2
        # A row v of the box is complete when its lowest point is above y_end.
        v = (y_end - box.cy - hw * abs(math.sin(t))) / max(math.cos(t), 1e-6)
        v = min(max(v, -hh), hh)
        c = np.array([box.cx, box.cy])
        pts = [c - hw * ux - hh * uy, c + hw * ux - hh * uy, c + hw * ux + v * uy, c - hw * ux + v * uy]
        return QPolygonF([QPointF(float(px), float(py)) for px, py in pts])

    def ensure_size(self, w: int, h: int) -> None:
        """Make the preview (w, h) pixels: a blank one if the current one differs."""
        cur = self.pixmap_item.pixmap().size() if self.pixmap_item is not None else None
        if self._old_pixmap is not None:
            cur = self._old_pixmap.size()
        if cur is None or (cur.width(), cur.height()) != (w, h):
            boxes = self.boxes()
            self.set_image(np.full((h, w, 3), BLANK, np.uint8))
            self.set_boxes(boxes)

    def begin_live(self) -> None:
        """A preview starts: the old picture and the crop boxes go (like VueScan);
        :meth:`live_area` then shows where the new one will come in."""
        if self.pixmap_item is None:
            return
        for it in (self._live, self._front):
            if it is not None and it.scene() is not None:
                self.scene().removeItem(it)
        self._live = self._front = None
        if self._old_pixmap is None:
            self._old_pixmap = self.pixmap_item.pixmap()
        blank = QPixmap(self._old_pixmap.size())
        blank.fill(QColor(*BLANK))
        self.pixmap_item.setPixmap(blank)
        for it in self.items_:  # the crop boxes are hidden while the scan comes in (maintainer)
            it.setVisible(False)

    def live_area(self, x: float, y: float, w: float, h: float, zoom: bool = False,
                  box: CropBox | None = None) -> None:
        """The first strip of a preview or scan arrived: grey placeholder over the area
        it will fill (as in VueScan); ``zoom`` shows just that area (a scan of a crop).
        With ``box`` (the one crop box being scanned) only that box is shown, turned
        straight, the way it will be saved (VueScan does the same)."""
        self.begin_live()
        if self.pixmap_item is None:
            return
        if self._placeholder_item is None:
            self._placeholder_item = self.scene().addPolygon(QPolygonF(), QPen(Qt.NoPen),
                                                             QBrush(QColor(*PLACEHOLDER)))
            self._placeholder_item.setZValue(0.5)
        if box is not None:
            self._placeholder_item.setPolygon(QPolygonF([QPointF(px, py) for px, py in box.corners()]))
        else:
            self._placeholder_item.setPolygon(QPolygonF(QRectF(x, y, w, h)))
        self._clip_live = box is not None
        self._live_box = box
        self._placeholder_item.setFlag(QGraphicsItem.ItemClipsChildrenToShape, self._clip_live)
        if zoom:
            # No scroll bars while zoomed in: they would take room after the fit and
            # push part of the area out of sight.
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            if box is None:
                self.fitInView(QRectF(x, y, w, h), Qt.KeepAspectRatio)
            else:
                self._show_straight(box)
            self._fitted = False
            self._zoomed_live = True

    def _show_straight(self, box: CropBox) -> None:
        """Turn and zoom the view so ``box`` fills it, upright (plus the view's quarter turns)."""
        turns = CropItem.view_turns
        bw, bh = (box.h, box.w) if turns % 2 else (box.w, box.h)
        vp = self.viewport().rect()
        s = max(min((vp.width() - 4) / max(bw, 1.0), (vp.height() - 4) / max(bh, 1.0)), 1e-3)
        self.resetTransform()
        self.rotate(90 * turns - box.angle)
        self.scale(s, s)
        # Room to centre the box even at the edge of the bed (the scene ends there).
        m = max(vp.width(), vp.height()) / s + max(box.w, box.h)
        r = QRectF(box.cx - m, box.cy - m, 2 * m, 2 * m)
        self.setSceneRect(r.united(self.scene().sceneRect()))
        self.centerOn(box.cx, box.cy)

    def end_live(self) -> None:
        """Put the picture from before the scan back, with its crop boxes."""
        if self._old_pixmap is not None and self.pixmap_item is not None:
            self.pixmap_item.setPixmap(self._old_pixmap)
        if self._live is not None and self._live.parentItem() is None:
            self.scene().removeItem(self._live)
        if self._placeholder_item is not None:
            self.scene().removeItem(self._placeholder_item)  # with the strips inside it
        self._live = self._old_pixmap = self._placeholder_item = None
        self._clip_live = False
        self._front = None
        for it in self.items_:
            it.setVisible(True)
        if self._zoomed_live:
            self._zoomed_live = False
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setSceneRect(QRectF())  # the scene's own again
            self._fitted = True
            self.set_rotation(CropItem.view_turns)  # straight again, fitted
            QTimer.singleShot(0, self._refit)  # again once the scroll bars are gone

    def set_boxes(self, boxes: list[CropBox]) -> None:
        for it in self.items_:
            self.scene().removeItem(it)
        self.items_ = []
        for i, b in enumerate(boxes):
            self._add_item(b)

    def _add_item(self, box: CropBox) -> CropItem:
        it = CropItem(box, len(self.items_))
        it.changed.connect(self.boxes_changed.emit)
        self.scene().addItem(it)
        self.items_.append(it)
        return it

    def boxes(self) -> list[CropBox]:
        return [it.box for it in self.items_]

    def selected_index(self) -> int | None:
        for i, it in enumerate(self.items_):
            if it.isSelected():
                return i
        return None

    def current_index(self) -> int | None:
        """The selected crop box, else the first one (None without boxes)."""
        i = self.selected_index()
        return i if i is not None else (0 if self.items_ else None)

    def set_box_angle(self, i: int, angle: float) -> None:
        """Turn crop box ``i`` about its centre to ``angle`` degrees (Skew field)."""
        it = self.items_[i]
        b0 = it.box
        nb = CropBox(b0.cx, b0.cy, b0.w, b0.h, float(angle), b0.confidence)
        nb.manual = True
        nb.meta = dict(b0.meta)
        it.prepareGeometryChange()
        it.box = nb
        it.update()
        self.boxes_changed.emit()

    def set_rotation(self, turns: int) -> None:
        """Show the preview turned by quarter turns clockwise (like the saved scans)."""
        turns = int(turns) % 4
        CropItem.view_turns = turns
        s = math.hypot(self.transform().m11(), self.transform().m12()) or 1.0
        self.resetTransform()
        self.rotate(90 * turns)
        if self.pixmap_item is None:
            for it in self.scene().items():  # the hint text stays upright
                it.setRotation(-90 * turns)
        if self._fitted:
            self.fit()
        else:
            self.scale(s, s)
        for it in self.items_:
            it.update()

    def delete_selected(self) -> None:
        keep = [it for it in self.items_ if not it.isSelected()]
        if len(keep) == len(self.items_):
            return
        for it in self.items_:
            if it.isSelected():
                self.scene().removeItem(it)
        self.items_ = keep
        for i, it in enumerate(self.items_):
            it.index = i
            it.update()
        self.boxes_changed.emit()

    # ------------------------------------------------------------------ view
    def fit(self) -> None:
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._fitted = True

    def _refit(self) -> None:
        if self._fitted:
            self.fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Only while showing the whole image: zooming in brings up the
        # scroll bars, which resizes the view, and must not undo the zoom.
        if self._fitted:
            self.fit()

    def zoom(self, factor: float) -> None:
        """Zoom by ``factor`` (between fitting the image and 32 screen pixels per pixel)."""
        if self.pixmap_item is None:
            return
        cur = math.hypot(self.transform().m11(), self.transform().m12())
        r = self.pixmap_item.boundingRect()
        rw, rh = (r.height(), r.width()) if CropItem.view_turns % 2 else (r.width(), r.height())
        vp = self.viewport().rect()
        fit = min(vp.width() / max(rw, 1), vp.height() / max(rh, 1))
        new = min(max(cur * factor, fit), 32.0)
        if new <= fit * 1.0001:
            self.fit()
            return
        self._fitted = False
        self.scale(new / cur, new / cur)

    def zoom_in(self) -> None:
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.zoom(1.25)
        self.setTransformationAnchor(anchor)

    def zoom_out(self) -> None:
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.zoom(1 / 1.25)
        self.setTransformationAnchor(anchor)

    def wheelEvent(self, event) -> None:
        dy = event.angleDelta().y()
        if dy:
            self.zoom(1.15 if dy > 0 else 1 / 1.15)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            # Pan a zoomed-in preview by dragging with the right or middle button.
            self._pan = event.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        item = self.itemAt(event.position().toPoint())
        if self.pixmap_item is not None and (item is None or item is self.pixmap_item) and event.button() == Qt.LeftButton:
            self.scene().clearSelection()
            self._new_box_start = self.mapToScene(event.position().toPoint())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan is not None:
            d = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._new_box_start is not None:
            p0 = self._new_box_start
            p1 = self.mapToScene(event.position().toPoint())
            w, h = abs(p1.x() - p0.x()), abs(p1.y() - p0.y())
            if w > 6 and h > 6:
                box = CropBox.from_xywh(min(p0.x(), p1.x()), min(p0.y(), p1.y()), w, h, manual=True)
                box.meta["drawn"] = True  # drawn from scratch: kept by the next preview
                if getattr(self, "_drawing", None) is None:
                    self._drawing = self._add_item(box)
                    self._drawing.setSelected(True)
                else:
                    self._drawing.prepareGeometryChange()
                    self._drawing.box = box
                    self._drawing.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan is not None:
            self._pan = None
            self.viewport().unsetCursor()
            return
        if self._new_box_start is not None:
            self._new_box_start = None
            if getattr(self, "_drawing", None) is not None:
                self._drawing = None
                self.boxes_changed.emit()
            return
        super().mouseReleaseEvent(event)


class CroppedView(QLabel):
    """The "Cropped area" tab (as in VueScan): only what is inside one crop box,
    straightened, turned and with the colour settings applied, as large as fits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("QLabel { background: #26262c; color: #c8c8d2; }")
        self._pm: QPixmap | None = None
        self.show_image(None, "Press Preview, then click a crop box to see it here.")

    def show_image(self, img: np.ndarray | None, text: str = "") -> None:
        self._pm = QPixmap.fromImage(to_qimage(img)) if img is not None else None
        self.setToolTip(text if img is not None else "")
        if self._pm is None:
            self.setPixmap(QPixmap())
            self.setText(text)
        else:
            self.setText("")
            self._rescale()

    def _rescale(self) -> None:
        if self._pm is not None and not self._pm.isNull():
            self.setPixmap(self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refit(self) -> None:
        if self._fitted:
            self.fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()
