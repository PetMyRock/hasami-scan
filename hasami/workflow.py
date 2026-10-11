"""Preview -> crop -> scan -> process -> save, shared by the CLI and the GUI.

Crop boxes are kept in *bed millimetres* so that they do not depend on the
resolution of the preview or the final scan.  The final scan covers only the
area around the boxes (when ``scan_crop_area_only`` is set) and the boxes are
re-refined on the high-resolution image before the items are cut out.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .core import autocrop
from .core.geometry import CropBox, extract
from .core.output import save_items, save_raw_items
from .core.processing import process
from .devices.base import (
    MM_PER_INCH,
    LiveImage,
    ProgressFn,
    Scanner,
    ScanRequest,
    ScannerError,
    ScanResult,
    open_device,
)
from .settings import AppSettings

StatusFn = Callable[[str], None]


def box_px_to_mm(box: CropBox, dpi: float, origin_mm: tuple[float, float] = (0.0, 0.0)) -> CropBox:
    """Image pixels (at ``dpi``, image starting at ``origin_mm``) -> bed mm."""
    f = MM_PER_INCH / dpi
    return replace(box, cx=(box.cx + 0.5) * f + origin_mm[0], cy=(box.cy + 0.5) * f + origin_mm[1],
                   w=box.w * f, h=box.h * f, meta=dict(box.meta))


def box_mm_to_px(box: CropBox, dpi: float, origin_mm: tuple[float, float] = (0.0, 0.0)) -> CropBox:
    """Bed mm -> pixels of an image scanned at ``dpi`` starting at ``origin_mm``."""
    f = dpi / MM_PER_INCH
    return replace(box, cx=(box.cx - origin_mm[0]) * f - 0.5, cy=(box.cy - origin_mm[1]) * f - 0.5,
                   w=box.w * f, h=box.h * f, meta=dict(box.meta))


@dataclass
class ScanOutcome:
    scan: ScanResult
    boxes: list[CropBox]  # in pixels of ``scan.image``
    items: list[np.ndarray]
    paths: list[Path] = field(default_factory=list)
    #: Raw files (cropped, no colour processing) when ``output.raw_file`` is on.
    raw_paths: list[Path] = field(default_factory=list)


class Session:
    """State that lives between a preview and the final scan."""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        from .devices import twain_backend

        twain_backend.use_settings(settings.input)  # changes count at once, not after a restart
        self.device: Scanner | None = None
        self.device_id: str = ""
        self.preview: ScanResult | None = None
        #: Crop boxes in bed millimetres.
        self.boxes_mm: list[CropBox] = []
        #: Called with each :class:`LiveImage` of a preview or scan (to show it coming in).
        self.live: Callable[[LiveImage], None] | None = None

    # ---------------------------------------------------------------- device
    def open(self, device_id: str | None = None) -> Scanner:
        device_id = device_id or self.settings.input.device or self._best_device()
        if self.device is not None and self.device_id == device_id:
            return self.device
        self.close()
        self.device = open_device(device_id)
        self.device_id = device_id
        return self.device

    @staticmethod
    def _best_device() -> str:
        """No scanner chosen yet: the best one found (see devices.choose)."""
        from .devices.base import list_devices
        from .devices.choose import NO_SCANNER, best

        d = best(list_devices())
        if d is None:
            raise ScannerError(NO_SCANNER)
        return d.id

    def close(self) -> None:
        if self.device is not None:
            self.device.close()
        self.device = None
        self.device_id = ""

    # ------------------------------------------------------------- profile
    def profile(self):
        """The scanner profile auto-crop uses: the one chosen in the Crop tab, or with "auto" the
        one matching the open scanner (default when none matches or no scanner is open)."""
        from .core import profiles

        name = self.settings.crop.profile
        if name and name != "auto":
            return profiles.get(name)
        info = self.device.info if self.device is not None else None
        return profiles.for_scanner(info.name, info.vendor, info.model) if info else profiles.DEFAULT

    def crop_settings(self):
        """The crop settings with the profile resolved (what detect / refine get)."""
        return replace(self.settings.crop, profile=self.profile().name)

    # ------------------------------------------------------------- preview
    def run_preview(self, progress: ProgressFn | None = None,
                    cancel: threading.Event | None = None) -> ScanResult:
        dev = self.open()
        inp = self.settings.input
        req = ScanRequest(dpi=inp.preview_dpi, mode="color", source=inp.source, preview=True, live=self.live)
        self.preview = dev.scan(req, progress, cancel)
        self.detect_on_preview()
        return self.preview

    def detect_on_preview(self) -> list[CropBox]:
        """(Re)run auto-crop on the current preview, keeping manual boxes."""
        if self.preview is None:
            return []
        p = self.preview
        found = autocrop.detect(p.image, self.crop_settings(), dpi=p.dpi)
        origin = (p.area_mm[0], p.area_mm[1])
        manual = [b for b in self.boxes_mm if b.manual]
        self.boxes_mm = manual + [box_px_to_mm(b, p.dpi, origin) for b in found]
        return self.boxes_mm

    def preview_boxes_px(self) -> list[CropBox]:
        if self.preview is None:
            return []
        p = self.preview
        return [box_mm_to_px(b, p.dpi, (p.area_mm[0], p.area_mm[1])) for b in self.boxes_mm]

    def set_preview_boxes_px(self, boxes: list[CropBox]) -> None:
        if self.preview is None:
            return
        p = self.preview
        self.boxes_mm = [box_px_to_mm(b, p.dpi, (p.area_mm[0], p.area_mm[1])) for b in boxes]

    # --------------------------------------------------------------- scan
    def _crop_boxes(self) -> list[CropBox]:
        """The boxes a scan is cropped to.  Any box counts, also with auto-crop off;
        then boxes drawn by hand replace the whole-bed box auto-crop "off" puts there."""
        if self.settings.crop.mode == "off" and any(b.manual for b in self.boxes_mm):
            return [b for b in self.boxes_mm if b.manual]
        return list(self.boxes_mm)

    def _scan_area(self) -> tuple[float, float, float, float] | None:
        st = self.settings
        crops = self._crop_boxes()
        if not crops or not st.input.scan_crop_area_only:
            return None
        pad = 3.0
        xs, ys = [], []
        for b in crops:
            x0, y0, x1, y1 = b.bounding_rect()
            xs += [x0, x1]
            ys += [y0, y1]
        caps = self.open().capabilities()
        bw, bh = caps.max_area_mm
        x0, y0 = max(0.0, min(xs) - pad), max(0.0, min(ys) - pad)
        x1, y1 = min(bw, max(xs) + pad), min(bh, max(ys) + pad)
        return (x0, y0, x1 - x0, y1 - y0)

    def run_scan(self, progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                 status: StatusFn | None = None, save: bool = True) -> list[ScanOutcome]:
        """Scan at full resolution, crop, process and save.

        Returns one outcome per page (several when scanning from a feeder).
        """
        st = self.settings
        dev = self.open()
        area = self._scan_area()
        raw = st.output.raw_file in ("only", "also")
        req = ScanRequest(dpi=st.input.scan_dpi, mode=st.input.mode, source=st.input.source,
                          area_mm=area, bit_depth=st.input.bit_depth, raw=raw, live=self.live)
        outcomes = []
        crops = self._crop_boxes()
        use_preview_boxes = bool(crops)
        for page_no, result in enumerate(dev.scan_batch(req, progress, cancel)):
            if status:
                h, w = result.image.shape[:2]
                status(f"Scanned {result.area_mm[2]:.0f} x {result.area_mm[3]:.0f} mm at {result.dpi:.0f} dpi "
                       f"({w} x {h} px). {result.note}".strip())
                status(f"Cropping page {page_no + 1}...")
            if use_preview_boxes and page_no == 0:
                origin = (result.area_mm[0], result.area_mm[1])
                boxes = [box_mm_to_px(b, result.dpi, origin) for b in crops]
                if st.crop.edge_refine:
                    boxes = autocrop.refine(result.image, boxes, self.crop_settings(), dpi=result.dpi)
            else:
                boxes = autocrop.detect(result.image, self.crop_settings(), dpi=result.dpi)
            outcomes.append(self._finish(result, boxes, save, status))
            if cancel is not None and cancel.is_set():
                break
        return outcomes

    def crop_image(self, image: np.ndarray, dpi: float, save: bool = True,
                   status: StatusFn | None = None) -> ScanOutcome:
        """Auto-crop, process and save an image that is already scanned."""
        result = ScanResult(image, dpi, (0.0, 0.0, image.shape[1] / dpi * MM_PER_INCH,
                                         image.shape[0] / dpi * MM_PER_INCH))
        boxes = autocrop.detect(image, self.crop_settings(), dpi=dpi)
        return self._finish(result, boxes, save, status)

    def _finish(self, result: ScanResult, boxes: list[CropBox], save: bool,
                status: StatusFn | None) -> ScanOutcome:
        st = self.settings
        raw_mode = st.output.raw_file if st.output.raw_file in ("only", "also") else "off"
        if raw_mode != "off" and status and st.input.bit_depth == 16 and result.image.dtype == np.uint8:
            status("The scanner delivered 8 bits per channel (24-bit colour), not 16; the raw file has 8 bits.")
        items, raws = [], []
        for i, b in enumerate(boxes):
            if status:
                status(f"Processing item {i + 1} of {len(boxes)}...")
            crop = extract(result.image, b)  # cropped and straightened, colours untouched
            turns = int(st.crop.rotate) % 4
            if turns:
                crop = np.ascontiguousarray(np.rot90(crop, -turns))  # clockwise
            if raw_mode != "off":
                raws.append(crop)
            if raw_mode != "only":
                items.append(process(crop, st.color, dpi=result.dpi))
        paths, raw_paths = [], []
        if save and items:
            paths = save_items(items, st.output, result.dpi)
        if save and raws:
            raw_paths = save_raw_items(raws, st.output, result.dpi, beside=paths if raw_mode == "also" else None)
        if status and (paths or raw_paths):
            status(f"Saved {len(paths) + len(raw_paths)} file(s) to {st.output.path}")
        if raw_mode == "only":
            return ScanOutcome(result, boxes, raws, raw_paths, raw_paths)
        return ScanOutcome(result, boxes, items, paths, raw_paths)
