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

from . import cases
from .core import autocrop, profiles, sidelines
from .core.geometry import SIDE_NAMES, CropBox, extract
from .core.output import next_filenames, save_image, save_items, save_raw_items
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
        #: The boxes auto-crop proposed (bed mm, with meta["auto_id"]): what a crop case compares against.
        self.auto_mm: list[CropBox] = []
        self._auto_seq = 0
        #: Booklet mode: the candidate lines around each auto box (by auto id, then side name; bed mm),
        #: what a dragged side snaps to.
        self.snap_lines: dict[str, dict[str, list[sidelines.SnapLine]]] = {}
        #: The last scan's crop case (crop cases on), its folder once written, and its status line.
        self.last_case: cases.CaseData | None = None
        self.last_case_path: Path | None = None
        self.case_note = ""
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
        # A new preview (usually the next page): boxes auto-crop found are replaced, also when
        # edited by hand (dragged, Skew); only boxes drawn from scratch stay.
        self.boxes_mm = [b for b in self.boxes_mm if b.meta.get("drawn")]
        self.last_case, self.last_case_path = None, None  # "Save this case" means the page on screen
        for b in self.boxes_mm:
            b.meta["carried"] = True  # from an earlier page: not a correction of this one
        self.snap_lines = {}  # a new page: none of the old lines apply
        self.detect_on_preview()
        return self.preview

    def detect_on_preview(self) -> list[CropBox]:
        """(Re)run auto-crop on the current preview, keeping manual boxes.  With auto-crop off no
        box is added (the whole bed is scanned unless a box is drawn by hand)."""
        if self.preview is None:
            return []
        p = self.preview
        off = self.settings.crop.mode == "off"
        found = [] if off else autocrop.detect(p.image, self.crop_settings(), dpi=p.dpi)
        origin = (p.area_mm[0], p.area_mm[1])
        manual = [b for b in self.boxes_mm if b.manual]
        found_mm = [box_px_to_mm(b, p.dpi, origin) for b in found]
        for b in found_mm:
            self._auto_seq += 1
            b.meta["auto_id"] = f"a{self._auto_seq}"
        kept = {b.meta.get("auto_id") for b in manual} - {None}
        self.snap_lines = {k: v for k, v in self.snap_lines.items() if k in kept}
        if self.settings.crop.mode == "booklet" and found:
            try:
                with profiles.use(self.profile()):
                    for b_px, b_mm in zip(found, found_mm):
                        lines = sidelines.box_lines(p.image, b_px, p.dpi)
                        self.snap_lines[b_mm.meta["auto_id"]] = self._snap_lines(b_px, lines)
                        check = self._check_sides(b_px, lines)
                        if check:
                            b_mm.meta["check_sides"] = check  # (before auto_mm copies the boxes: cases see it)
            except Exception:  # snapping is a help: never let it break the preview
                pass
        self.auto_mm = ([a for a in self.auto_mm if a.meta.get("auto_id") in kept]
                        + [replace(b, meta=dict(b.meta)) for b in found_mm])
        self.boxes_mm = manual + found_mm
        return self.boxes_mm

    def _check_sides(self, box_px: CropBox, lines: dict) -> dict[str, str]:
        """The sides auto-crop was unsure about (``sidelines.CHECK_RULE``) and why."""
        spines = sidelines.spine_names(box_px)
        out = {}
        for name in SIDE_NAMES:
            sig = sidelines.signals(lines[name], self.settings.crop.margin_mm, name in spines)
            reason = sidelines.check_reason(sig, sidelines.CHECK_RULE)
            if reason:
                out[name] = reason
        return out

    def _snap_lines(self, box_px: CropBox, lines: dict) -> dict[str, list[sidelines.SnapLine]]:
        """A box's candidate lines (preview px, relative to its sides) as absolute bed-mm lines, moved in
        by the crop margin."""
        p = self.preview
        px_mm, f = p.dpi / MM_PER_INCH, MM_PER_INCH / p.dpi
        out = {}
        for name, ls in lines.items():
            out[name] = []
            for ln in ls:
                q, dd = sidelines.to_image(ln, box_px, name, px_mm, inset_mm=self.settings.crop.margin_mm)
                out[name].append(sidelines.SnapLine(((q[0] + 0.5) * f + p.area_mm[0], (q[1] + 0.5) * f + p.area_mm[1]),
                                                    (float(dd[0]), float(dd[1])), ln.support, ln.kind))
        return out

    def snap_lines_px(self) -> dict[str, dict[str, list[sidelines.SnapLine]]]:
        """``snap_lines`` in preview pixels (as ``preview_boxes_px``)."""
        if self.preview is None:
            return {}
        p = self.preview
        f = p.dpi / MM_PER_INCH
        return {aid: {name: [replace(ln, p=((ln.p[0] - p.area_mm[0]) * f - 0.5, (ln.p[1] - p.area_mm[1]) * f - 0.5))
                             for ln in ls] for name, ls in sides.items()}
                for aid, sides in self.snap_lines.items()}

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
        """The boxes a scan is cropped to (also boxes drawn by hand with auto-crop off; none:
        the whole bed)."""
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
            if page_no == 0:
                self._capture(result, crops, status)
            outcomes.append(self._finish(result, boxes, save, status))
            if cancel is not None and cancel.is_set():
                break
        return outcomes

    def _capture(self, result: ScanResult, final_mm: list[CropBox], status: StatusFn | None) -> None:
        """Crop cases: keep this scan's case and write it when an auto box was corrected.  Never raises."""
        self.last_case, self.last_case_path, self.case_note = None, None, ""
        if not self.settings.cases.enabled or self.preview is None or self.device_id.startswith("file:"):
            return  # (an opened image: no lid, not the scanner frame, and its id is a local path)
        try:
            scan, saved_dpi = cases.downscale(result.image, result.dpi)
            info = self.device.info if self.device is not None else None
            c = self.settings.crop
            self.last_case = cases.CaseData(
                preview=self.preview.image, preview_dpi=self.preview.dpi,
                preview_area_mm=tuple(self.preview.area_mm), scan=scan, scan_dpi_saved=saved_dpi,
                scan_dpi=result.dpi, scan_area_mm=tuple(result.area_mm),
                auto=[replace(b, meta=dict(b.meta)) for b in self.auto_mm],
                final=[replace(b, meta=dict(b.meta)) for b in final_mm],
                scanner_id=self.device_id, scanner_name=info.name if info else "", profile=self.profile().name,
                crop={"mode": c.mode, "booklet_output": c.booklet_output, "booklet_rtl": c.booklet_rtl,
                      "margin_mm": c.margin_mm, "edge_refine": c.edge_refine, "rotate": c.rotate})
            edits = cases.diff_boxes(self.last_case.auto, self.last_case.final)
            if c.mode != "off" and cases.corrected(edits):  # Off: drawing boxes is how it works
                self.last_case_path = cases.write_case(cases.folder_of(self.settings.cases), self.last_case)
                self.case_note = f"Saved crop case {self.last_case_path.name} ({cases.summary(edits, c.rotate)})"
        except Exception as e:  # noqa: BLE001 - a crop case must never break a scan
            self.case_note = f"Could not save the crop case: {e}"
        if status and self.case_note:
            status(self.case_note)

    def save_last_case(self) -> Path | None:
        """"Save this case": write the last scan's case (reason "saved"); its path when already written,
        None without one (crop cases off, no scan yet)."""
        if self.last_case is None:
            return None
        if self.last_case_path is None:
            self.last_case.reason = "saved"
            self.last_case_path = cases.write_case(cases.folder_of(self.settings.cases), self.last_case)
        return self.last_case_path

    def run_test_scan(self, progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                      status: StatusFn | None = None) -> tuple[ScanResult, Path]:
        """Scan the whole bed at ``output.test_dpi`` (crop boxes and the scan resolution are
        ignored) and save it uncropped, as the scanner delivered it, as a TIFF named
        ``output.test_prefix`` + number in ``output.test_folder``: test data for tuning the
        auto-crop.  Nothing else changes (preview, boxes, the normal output numbering)."""
        st = self.settings
        out = st.output
        raw = out.raw_file in ("only", "also")  # the same driver settings as the normal scans
        req = ScanRequest(dpi=out.test_dpi, mode="color", source=st.input.source, area_mm=None,
                          bit_depth=8, raw=raw, live=self.live)
        result = self.open().scan(req, progress, cancel)
        if status:
            h, w = result.image.shape[:2]
            status(f"Scanned {result.area_mm[2]:.0f} x {result.area_mm[3]:.0f} mm at {result.dpi:.0f} dpi "
                   f"({w} x {h} px). Saving...")
        folder = out.test_path
        folder.mkdir(parents=True, exist_ok=True)
        (path,) = next_filenames(folder, out.test_prefix, "tif", 1, out.digits)
        save_image(result.image, path, out, dpi=result.dpi)
        return result, path

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
