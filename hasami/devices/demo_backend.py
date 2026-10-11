"""A virtual scanner that 'scans' synthetic beds, for trying the app without hardware.

Devices:

* ``demo:anime`` - illustration prints, trading cards and shikishi boards
* ``demo:photos`` - ordinary photos, some with white borders
* ``demo:booklet`` - an open artbook that is too wide for the bed
* ``demo:document`` - a single page

Preview and final scan show the same bed; the arrangement changes after each
final scan, as if new items had been placed on the glass.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..demo import BookletOptions, SynthOptions, synth_booklet, synth_scan
from .base import (
    Backend,
    Capabilities,
    DeviceInfo,
    LiveImage,
    ProgressFn,
    ScanCancelled,
    Scanner,
    ScanRequest,
    ScanResult,
    crop_and_resample,
)

BED_DPI = 200.0
BED_MM = (215.9, 297.0)
KINDS = {
    "anime": "Demo scanner - anime prints and cards",
    "photos": "Demo scanner - photos",
    "booklet": "Demo scanner - artbook too wide for the bed",
    "document": "Demo scanner - document",
}


class DemoScanner(Scanner):
    #: Seconds per strip when the scan is shown live (so it can be seen coming in).
    strip_delay = 0.08

    def __init__(self, kind: str, seed: int = 1):
        if kind not in KINDS:
            kind = "anime"
        self.kind = kind
        self.seed = seed
        self.info = DeviceInfo(f"demo:{kind}", KINDS[kind], "", KINDS[kind], "demo")
        self._bed: np.ndarray | None = None
        self._bed_seed: int | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            sources=["Flatbed"],
            modes=["color", "gray", "lineart"],
            resolutions=[75, 100, 150, 200, 300, 400, 600],
            bit_depths=[8, 16],
            max_area_mm=BED_MM,
        )

    def bed(self) -> np.ndarray:
        if self._bed is None or self._bed_seed != self.seed:
            w = int(round(BED_MM[0] / 25.4 * BED_DPI))
            h = int(round(BED_MM[1] / 25.4 * BED_DPI))
            if self.kind == "booklet":
                img = synth_booklet(BookletOptions(width=w, height=h, content="artbook"), seed=self.seed).image
            elif self.kind == "document":
                img = synth_scan(SynthOptions(width=w, height=h, document=True, theme="anime"), seed=self.seed).image
            elif self.kind == "photos":
                img = synth_scan(SynthOptions(width=w, height=h), seed=self.seed).image
            else:
                img = synth_scan(SynthOptions(width=w, height=h, theme="anime", n_photos=4), seed=self.seed).image
            self._bed, self._bed_seed = img, self.seed
        return self._bed

    def scan(self, request: ScanRequest, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> ScanResult:
        bed = self.bed()
        result = crop_and_resample(bed, BED_DPI, request, BED_MM)
        h = result.image.shape[0]
        steps = 12
        for i in range(steps):  # pretend the carriage moves, delivering the image in strips
            if cancel is not None and cancel.is_set():
                raise ScanCancelled("scan cancelled")
            rows = h * (i + 1) // steps
            if progress:
                progress(rows / h)
            if request.live:
                part = np.zeros_like(result.image)
                part[:rows] = result.image[:rows]
                request.live(LiveImage(part, rows, result.dpi, result.area_mm))
                time.sleep(self.strip_delay)
        if progress:
            progress(1.0)
        if not request.preview:
            self.seed += 1  # next time, different items are on the glass
        return result


class DemoBackend(Backend):
    name = "demo"

    def available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        return [DeviceInfo(f"demo:{k}", v, "", v, "demo") for k, v in KINDS.items()]

    def open(self, native_id: str) -> Scanner:
        kind, _, seed = native_id.partition("@")
        return DemoScanner(kind or "anime", int(seed) if seed.isdigit() else 1)
