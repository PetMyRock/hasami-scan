"""Treat an image file (or a folder of them) as the scanner bed.

Useful for re-cropping old scans with the current settings: ``file:/path/a.tif``
behaves like a flatbed holding that image; ``file:/path/folder`` behaves like a
document feeder that delivers every image in the folder.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.output import load_image
from .base import (
    Backend,
    Capabilities,
    DeviceInfo,
    ProgressFn,
    ScanCancelled,
    Scanner,
    ScannerError,
    ScanRequest,
    ScanResult,
    crop_and_resample,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_DPI = 300.0


class FileScanner(Scanner):
    def __init__(self, path: Path):
        self.path = path
        if path.is_dir():
            self.files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.files:
                raise ScannerError(f"no images in {path}")
        elif path.is_file():
            self.files = [path]
        else:
            raise ScannerError(f"no such file or folder: {path}")
        self.info = DeviceInfo(f"file:{path}", path.name, "File", "", "file")
        self._cache: dict[Path, tuple[np.ndarray, float]] = {}

    def _load(self, p: Path) -> tuple[np.ndarray, float]:
        if p not in self._cache:
            img, dpi = load_image(p)
            self._cache = {p: (img, dpi or DEFAULT_DPI)}  # keep one image in memory
        return self._cache[p]

    def capabilities(self) -> Capabilities:
        img, dpi = self._load(self.files[0])
        h, w = img.shape[:2]
        return Capabilities(
            sources=["Flatbed"] if len(self.files) == 1 else ["Flatbed", "ADF"],
            modes=["color", "gray", "lineart"],
            resolutions=sorted({75, 150, 300, 600, int(round(dpi))}),
            bit_depths=[8, 16],
            max_area_mm=(w / dpi * 25.4, h / dpi * 25.4),
        )

    def _scan_file(self, p: Path, request: ScanRequest) -> ScanResult:
        img, dpi = self._load(p)
        h, w = img.shape[:2]
        return crop_and_resample(img, dpi, request, (w / dpi * 25.4, h / dpi * 25.4))

    def scan(self, request: ScanRequest, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> ScanResult:
        if cancel is not None and cancel.is_set():
            raise ScanCancelled("scan cancelled")
        result = self._scan_file(self.files[0], request)
        if progress:
            progress(1.0)
        return result

    def scan_batch(self, request: ScanRequest, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> Iterator[ScanResult]:
        files = self.files if "adf" in request.source.lower() or len(self.files) > 1 else self.files[:1]
        for i, p in enumerate(files):
            if cancel is not None and cancel.is_set():
                raise ScanCancelled("scan cancelled")
            yield self._scan_file(p, request)
            if progress:
                progress((i + 1) / len(files))


class FileBackend(Backend):
    name = "file"

    def available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        return []  # files are opened explicitly with file:<path>

    def open(self, native_id: str) -> Scanner:
        return FileScanner(Path(native_id).expanduser())
