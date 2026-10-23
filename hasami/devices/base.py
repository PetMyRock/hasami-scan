"""Common scanner interface and the backend registry.

Device ids have the form ``backend:native-id``, e.g.
``sane:epson2:libusb:001:004``, ``escl:http://192.168.1.20:80/eSCL``,
``wia:{6BDD1FC6-810F-11D0-BEC7-08002BE2092F}\\0000``, ``file:/scans/a.tif``
or ``demo:anime``.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np

MM_PER_INCH = 25.4

ProgressFn = Callable[[float], None]


class ScannerError(RuntimeError):
    """Something went wrong talking to a scanner."""


class ScanCancelled(ScannerError):
    """The scan was cancelled by the user."""

    #: What the user should know (e.g. the driver had to be killed), or "".
    warning = ""


@dataclass
class DeviceInfo:
    id: str
    name: str
    vendor: str = ""
    model: str = ""
    backend: str = ""

    def label(self) -> str:
        parts = [p for p in (self.vendor, self.model) if p]
        desc = " ".join(parts) if parts else self.name
        return f"{desc} ({self.backend})"


@dataclass
class Capabilities:
    sources: list[str] = field(default_factory=lambda: ["Flatbed"])
    modes: list[str] = field(default_factory=lambda: ["color", "gray", "lineart"])
    resolutions: list[int] = field(default_factory=lambda: [75, 150, 300, 600, 1200])
    bit_depths: list[int] = field(default_factory=lambda: [8])
    #: Width and height of the scan area in millimetres.
    max_area_mm: tuple[float, float] = (215.9, 297.0)

    def nearest_resolution(self, dpi: int) -> int:
        if not self.resolutions:
            return dpi
        return min(self.resolutions, key=lambda r: (abs(r - dpi), -r))


@dataclass
class ScanRequest:
    dpi: int = 300
    #: "color", "gray" or "lineart".
    mode: str = "color"
    #: Input source name as offered by the device; empty for the default.
    source: str = ""
    #: ``(x, y, width, height)`` in mm from the top-left of the bed; ``None``
    #: scans the whole area.
    area_mm: tuple[float, float, float, float] | None = None
    bit_depth: int = 8
    preview: bool = False
    #: Ask the scanner for its data with as little correction as it allows
    #: (neutral brightness/contrast, no colour correction) for a raw file.
    raw: bool = False
    #: Called with a :class:`LiveImage` while the image comes in, by backends
    #: that can deliver it in strips (TWAIN, demo), to show the scan live.
    live: "Callable[[LiveImage], None] | None" = field(default=None, repr=False, compare=False)


@dataclass
class LiveImage:
    """The part of a scan received so far."""

    image: np.ndarray  # the whole image; rows from ``rows`` on are not filled yet
    rows: int
    dpi: float
    area_mm: tuple[float, float, float, float]

    @property
    def done(self) -> bool:
        return self.rows >= self.image.shape[0]


@dataclass
class ScanResult:
    image: np.ndarray  # RGB or grey, uint8 or uint16
    dpi: float
    #: Area actually scanned, ``(x, y, width, height)`` in mm.
    area_mm: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: Anything the user should know about this scan (e.g. the driver
    #: ignored the requested area).
    note: str = ""


class Scanner(ABC):
    """An open scanner."""

    info: DeviceInfo

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def scan(self, request: ScanRequest, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> ScanResult: ...

    def scan_batch(self, request: ScanRequest, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> Iterator[ScanResult]:
        """Scan every page in a feeder; flatbeds yield a single page."""
        yield self.scan(request, progress, cancel)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Backend(ABC):
    name: str = ""

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def list_devices(self) -> list[DeviceInfo]: ...

    @abstractmethod
    def open(self, native_id: str) -> Scanner: ...

    def unavailable_reason(self) -> str:
        return ""


# ------------------------------------------------------------------ helpers
def match_mode(wanted: str, offered: list[str]) -> str | None:
    """Map our mode name to a device's own spelling ("Color", "24bit Color"...)."""
    keys = {
        "color": ("color", "colour", "rgb"),
        "gray": ("gray", "grey", "grayscale"),
        "lineart": ("lineart", "binary", "black", "bw", "monochrome"),
    }[wanted]
    for name in offered:
        low = name.lower().replace(" ", "")
        if any(k in low for k in keys):
            return name
    return None


def normalise_mode(name: str) -> str | None:
    for m in ("color", "gray", "lineart"):
        if match_mode(m, [name]):
            return m
    return None


def match_source(wanted: str, offered: list[str]) -> str | None:
    """Find the device's spelling of a source such as "Flatbed" or "ADF"."""
    if not offered:
        return None
    if not wanted:
        return offered[0]
    w = wanted.lower()
    for name in offered:
        if name.lower() == w:
            return name
    aliases = {
        "flatbed": ("flatbed", "platen", "normal", "document table"),
        "adf": ("adf", "feeder", "automatic document"),
        "duplex": ("duplex",),
        "transparency": ("transparency", "tpu", "film", "slide", "negative"),
    }
    for key, words in aliases.items():
        if key in w or any(x in w for x in words):
            hits = [n for n in offered if any(x in n.lower() for x in words)]
            if key == "adf":
                hits = [n for n in hits if "duplex" not in n.lower()] or hits
            if hits:
                return hits[0]
    return None


def crop_and_resample(bed: np.ndarray, bed_dpi: float, request: ScanRequest,
                      bed_mm: tuple[float, float]) -> ScanResult:
    """Cut the requested area out of a bed image and resample it to the dpi.

    Used by backends that hold the whole bed as an image (file, demo).
    """
    H, W = bed.shape[:2]
    x, y, w, h = request.area_mm or (0.0, 0.0, bed_mm[0], bed_mm[1])
    x = max(0.0, min(x, bed_mm[0]))
    y = max(0.0, min(y, bed_mm[1]))
    w = max(1.0, min(w, bed_mm[0] - x))
    h = max(1.0, min(h, bed_mm[1] - y))
    px = bed_dpi / MM_PER_INCH
    x0, y0 = int(round(x * px)), int(round(y * px))
    x1, y1 = min(W, int(round((x + w) * px))), min(H, int(round((y + h) * px)))
    part = bed[y0:y1, x0:x1]
    scale = request.dpi / bed_dpi
    out_w = max(1, int(round(part.shape[1] * scale)))
    out_h = max(1, int(round(part.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(part, (out_w, out_h), interpolation=interp)
    if request.mode != "color" and img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = g if request.mode == "gray" else np.where(g > 127, 255, 0).astype(np.uint8)
    if request.bit_depth == 16 and img.dtype == np.uint8 and request.mode != "lineart":
        img = img.astype(np.uint16) * 257
    elif request.bit_depth == 8 and img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)
    return ScanResult(img, float(request.dpi), (x0 / px, y0 / px, (x1 - x0) / px, (y1 - y0) / px))


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/TIFF/BMP bytes to an RGB (or grey) array."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ScannerError("could not decode image data from hasami")
    if arr.ndim == 3:
        code = cv2.COLOR_BGRA2RGB if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB
        arr = cv2.cvtColor(arr, code)
    return arr


# ----------------------------------------------------------------- registry
def backends() -> list[Backend]:
    from .demo_backend import DemoBackend
    from .escl_backend import EsclBackend
    from .file_backend import FileBackend
    from .sane_backend import SaneBackend
    from .twain_backend import TwainBackend
    from .wia_backend import WiaBackend

    from .private import private_backends

    return [SaneBackend(), TwainBackend(), WiaBackend(), EsclBackend(), DemoBackend(), FileBackend(),
            *private_backends()]


def get_backend(name: str) -> Backend:
    for b in backends():
        if b.name == name:
            return b
    raise ScannerError(f"unknown backend: {name}")


def list_devices(include_unavailable: bool = False) -> list[DeviceInfo]:
    out: list[DeviceInfo] = []
    for b in backends():
        if not b.available():
            continue
        try:
            out.extend(b.list_devices())
        except Exception as e:  # one broken backend must not hide the others
            if include_unavailable:
                out.append(DeviceInfo(f"{b.name}:", f"error: {e}", backend=b.name))
    return out


def open_device(device_id: str) -> Scanner:
    backend, _, native = device_id.partition(":")
    b = get_backend(backend)
    if not b.available():
        raise ScannerError(f"{backend} backend is not available: {b.unavailable_reason()}")
    return b.open(native)
