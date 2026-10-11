"""SANE backend (Linux, BSD, macOS with sane-backends) via ``python-sane``."""

from __future__ import annotations

import threading
from typing import Iterator

import numpy as np

from .base import (
    MM_PER_INCH,
    Backend,
    Capabilities,
    DeviceInfo,
    ProgressFn,
    ScanCancelled,
    Scanner,
    ScannerError,
    ScanRequest,
    ScanResult,
    match_mode,
    match_source,
    normalise_mode,
)

try:  # pragma: no cover - depends on the system
    import sane as _sane_mod
except Exception:  # ImportError, or OSError when libsane is missing
    _sane_mod = None

UNIT_PIXEL, UNIT_MM, UNIT_DPI = 1, 3, 4
_init_lock = threading.Lock()
_initialised = False


def _ensure_init() -> None:
    global _initialised
    with _init_lock:
        if not _initialised:
            _sane_mod.init()
            _initialised = True


def _resolutions(constraint) -> list[int]:
    if isinstance(constraint, tuple) and len(constraint) == 3:  # (min, max, quant) range
        lo, hi, _quant = constraint
        common = [50, 75, 100, 150, 200, 240, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400]
        vals = [r for r in common if lo <= r <= hi]
        return vals or [int(lo), int(hi)]
    if isinstance(constraint, list):
        return sorted({int(v) for v in constraint})
    return [75, 150, 300, 600]


class SaneScanner(Scanner):
    def __init__(self, name: str, vendor: str = "", model: str = ""):
        _ensure_init()
        try:
            self.dev = _sane_mod.open(name)
        except Exception as e:
            raise ScannerError(f"cannot open SANE device {name}: {e}") from e
        self.info = DeviceInfo(f"sane:{name}", name, vendor, model, "sane")
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- options
    def _opt(self, key: str):
        o = self.dev.opt.get(key)
        if o is None or not o.is_active() or not o.is_settable():
            return None
        return o

    def _set(self, key: str, value) -> bool:
        o = self._opt(key)
        if o is None:
            return False
        try:
            setattr(self.dev, key, value)
            return True
        except Exception:
            return False

    def capabilities(self) -> Capabilities:
        caps = Capabilities()
        o = self.dev.opt.get("source")
        if o is not None and isinstance(o.constraint, list):
            caps.sources = list(o.constraint)
        o = self.dev.opt.get("mode")
        if o is not None and isinstance(o.constraint, list):
            modes = [normalise_mode(m) for m in o.constraint]
            caps.modes = [m for m in ("color", "gray", "lineart") if m in modes]
        o = self.dev.opt.get("resolution")
        if o is not None:
            caps.resolutions = _resolutions(o.constraint)
        o = self.dev.opt.get("depth")
        if o is not None and isinstance(o.constraint, list):
            caps.bit_depths = [d for d in (8, 16) if d in o.constraint] or [8]
        bx, by = self.dev.opt.get("br_x"), self.dev.opt.get("br_y")
        if bx is not None and by is not None and isinstance(bx.constraint, tuple):
            w, h = float(bx.constraint[1]), float(by.constraint[1])
            if bx.unit == UNIT_PIXEL:  # some backends measure in pixels
                res = self.dev.opt.get("resolution")
                dpi = float(getattr(self.dev, "resolution", 300) or 300) if res else 300.0
                w, h = w / dpi * MM_PER_INCH, h / dpi * MM_PER_INCH
            caps.max_area_mm = (w, h)
        return caps

    def _configure(self, request: ScanRequest) -> float:
        caps = self.capabilities()
        mode_opt = self.dev.opt.get("mode")
        if mode_opt is not None and isinstance(mode_opt.constraint, list):
            m = match_mode(request.mode, mode_opt.constraint)
            if m:
                self._set("mode", m)
        src_opt = self.dev.opt.get("source")
        if request.source and src_opt is not None and isinstance(src_opt.constraint, list):
            s = match_source(request.source, src_opt.constraint)
            if s:
                self._set("source", s)
        if request.mode != "lineart":
            self._set("depth", 16 if request.bit_depth == 16 and 16 in caps.bit_depths else 8)
        if request.raw:
            # Drivers that offer it (e.g. epson2) can skip their colour correction.
            cc = self.dev.opt.get("color_correction")
            if cc is not None and isinstance(cc.constraint, list):
                none = [v for v in cc.constraint if str(v).lower() in ("none", "off", "disabled")]
                if none:
                    self._set("color_correction", none[0])
        dpi = caps.nearest_resolution(request.dpi)
        res = self._opt("resolution")
        if res is not None:
            self._set("resolution", float(dpi) if res.type == 2 else int(dpi))
        else:
            for k in ("x_resolution", "y_resolution"):
                self._set(k, int(dpi))
        self._set("preview", bool(request.preview))
        x, y, w, h = request.area_mm or (0.0, 0.0, *caps.max_area_mm)
        bx = self.dev.opt.get("br_x")
        to_unit = 1.0
        if bx is not None and bx.unit == UNIT_PIXEL:
            to_unit = dpi / MM_PER_INCH
        # Set the far corner first so that tl < br holds at every step.
        for key, val in (("br_x", x + w), ("br_y", y + h), ("tl_x", x), ("tl_y", y)):
            o = self._opt(key)
            if o is None:
                continue
            v = val * to_unit
            if isinstance(o.constraint, tuple):
                v = min(max(v, o.constraint[0]), o.constraint[1])
            self._set(key, float(v) if o.type == 2 else int(round(v)))
        return float(dpi)

    def _read(self, progress: ProgressFn | None, cancel: threading.Event | None) -> np.ndarray:
        stop = threading.Event()

        def watch():
            while not stop.wait(0.1):
                if cancel is not None and cancel.is_set():
                    try:
                        self.dev.cancel()
                    except Exception:
                        pass
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            def cb(done, total):
                if progress and total:
                    progress(min(1.0, done / total))

            arr = self.dev.arr_snap(progress=cb)
        except Exception as e:
            if cancel is not None and cancel.is_set():
                raise ScanCancelled("scan cancelled") from e
            raise ScannerError(f"scan failed: {e}") from e
        finally:
            stop.set()
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return np.ascontiguousarray(arr)

    def scan(self, request: ScanRequest, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> ScanResult:
        with self._lock:
            dpi = self._configure(request)
            try:
                self.dev.start()
            except Exception as e:
                raise ScannerError(f"cannot start scan: {e}") from e
            img = self._read(progress, cancel)
            x, y, w, h = request.area_mm or (0.0, 0.0, *self.capabilities().max_area_mm)
            return ScanResult(img, dpi, (x, y, w, h))

    def scan_batch(self, request: ScanRequest, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> Iterator[ScanResult]:
        with self._lock:
            dpi = self._configure(request)
            area = request.area_mm or (0.0, 0.0, *self.capabilities().max_area_mm)
            feeder = "adf" in request.source.lower() or "feeder" in request.source.lower() or "duplex" in request.source.lower()
            while True:
                if cancel is not None and cancel.is_set():
                    raise ScanCancelled("scan cancelled")
                try:
                    self.dev.start()
                except Exception as e:
                    msg = str(e).lower()
                    if "no more" in msg or "empty" in msg or "no docs" in msg or "out of documents" in msg:
                        return  # feeder is empty
                    raise ScannerError(f"cannot start scan: {e}") from e
                yield ScanResult(self._read(progress, cancel), dpi, area)
                if not feeder:
                    return

    def close(self) -> None:
        try:
            self.dev.close()
        except Exception:
            pass


class SaneBackend(Backend):
    name = "sane"

    def available(self) -> bool:
        return _sane_mod is not None

    def unavailable_reason(self) -> str:
        return "install sane-backends and the python-sane package (pip install python-sane)"

    def list_devices(self) -> list[DeviceInfo]:
        _ensure_init()
        out = []
        for name, vendor, model, kind in _sane_mod.get_devices():
            out.append(DeviceInfo(f"sane:{name}", name, vendor, model, "sane"))
        return out

    def open(self, native_id: str) -> Scanner:
        vendor = model = ""
        try:
            for name, v, m, _k in _sane_mod.get_devices():
                if name == native_id:
                    vendor, model = v, m
        except Exception:
            pass
        return SaneScanner(native_id, vendor, model)
