"""Windows Image Acquisition (WIA) backend, via ``pywin32``.

WIA is the scanner API built into Windows; every scanner with a Windows driver
supports it.  Uses the WIA Automation layer (``WIA.DeviceManager``).

16 bits per channel (48-bit colour, 16-bit grey) is used when the driver lists
that depth; such scans are transferred as TIFF (BMP has no 16-bit colour).
For raw scans brightness and contrast are set to neutral.  WIA has no
standard way to turn off a driver's own gamma or colour correction, so a raw
WIA scan is the driver's output with nothing added by this program.
"""

from __future__ import annotations

import sys
import threading
from typing import Iterator

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
    decode_image,
)

SCANNER_DEVICE_TYPE = 1
# Property ids (wiadef.h).
WIA_DPS_HORIZONTAL_BED_SIZE = 3074  # thousandths of an inch
WIA_DPS_VERTICAL_BED_SIZE = 3075
WIA_DPS_DOCUMENT_HANDLING_SELECT = 3088
WIA_IPA_DEPTH = 4104
WIA_IPS_CUR_INTENT = 6146
WIA_IPS_BRIGHTNESS = 6154
WIA_IPS_CONTRAST = 6155
WIA_IPS_XRES = 6147
WIA_IPS_YRES = 6148
WIA_IPS_XPOS = 6149
WIA_IPS_YPOS = 6150
WIA_IPS_XEXTENT = 6151
WIA_IPS_YEXTENT = 6152
FEEDER, FLATBED, DUPLEX = 1, 2, 4
INTENT = {"color": 1, "gray": 2, "lineart": 4}
FORMAT_PNG = "{B96B3CAF-0728-11D3-9D7B-0000F81EF32E}"
FORMAT_BMP = "{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}"
FORMAT_TIFF = "{B96B3CB1-0728-11D3-9D7B-0000F81EF32E}"
RANGE_SUBTYPE, LIST_SUBTYPE = 1, 2  # WiaSubType
STANDARD_DPI = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800)
#: Bits per pixel for (mode, bits per channel).
DEPTH = {("color", 8): 24, ("color", 16): 48, ("gray", 8): 8, ("gray", 16): 16, ("lineart", 8): 1, ("lineart", 16): 1}
WIA_ERROR_PAPER_EMPTY = -2145320957  # 0x80210003


def _com():
    import pythoncom  # noqa: F401  (pywin32; COM must be importable)
    import win32com.client

    return win32com.client


def _props(obj) -> dict:
    out = {}
    for i in range(1, obj.Properties.Count + 1):
        p = obj.Properties.Item(i)
        out[int(p.PropertyID)] = p
    return out


def _vector(v) -> list:
    """A WIA Vector (1-based ``Item``) or any iterable as a list."""
    try:
        return [v.Item(i) for i in range(1, int(v.Count) + 1)]
    except Exception:
        try:
            return list(v)
        except Exception:
            return []


def valid_values(p) -> list[int] | None:
    """The values a WIA property accepts, if it lists them (``None`` if unknown).

    A range is expanded to the standard resolutions it contains.
    """
    if p is None:
        return None
    try:
        sub = int(p.SubType)
        if sub == LIST_SUBTYPE:
            return sorted({int(x) for x in _vector(p.SubTypeValues)})
        if sub == RANGE_SUBTYPE:
            lo, hi, step = int(p.SubTypeMin), int(p.SubTypeMax), max(1, int(p.SubTypeStep))
            return [d for d in STANDARD_DPI if lo <= d <= hi and (d - lo) % step == 0]
    except Exception:
        return None
    return None


HRESULTS = {
    0x80210001: "general scanner error", 0x80210002: "paper jam", 0x80210003: "no paper",
    0x80210004: "paper problem", 0x80210005: "scanner is offline (off, unplugged or asleep?)",
    0x80210006: "scanner is busy (is another scanning program open?)", 0x80210007: "scanner is warming up",
    0x80210008: "the scanner needs attention", 0x8021000A: "cannot communicate with the scanner",
    0x8021000B: "invalid command", 0x8021000C: "incorrect hardware setting", 0x8021000D: "scanner is locked",
    0x8021000E: "the scanner driver failed", 0x8021000F: "invalid driver response",
    0x80004005: "unspecified error", 0x80070057: "invalid setting", 0x8001010E: "called from the wrong thread",
    0x800401F0: "COM not initialised",
}


def describe(e: Exception) -> str:
    """A COM error as text, with the driver's own error code decoded."""
    args = getattr(e, "args", ())
    codes = []
    if args and isinstance(args[0], int):
        codes.append(args[0])
    if len(args) > 2 and isinstance(args[2], tuple) and len(args[2]) > 5 and isinstance(args[2][5], int):
        codes.insert(0, args[2][5])  # the driver's code inside DISP_E_EXCEPTION
    for c in codes:
        u = c & 0xFFFFFFFF
        if u in HRESULTS:
            return f"{HRESULTS[u]} (0x{u:08X})"
    if codes:
        return f"error 0x{codes[0] & 0xFFFFFFFF:08X}: {e}"
    return str(e) or e.__class__.__name__


def _step(what: str, fn, *args):
    """Run one WIA call; on failure say which step failed."""
    try:
        return fn(*args)
    except ScannerError:
        raise
    except Exception as e:
        raise ScannerError(f"WIA: {what} failed: {describe(e)}") from e


def _range(p) -> tuple[int, int] | None:
    try:
        if int(p.SubType) == RANGE_SUBTYPE:
            return int(p.SubTypeMin), int(p.SubTypeMax)
    except Exception:
        pass
    return None


def _set_prop(obj, prop_id: int, value, log: list | None = None) -> bool:
    p = _props(obj).get(prop_id)
    if p is None:
        if log is not None:
            log.append(f"property {prop_id}: not offered")
        return False
    rng = _range(p)
    if rng is not None and isinstance(value, int):
        value = min(max(value, rng[0]), rng[1])
    try:
        p.Value = value
        if log is not None:
            log.append(f"property {prop_id} = {value}: ok")
        return True
    except Exception as e:
        if log is not None:
            log.append(f"property {prop_id} = {value}: refused ({describe(e)})")
        return False


class _ComThread:
    """One thread that owns every WIA object.

    WIA objects live in a single-threaded COM apartment and must be used from
    the thread that created them; the GUI runs each job on a different pool
    thread, so all calls are handed to this thread instead.
    """

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor

        def init():
            import pythoncom

            pythoncom.CoInitialize()

        self.pool = ThreadPoolExecutor(max_workers=1, initializer=init, thread_name_prefix="wia")

    def call(self, fn, *args):
        return self.pool.submit(fn, *args).result()


_THREAD: _ComThread | None = None


def _com_thread() -> _ComThread:
    global _THREAD
    if _THREAD is None:
        _THREAD = _ComThread()
    return _THREAD


class WiaScanner(Scanner):
    def __init__(self, device_id: str):
        self.device_id = device_id
        self._thread = _com_thread()
        self._thread.call(self._connect)

    def _call(self, fn, *args):
        thread = getattr(self, "_thread", None)
        return thread.call(fn, *args) if thread is not None else fn(*args)

    def _connect(self) -> None:
        dm = _step("starting Windows Image Acquisition", _com().Dispatch, "WIA.DeviceManager")
        self.dev = None
        name = self.device_id
        for i in range(1, dm.DeviceInfos.Count + 1):
            info = dm.DeviceInfos.Item(i)
            if info.DeviceID == self.device_id:
                name = str(info.Properties("Name").Value)
                self.dev = _step("connecting to the scanner", info.Connect)
                break
        if self.dev is None:
            raise ScannerError(f"WIA device not found: {self.device_id}")
        self.info = DeviceInfo(f"wia:{self.device_id}", name, "", name, "wia")

    def _item(self):
        return _step("opening the scanner's flatbed item", lambda: self.dev.Items(1))

    def capabilities(self) -> Capabilities:
        return self._call(self._capabilities)

    def _capabilities(self) -> Capabilities:
        props = _step("reading the scanner's settings", _props, self.dev)
        size = (215.9, 297.0)
        try:
            w = props.get(WIA_DPS_HORIZONTAL_BED_SIZE)
            h = props.get(WIA_DPS_VERTICAL_BED_SIZE)
            if w is not None and h is not None:
                size = (float(w.Value) / 1000 * MM_PER_INCH, float(h.Value) / 1000 * MM_PER_INCH)
        except Exception:
            pass
        sources = ["Flatbed"]
        if WIA_DPS_DOCUMENT_HANDLING_SELECT in props:
            sources += ["ADF", "ADF Duplex"]
        item_props = _step("reading the flatbed settings", _props, self._item())
        resolutions = valid_values(item_props.get(WIA_IPS_XRES)) or [75, 100, 150, 200, 300, 600, 1200]
        depths = valid_values(item_props.get(WIA_IPA_DEPTH)) or []
        bit_depths = [8, 16] if (48 in depths or 16 in depths) else [8]
        return Capabilities(sources=sources, modes=["color", "gray", "lineart"],
                            resolutions=resolutions, bit_depths=bit_depths, max_area_mm=size)

    def _configure(self, request: ScanRequest, log: list | None = None):
        src = (request.source or "Flatbed").lower()
        handling = FLATBED
        if "adf" in src or "feeder" in src:
            handling = FEEDER | (DUPLEX if "duplex" in src else 0)
        if handling != FLATBED:
            _set_prop(self.dev, WIA_DPS_DOCUMENT_HANDLING_SELECT, handling, log)
        item = self._item()
        dpi = int(request.dpi)
        _set_prop(item, WIA_IPS_CUR_INTENT, INTENT.get(request.mode, 1), log)
        # The intent sets the data type, and with it the depths on offer.
        depth = DEPTH[(request.mode, 16 if request.bit_depth == 16 else 8)]
        offered = valid_values(_props(item).get(WIA_IPA_DEPTH))
        if offered is not None and depth not in offered:
            depth = DEPTH[(request.mode, 8)]
        _set_prop(item, WIA_IPA_DEPTH, depth, log)
        self._sixteen = depth in (16, 48)
        if request.raw:
            _set_prop(item, WIA_IPS_BRIGHTNESS, 0, log)
            _set_prop(item, WIA_IPS_CONTRAST, 0, log)
        _set_prop(item, WIA_IPS_XRES, dpi, log)
        _set_prop(item, WIA_IPS_YRES, dpi, log)
        x, y, w, h = request.area_mm or (0.0, 0.0, *self._capabilities().max_area_mm)
        px = dpi / MM_PER_INCH
        # Position first with the extent at its smallest is not needed by every
        # driver; the extent is clamped to what the driver allows at this dpi.
        _set_prop(item, WIA_IPS_XPOS, int(round(x * px)), log)
        _set_prop(item, WIA_IPS_YPOS, int(round(y * px)), log)
        _set_prop(item, WIA_IPS_XEXTENT, int(round(w * px)), log)
        _set_prop(item, WIA_IPS_YEXTENT, int(round(h * px)), log)
        return item, float(dpi), (x, y, w, h)

    def _transfer(self, item) -> bytes:
        formats = [FORMAT_TIFF, FORMAT_PNG, FORMAT_BMP] if getattr(self, "_sixteen", False) else [FORMAT_PNG, FORMAT_BMP]
        errors = []
        for fmt in formats:
            try:
                img = item.Transfer(fmt)
                return bytes(img.FileData.BinaryData)
            except Exception as e:
                errors.append(e)
                code = getattr(e, "hresult", None) or (e.args[0] if e.args else None)
                if code == WIA_ERROR_PAPER_EMPTY:
                    raise
        raise errors[-1]

    def _scan_once(self, request: ScanRequest) -> tuple[bytes, float, tuple]:
        item, dpi, area = self._configure(request)
        data = _step("scanning", self._transfer, item)
        return data, dpi, area

    def scan_batch(self, request: ScanRequest, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> Iterator[ScanResult]:
        feeder = "adf" in (request.source or "").lower()
        while True:
            if cancel is not None and cancel.is_set():
                raise ScanCancelled("scan cancelled")
            try:
                data, dpi, area = self._call(self._scan_once, request)
            except ScannerError as e:
                cause = e.__cause__
                code = getattr(cause, "hresult", None) or (cause.args[0] if cause is not None and cause.args else None)
                if feeder and code == WIA_ERROR_PAPER_EMPTY:
                    return
                raise
            if progress:
                progress(1.0)
            yield ScanResult(decode_image(data), dpi, area)
            if not feeder:
                return

    def scan(self, request: ScanRequest, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> ScanResult:
        for r in self.scan_batch(request, progress, cancel):
            return r
        raise ScannerError("no page scanned")


def _dump_props(obj, out: list[str], indent: str = "  ") -> None:
    for pid, p in sorted(_props(obj).items()):
        try:
            name = str(p.Name)
        except Exception:
            name = "?"
        try:
            value = p.Value
        except Exception as e:
            value = f"<unreadable: {describe(e)}>"
        allowed = ""
        try:
            sub = int(p.SubType)
            if sub == LIST_SUBTYPE:
                allowed = f"  one of {[x for x in _vector(p.SubTypeValues)]}"
            elif sub == RANGE_SUBTYPE:
                allowed = f"  range {p.SubTypeMin}..{p.SubTypeMax} step {p.SubTypeStep}"
        except Exception:
            pass
        out.append(f"{indent}{pid:5d} {name} = {value}{allowed}")


def diagnose(device_id: str | None = None, save_to: str | None = None) -> str:
    """Step-by-step report on the WIA scanners: settings offered, and a test
    preview scan (75 dpi, whole bed) with what the driver accepted."""
    import traceback

    out: list[str] = []

    def run():
        dm = _com().Dispatch("WIA.DeviceManager")
        out.append(f"WIA devices: {dm.DeviceInfos.Count}")
        target = None
        for i in range(1, dm.DeviceInfos.Count + 1):
            info = dm.DeviceInfos.Item(i)
            name = str(info.Properties("Name").Value)
            out.append(f"- {name}  type={int(info.Type)}  id={info.DeviceID}")
            if int(info.Type) == SCANNER_DEVICE_TYPE and (device_id is None or info.DeviceID == device_id) and target is None:
                target = info
        if target is None:
            out.append("No WIA scanner to test.")
            return
        sc = object.__new__(WiaScanner)
        sc.device_id = str(target.DeviceID)
        sc._thread = None  # already on the COM thread
        try:
            sc._connect()
            out.append("Connect: ok")
        except Exception as e:
            out.append(f"Connect: FAILED {describe(e)}")
            return
        out.append("Device properties:")
        _dump_props(sc.dev, out)
        try:
            item = sc._item()
            out.append(f"Items: {sc.dev.Items.Count}")
            out.append("Flatbed item properties:")
            _dump_props(item, out)
        except Exception as e:
            out.append(f"Flatbed item: FAILED {describe(e)}")
            return
        try:
            caps = sc._capabilities()
            out.append(f"Capabilities: bed {caps.max_area_mm[0]:.1f} x {caps.max_area_mm[1]:.1f} mm, "
                       f"resolutions {caps.resolutions}, bits {caps.bit_depths}")
        except Exception as e:
            out.append(f"Capabilities: FAILED {describe(e)}")
        log: list[str] = []
        try:
            item, dpi, area = sc._configure(ScanRequest(dpi=75, preview=True), log)
            out.append("Setting up a 75 dpi preview:")
            out.extend("  " + x for x in log)
            out.append("Flatbed item properties after setup:")
            _dump_props(item, out)
        except Exception as e:
            out.extend("  " + x for x in log)
            out.append(f"Setup: FAILED {describe(e)}")
            out.append(traceback.format_exc())
            return
        for fmt, label in ((FORMAT_PNG, "PNG"), (FORMAT_BMP, "BMP"), (FORMAT_TIFF, "TIFF")):
            try:
                img = item.Transfer(fmt)
                data = bytes(img.FileData.BinaryData)
                arr = decode_image(data)
                out.append(f"Transfer as {label}: ok, {len(data)} bytes, image {arr.shape} {arr.dtype}")
                if save_to:
                    with open(save_to, "wb") as f:
                        f.write(data)
                    out.append(f"Saved the test scan to {save_to}")
                break
            except Exception as e:
                out.append(f"Transfer as {label}: FAILED {describe(e)}")
                out.append("  " + repr(e))

    try:
        _com_thread().call(run)
    except Exception as e:
        out.append(f"FAILED: {describe(e)}")
        out.append(traceback.format_exc())
    return "\n".join(out)


class WiaBackend(Backend):
    name = "wia"

    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            _com()
            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        if sys.platform != "win32":
            return "WIA is only available on Windows"
        return "install pywin32 (pip install pywin32)"

    def list_devices(self) -> list[DeviceInfo]:
        return _com_thread().call(self._list)

    def _list(self) -> list[DeviceInfo]:
        dm = _step("starting Windows Image Acquisition", _com().Dispatch, "WIA.DeviceManager")
        out = []
        for i in range(1, dm.DeviceInfos.Count + 1):
            info = dm.DeviceInfos.Item(i)
            if int(info.Type) != SCANNER_DEVICE_TYPE:
                continue
            name = str(info.Properties("Name").Value)
            out.append(DeviceInfo(f"wia:{info.DeviceID}", name, "", name, "wia"))
        return out

    def open(self, native_id: str) -> Scanner:
        return WiaScanner(native_id)
