"""TWAIN helper: talks to one TWAIN driver and exits.

Run as a separate process so that it can use a different Python than the
app: many scanner drivers (e.g. Epson Scan for the Perfection V600) are 32-bit
only, and a 64-bit program cannot load them.  It needs only the standard
library and ``pytwain`` (``python -m pip install pytwain``), so a 32-bit
Python with just that package installed is enough.

    python twain_helper.py '{"cmd": "sources"}'
    python twain_helper.py '{"cmd": "info", "source": "EPSON Perfection V600"}'
    python twain_helper.py '{"cmd": "scan", "source": ..., "dpi": 600, "mode": "color",
                             "bits": 8, "frame": [l, t, r, b], "raw": false,
                             "show_ui": false, "out": "C:/.../scan"}'

The answer is one line of JSON on stdout: ``{"ok": true, ...}`` or
``{"ok": false, "error": "...", "trace": "..."}``.  Every answer includes the
helper's ``bits`` (32 or 64) and Python version.  Frames are in inches.

Cancel: a scan request may name a ``cancel_file``.  When the app creates that
file the helper stops the scan the TWAIN way - no more strips, the transfer
ended (MSG_ENDXFER / MSG_RESET), the source disabled and closed, the source
manager closed - and answers ``{"cancelled": true}``.  Killing the helper in
the middle of a transfer instead left the Epson V600 hung until its power
cable was pulled.  While the helper is inside a driver call that cannot be
interrupted (a strip, a one-block transfer, closing down) ``<cancel_file>.busy``
exists, so the app knows to wait rather than kill it.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import sys
import threading
import time
import traceback
from contextlib import contextmanager

T0 = time.monotonic()  # when the helper started: timings in the answer are from here

BITS = struct.calcsize("P") * 8
WM_QUIT = 0x0012


def _thread_id() -> int:
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetCurrentThreadId())
    except Exception:
        return 0


def _post_quit(thread_id: int) -> None:
    """End the TWAIN message loop of that thread (pytwain's GetMessage loop stops on WM_QUIT)."""
    try:
        import ctypes

        ctypes.windll.user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0)
    except Exception:
        pass


class Cancel:
    """The app's Cancel button, seen from the helper: the cancel file appearing.

    The main thread checks :meth:`is_set` between strips and before a
    transfer.  While it waits for the driver to get ready (state 5, in
    pytwain's message loop) a WM_QUIT ends that loop, which disables the source.
    """

    def __init__(self, path: str | None, log: list):
        self.path, self.log = path, log
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.waiting = False  # in the message loop, before the driver has the image ready
        self.tid = _thread_id()
        if path:
            threading.Thread(target=self._watch, daemon=True).start()

    def is_set(self) -> bool:
        return self.event.is_set()

    def _watch(self) -> None:
        while not os.path.exists(self.path):
            time.sleep(0.05)
        with self.lock:
            self.event.set()
            self.log.append(f"cancel asked after {time.monotonic() - T0:.2f} s"
                            + (" (waiting for the driver)" if self.waiting else ""))
            if self.waiting:
                _post_quit(self.tid)

    def set_waiting(self, on: bool) -> None:
        with self.lock:
            self.waiting = on
            if on and self.event.is_set():
                _post_quit(self.tid)

    def _busy(self, what: str | None) -> None:
        if not self.path:
            return
        try:
            if what is None:
                os.remove(self.path + ".busy")
            else:
                with open(self.path + ".busy", "w") as f:
                    f.write(what)
        except OSError:
            pass

    @contextmanager
    def driver_call(self, what: str):
        """Mark a driver call that cannot be interrupted (the app waits instead of killing)."""
        self._busy(what)
        try:
            yield
        finally:
            if not self.is_set():  # once cancelled, closing down counts as busy until the exit
                self._busy(None)

    def closing(self) -> None:
        self._busy("closing")


def _abort_transfer(src, log: list, mid_image: bool) -> None:
    """End the transfer after a cancel: back to state 5 (enabled), each step tried on its own."""
    if mid_image:  # state 7: MSG_ENDXFER aborts the image being sent
        try:
            more = src._end_xfer()
            log.append(f"cancel: MSG_ENDXFER ok ({more} more)")
        except Exception as e:
            more = -1
            log.append(f"cancel: MSG_ENDXFER {e.__class__.__name__} {e}")
        if not more:
            return
    try:  # state 6: no more images
        src._end_all_xfers()
        log.append("cancel: MSG_RESET ok")
    except Exception as e:
        log.append(f"cancel: MSG_RESET {e.__class__.__name__} {e}")


def _hidden_window() -> int:
    """A window handle for the driver to attach to (some drivers need one)."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.CreateWindowExW.restype = ctypes.c_void_p
        hwnd = user32.CreateWindowExW(0, "STATIC", "scanner-twain", 0, 0, 0, 0, 0, None, None, None, None)
        return int(hwnd or 0)
    except Exception:
        return 0


def _fix(v) -> float:
    """TW_FIX32 (struct, or packed into a uint32 inside a TW_RANGE) to float."""
    if hasattr(v, "Whole"):
        return float(v.Whole) + float(v.Frac) / 65536.0
    v = int(v) & 0xFFFFFFFF
    whole = v & 0xFFFF
    if whole >= 0x8000:
        whole -= 0x10000
    return whole + (v >> 16) / 65536.0


def _values(src, cap, fix: bool = False):
    """Allowed values of a capability: a list, or (min, max, step) for a range, or None."""
    try:
        r = src.get_capability(cap)
    except Exception:
        return None
    conv = _fix if fix else (lambda x: x)
    if isinstance(r, dict):  # range
        return {"min": conv(r["MinValue"]), "max": conv(r["MaxValue"]), "step": conv(r["StepSize"])}
    type_id, val = r
    if isinstance(val, tuple) and len(val) == 3 and isinstance(val[2], list):  # enumeration
        return [conv(x) for x in val[2]]
    if isinstance(val, list):  # array
        return [conv(x) for x in val]
    return [conv(val) if not isinstance(val, float) else val]


def _current(src, cap, fix: bool = False):
    try:
        type_id, val = src.get_capability_current(cap)
        return _fix(val) if fix and not isinstance(val, float) else val
    except Exception:
        return None


def _scalar(src, cap):
    """Current value of a capability as [type, value], or None if it is not a single number."""
    try:
        type_id, val = src.get_capability_current(cap)
    except Exception:
        return None
    if isinstance(val, tuple) and len(val) == 3 and isinstance(val[2], list):  # enumeration
        val = val[2][val[0]]
    if hasattr(val, "Whole"):
        val = _fix(val)
    if isinstance(val, (bool, int, float)):
        return [int(type_id), val]
    return None


def _driver_caps(src, C) -> dict:
    """The driver's own settings: its custom capabilities (ids >= 0x8000) -> [type, value].

    These hold what only the driver's window offers, such as Epson Scan 2's
    "Scanning Quality" (Standard / High), which changes how the scanner moves.
    """
    out = {}
    for cap in _values(src, C.CAP_SUPPORTEDCAPS) or []:
        if int(cap) >= 0x8000:
            v = _scalar(src, int(cap))
            if v is not None:
                out[str(int(cap))] = v
    return out


def _try_set(src, cap, type_id, value, log: list) -> bool:
    try:
        src.set_capability(cap, type_id, value)
        log.append(f"{cap:#06x} = {value}: ok")
        return True
    except Exception as e:  # CheckStatus means "set to a close value"
        log.append(f"{cap:#06x} = {value}: {e.__class__.__name__} {e}")
        return e.__class__.__name__ == "CheckStatus"


def _set_frame(src, C, frame, log: list) -> None:
    """Scan area in inches (left, top, right, bottom)."""
    # Drivers use a paper-size preset or detect the document by
    # themselves unless told not to, and then ignore the layout.
    _try_set(src, C.ICAP_SUPPORTEDSIZES, C.TWTY_UINT16, C.TWSS_NONE, log)
    _try_set(src, C.ICAP_AUTOMATICBORDERDETECTION, C.TWTY_BOOL, False, log)
    _try_set(src, C.ICAP_AUTOSIZE, C.TWTY_UINT16, C.TWAS_NONE, log)
    try:
        src.set_image_layout(tuple(float(v) for v in frame))
        log.append(f"frame {list(frame)}: ok")
    except Exception as e:
        log.append(f"frame {list(frame)}: {e.__class__.__name__} {e}")


def _xfer_memory(src, C, out: str, cancel: Cancel | None = None, log: list | None = None) -> dict:
    """Buffered memory transfer: the image arrives in strips while the scanner moves.

    ``<out>.json`` (width, height, bits per pixel, channels) is written first,
    then every strip goes straight into ``<out>.raw`` (rows of chunky RGB or
    grey, top to bottom, 16-bit little-endian), so the app can show the scan
    as it comes in.  Returns the header; ``cancelled`` is set when the driver
    or the app (``cancel``, checked between strips) stopped the scan.
    """
    cancel = cancel or Cancel(None, [])
    log = log if log is not None else []
    import ctypes as ct

    u16, u32 = ct.c_uint16, ct.c_uint32

    class SETUPMEMXFER(ct.Structure):
        _pack_ = 2
        _fields_ = [("MinBufSize", u32), ("MaxBufSize", u32), ("Preferred", u32)]

    class MEMORY(ct.Structure):
        _pack_ = 2
        _fields_ = [("Flags", u32), ("Length", u32), ("TheMem", ct.c_void_p)]

    class IMAGEMEMXFER(ct.Structure):
        _pack_ = 2
        _fields_ = [("Compression", u16), ("BytesPerRow", u32), ("Columns", u32), ("Rows", u32),
                    ("XOffset", u32), ("YOffset", u32), ("BytesWritten", u32), ("Memory", MEMORY)]

    info = src.image_info
    head = {"width": int(info["ImageWidth"]), "height": int(info["ImageLength"]),
            "bpp": int(info["BitsPerPixel"]), "channels": int(info.get("SamplesPerPixel") or 3)}
    head["strips"] = []
    row_bytes = (head["width"] * head["bpp"] + 7) // 8
    setup = SETUPMEMXFER()
    src._call(C.DG_CONTROL, C.DAT_SETUPMEMXFER, C.MSG_GET, ct.byref(setup))
    size = setup.Preferred or setup.MinBufSize or (1 << 20)
    size = max(size, row_bytes, setup.MinBufSize)
    if setup.MaxBufSize not in (0, 0xFFFFFFFF):
        size = min(size, setup.MaxBufSize)
    buf = ct.create_string_buffer(size)
    with open(out + ".json", "w") as f:
        json.dump(head, f)
    with open(out + ".raw", "wb") as f:
        while True:
            if cancel.is_set():
                head["cancelled"] = True
                log.append(f"cancel: stopped after {len(head['strips'])} strips")
                cancel.closing()
                _abort_transfer(src, log, mid_image=True)
                return head
            mx = IMAGEMEMXFER()
            mx.Memory.Flags = C.TWMF_APPOWNS | C.TWMF_POINTER
            mx.Memory.Length = size
            mx.Memory.TheMem = ct.addressof(buf)
            with cancel.driver_call("strip"):
                rv = src._call(C.DG_IMAGE, C.DAT_IMAGEMEMXFER, C.MSG_GET, ct.byref(mx),
                               (C.TWRC_XFERDONE, C.TWRC_CANCEL))
            if rv == C.TWRC_CANCEL:
                head["cancelled"] = True
                break
            if len(head["strips"]) < 400:  # [seconds since start, first row, rows]
                head["strips"].append([round(time.monotonic() - T0, 2), mx.YOffset, mx.Rows])
            data = buf.raw[:mx.BytesWritten]
            used = (mx.Columns * head["bpp"] + 7) // 8
            for r in range(mx.Rows):
                f.seek((mx.YOffset + r) * row_bytes + (mx.XOffset * head["bpp"]) // 8)
                f.write(data[r * mx.BytesPerRow:r * mx.BytesPerRow + used])
            f.flush()
            if rv == C.TWRC_XFERDONE:
                break
    head["more"] = src._end_xfer()
    return head


def _open(req):
    import twain

    # Still "Scanner" after the rename to Hasami Scan: drivers may keep settings per program name.
    sm = twain.SourceManager(_hidden_window(), ProductName="Scanner")
    return twain, sm


def _identities(twain, sm) -> list:
    """(identity, info) for every source the TWAIN manager knows, without opening them.

    Two drivers for the same scanner can have the same name (the Epson
    Perfection V600 has "EPSON Scan" and "EPSON Scan 2"); their family tells
    them apart.
    """
    import ctypes as ct

    dec = getattr(sm, "_decode", lambda b: b.decode("latin-1", "replace"))
    out = []
    try:
        C, S = twain.lowlevel.constants, twain.lowlevel.structs
        ds = S.TW_IDENTITY()
        rv = sm._call(None, C.DG_CONTROL, C.DAT_IDENTITY, C.MSG_GETFIRST, ct.byref(ds),
                      (C.TWRC_SUCCESS, C.TWRC_ENDOFLIST))
        while rv != C.TWRC_ENDOFLIST:
            ident = S.TW_IDENTITY.from_buffer_copy(ds)
            out.append((ident, {"name": dec(ds.ProductName), "family": dec(ds.ProductFamily),
                                "manufacturer": dec(ds.Manufacturer), "version": dec(ds.Version.Info)}))
            rv = sm._call(None, C.DG_CONTROL, C.DAT_IDENTITY, C.MSG_GETNEXT, ct.byref(ds),
                          (C.TWRC_SUCCESS, C.TWRC_ENDOFLIST))
    except Exception:
        return []
    return out


def cmd_sources(req) -> dict:
    twain, sm = _open(req)
    try:
        answer = {"sources": list(sm.source_list)}
        answer["identities"] = [info for _, info in _identities(twain, sm)]
        return answer
    finally:
        sm.close()


def _source(twain, sm, name, family=None):
    """Open the source called ``name`` (of driver family ``family``, when given)."""
    if family:
        for ident, info in _identities(twain, sm):
            if info["name"] == name and info["family"] == family:
                sm._open_ds(ident)
                src = twain.Source(sm, ident)
                sm._sources.add(src)
                return src
        raise RuntimeError(f"TWAIN source not found: {name} ({family})")
    src = sm.open_source(name)
    if src is None:
        raise RuntimeError(f"TWAIN source not found: {name}")
    return src


def cmd_info(req) -> dict:
    twain, sm = _open(req)
    C = twain.lowlevel.constants
    try:
        src = _source(twain, sm, req.get("source"), req.get("family"))
        try:
            log: list = []
            _try_set(src, C.ICAP_UNITS, C.TWTY_UINT16, C.TWUN_INCHES, log)
            info = {
                "identity": {k: (v.decode(errors="replace") if isinstance(v, bytes) else v)
                             for k, v in src.identity.items() if isinstance(v, (str, bytes, int))},
                "resolutions": _values(src, C.ICAP_XRESOLUTION, fix=True),
                "pixel_types": _values(src, C.ICAP_PIXELTYPE),
                "xfer_mechs": _values(src, C.ICAP_XFERMECH),
                "physical": [_current(src, C.ICAP_PHYSICALWIDTH, True), _current(src, C.ICAP_PHYSICALHEIGHT, True)],
                "ui_controllable": _current(src, C.CAP_UICONTROLLABLE),
                "gamma": _values(src, C.ICAP_GAMMA, fix=True),
                "supported_caps": _values(src, C.CAP_SUPPORTEDCAPS),
                "driver_caps": _driver_caps(src, C),
            }
            info["driver_cap_values"] = {k: _values(src, int(k)) for k in info["driver_caps"]}
            depths = {}
            for name, pt in (("color", C.TWPT_RGB), ("gray", C.TWPT_GRAY)):
                if _try_set(src, C.ICAP_PIXELTYPE, C.TWTY_UINT16, pt, log):
                    depths[name] = _values(src, C.ICAP_BITDEPTH)
            info["bit_depths"] = depths
            if req.get("probe_dpi"):
                # The highest resolution can depend on the scan area (old EPSON Scan: 1184 dpi
                # for the whole bed, its window offers 12800 for small areas).  Ask again for
                # a 1-inch area, and set each resolution and read it back.
                _try_set(src, C.ICAP_PIXELTYPE, C.TWTY_UINT16, C.TWPT_RGB, log)
                _set_frame(src, C, [0.0, 0.0, 1.0, 1.0], log)
                info["resolutions_small_area"] = _values(src, C.ICAP_XRESOLUTION, fix=True)
                info["bit_depths_small_area"] = _values(src, C.ICAP_BITDEPTH)
                info["takes_48_bit"] = _try_set(src, C.ICAP_BITDEPTH, C.TWTY_UINT16, 48, log) and \
                    _current(src, C.ICAP_BITDEPTH) == 48
                _try_set(src, C.ICAP_BITDEPTH, C.TWTY_UINT16, 24, log)
                takes = {}
                for d in req["probe_dpi"]:
                    _try_set(src, C.ICAP_XRESOLUTION, C.TWTY_FIX32, float(d), log)
                    takes[d] = _current(src, C.ICAP_XRESOLUTION, True)
                info["resolution_set_then_read"] = takes
                # Whole bed: the highest resolution with each way of transferring the image.
                by_mech = {}
                phys = [_current(src, C.ICAP_PHYSICALWIDTH, True), _current(src, C.ICAP_PHYSICALHEIGHT, True)]
                if all(phys):
                    for name, mech in (("native", C.TWSX_NATIVE), ("file", C.TWSX_FILE), ("memory", C.TWSX_MEMORY)):
                        if mech in (info["xfer_mechs"] or []) and _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, mech, log):
                            _set_frame(src, C, [0.0, 0.0, phys[0], phys[1]], log)
                            r = _values(src, C.ICAP_XRESOLUTION, fix=True)
                            by_mech[name] = r.get("max") if isinstance(r, dict) else r
                    _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_NATIVE, log)
                info["whole_bed_max_dpi_by_transfer"] = by_mech
                info["accepted_dpi"] = [d for d, v in takes.items() if v is not None and abs(float(v) - d) < 0.5]
            if C.TWSX_FILE in (info["xfer_mechs"] or []):
                _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_FILE, log)
                info["file_formats"] = _values(src, C.ICAP_IMAGEFILEFORMAT)
            info["log"] = log
            return info
        finally:
            src.close()
    finally:
        sm.close()


def cmd_scan(req) -> dict:
    twain, sm = _open(req)
    C = twain.lowlevel.constants
    log: list = []
    cancel = Cancel(req.get("cancel_file"), log)
    try:
        src = _source(twain, sm, req.get("source"), req.get("family"))
        try:
            show_ui = bool(req.get("show_ui"))
            if not show_ui:
                # What the user last chose in the driver's window (e.g. Scanning
                # Quality), first: the settings below take precedence.
                for cap, (type_id, value) in (req.get("driver_caps") or {}).items():
                    _try_set(src, int(cap), int(type_id), value, log)
            mode = req.get("mode", "color")
            bits = int(req.get("bits", 8))
            pixel = {"color": C.TWPT_RGB, "gray": C.TWPT_GRAY, "lineart": C.TWPT_BW}[mode]
            depth = {("color", 8): 24, ("color", 16): 48, ("gray", 8): 8, ("gray", 16): 16}.get((mode, bits), 1)
            _try_set(src, C.ICAP_UNITS, C.TWTY_UINT16, C.TWUN_INCHES, log)
            _try_set(src, C.ICAP_PIXELTYPE, C.TWTY_UINT16, pixel, log)
            if not _try_set(src, C.ICAP_BITDEPTH, C.TWTY_UINT16, depth, log) and depth in (48, 16):
                depth = 24 if mode == "color" else 8
                _try_set(src, C.ICAP_BITDEPTH, C.TWTY_UINT16, depth, log)
            # The transfer first: the highest resolution can depend on it (old EPSON Scan
            # limits images to ~400 MB, at least for native transfers).
            mechs = _values(src, C.ICAP_XFERMECH) or []
            # Memory transfer delivers the image in strips (shown live); not for 1-bit.
            use_memory = bool(req.get("live")) and depth >= 8 and C.TWSX_MEMORY in mechs
            if use_memory:
                use_memory = _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_MEMORY, log)
                if use_memory:
                    _try_set(src, C.ICAP_COMPRESSION, C.TWTY_UINT16, C.TWCP_NONE, log)
            use_file = not use_memory and depth in (48, 16) and C.TWSX_FILE in mechs
            if use_file:
                use_file = _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_FILE, log)
            dpi = float(req.get("dpi", 300))
            frame = req.get("frame")

            def area_and_resolution() -> bool:
                # The area first: some drivers allow high resolutions only for small areas
                # (old EPSON Scan), then again after the resolution (others clip it to it).
                if frame:
                    _set_frame(src, C, frame, log)
                _try_set(src, C.ICAP_XRESOLUTION, C.TWTY_FIX32, dpi, log)
                _try_set(src, C.ICAP_YRESOLUTION, C.TWTY_FIX32, dpi, log)
                if frame:
                    _set_frame(src, C, frame, log)
                got = _current(src, C.ICAP_XRESOLUTION, True)
                return got is None or abs(float(got) - dpi) < 0.5

            if not area_and_resolution() and not use_file and C.TWSX_FILE in mechs and not show_ui:
                # Too big for a transfer through memory: a file may be allowed more (Epson
                # Scan's own window saves 800 MB images).  No live view then.
                log.append(f"{dpi:g} dpi refused for this area with "
                           f"{'memory' if use_memory else 'native'} transfer: trying a file")
                if _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_FILE, log):
                    use_memory, use_file = False, True
                    if not area_and_resolution():
                        log.append(f"{dpi:g} dpi refused with a file too")
            _try_set(src, C.CAP_INDICATORS, C.TWTY_BOOL, False, log)  # no progress window
            if req.get("raw"):
                _try_set(src, C.ICAP_BRIGHTNESS, C.TWTY_FIX32, 0.0, log)
                _try_set(src, C.ICAP_CONTRAST, C.TWTY_FIX32, 0.0, log)
            try:
                layout = src.get_image_layout()[0]
                if frame and max(abs(a - b) for a, b in zip(layout, frame)) > 0.05:
                    log.append(f"the driver changed the scan area to {list(layout)}")
            except Exception:
                layout = None
            out = req["out"]
            log.append(f"live view asked: {bool(req.get('live'))}, transfer mechanisms {mechs}: using "
                       + ("memory (strips)" if use_memory else "file" if use_file else "native (one block)"))
            files: list[str] = []

            image_info: dict = {}
            raw_header: dict = {}

            def ready() -> int:
                cancel.set_waiting(False)
                image_info.setdefault("ready_after_s", round(time.monotonic() - T0, 2))
                if cancel.is_set():  # state 6: the image is ready, nothing sent yet
                    cancel.closing()
                    _abort_transfer(src, log, mid_image=False)
                    return 0
                try:
                    image_info.update(src.image_info)
                except Exception:
                    pass
                if use_memory:
                    head = _xfer_memory(src, C, out, cancel, log)
                    if head.get("cancelled"):
                        return 0
                    raw_header.update(head)
                    path, more = out + ".raw", head["more"]
                elif use_file:
                    # TIFF when offered (16-bit), else BMP (old EPSON Scan offers only BMP).
                    fmts = _values(src, C.ICAP_IMAGEFILEFORMAT) or [C.TWFF_TIFF]
                    fmt, ext = (C.TWFF_TIFF, ".tif") if C.TWFF_TIFF in fmts else (C.TWFF_BMP, ".bmp")
                    path = out + ext
                    src.file_xfer_params = (path, fmt)
                    with cancel.driver_call("transfer"):  # one block: cannot be stopped halfway
                        more = src.xfer_image_by_file()
                else:
                    with cancel.driver_call("transfer"):
                        handle, more = src.xfer_image_natively()
                    path = out + ".bmp"
                    with open(path, "wb") as f:
                        f.write(twain.windows.dib_to_bm_file(handle))
                    twain.windows.global_handle_free(handle)
                files.append(path)
                if more:
                    src._end_all_xfers()
                return 0

            if not use_file and not use_memory:
                _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_NATIVE, log)

            def acquire(show_ui: bool) -> None:
                # Waiting for the driver to get the image ready: a cancel ends the message loop.
                if cancel.is_set():
                    return
                cancel.set_waiting(True)
                try:
                    src._acquire(ready, show_ui=show_ui, modal=False)
                finally:
                    cancel.set_waiting(False)

            try:
                acquire(show_ui)
            except Exception as e:
                err = e
                if cancel.is_set():
                    log.append(f"after the cancel: {e.__class__.__name__} {e}")
                    err = None
                elif use_memory and not files:
                    # The strip-by-strip transfer failed: scan again the usual way.
                    log.append(f"memory transfer failed ({e.__class__.__name__} {e}); retrying without it")
                    use_memory, raw_header = False, {}
                    use_file = depth in (48, 16) and C.TWSX_FILE in mechs and _try_set(
                        src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_FILE, log)
                    if not use_file:
                        _try_set(src, C.ICAP_XFERMECH, C.TWTY_UINT16, C.TWSX_NATIVE, log)
                    try:
                        acquire(show_ui)
                        err = None
                    except Exception as e2:
                        err = None if cancel.is_set() else e2
                if err is not None:
                    if show_ui:
                        raise err
                    # Some drivers cannot scan without their own window.
                    log.append(f"scan without the driver's window failed ({err.__class__.__name__} {err}); "
                               "retrying with it")
                    show_ui = True
                    acquire(True)
            if cancel.is_set():
                return {"cancelled": True, "by_app": True, "log": log}
            if not files:
                return {"cancelled": True, "log": log}
            answer = {"file": files[0], "dpi": _current(src, C.ICAP_XRESOLUTION, True) or dpi,
                      "frame": list(layout) if layout else frame, "depth": depth, "show_ui": show_ui,
                      "image_info": image_info, "log": log}
            if raw_header:
                answer["raw"] = raw_header
            answer["done_after_s"] = round(time.monotonic() - T0, 2)
            if show_ui:
                # The window's own resolution and area replace the requested ones.
                try:
                    answer["dpi"] = float(image_info["XResolution"])
                    answer["frame"] = list(src.get_image_layout()[0])
                except Exception:
                    pass
                answer["driver_caps"] = _driver_caps(src, C)
            return answer
        finally:
            if cancel.is_set():
                cancel.closing()
            _close(src, "source", log, cancel)
    finally:
        _close(sm, "source manager", log, cancel)


def _close(obj, what: str, log: list, cancel: Cancel) -> None:
    """Close the source (disabling it first if needed) or the source manager; logged after a cancel."""
    try:
        obj.close()
        if cancel.is_set():
            log.append(f"cancel: {what} closed")
    except Exception as e:
        if not cancel.is_set():
            raise
        log.append(f"cancel: closing the {what}: {e.__class__.__name__} {e}")


COMMANDS = {"sources": cmd_sources, "info": cmd_info, "scan": cmd_scan}


def main(argv: list[str]) -> int:
    base = {"bits": BITS, "python": platform.python_version(), "executable": sys.executable}
    try:
        req = json.loads(argv[1]) if len(argv) > 1 else {"cmd": "sources"}
        try:
            import twain  # noqa: F401
        except ImportError:
            raise RuntimeError(f"pytwain is not installed for {sys.executable} "
                               f"({BITS}-bit): run  \"{sys.executable}\" -m pip install pytwain")
        result = COMMANDS[req.get("cmd", "sources")](req)
        result.update(base, ok=True)
    except Exception as e:
        result = dict(base, ok=False, error=f"{e.__class__.__name__}: {e}", trace=traceback.format_exc())
    sys.stdout.write(json.dumps(result, default=str) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
