"""A fake ``pytwain`` for tests: one simulated Epson-like flatbed.

Behaves like pytwain 2.3.0's SourceManager/Source for the calls the helper
makes.  Resolutions are a FIX32 range 50..6400 (packed like a real TW_RANGE),
colour depths 24 and 48; the 'scan' is a gradient of the requested frame.
Set FAKE_TWAIN_LOG to a file to record the capabilities that were set.
Custom capability 0x8030 plays Epson Scan 2's "Scanning Quality" (0 standard,
1 high): its window sets it to 1.

TWAIN states (4 open, 5 enabled, 6 image ready, 7 sending) are checked like
the source manager does: MSG_RESET only in 6, MSG_DISABLEDS only in 5.
FAKE_TWAIN_DEVICE names a file that plays the scanner itself: "busy" from
enabling the source until it is disabled again.  A helper killed in between
leaves it busy, and the next scan fails like the real V600 did ("does not
respond") until the test resets the file (= pulling the power cable).
FAKE_TWAIN_CALLS: a file the TWAIN calls are appended to (ENABLEDS, STRIP,
ENDXFER, RESET, DISABLEDS, CLOSEDS, CLOSEDSM).  FAKE_TWAIN_WARMUP: seconds
before the image is ready (state 5); a WM_QUIT for the thread ends the
wait, as it ends pytwain's message loop (Windows only).  FAKE_TWAIN_XFER_DELAY:
seconds a native / file transfer takes.
"""
import json
import os
import sys
import time

QUALITY = 0x8030

from . import windows  # noqa: F401
from .lowlevel import constants as C


class CheckStatus(Exception):
    pass


class CapSeqError(Exception):
    pass


class SeqError(Exception):
    """A TWAIN call in the wrong state."""


def _calls(name):
    path = os.environ.get("FAKE_TWAIN_CALLS")
    if path:
        with open(path, "a") as f:
            f.write(name + "\n")


def _device(state=None):
    """The fake scanner: "idle" or "busy" (read, or set to ``state``)."""
    path = os.environ.get("FAKE_TWAIN_DEVICE")
    if not path:
        return "idle"
    if state is not None:
        with open(path, "w") as f:
            f.write(state)
        return state
    try:
        with open(path) as f:
            return f.read().strip() or "idle"
    except OSError:
        return "idle"


def _quit_posted():
    """True when a WM_QUIT is waiting for this thread (what ends pytwain's GetMessage loop)."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    msg = wintypes.MSG()
    while ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE
        if msg.message == 0x0012:
            return True
    return False


def _width_limit(frame, mech=None):
    """FAKE_TWAIN_MAX_WIDTH_PX: the highest resolution depends on the scan area.
    FAKE_TWAIN_MAX_BYTES: like old EPSON Scan, images (24-bit) over this size only
    with a file transfer (whole bed: ~1184 dpi through memory, Epson Scan's own
    window saves 800 MB)."""
    px = os.environ.get("FAKE_TWAIN_MAX_WIDTH_PX")
    lim = float(px) / max(frame[2] - frame[0], 1e-6) if px else float("inf")
    mb = os.environ.get("FAKE_TWAIN_MAX_BYTES")
    if mb and mech != C.TWSX_FILE:
        area = max((frame[2] - frame[0]) * (frame[3] - frame[1]), 1e-6)
        lim = min(lim, (float(mb) / 3 / area) ** 0.5)
    return lim


def _pack_fix(v):
    whole = int(v)
    return (whole & 0xFFFF) | (int(round((v - whole) * 65536)) << 16)


class Source:
    def __init__(self, name, ident=None):
        if ident is not None:  # Source(sm, identity), as pytwain's own code makes it
            name = ident.ProductName.decode()
            self.family = ident.ProductFamily.decode()
        else:
            self.family = None
        self.name = name
        self.caps = {C.ICAP_PIXELTYPE: C.TWPT_RGB, C.ICAP_BITDEPTH: 24, C.ICAP_XRESOLUTION: 300.0,
                     C.ICAP_XFERMECH: C.TWSX_NATIVE, QUALITY: 0}
        self.frame = (0.0, 0.0, 8.5, 11.7)
        self.set_log = []
        self.file_xfer_params = None
        if _device() == "busy":
            raise Exception("the scanner does not respond (left in the middle of a scan)")
        self.state = 4

    def _to(self, state):
        self.state = state
        _device("busy" if state >= 5 else "idle")

    @property
    def identity(self):
        return {"ProductName": self.name, "Manufacturer": "EPSON"}

    def get_capability(self, cap):
        if cap == C.ICAP_XRESOLUTION:
            # FAKE_TWAIN_LISTED_MAX: a driver that lists less than it takes (old EPSON Scan).
            listed = min(float(os.environ.get("FAKE_TWAIN_LISTED_MAX", 6400)), _width_limit(self.frame, self.caps.get(C.ICAP_XFERMECH)))
            listed = float(int(listed))
            return {"MinValue": _pack_fix(50), "MaxValue": _pack_fix(listed), "StepSize": _pack_fix(1),
                    "DefaultValue": _pack_fix(300), "CurrentValue": _pack_fix(300)}
        if cap == C.ICAP_PIXELTYPE:
            return C.TWTY_UINT16, (0, 0, [C.TWPT_BW, C.TWPT_GRAY, C.TWPT_RGB])
        if cap == C.ICAP_BITDEPTH:
            return C.TWTY_UINT16, (0, 0, [24, 48] if self.caps[C.ICAP_PIXELTYPE] == C.TWPT_RGB else [8, 16])
        if cap == C.ICAP_XFERMECH:
            mechs = [C.TWSX_NATIVE, C.TWSX_FILE] + ([] if os.environ.get("FAKE_TWAIN_NO_MEMORY") else [C.TWSX_MEMORY])
            return C.TWTY_UINT16, (0, 0, mechs)
        if cap == C.ICAP_IMAGEFILEFORMAT:  # FAKE_TWAIN_FILE_FORMATS="2": BMP only (old EPSON Scan)
            fmts = [int(v) for v in os.environ.get("FAKE_TWAIN_FILE_FORMATS", "0,2").split(",")]
            return C.TWTY_UINT16, (0, 0, fmts)
        if cap == C.CAP_SUPPORTEDCAPS:
            return C.TWTY_UINT16, [C.ICAP_XRESOLUTION, C.ICAP_BITDEPTH, QUALITY]
        if cap == QUALITY:
            return C.TWTY_UINT16, (self.caps[QUALITY], 0, [0, 1])
        raise Exception(f"capability {cap} not supported")

    def get_capability_current(self, cap):
        if cap in (C.ICAP_PHYSICALWIDTH, C.ICAP_PHYSICALHEIGHT):
            return C.TWTY_FIX32, 8.5 if cap == C.ICAP_PHYSICALWIDTH else 11.7
        if cap == C.ICAP_XRESOLUTION:
            return C.TWTY_FIX32, float(self.caps[C.ICAP_XRESOLUTION])
        if cap == C.CAP_UICONTROLLABLE:
            return C.TWTY_BOOL, True
        if cap == QUALITY:
            return C.TWTY_UINT16, self.caps[QUALITY]
        if cap == C.ICAP_BITDEPTH:
            return C.TWTY_UINT16, self.caps[C.ICAP_BITDEPTH]
        raise Exception(f"capability {cap} not supported")

    def set_capability(self, cap, type_id, value):
        self.set_log.append([cap, value])
        if cap == C.ICAP_GAMMA:
            raise Exception("not supported")
        if cap == C.ICAP_XRESOLUTION:  # a real driver takes the nearest value it can do
            if float(value) > _width_limit(self.frame, self.caps.get(C.ICAP_XFERMECH)):
                raise CapSeqError("")
            value = min(float(value), float(os.environ.get("FAKE_TWAIN_REAL_MAX", 6400)))
        self.caps[cap] = value

    def set_image_layout(self, frame):
        # FAKE_TWAIN_IGNORES_FRAME: a driver that keeps scanning the whole bed
        # unless its paper-size preset is switched off (like many real ones).
        if os.environ.get("FAKE_TWAIN_IGNORES_FRAME") == "always" or (
                os.environ.get("FAKE_TWAIN_IGNORES_FRAME") and self.caps.get(C.ICAP_SUPPORTEDSIZES) != C.TWSS_NONE):
            return
        self.frame = frame

    def get_image_layout(self):
        return self.frame, 1, 1, 1

    @property
    def image_info(self):
        h, w = self._image().shape[:2]
        return {"BitsPerPixel": self.caps[C.ICAP_BITDEPTH], "ImageWidth": w, "ImageLength": h,
                "SamplesPerPixel": 3 if self.caps[C.ICAP_PIXELTYPE] == C.TWPT_RGB else 1}

    # Memory transfer (strips), like a real driver: DAT_SETUPMEMXFER, then
    # DAT_IMAGEMEMXFER until TWRC_XFERDONE.  FAKE_TWAIN_STRIP_DELAY slows it.
    def _call(self, dg, dat, msg, buf, expected_returns=()):
        import ctypes as ct
        import time

        import numpy as np

        obj = buf._obj
        if os.environ.get("FAKE_TWAIN_MEMORY_FAILS"):
            raise Exception("memory transfer not working")
        if self.state not in (6, 7):
            raise SeqError(f"memory transfer in state {self.state}")
        if dat == C.DAT_SETUPMEMXFER:
            img = self._mem_image()
            obj.MinBufSize, obj.MaxBufSize, obj.Preferred = 0, 0xFFFFFFFF, img.shape[1] * img[0, 0].nbytes * 7
            return C.TWRC_SUCCESS
        assert dat == C.DAT_IMAGEMEMXFER
        img = self._mem_image()
        row_bytes = img.shape[1] * img[0, 0].nbytes
        y = getattr(self, "_mem_y", 0)
        rows = min(obj.Memory.Length // row_bytes, img.shape[0] - y)
        data = img[y:y + rows].astype("<u2" if img.dtype == np.uint16 else np.uint8).tobytes()
        ct.memmove(obj.Memory.TheMem, data, len(data))
        obj.BytesPerRow, obj.Columns, obj.Rows, obj.XOffset, obj.YOffset = row_bytes, img.shape[1], rows, 0, y
        obj.BytesWritten = len(data)
        self._mem_y = y + rows
        self.strips = getattr(self, "strips", 0) + 1
        self.state = 7
        _calls("STRIP")
        time.sleep(float(os.environ.get("FAKE_TWAIN_STRIP_DELAY", "0")))
        return C.TWRC_XFERDONE if self._mem_y >= img.shape[0] else C.TWRC_SUCCESS

    def _mem_image(self):
        import numpy as np

        if getattr(self, "_mem", None) is None:
            img = self._image()
            self._mem = ((img * 65535 + 0.5).astype(np.uint16) if self.caps[C.ICAP_BITDEPTH] == 48
                         else (img * 255 + 0.5).astype(np.uint8))
        return self._mem

    def _end_xfer(self):
        if self.state not in (6, 7):
            raise SeqError(f"MSG_ENDXFER in state {self.state}")
        _calls("ENDXFER")
        self._to(5)
        return 0

    def _image(self):
        import numpy as np

        dpi = float(self.caps[C.ICAP_XRESOLUTION])
        l, t, r, b = self.frame
        w, h = max(1, int(round((r - l) * dpi))), max(1, int(round((b - t) * dpi)))
        img = np.zeros((h, w, 3), np.float32)
        img[:, :, 0] = np.linspace(0, 1, w)[None, :]
        img[:, :, 1] = np.linspace(0, 1, h)[:, None]
        img[:, :, 2] = 0.5
        return img

    def _acquire(self, ready, show_ui=False, modal=False):
        if os.environ.get("FAKE_TWAIN_NEEDS_UI") and not show_ui:
            raise Exception("the driver needs its own window")
        if show_ui:
            self.caps[QUALITY] = 1  # the user picks "High" in the driver's window
        log = os.environ.get("FAKE_TWAIN_LOG")
        if log:
            with open(log, "w") as f:
                json.dump({"set": self.set_log, "show_ui": show_ui, "quality": self.caps[QUALITY],
                           "xfermech": self.caps.get(C.ICAP_XFERMECH), "family": self.family}, f)
        # Like pytwain: enable, message loop until the image is ready, then the
        # callback while it says there are more; disabled at the end in any case.
        _calls("ENABLEDS")
        self._to(5)
        try:
            end = time.monotonic() + float(os.environ.get("FAKE_TWAIN_WARMUP", "0"))
            while time.monotonic() < end:
                if _quit_posted():
                    return
                time.sleep(0.02)
            self._to(6)
            more = 1
            while more:
                more = ready()
        finally:
            self._disable()

    def _disable(self):
        if self.state != 5:
            raise SeqError(f"MSG_DISABLEDS in state {self.state}")
        _calls("DISABLEDS")
        self._to(4)

    def xfer_image_natively(self):
        import cv2
        import numpy as np

        self._one_block()
        img = (self._image() * 255 + 0.5).astype(np.uint8)
        ok, enc = cv2.imencode(".bmp", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self._end_xfer()  # pytwain ends the transfer itself
        return enc.tobytes(), 0

    def _one_block(self):
        if self.state != 6:
            raise SeqError(f"transfer in state {self.state}")
        self.state = 7
        time.sleep(float(os.environ.get("FAKE_TWAIN_XFER_DELAY", "0")))

    def xfer_image_by_file(self):
        import cv2
        import numpy as np

        path, fmt = self.file_xfer_params
        self._one_block()
        self.file_formats_used = getattr(self, "file_formats_used", []) + [fmt]
        if fmt == C.TWFF_BMP or self.caps[C.ICAP_BITDEPTH] in (24, 8):
            img = (self._image() * 255 + 0.5).astype(np.uint8)
        else:
            img = (self._image() * 65535 + 0.5).astype(np.uint16)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self._end_xfer()
        return 0

    def _end_all_xfers(self):
        if self.state != 6:
            raise SeqError(f"MSG_RESET in state {self.state}")
        _calls("RESET")
        self._to(5)

    def close(self):
        # Like pytwain: end pending transfers / disable first, then close the source.
        if self.state == 6:
            self._end_all_xfers()
        if self.state == 5:
            self._disable()
        if self.state == 4:
            _calls("CLOSEDS")
            self.state = 3


def _families():
    """FAKE_TWAIN_FAMILIES="EPSON Scan 2,EPSON Scan": two drivers of the same name."""
    fams = os.environ.get("FAKE_TWAIN_FAMILIES")
    return fams.split(",") if fams else None


class SourceManager:
    def __init__(self, parent_window=None, **kw):
        self._sources = set()
        self._next = 0

    @property
    def source_list(self):
        return ["EPSON Perfection V600"] * len(_families() or [1])

    def _call(self, dest, dg, dat, msg, buf, expected=()):
        fams = _families()
        if fams is None or dat != C.DAT_IDENTITY:
            raise Exception("not supported by the fake")
        self._next = 0 if msg == C.MSG_GETFIRST else self._next + 1
        if self._next >= len(fams):
            return C.TWRC_ENDOFLIST
        ds = buf._obj
        ds.Id, ds.ProductName, ds.ProductFamily = self._next + 1, b"EPSON Perfection V600", fams[self._next].encode()
        ds.Manufacturer = b"EPSON"
        return C.TWRC_SUCCESS

    def _open_ds(self, ident):
        pass

    def open_source(self, name=None):
        if name not in self.source_list:
            return None
        src = Source(name)
        self._sources.add(src)
        return src

    def close(self):
        while self._sources:
            self._sources.pop().close()
        _calls("CLOSEDSM")


import types as _types  # noqa: E402

from .lowlevel import structs as _structs  # noqa: E402

lowlevel = _types.SimpleNamespace(constants=C, structs=_structs)
