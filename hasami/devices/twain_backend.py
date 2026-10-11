"""TWAIN scanners (Windows), through a helper process.

TWAIN is the driver interface most scanning programs use on Windows.  Compared
with WIA it usually exposes everything the scanner can do (e.g. the Epson
Perfection V600's full 6400 dpi, which its WIA driver limits to 1200).

The driver is loaded by ``twain_helper.py`` in a separate process: many TWAIN
drivers are 32-bit only and cannot be loaded by a 64-bit Python.  The helper
needs only ``pytwain``; it is run with this Python, or with a 32-bit Python
found through the ``py`` launcher (or the one set in ``input.twain_python``).
The installed app (a PyInstaller build, no Python) ships the helper as two
programs instead: ``twain64	wain_helper64.exe`` and ``twain32	wain_helper32.exe``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import (
    MM_PER_INCH,
    Backend,
    Capabilities,
    DeviceInfo,
    LiveImage,
    ProgressFn,
    ScanCancelled,
    Scanner,
    ScannerError,
    ScanRequest,
    ScanResult,
)

HELPER = Path(__file__).with_name("twain_helper.py")
STANDARD_DPI = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800)
_NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console flashing up for the helper
#: After Cancel, how long the helper may take to close the driver down when it is
#: not inside a driver call - it is killed after that, as a last resort.
CANCEL_GRACE = 15.0
#: The longest wait for a driver call that cannot be stopped (a whole image sent
#: in one block can take minutes at high resolutions).
CANCEL_LIMIT = 600.0
FORCED_STOP = ("The scanner driver did not stop and had to be ended by force. If the scanner no "
               "longer responds, close this app, unplug the scanner's power cable for 30 seconds "
               "and plug it back in (its power button is not enough).")

#: device id -> Python that sees it (filled by ``list_devices``).  The id is the
#: source's name, or "name [family]" for a second driver with the same name.
_SOURCES: dict[str, str] = {}
#: device id -> (source name, driver family or None).
_IDENT: dict[str, tuple[str, str | None]] = {}


#: The input settings the app is using right now (set by ``use_settings``).
_LIVE_SETTINGS = None


def use_settings(input_settings) -> None:
    """Read the TWAIN options (driver window, helper Python) from these settings.

    Without this the saved settings file is read, which the app writes only on
    exit: ticking "show the scanner's own window" did nothing until a restart.
    """
    global _LIVE_SETTINGS
    _LIVE_SETTINGS = input_settings


def _settings():
    if _LIVE_SETTINGS is not None:
        return _LIVE_SETTINGS
    try:
        from ..settings import load_settings

        return load_settings().input
    except Exception:
        return None


def parse_py_launcher(text: str) -> list[tuple[str, str]]:
    """``py -0p`` output -> [(tag, path)], e.g. ("3.12-32", "C:\\...\\python.exe")."""
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*-(?:V:)?(\S+?)\s+(?:\*\s+)?(\S.*?python\w*\.exe)\s*$", line, re.I)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def console_python(exe: str) -> str:
    """python.exe instead of pythonw.exe (what the app's window runs on).

    Scans run from the app's window - helper started with pythonw.exe - made
    the V600 sound different and sent no strips while scanning, unlike the
    same scan from ``scanner diag`` (python.exe).  The helper runs without a
    console window either way (CREATE_NO_WINDOW).
    """
    p = Path(exe)
    if p.name.lower() == "pythonw.exe":
        console = p.with_name("python.exe")
        if console.exists():
            return str(console)
    return exe


def app_dir() -> Path | None:
    """The installed app's folder (a frozen PyInstaller build), else None."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def bundled_helpers(folder: Path | None = None) -> list[str]:
    """The helper programs shipped with the installed app: 64-bit first, then 32-bit."""
    folder = folder or app_dir()
    if folder is None:
        return []
    exes = (folder / "twain64" / "twain_helper64.exe", folder / "twain32" / "twain_helper32.exe")
    return [str(p) for p in exes if p.is_file()]


def is_helper_exe(path: str) -> bool:
    """A built helper program (takes the request itself), not a Python."""
    return Path(path).name.lower().startswith("twain_helper")


def helper_command(python: str, request: dict) -> list[str]:
    """Command line that runs one helper request with ``python`` (a Python or a helper program)."""
    if is_helper_exe(python):
        return [python, json.dumps(request)]
    return [python, str(HELPER), json.dumps(request)]


def candidate_pythons() -> list[str]:
    """Pythons (or bundled helper programs) to run the helper with, best first."""
    st = _settings()
    if st is not None and st.twain_python:
        return [st.twain_python]
    if app_dir() is not None:
        return bundled_helpers()  # the installed app: no Python to run the script with
    found = [console_python(sys.executable)]
    try:
        r = subprocess.run(["py", "-0p"], capture_output=True, text=True, timeout=10,
                           creationflags=_NO_WINDOW if sys.platform == "win32" else 0)
        for tag, path in parse_py_launcher(r.stdout + r.stderr):
            if tag.endswith("-32") and path not in found:
                found.append(path)
    except (OSError, subprocess.SubprocessError):
        pass
    return found


def stop_helper(proc: subprocess.Popen, cancel_file: str | None) -> bool:
    """Ask the helper to stop the scan (the cancel file) and wait for it to exit.

    True when it closed the driver down by itself.  It is killed only when it
    sits for ``CANCEL_GRACE`` seconds without exiting outside a driver call
    (``<cancel_file>.busy``), or after ``CANCEL_LIMIT`` in all: a helper killed
    in the middle of a transfer left the Epson V600 hung until its power cable
    was pulled.
    """
    if cancel_file:
        try:
            Path(cancel_file).touch()
        except OSError:
            cancel_file = None
    start = active = time.monotonic()
    while proc.poll() is None:
        now = time.monotonic()
        if cancel_file and os.path.exists(cancel_file + ".busy"):
            active = now
        if not cancel_file or now - active > CANCEL_GRACE or now - start > CANCEL_LIMIT:
            proc.kill()
            proc.wait()
            return False
        time.sleep(0.05)
    return True


def run_helper(python: str, request: dict, cancel: threading.Event | None = None,
               timeout: float | None = 60.0, poll=None) -> dict:
    """Run one helper command; returns its JSON answer (raises on failure).

    ``poll()`` is called about ten times a second while the helper runs.  When
    ``cancel`` is set the helper is asked to stop (see :func:`stop_helper`) and
    :class:`ScanCancelled` is raised; its ``answer`` is the helper's answer.
    """
    # The answer and any messages go to files, not pipes: the answer of a large
    # scan (the time of every strip) filled the pipe, which is only read at the
    # end, so the helper waited for the app and the app for the helper ("stuck
    # at 99%").
    with tempfile.TemporaryDirectory(prefix="scanner-helper-") as tmp, \
            tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        if cancel is not None:
            request = dict(request, cancel_file=os.path.join(tmp, "cancel"))
        cmd = helper_command(python, request)
        try:
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f,
                                    creationflags=_NO_WINDOW if sys.platform == "win32" else 0)
        except OSError as e:
            raise ScannerError(f"TWAIN: cannot start {python}: {e}") from e
        start = time.monotonic()
        stopped = None  # None: ran to the end; True / False: cancelled, stopped cleanly / killed
        try:
            while proc.poll() is None:
                if cancel is not None and cancel.is_set():
                    stopped = stop_helper(proc, request.get("cancel_file"))
                    break
                if timeout is not None and time.monotonic() - start > timeout:
                    proc.kill()
                    proc.wait()
                    raise ScannerError(f"TWAIN: the scanner driver did not answer within {timeout:.0f} s")
                if poll is not None:
                    poll()
                time.sleep(0.1)
        except KeyboardInterrupt:  # Ctrl+C in the command line: stop the scan properly too
            stop_helper(proc, request.get("cancel_file"))
            raise
        out_f.seek(0)
        err_f.seek(0)
        out = out_f.read().decode("utf-8", "replace")
        err = err_f.read().decode("utf-8", "replace")
    lines = [x for x in out.splitlines() if x.startswith("{")]
    answer = json.loads(lines[-1]) if lines else {}
    if stopped is not None:
        e = ScanCancelled("scan cancelled")
        e.answer = answer
        e.forced = not stopped
        if e.forced:
            e.warning = FORCED_STOP
        raise e
    if not lines:
        raise ScannerError(f"TWAIN helper ({python}) gave no answer (exit code {proc.returncode}): {err.strip()[-500:]}")
    if not answer.get("ok"):
        raise ScannerError(f"TWAIN ({answer.get('bits')}-bit Python): {answer.get('error')}")
    return answer


def _dpi_list(res) -> list[int]:
    if isinstance(res, dict):
        lo, hi = float(res["min"]), float(res["max"])
        step = float(res.get("step") or 1) or 1
        return [d for d in STANDARD_DPI if lo <= d <= hi and abs((d - lo) / step - round((d - lo) / step)) < 1e-6]
    if isinstance(res, list):
        return sorted({int(round(float(v))) for v in res if float(v) > 0})
    return []


class RawStrips:
    """Reads the image the helper writes strip by strip (``<out>.json`` + ``<out>.raw``)."""

    def __init__(self, out: str, expected_height: int = 0):
        self.out = out
        self.expected_height = expected_height
        self.head: dict | None = None
        self.image: np.ndarray | None = None
        self.rows = 0
        self._pos = 0

    def _shape(self):
        h = self.head
        dtype = np.dtype("<u2" if h["bpp"] // max(1, h["channels"]) == 16 else np.uint8)
        return h["width"], h["channels"], dtype

    def read(self) -> int:
        """Take in what arrived since the last call; returns the number of rows received."""
        if self.head is None:
            try:
                with open(self.out + ".json", encoding="utf-8") as f:
                    self.head = json.load(f)
            except (OSError, ValueError):
                return 0
        try:
            with open(self.out + ".raw", "rb") as f:
                f.seek(self._pos)
                data = f.read()
        except OSError:
            return self.rows
        width, channels, dtype = self._shape()
        row_bytes = width * channels * dtype.itemsize
        rows = (self._pos + len(data)) // row_bytes
        if rows <= self.rows:
            return self.rows
        height = max(int(self.head.get("height") or 0), self.expected_height, rows)
        if self.image is None or self.image.shape[0] < height:
            img = np.zeros((height, width, channels) if channels > 1 else (height, width), dtype)
            if self.image is not None:
                img[:self.rows] = self.image[:self.rows]
            self.image = img
        data = data[:(rows * row_bytes) - self._pos]
        new = np.frombuffer(data, dtype).reshape((rows - self.rows, width, channels)[:3 if channels > 1 else 2])
        self.image[self.rows:rows] = new
        self._pos, self.rows = rows * row_bytes, rows
        return rows

    def final(self) -> np.ndarray:
        self.read()
        if self.image is None:
            raise ScannerError("TWAIN: the scanner sent no image data")
        return self.image[:self.rows]


def write_scan_log(req: dict, answer: dict, log: list[str], python: str) -> None:
    """Append what happened in a TWAIN scan to hasami.log in the settings folder."""
    import datetime

    from ..settings import config_dir

    try:
        path = config_dir() / "hasami.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        info = answer.get("image_info") or {}
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S} TWAIN scan with {python}\n")
            f.write(f"  asked: {({k: v for k, v in req.items() if k not in ('out', 'driver_caps')})}, "
                    f"{len(req.get('driver_caps') or {})} remembered driver settings\n")
            f.write(f"  driver ready to send after {info.get('ready_after_s')} s, helper done after "
                    f"{answer.get('done_after_s')} s, dpi {answer.get('dpi')}, frame {answer.get('frame')}\n")
            for line in log:
                f.write(f"  {line}\n")
    except OSError:
        pass


def _driver_settings_file() -> Path:
    from ..settings import config_dir

    return config_dir() / "twain_driver_settings.json"


def load_driver_caps(source: str) -> dict:
    """Settings last chosen in the driver's own window for ``source`` (custom capabilities)."""
    try:
        return json.loads(_driver_settings_file().read_text(encoding="utf-8")).get(source) or {}
    except Exception:
        return {}


def save_driver_caps(source: str, caps: dict) -> None:
    path = _driver_settings_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data[source] = caps
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


class TwainScanner(Scanner):
    def __init__(self, source: str, python: str, family: str | None = None, dev_id: str | None = None):
        self.source, self.python, self.family = source, python, family
        dev_id = dev_id or source
        self.info = DeviceInfo(f"twain:{dev_id}", source, "", source, "twain")
        self.key = dev_id  # remembered driver settings are kept per driver
        self._caps: Capabilities | None = None
        self.last_log: list[str] = []

    def capabilities(self) -> Capabilities:
        if self._caps is None:
            info = run_helper(self.python, {"cmd": "info", "source": self.source, "family": self.family,
                                            "probe_dpi": list(STANDARD_DPI[STANDARD_DPI.index(1200):])})
            # A small area may allow more than the whole bed (old EPSON Scan: 1184 dpi).
            listed = _dpi_list(info.get("resolutions_small_area")) or _dpi_list(info.get("resolutions"))
            # Resolutions above the listed ones that the driver takes all the same.
            extra = [int(d) for d in info.get("accepted_dpi") or [] if not listed or int(d) > max(listed)]
            resolutions = sorted(set(listed + extra)) or [75, 100, 150, 200, 300, 600, 1200]
            depths = info.get("bit_depths") or {}
            color = depths.get("color") or []
            bit_depths = [8, 16] if (48 in (color if isinstance(color, list) else [])
                                     or info.get("takes_48_bit")) else [8]
            w, h = info.get("physical") or [None, None]
            size = (float(w) * MM_PER_INCH, float(h) * MM_PER_INCH) if w and h else (215.9, 297.0)
            self._caps = Capabilities(sources=["Flatbed"], modes=["color", "gray", "lineart"],
                                      resolutions=resolutions, bit_depths=bit_depths, max_area_mm=size)
        return self._caps

    def scan_batch(self, request: ScanRequest, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> Iterator[ScanResult]:
        yield self.scan(request, progress, cancel)

    def scan(self, request: ScanRequest, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> ScanResult:
        from ..core.output import load_image

        caps = self.capabilities()
        x, y, w, h = request.area_mm or (0.0, 0.0, *caps.max_area_mm)
        frame = [x / MM_PER_INCH, y / MM_PER_INCH, (x + w) / MM_PER_INCH, (y + h) / MM_PER_INCH]
        st = _settings()
        with tempfile.TemporaryDirectory(prefix="scanner-twain-") as tmp:
            req = {"cmd": "scan", "source": self.source, "family": self.family, "dpi": int(request.dpi),
                   "mode": request.mode,
                   "bits": 16 if request.bit_depth == 16 else 8, "frame": frame, "raw": bool(request.raw),
                   "show_ui": bool(st.twain_show_ui) if st is not None else False,
                   "out": os.path.join(tmp, "scan")}
            if not req["show_ui"]:
                # Scans without the driver's window use what was last chosen in it
                # (e.g. Epson Scan 2's Scanning Quality: the motor runs differently).
                req["driver_caps"] = load_driver_caps(self.key)
            poll = None
            if request.live is not None:
                req["live"] = True
                strips = RawStrips(req["out"], int(round((frame[3] - frame[1]) * request.dpi)))
                area_mm = (x, y, w, h)
                shown = [0, 0.0]  # rows, time of the last update

                def poll():
                    rows = strips.read()
                    # A few updates a second: each one is redrawn by the app.
                    if rows > shown[0] and time.monotonic() - shown[1] > 0.25:
                        shown[:] = [rows, time.monotonic()]
                        if progress:
                            progress(min(0.99, rows / strips.image.shape[0]))
                        request.live(LiveImage(strips.image, rows, float(request.dpi), area_mm))

            try:
                answer = run_helper(self.python, req, cancel, timeout=None, poll=poll)
            except ScanCancelled as e:
                answer = getattr(e, "answer", {})
                self.last_log = answer.get("log", []) + (
                    ["the helper did not stop and was killed"] if getattr(e, "forced", False) else [])
                write_scan_log(req, answer, self.last_log, self.python)
                raise
            self.last_log = answer.get("log", [])
            if answer.get("driver_caps"):
                save_driver_caps(self.key, answer["driver_caps"])
                self.last_log.append(f"remembered {len(answer['driver_caps'])} settings of the driver's window")
            if answer.get("cancelled"):
                raise ScanCancelled("scan cancelled in the scanner's window")
            info = answer.get("image_info") or {}
            if answer.get("raw"):
                st_ = answer["raw"].get("strips") or []
                if st_:
                    self.last_log.append(
                        f"memory transfer: {len(st_)} strips from {st_[0][0]} s to {st_[-1][0]} s; the driver "
                        f"was ready to send after {info.get('ready_after_s')} s, done after "
                        f"{answer.get('done_after_s')} s (times from the helper's start); strip times: "
                        f"{[t for t, _, _ in st_[:60]]}")
                img = strips.final()
            else:
                img, _ = load_image(answer["file"])
            if request.live is not None:
                request.live(LiveImage(img, img.shape[0], float(request.dpi), area_mm))
            write_scan_log(req, answer, self.last_log, self.python)
        if progress:
            progress(1.0)
        if request.mode != "color" and img.ndim == 3:
            import cv2

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # grey BMPs come back as RGB
        dpi = float(answer.get("dpi") or request.dpi)
        fr = answer.get("frame") or frame
        area = (fr[0] * MM_PER_INCH, fr[1] * MM_PER_INCH, (fr[2] - fr[0]) * MM_PER_INCH, (fr[3] - fr[1]) * MM_PER_INCH)
        area, note = check_area(img.shape[1], img.shape[0], dpi, area, (x, y, w, h), caps.max_area_mm)
        if not request.preview and abs(dpi - request.dpi) > 0.5 and not answer.get("show_ui"):
            note = (f"The driver would not scan this area at {request.dpi:g} dpi ({dpi:g} instead): "
                    "high resolutions need a smaller area - tick 'Scan only the crop area'. " + note).strip()
        return ScanResult(img, dpi, area, note)


def check_area(width_px: int, height_px: int, dpi: float, reported: tuple, requested: tuple,
               bed_mm: tuple[float, float]) -> tuple[tuple, str]:
    """The area a scan really covers, judged by the image size.

    Drivers may ignore the requested area and scan the whole bed while
    reporting either area.  The crop boxes are placed by the area's origin,
    so an image the size of the whole bed is taken to start at the bed's
    corner.  Returns ``(area_mm, note)``.
    """
    w_mm, h_mm = width_px / dpi * MM_PER_INCH, height_px / dpi * MM_PER_INCH

    def fits(a):
        return abs(a[2] - w_mm) <= max(2.0, 0.03 * a[2]) and abs(a[3] - h_mm) <= max(2.0, 0.03 * a[3])

    bed = (0.0, 0.0, bed_mm[0], bed_mm[1])
    wanted_part = requested[2] < 0.9 * bed_mm[0] or requested[3] < 0.9 * bed_mm[1]
    if fits(reported):
        if wanted_part and fits(bed) and not fits(requested):
            return reported, "The scanner driver ignored the scan area and scanned the whole bed."
        return reported, ""
    if fits(bed):
        note = "The scanner driver ignored the scan area and scanned the whole bed." if wanted_part else ""
        return bed, note
    return (reported[0], reported[1], w_mm, h_mm), f"The scanner returned {w_mm:.0f} x {h_mm:.0f} mm."


class TwainBackend(Backend):
    name = "twain"

    def available(self) -> bool:
        return sys.platform == "win32"

    def unavailable_reason(self) -> str:
        return "TWAIN is only used on Windows"

    def list_devices(self) -> list[DeviceInfo]:
        return [d for d, _ in self._listing()]

    def _listing(self) -> list[tuple[DeviceInfo, str]]:
        """(device, line for scanner diag) for every TWAIN source of every helper Python."""
        out, errors, lines = [], [], []
        sources: dict[str, str] = {}
        idents: dict[str, tuple[str, str | None]] = {}
        seen = set()
        for python in candidate_pythons():
            try:
                answer = run_helper(python, {"cmd": "sources"})
            except ScannerError as e:
                errors.append(str(e))
                continue
            bits = answer.get("bits")
            found = answer.get("identities") or [{"name": n, "family": None} for n in answer.get("sources", [])]
            for ident in found:
                name, family = ident["name"], ident.get("family") or None
                if (name, family) in seen:  # the same driver, seen by another Python
                    continue
                seen.add((name, family))
                dev_id = name
                if dev_id in sources:  # a second driver of that name (e.g. old Epson Scan)
                    dev_id = f"{name} [{family or f'{bits}-bit'}]"
                    if dev_id in sources:
                        dev_id = f"{name} [{family}, {bits}-bit]"
                sources[dev_id], idents[dev_id] = python, (name, family)
                model = f"{name} - {family}, {bits}-bit" if family else f"{name}, {bits}-bit"
                out.append((DeviceInfo(f"twain:{dev_id}", name, "", model, "twain"),
                            f"{'Helper' if is_helper_exe(python) else 'Python'} {python} ({bits}-bit {answer.get('python')}): {dev_id!r}"
                            + (f" = {family} {ident.get('version', '')}".rstrip() if family else "")))
        _SOURCES.clear()
        _SOURCES.update(sources)
        _IDENT.clear()
        _IDENT.update(idents)
        if not out and errors:
            raise ScannerError("; ".join(errors))
        self.errors = errors
        return out

    def open(self, native_id: str) -> Scanner:
        if native_id not in _SOURCES:
            self.list_devices()
        if native_id not in _SOURCES:
            raise ScannerError(f"TWAIN source not found: {native_id}")
        name, family = _IDENT.get(native_id, (native_id, None))
        return TwainScanner(name, _SOURCES[native_id], family, native_id)


def installed_drivers() -> dict[str, list[str]]:
    """TWAIN driver files (*.ds) in the 32-bit and 64-bit TWAIN folders."""
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    out = {}
    for bits, folder in (("32-bit", "twain_32"), ("64-bit", "twain_64")):
        d = windir / folder
        out[bits] = sorted(str(p.relative_to(d)) for p in d.rglob("*.ds")) if d.is_dir() else []
    return out


def pick_source(sources: list[str], device: str | None = None) -> str | None:
    """The source to test: the one asked for, else the scanner chosen in the app.

    Never another scanning program's own TWAIN source by default (VueScan
    installs one, and opening it starts VueScan).
    """
    if device is not None:
        return device if device in sources else None
    st = _settings()
    saved = getattr(st, "device", "") or ""
    if saved.startswith("twain:") and saved[6:] in sources:
        return saved[6:]
    hardware = [s for s in sources if not any(app in s.lower() for app in SOFTWARE_SOURCES)]
    # Windows' WIA-to-TWAIN bridge ("WIA-<scanner>") only if there is no real driver.
    hardware.sort(key=lambda s: s.lower().startswith("wia-"))
    return (hardware or [None])[0]


#: Names of TWAIN sources that are programs, not scanners.
SOFTWARE_SOURCES = ("vuescan", "silverfast", "naps2")


def live_test(device: str | None = None, dpi: int = 300, height_mm: float = 100.0) -> str:
    """Does the driver hand over the image while the scanner moves?

    Scans a band of the bed with the strip-by-strip (memory) transfer and
    compares when the strips arrived with how long the scan took.
    """
    ids = [d.id[6:] for d in TwainBackend().list_devices()]
    name = pick_source(ids, device)
    if name is None:
        return f"No TWAIN scanner to test (sources: {ids})."
    sc = TwainBackend().open(name)
    caps = sc.capabilities()
    seen: list = []
    start = time.monotonic()
    res = sc.scan(ScanRequest(dpi=dpi, area_mm=(0.0, 0.0, caps.max_area_mm[0], height_mm),
                              live=lambda li: seen.append((round(time.monotonic() - start, 1), li.rows))))
    total = time.monotonic() - start
    lines = [f"Live test on {name!r}: {dpi} dpi, {caps.max_area_mm[0]:.0f} x {height_mm:.0f} mm, "
             f"{res.image.shape[1]} x {res.image.shape[0]} px, {total:.1f} s in all",
             f"Updates in the app (seconds, rows): {seen}"]
    lines += sc.last_log
    if len(seen) >= 2:
        lines.append(f"The image came in between {seen[0][0]} s and {seen[-2][0] if len(seen) > 2 else seen[0][0]} s "
                     f"after the start ({total:.1f} s in all). Compare with when the scanner's carriage was "
                     "moving: during the movement = live view can work; only after it stopped = the driver "
                     "keeps the image until the end.")
    return "\n".join(lines)


def diagnose(device: str | None = None, save_to: str | None = None) -> str:
    """Report TWAIN drivers, helper Pythons, sources and a test scan."""
    lines = []
    for bits, files in installed_drivers().items():
        lines.append(f"TWAIN drivers ({bits}): {', '.join(files) if files else 'none'}")
    backend = TwainBackend()
    try:
        listing = backend._listing()
    except ScannerError as e:
        listing = []
        lines.append(f"No TWAIN sources: {e}")
    lines += [line for _, line in listing] + [f"  ({e})" for e in getattr(backend, "errors", [])]
    first = pick_source(list(_SOURCES), device)
    if first is None:
        lines.append("No TWAIN source to test (choose one with  scanner diag -d \"twain:NAME\").")
        return "\n".join(lines)
    python = _SOURCES[first]
    name, family = _IDENT.get(first, (first, None))
    try:
        info = run_helper(python, {"cmd": "info", "source": name, "family": family,
                                     "probe_dpi": list(STANDARD_DPI[STANDARD_DPI.index(1200):])})
        lines.append(f"Source {first!r} via {python}:")
        for k, v in info.items():
            if k not in ("ok",):
                lines.append(f"  {k}: {v}")
    except ScannerError as e:
        lines.append(f"Source {first!r}: {e}")
        return "\n".join(lines)
    try:
        sc = TwainScanner(name, python, family, first)
        seen = []
        start = time.monotonic()
        res = sc.scan(ScanRequest(dpi=75, preview=True,
                                  live=lambda li: seen.append((round(time.monotonic() - start, 1), li.rows))))
        lines.append(f"Live view: {len(seen)} updates (seconds, rows): {seen[:12]}"
                     + (" ..." if len(seen) > 12 else ""))
        h, w = res.image.shape[:2]
        lines.append(f"Test scan (asked for 75 dpi, whole bed; the driver's window, when shown, decides): "
                     f"{w} x {h} px {res.image.dtype} at {res.dpi:g} dpi = "
                     f"{w / res.dpi * MM_PER_INCH:.0f} x {h / res.dpi * MM_PER_INCH:.0f} mm")
        lines.extend("  " + x for x in sc.last_log)
        if save_to:
            from ..core.output import save_image

            save_image(res.image, Path(save_to), dpi=res.dpi)
            lines.append(f"Saved the test scan to {save_to}")
    except Exception as e:
        lines.append(f"Test scan: FAILED {e}")
    return "\n".join(lines)
