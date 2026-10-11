"""Application settings and their persistence as JSON.

Settings live in the platform's configuration folder:

* Linux: ``$XDG_CONFIG_HOME/hasami-scan/settings.json`` (``~/.config/...``)
* macOS: ``~/Library/Application Support/HasamiScan/settings.json``
* Windows: ``%APPDATA%\\HasamiScan\\settings.json``

Before 0.4.2 (when the app was called "Scanner") the folder was ``scanner``:
its settings are copied over the first time (the old folder is left as it was).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from .core.autocrop import AutoCropSettings
from .core.output import OutputSettings
from .core.processing import ProcessSettings


@dataclass
class InputSettings:
    #: Device id as listed by ``scanner list`` (e.g. ``sane:epson2:libusb:001:004``,
    #: ``escl:http://192.168.1.20/eSCL``, ``demo:anime``).  Empty: the best scanner found.
    device: str = ""
    #: "Flatbed", "ADF", "ADF Duplex", "Transparency"... as offered by the device.
    source: str = "Flatbed"
    #: "color", "gray" or "lineart".
    mode: str = "color"
    preview_dpi: int = 75
    scan_dpi: int = 600
    #: 8 or 16 bits per channel from the scanner.
    bit_depth: int = 8
    #: Scan only the area around the crop boxes found in the preview.
    scan_crop_area_only: bool = True
    #: TWAIN scanners: the Python that runs the TWAIN helper; empty = find one
    #: (this Python, then any 32-bit Python the ``py`` launcher knows).
    twain_python: str = ""
    #: TWAIN scanners: show the scanner driver's own window for each scan.
    twain_show_ui: bool = False
    #: Show previews and scans in the preview while they come in (TWAIN, demo).
    live_view: bool = True
    #: List every connection of a scanner (WIA twin, WIA-to-TWAIN bridge, VueScan's
    #: TWAIN source), not just the best one.
    show_all_devices: bool = False


@dataclass
class UpdateSettings:
    #: The installed app asks GitHub for a newer release at most once a day on start.
    check_on_start: bool = True
    #: When it last asked (time.time()).
    last_check: float = 0.0
    #: A version the user chose to skip: not offered again on start.
    skipped: str = ""


@dataclass
class AppSettings:
    input: InputSettings = field(default_factory=InputSettings)
    crop: AutoCropSettings = field(default_factory=AutoCropSettings)
    color: ProcessSettings = field(default_factory=ProcessSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    updates: UpdateSettings = field(default_factory=UpdateSettings)


#: Files carried over from the old settings folder ("scanner", before 0.4.2).
CARRIED_OVER = ("settings.json", "twain_driver_settings.json")


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        name = "HasamiScan"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
        name = "HasamiScan"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        name = "hasami-scan"
    folder = base / name
    old = base / "scanner"
    if not folder.exists() and old.is_dir():
        try:  # first start after the rename: keep the settings (copied, the old folder stays)
            folder.mkdir(parents=True)
            for f in CARRIED_OVER:
                if (old / f).is_file():
                    shutil.copy2(old / f, folder / f)
        except OSError:
            return old
    return folder


def settings_path() -> Path:
    return config_dir() / "settings.json"


def _fill(obj, data: dict):
    """Copy known keys from ``data`` into dataclass ``obj`` (recursively)."""
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        val = data[f.name]
        if is_dataclass(cur) and isinstance(val, dict):
            _fill(cur, val)
        elif isinstance(cur, bool):
            setattr(obj, f.name, bool(val))
        elif isinstance(cur, (int, float)) and not isinstance(cur, bool) and isinstance(val, (int, float)):
            setattr(obj, f.name, type(cur)(val))
        elif isinstance(cur, str) and isinstance(val, str):
            setattr(obj, f.name, val)
    return obj


def load_settings(path: Path | None = None) -> AppSettings:
    path = path or settings_path()
    st = AppSettings()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return st
    if isinstance(data, dict):
        _fill(st, data)
    return st


def save_settings(st: AppSettings, path: Path | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(st), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
