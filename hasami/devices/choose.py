"""Which scanner connection to offer first, and which to hide.

One scanner often shows up several times on Windows: its TWAIN driver (or two,
e.g. the old EPSON Scan and Epson Scan 2), its WIA driver, Windows' WIA-to-TWAIN
bridge ("WIA-<scanner>") and programs that pose as TWAIN scanners (VueScan).
The app lists the useful one first and hides the rest unless asked
("Show all scanner connections").  Demo scanners are for tests: never listed.
"""

from __future__ import annotations

from .base import DeviceInfo
from .twain_backend import SOFTWARE_SOURCES  # TWAIN sources that are other programs

NO_SCANNER = ("No scanner found.  Install your scanner's own driver first (for an Epson: Epson Scan, "
              "from the support page of your model on epson.com), connect the scanner with USB, switch "
              "it on and press Refresh.")


def is_software(d: DeviceInfo) -> bool:
    return d.backend == "twain" and any(app in d.name.lower() for app in SOFTWARE_SOURCES)


def is_bridge(d: DeviceInfo) -> bool:
    """Windows' WIA-to-TWAIN bridge: WIA's limits under a TWAIN name."""
    return d.backend == "twain" and d.name.lower().startswith("wia-")


def _twain_driver(d: DeviceInfo) -> bool:
    return d.backend == "twain" and not is_software(d) and not is_bridge(d)


def is_wia_twin(d: DeviceInfo, devices: list[DeviceInfo]) -> bool:
    """A WIA scanner that also has a real TWAIN driver (TWAIN is better: live view, full dpi)."""
    if d.backend != "wia":
        return False
    text = f"{d.name} {d.vendor} {d.model}".lower()
    return any(_twain_driver(t) and t.name.lower() in text for t in devices)


def is_extra(d: DeviceInfo, devices: list[DeviceInfo]) -> bool:
    """Hidden unless all connections are shown."""
    return is_software(d) or is_bridge(d) or is_wia_twin(d, devices)


def order_key(d: DeviceInfo, devices: list[DeviceInfo]) -> tuple:
    """TWAIN drivers first, then other scanners (SANE, network), WIA, bridges, programs, files.

    Two TWAIN drivers of the same name (old EPSON Scan + Epson Scan 2): the 32-bit
    one first - the old EPSON Scan is 32-bit only, Epson Scan 2 is found by the
    64-bit helper first (maintainer prefers the old one: closer to Epson Scan's colours)."""
    if is_software(d):
        group = 4
    elif is_bridge(d):
        group = 3
    elif d.backend == "twain":
        group = 0
    else:
        group = {"wia": 2, "demo": 5, "file": 6}.get(d.backend, 1)
    same_name = sum(1 for t in devices if _twain_driver(t) and t.name == d.name) > 1
    second = 1 if same_name and _twain_driver(d) and "32-bit" not in (d.model or "") else 0
    return (group, second)


def offered(devices: list[DeviceInfo], show_all: bool = False, keep: str = "") -> list[DeviceInfo]:
    """The devices to list, best first.  ``keep``: a device id that stays listed
    (the one chosen before, even if it would be hidden)."""
    out = [d for d in devices
           if d.id == keep or (d.backend != "demo" and (show_all or not is_extra(d, devices)))]
    return sorted(out, key=lambda d: order_key(d, devices))


def best(devices: list[DeviceInfo]) -> DeviceInfo | None:
    """The scanner to use when none has been chosen (None: no scanner)."""
    real = [d for d in offered(devices) if d.backend not in ("demo", "file")]
    return real[0] if real else None
