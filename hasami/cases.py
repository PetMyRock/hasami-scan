"""Crop cases: what a corrected (or kept) auto-crop shows, saved for tuning (opt-in, Crop tab).

A case is a folder: ``preview.jpg`` (the whole-bed preview), ``scan.jpg`` (the uncropped scanned area,
<= 400 dpi) and ``case.json`` (scanner, settings, the boxes auto-crop proposed and the boxes scanned, in bed
mm).  Spec: docs/specs/2026-10-15-crop-cases-design.md.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import __version__
from .core.geometry import CropBox

MOVED_MM = 0.1     # a side moved by more than this is a correction
TURNED_DEG = 0.05  # so is a turn by more than this
SIDES = ("top", "right", "bottom", "left")


def default_folder() -> Path:
    """Documents/Hasami Scan/Crop cases."""
    docs = os.environ.get("XDG_DOCUMENTS_DIR") if sys.platform.startswith("linux") else None
    base = Path(docs) if docs else Path.home() / "Documents"
    return base / "Hasami Scan" / "Crop cases"


def folder_of(cs) -> Path:
    """The case folder of a ``CaseSettings`` ("" = the default)."""
    return Path(os.path.expanduser(cs.folder)) if cs.folder else default_folder()


def side_shifts(auto: CropBox, final: CropBox) -> dict[str, float]:
    """How far each side of ``final`` lies outside ``auto``'s (mm, + = outwards), in auto's frame."""
    ux, uy = auto.axes
    d = np.array([final.cx - auto.cx, final.cy - auto.cy])
    cu, cv = float(d @ ux), float(d @ uy)
    return {"top": final.h / 2 - cv - auto.h / 2, "right": cu + final.w / 2 - auto.w / 2,
            "bottom": cv + final.h / 2 - auto.h / 2, "left": final.w / 2 - cu - auto.w / 2}


def diff_boxes(auto: list[CropBox], final: list[CropBox]) -> list[dict]:
    """Per auto box: unchanged / moved (``sides_mm``, ``angle_deg``) / deleted; plus each box drawn since the
    preview (no auto id; boxes carried over from an earlier page, ``meta["carried"]``, are not edits)."""
    ids = {a.meta.get("auto_id") for a in auto}
    by_id = {b.meta["auto_id"]: b for b in final if b.meta.get("auto_id") in ids}
    edits = []
    for a in auto:
        f = by_id.get(a.meta.get("auto_id"))
        if f is None:
            edits.append({"id": a.meta.get("auto_id"), "kind": "deleted"})
            continue
        s = {k: round(v, 3) for k, v in side_shifts(a, f).items()}
        turn = round(f.angle - a.angle, 3)
        moved = any(abs(v) > MOVED_MM for v in s.values()) or abs(turn) > TURNED_DEG
        edits.append({"id": a.meta.get("auto_id"), "kind": "moved" if moved else "unchanged",
                      "sides_mm": s, "angle_deg": turn})
    edits += [{"id": None, "kind": "drawn"} for b in final
              if b.meta.get("auto_id") not in ids and not b.meta.get("carried")]
    return edits


def corrected(edits: list[dict]) -> bool:
    return any(e["kind"] != "unchanged" for e in edits)


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'es' if n > 1 else ''}"


def summary(edits: list[dict], turns: int = 0) -> str:
    """'bottom edge 4.2 mm out; 1 box deleted; 1 box drawn' (status bar, share dialog).  ``turns``: the
    clockwise quarter turns the preview is shown with (``crop.rotate``), so sides are named as the user sees
    them; case.json keeps the scanner frame."""
    parts = []
    for e in edits:
        if e["kind"] != "moved":
            continue
        for side in SIDES:
            v = e["sides_mm"][side]
            if abs(v) > MOVED_MM:
                shown = SIDES[(SIDES.index(side) + int(turns)) % 4]
                parts.append(f"{shown} edge {abs(v):.1f} mm {'out' if v > 0 else 'in'}")
        if abs(e["angle_deg"]) > TURNED_DEG:
            parts.append(f"turned {e['angle_deg']:+.2f}°")
    for kind in ("deleted", "drawn"):
        n = sum(e["kind"] == kind for e in edits)
        if n:
            parts.append(f"{_count(n, 'box')} {kind}")
    return "; ".join(parts) or "no corrections"


FORMAT = 1
SCAN_DPI_MAX = 400.0
JPEG_QUALITY = 95
SHARE_DIR = "To share"
CASE_NAME = re.compile(r"^\d{4}-\d\d-\d\d_\d{6}(_\d+)?$")  # write_case's folder names
CASE_FILES = ("case.json", "preview.jpg", "scan.jpg")
README = """Hasami Scan crop case (format 1)

A crop case keeps what an auto-crop correction shows, to tune auto-crop for your scanner:
- preview.jpg: the whole-bed preview, as scanned (scanner frame, not turned);
- scan.jpg: the uncropped scanned area (at most 400 dpi);
- case.json: scanner, settings, the boxes auto-crop proposed and the boxes you scanned (bed mm).

If you post this zip in the tuning channel on the Hasami Scan Discord, the members of that channel can see
it. Cases are used only to tune auto-crop and are not published anywhere else.
"""


@dataclass
class CaseData:
    preview: np.ndarray
    preview_dpi: float
    preview_area_mm: tuple
    scan: np.ndarray
    scan_dpi_saved: float
    scan_dpi: float
    scan_area_mm: tuple
    auto: list
    final: list
    scanner_id: str = ""
    scanner_name: str = ""
    profile: str = ""
    crop: dict = field(default_factory=dict)
    reason: str = "corrected"
    created: datetime = field(default_factory=datetime.now)


def downscale(img: np.ndarray, dpi: float, max_dpi: float = SCAN_DPI_MAX) -> tuple[np.ndarray, float]:
    """``img`` at most ``max_dpi`` (INTER_AREA), and its dpi."""
    if dpi <= max_dpi:
        return img, float(dpi)
    f = max_dpi / dpi
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), float(max_dpi)


def _save_jpeg(path: Path, img: np.ndarray, dpi: float) -> None:
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(img)).save(path, "JPEG", quality=JPEG_QUALITY, dpi=(dpi, dpi))


def _box(b: CropBox) -> list[float]:
    return [round(float(v), 3) for v in (b.cx, b.cy, b.w, b.h, b.angle)]

def _marks(b: CropBox, *keys: str) -> dict:
    """The box's optional case keys (flags), omitted when empty."""
    return {k: b.meta[k] for k in keys if b.meta.get(k)}


def write_case(folder: Path, data: CaseData) -> Path:
    """Write a case into ``folder`` (created); returns its path.  Written as ``<name>.partial`` and renamed at
    the end, so a half-written case is never listed."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = data.created.strftime("%Y-%m-%d_%H%M%S")
    name, n = base, 2
    while (folder / name).exists() or (folder / f"{name}.partial").exists():
        name, n = f"{base}_{n}", n + 1
    tmp = folder / f"{name}.partial"
    tmp.mkdir()
    try:
        _save_jpeg(tmp / "preview.jpg", data.preview, data.preview_dpi)
        _save_jpeg(tmp / "scan.jpg", data.scan, data.scan_dpi_saved)
        doc = {"format": FORMAT, "app_version": __version__, "created": data.created.isoformat(timespec="seconds"),
               "reason": data.reason, "scanner": {"id": data.scanner_id, "name": data.scanner_name},
               "profile": data.profile, "crop": data.crop,
               "preview": {"dpi": float(data.preview_dpi), "area_mm": [float(v) for v in data.preview_area_mm],
                           "file": "preview.jpg"},
               "scan": {"scan_dpi": float(data.scan_dpi), "dpi": float(data.scan_dpi_saved),
                        "area_mm": [float(v) for v in data.scan_area_mm], "file": "scan.jpg"},
               "auto": [{"id": b.meta.get("auto_id"), "box_mm": _box(b), "confidence": round(float(b.confidence), 3),
                         **_marks(b, "check_sides")} for b in data.auto],
               "final": [{"id": b.meta.get("auto_id"), "box_mm": _box(b)}
                         for b in data.final],
               "edits": diff_boxes(data.auto, data.final)}
        (tmp / "case.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.rename(folder / name)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return folder / name


def list_cases(folder: Path) -> list[Path]:
    """Case folders (named as ``write_case`` names them), newest first."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_dir() and CASE_NAME.match(p.name)),
                  key=lambda p: p.name, reverse=True)


def delete_case(path: Path) -> None:
    """Remove a case: its own files, then the folder; anything else in it stays (OSError then)."""
    path = Path(path)
    for name in CASE_FILES:
        (path / name).unlink(missing_ok=True)
    path.rmdir()


def read_case(path: Path) -> dict:
    """``case.json`` of a case; ValueError when it is missing, broken or from a newer format."""
    try:
        doc = json.loads((Path(path) / "case.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"unreadable case ({e})") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("format"), int) or "edits" not in doc:
        raise ValueError("unreadable case (not a crop case)")
    if doc["format"] > FORMAT:
        raise ValueError("made by a newer Hasami Scan")
    return doc


def pack(cases: list[Path], out_dir: Path, limit_mb: float = 9.5) -> list[Path]:
    """One zip per case (its files + README.txt) in ``out_dir``; a case over ``limit_mb`` is split into a
    ``scan.jpg`` zip and the rest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = []
    for case in map(Path, cases):
        files = sorted(p for p in case.iterdir() if p.is_file())
        size_mb = sum(p.stat().st_size for p in files) / 2 ** 20
        groups = ([files] if size_mb <= limit_mb else
                  [[p for p in files if p.name == "scan.jpg"], [p for p in files if p.name != "scan.jpg"]])
        for k, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"-part{k + 1}"
            z = out_dir / f"hasami-case-{case.name}{suffix}.zip"
            with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:  # JPEGs do not shrink
                zf.writestr("README.txt", README)
                for p in group:
                    zf.write(p, f"{case.name}/{p.name}")
            zips.append(z)
    return zips
