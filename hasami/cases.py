"""Crop cases: what a corrected (or kept) auto-crop shows, saved for tuning (opt-in, Crop tab).

A case is a folder: ``preview.jpg`` (the whole-bed preview), ``scan.jpg`` (the uncropped scanned area,
<= 400 dpi) and ``case.json`` (scanner, settings, the boxes auto-crop proposed and the boxes scanned, in bed
mm).  Spec: docs/specs/2026-10-15-crop-cases-design.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

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


def summary(edits: list[dict]) -> str:
    """'bottom edge 4.2 mm out; 1 box deleted; 1 box drawn' (status bar, share dialog)."""
    parts = []
    for e in edits:
        if e["kind"] != "moved":
            continue
        for side in SIDES:
            v = e["sides_mm"][side]
            if abs(v) > MOVED_MM:
                parts.append(f"{side} edge {abs(v):.1f} mm {'out' if v > 0 else 'in'}")
        if abs(e["angle_deg"]) > TURNED_DEG:
            parts.append(f"turned {e['angle_deg']:+.2f}°")
    for kind in ("deleted", "drawn"):
        n = sum(e["kind"] == kind for e in edits)
        if n:
            parts.append(f"{_count(n, 'box')} {kind}")
    return "; ".join(parts) or "no corrections"
