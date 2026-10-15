# Crop cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in saving of auto-crop corrections as "crop cases" (whole-bed preview + uncropped scanned area at <= 400 dpi + auto and final boxes) and a dialog that zips chosen cases for sharing on the Discord server.

**Architecture:** A Qt-free module `hasami/cases.py` holds the case format: diffing auto vs final boxes, writing / reading / listing case folders, zipping. `Session` (hasami/workflow.py) tags auto boxes with ids, keeps a snapshot, and after the first page of a scan builds the case and writes it when a box was corrected; `save_last_case()` serves "Save this case". The GUI adds two Crop-tab settings, Scan > "Save this case", File > "Share crop cases..." (`hasami/gui/cases_dialog.py`) and a status line.

**Tech Stack:** Python 3.14, numpy, OpenCV, Pillow, PySide6, pytest (`F:/HasamiDev/.venv/Scripts/python.exe`, written `$PY`).

**Spec:** `docs/specs/2026-10-15-crop-cases-design.md` (approved Oct 15 2026). Read it first.

## Global Constraints

- Opt-in: `AppSettings.cases.enabled` defaults to False; with it off nothing is kept, computed or written.
- Default folder: Documents/Hasami Scan/Crop cases (`cases.folder` "" = default).
- Case folder name: local time `YYYY-MM-DD_HHMMSS`, `_2`, `_3` ... when taken; written as `<name>.partial` and renamed at the end; `.partial` folders are never listed.
- Files: `preview.jpg` (whole-bed preview at its own dpi), `scan.jpg` (the first page's delivered image, `INTER_AREA` to min(scan dpi, 400), 16-bit reduced to 8), both JPEG quality 95 with their dpi in the JPEG; `case.json` with `"format": 1` and the fields of the spec's example.
- Boxes in bed mm `[cx, cy, w, h, angle]`; sides named in the box's frame (top = towards -y at angle 0); `sides_mm` signed, + = outwards. Moved = any side > 0.1 mm or angle > 0.05 deg.
- Capture only from `Session.run_scan`, page 1, when a preview exists. Not from opened images, test scans or the CLI.
- A case can never break a scan: every error in building or writing it becomes a status line "Could not save the crop case: ..."; the scan's outputs are saved as before.
- Zips: one per case `hasami-case-<name>.zip` with `README.txt`, under `<folder>/To share`; a case over 9.5 MB is split into a `scan.jpg` zip and the rest.
- Dialog text (verbatim): "Each case contains scans of your item (the preview and the scanned area). Share only what you are happy to share. Post the zips in the tuning channel on the Hasami Scan Discord: https://discord.gg/TAAV8B9WWg".
- Tests use generated scans only (demo backend `demo:booklet`, synthetic arrays); GUI tests offscreen with the temporary APPDATA of `tests/conftest.py`.
- Version 0.5.16 -> 0.5.17 (`hasami/__init__.py`, `pyproject.toml`). Branch `feat/crop-cases`, PR the owner merges; never `fix:` / `closes` / `fixes` + `#N` in PR text.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; the PR body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- No crop-engine change: the real-scan benchmarks are not needed; run `benchmarks/synthetic_*.py` once before the PR and report (identical expected).

## Review Focus

- Auto-crop Off with a box drawn by hand: drawing is how Off works, not a correction; no case is written automatically (Task 3: `test_off_mode_never_writes_a_case_by_itself`).
- A box drawn on an earlier page and carried over by the next preview (`meta["drawn"]`): counted as "drawn" only on the scan of the preview it was drawn on, not on every later page (Task 3: `test_a_drawn_box_carried_to_the_next_page_is_not_a_new_correction`).
- Auto-crop run again after an edit (a Crop setting changed, Ctrl+R): the edited box keeps its first auto id and its original auto box; new boxes get new ids, no id is reused (Task 3: `test_redetect_keeps_the_edited_boxs_auto_original`).
- 16-bit or grey scans: `scan.jpg` is still an 8-bit JPEG Pillow reads (Task 2: `test_write_case_handles_16_bit_grey_scans`).
- Two cases in the same second (Save this case right after a corrected scan of another page): distinct folders (Task 2: `test_two_cases_in_one_second_get_distinct_names`).

---

### Task 1: case settings, box diff and summary

**Files:**
- Modify: `hasami/settings.py` (add `CaseSettings`, `AppSettings.cases`)
- Create: `hasami/cases.py`
- Test: `tests/test_cases.py`

**Interfaces:**
- Produces: `hasami.settings.CaseSettings(enabled: bool = False, folder: str = "")`, `AppSettings.cases`; `hasami.cases.default_folder() -> Path`, `folder_of(cs: CaseSettings) -> Path`, `side_shifts(auto: CropBox, final: CropBox) -> dict[str, float]`, `diff_boxes(auto: list[CropBox], final: list[CropBox]) -> list[dict]`, `corrected(edits: list[dict]) -> bool`, `summary(edits: list[dict]) -> str`; constants `MOVED_MM = 0.1`, `TURNED_DEG = 0.05`, `SIDES = ("top", "right", "bottom", "left")`.

- [ ] **Step 1: Branch**

```bash
cd F:/HasamiDev && git checkout main && git pull -q && git checkout -b feat/crop-cases
```

- [ ] **Step 2: Write the failing tests** (`tests/test_cases.py`)

```python
from dataclasses import replace
from pathlib import Path

import pytest

from hasami.core.geometry import CropBox


def _auto(i=1, **kw):
    b = CropBox(100.0, 150.0, 120.0, 180.0, 0.0, 0.9, **kw)
    b.meta["auto_id"] = f"a{i}"
    return b


def _edit(b, **kw):
    return replace(b, manual=True, meta=dict(b.meta), **kw)


def test_case_settings_are_off_by_default_and_load_back(tmp_path):
    from hasami.settings import AppSettings, load_settings, save_settings

    st = AppSettings()
    assert st.cases.enabled is False and st.cases.folder == ""
    st.cases.enabled, st.cases.folder = True, str(tmp_path / "cc")
    save_settings(st, tmp_path / "s.json")
    back = load_settings(tmp_path / "s.json")
    assert back.cases.enabled is True and back.cases.folder == str(tmp_path / "cc")


def test_folder_of_uses_the_setting_or_documents():
    from hasami import cases
    from hasami.settings import CaseSettings

    assert cases.folder_of(CaseSettings(folder="X:/somewhere")) == Path("X:/somewhere")
    d = cases.folder_of(CaseSettings())
    assert d == cases.default_folder() and d.parts[-2:] == ("Hasami Scan", "Crop cases")


def test_an_untouched_box_is_unchanged():
    from hasami import cases

    a = _auto()
    edits = cases.diff_boxes([a], [_edit(a)])
    assert [e["kind"] for e in edits] == ["unchanged"]
    assert not cases.corrected(edits)


def test_moved_sides_are_signed_outwards():
    from hasami import cases

    a = _auto()
    f = _edit(a, h=a.h + 4.0, cy=a.cy + 2.0)  # bottom 4 mm further out
    e = cases.diff_boxes([a], [f])[0]
    assert e["kind"] == "moved"
    assert e["sides_mm"]["bottom"] == pytest.approx(4.0, abs=1e-6)
    assert all(abs(e["sides_mm"][s]) < 1e-6 for s in ("top", "left", "right"))
    g = _edit(a, w=a.w - 3.0, cx=a.cx - 1.5)  # right side 3 mm in
    assert cases.diff_boxes([a], [g])[0]["sides_mm"]["right"] == pytest.approx(-3.0, abs=1e-6)
    assert cases.corrected(cases.diff_boxes([a], [f]))


def test_tiny_moves_and_turns_stay_unchanged_but_a_turn_counts():
    from hasami import cases

    a = _auto()
    assert cases.diff_boxes([a], [_edit(a, cx=a.cx + 0.05)])[0]["kind"] == "unchanged"
    assert cases.diff_boxes([a], [_edit(a, angle=0.04)])[0]["kind"] == "unchanged"
    e = cases.diff_boxes([a], [_edit(a, angle=-0.7)])[0]
    assert e["kind"] == "moved" and e["angle_deg"] == pytest.approx(-0.7)


def test_deleted_and_drawn_boxes():
    from hasami import cases

    a1, a2 = _auto(1), _auto(2)
    drawn = CropBox.from_xywh(5, 5, 20, 20, manual=True)
    drawn.meta["drawn"] = True
    carried = CropBox.from_xywh(40, 5, 20, 20, manual=True)
    carried.meta.update(drawn=True, carried=True)
    edits = cases.diff_boxes([a1, a2], [_edit(a1), drawn, carried])
    assert [(e["id"], e["kind"]) for e in edits] == [("a1", "unchanged"), ("a2", "deleted"), (None, "drawn")]


def test_summary_wording():
    from hasami import cases

    a, b = _auto(1), _auto(2)
    edits = cases.diff_boxes([a, b], [_edit(a, h=a.h + 4.2, cy=a.cy + 2.1), CropBox.from_xywh(1, 1, 5, 5, manual=True)])
    assert cases.summary(edits) == "bottom edge 4.2 mm out; 1 box deleted; 1 box drawn"
    assert cases.summary(cases.diff_boxes([a], [_edit(a)])) == "no corrections"
```

- [ ] **Step 3: Run them to see them fail**

Run: `$PY -m pytest -q tests/test_cases.py`
Expected: FAIL: `ImportError: cannot import name 'cases'` / `AttributeError: 'AppSettings' object has no attribute 'cases'`.

- [ ] **Step 4: Add `CaseSettings`** (`hasami/settings.py`, above `class AppSettings`, and the field in `AppSettings`)

```python
@dataclass
class CaseSettings:
    #: Save crop cases (preview + scanned area + boxes) when auto-crop boxes were corrected.
    enabled: bool = False
    #: Where cases go; "" = Documents/Hasami Scan/Crop cases.
    folder: str = ""
```

and in `AppSettings` after `updates`:

```python
    cases: CaseSettings = field(default_factory=CaseSettings)
```

- [ ] **Step 5: Create `hasami/cases.py` (first part)**

```python
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
```

- [ ] **Step 6: Run the tests**

Run: `$PY -m pytest -q tests/test_cases.py`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add hasami/settings.py hasami/cases.py tests/test_cases.py
git commit -m "Crop cases: settings, auto/final box diff and its summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: writing, reading, listing and zipping cases

**Files:**
- Modify: `hasami/cases.py`
- Test: `tests/test_cases.py`

**Interfaces:**
- Consumes: Task 1's `diff_boxes`.
- Produces: `CaseData` (dataclass: `preview, preview_dpi, preview_area_mm, scan, scan_dpi_saved, scan_dpi, scan_area_mm, auto, final, scanner_id="", scanner_name="", profile="", crop={}, reason="corrected", created=datetime.now()`); `downscale(img, dpi, max_dpi=SCAN_DPI_MAX) -> tuple[np.ndarray, float]`; `write_case(folder: Path, data: CaseData) -> Path`; `list_cases(folder: Path) -> list[Path]`; `read_case(path: Path) -> dict` (raises `ValueError`); `pack(cases: list[Path], out_dir: Path, limit_mb: float = 9.5) -> list[Path]`; constants `FORMAT = 1`, `SCAN_DPI_MAX = 400.0`, `SHARE_DIR = "To share"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cases.py`)

```python
import json
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image


def _data(**kw):
    from hasami import cases

    rng = np.random.default_rng(0)
    a = _auto()
    d = dict(preview=rng.integers(0, 255, (300, 220, 3), np.uint8), preview_dpi=50.0,
             preview_area_mm=(0.0, 0.0, 216.0, 297.0),
             scan=rng.integers(0, 255, (500, 400, 3), np.uint8), scan_dpi_saved=150.0, scan_dpi=150.0,
             scan_area_mm=(10.0, 20.0, 67.7, 84.7), auto=[a], final=[_edit(a, h=a.h + 4.0, cy=a.cy + 2.0)],
             scanner_id="demo:booklet", scanner_name="Demo booklet", profile="default",
             crop={"mode": "booklet", "rotate": 0}, created=datetime(2026, 10, 15, 14, 32, 10))
    d.update(kw)
    return cases.CaseData(**d)


def test_write_and_read_a_case(tmp_path):
    from hasami import cases

    path = cases.write_case(tmp_path, _data())
    assert path == tmp_path / "2026-10-15_143210"
    assert sorted(p.name for p in path.iterdir()) == ["case.json", "preview.jpg", "scan.jpg"]
    doc = cases.read_case(path)
    assert doc["format"] == 1 and doc["reason"] == "corrected" and doc["scanner"]["name"] == "Demo booklet"
    assert doc["preview"] == {"dpi": 50.0, "area_mm": [0.0, 0.0, 216.0, 297.0], "file": "preview.jpg"}
    assert doc["scan"]["dpi"] == 150.0 and doc["scan"]["scan_dpi"] == 150.0
    assert doc["auto"][0]["id"] == "a1" and doc["auto"][0]["box_mm"] == [100.0, 150.0, 120.0, 180.0, 0.0]
    assert doc["final"][0]["box_mm"] == [100.0, 152.0, 120.0, 184.0, 0.0]
    assert doc["edits"][0]["kind"] == "moved" and doc["edits"][0]["sides_mm"]["bottom"] == 4.0
    with Image.open(path / "scan.jpg") as im:
        assert im.size == (400, 500) and round(im.info["dpi"][0]) == 150
    with Image.open(path / "preview.jpg") as im:
        assert im.size == (220, 300) and round(im.info["dpi"][0]) == 50
    assert cases.list_cases(tmp_path) == [path]


def test_two_cases_in_one_second_get_distinct_names(tmp_path):
    from hasami import cases

    p1, p2 = cases.write_case(tmp_path, _data()), cases.write_case(tmp_path, _data(reason="saved"))
    assert (p1.name, p2.name) == ("2026-10-15_143210", "2026-10-15_143210_2")
    assert cases.list_cases(tmp_path) == [p2, p1]  # newest first


def test_partial_folders_and_the_share_folder_are_not_cases(tmp_path):
    from hasami import cases

    (tmp_path / "2026-10-15_150000.partial").mkdir()
    (tmp_path / cases.SHARE_DIR).mkdir()
    p = cases.write_case(tmp_path, _data())
    assert cases.list_cases(tmp_path) == [p]
    assert cases.list_cases(tmp_path / "missing") == []


def test_read_case_refuses_broken_and_newer_cases(tmp_path):
    from hasami import cases

    p = cases.write_case(tmp_path, _data())
    doc = json.loads((p / "case.json").read_text(encoding="utf-8"))
    (p / "case.json").write_text(json.dumps(dict(doc, format=2)), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        cases.read_case(p)
    (p / "case.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        cases.read_case(p)
    (p / "case.json").unlink()
    with pytest.raises(ValueError):
        cases.read_case(p)


def test_downscale_to_400_dpi():
    from hasami import cases

    img = np.zeros((1200, 800, 3), np.uint8)
    small, dpi = cases.downscale(img, 1200.0)
    assert dpi == 400.0 and small.shape[:2] == (400, 267)
    same, dpi = cases.downscale(img, 300.0)
    assert same is img and dpi == 300.0


def test_write_case_handles_16_bit_grey_scans(tmp_path):
    from hasami import cases

    grey16 = (np.arange(500 * 400, dtype=np.uint32).reshape(500, 400) % 65536).astype(np.uint16)
    p = cases.write_case(tmp_path, _data(scan=grey16))
    with Image.open(p / "scan.jpg") as im:
        assert im.mode == "L" and im.size == (400, 500)


def test_pack_zips_each_case_with_a_readme_and_splits_big_ones(tmp_path):
    from hasami import cases

    p1 = cases.write_case(tmp_path / "cc", _data())
    p2 = cases.write_case(tmp_path / "cc", _data(created=datetime(2026, 10, 15, 15, 0, 0)))
    zips = cases.pack([p1, p2], tmp_path / "out")
    assert [z.name for z in zips] == ["hasami-case-2026-10-15_143210.zip", "hasami-case-2026-10-15_150000.zip"]
    with zipfile.ZipFile(zips[0]) as zf:
        assert sorted(zf.namelist()) == ["2026-10-15_143210/case.json", "2026-10-15_143210/preview.jpg",
                                         "2026-10-15_143210/scan.jpg", "README.txt"]
        assert "crop case" in zf.read("README.txt").decode()
    split = cases.pack([p1], tmp_path / "out2", limit_mb=0.0001)
    assert [z.name for z in split] == ["hasami-case-2026-10-15_143210-part1.zip",
                                       "hasami-case-2026-10-15_143210-part2.zip"]
    with zipfile.ZipFile(split[0]) as zf:
        assert sorted(zf.namelist()) == ["2026-10-15_143210/scan.jpg", "README.txt"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest -q tests/test_cases.py`
Expected: the 7 Task 1 tests pass; the new ones FAIL with `AttributeError: module 'hasami.cases' has no attribute 'CaseData'` (and similar).

- [ ] **Step 3: Implement** (append to `hasami/cases.py`; add the imports at the top: `import json`, `import shutil`, `import zipfile`, `from dataclasses import dataclass, field`, `from datetime import datetime`, `import cv2`, `from PIL import Image`, `from . import __version__`)

```python
FORMAT = 1
SCAN_DPI_MAX = 400.0
JPEG_QUALITY = 95
SHARE_DIR = "To share"
README = """Hasami Scan crop case (format 1)

A crop case keeps what an auto-crop correction shows, to tune auto-crop for your scanner:
- preview.jpg: the whole-bed preview, as scanned (scanner frame, not turned);
- scan.jpg: the uncropped scanned area (at most 400 dpi);
- case.json: scanner, settings, the boxes auto-crop proposed and the boxes you scanned (bed mm).

The scans are used to tune auto-crop and are kept private: they are never published.
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
               "auto": [{"id": b.meta.get("auto_id"), "box_mm": _box(b), "confidence": round(float(b.confidence), 3)}
                        for b in data.auto],
               "final": [{"id": b.meta.get("auto_id"), "box_mm": _box(b)} for b in data.final],
               "edits": diff_boxes(data.auto, data.final)}
        (tmp / "case.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.rename(folder / name)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return folder / name


def list_cases(folder: Path) -> list[Path]:
    """Case folders, newest first (``.partial`` and the share folder left out)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir()
                   if p.is_dir() and not p.name.endswith(".partial") and p.name != SHARE_DIR),
                  key=lambda p: p.name, reverse=True)


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
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest -q tests/test_cases.py`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add hasami/cases.py tests/test_cases.py
git commit -m "Crop cases: write, read, list and zip cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: capture in the Session

**Files:**
- Modify: `hasami/workflow.py` (`Session.__init__`, `run_preview`, `detect_on_preview`, `run_scan`; new `_capture`, `save_last_case`)
- Test: `tests/test_cases.py`

**Interfaces:**
- Consumes: Task 1 and 2's `cases` functions and `CaseData`.
- Produces: `Session.auto_mm: list[CropBox]`, `Session.last_case: cases.CaseData | None`, `Session.last_case_path: Path | None`, `Session.case_note: str` (the status line of the last scan's case, "" when none), `Session.save_last_case() -> Path | None`; auto boxes carry `meta["auto_id"]` ("a<n>", unique per Session); boxes kept by a new preview carry `meta["carried"] = True`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cases.py`)

```python
def _session(tmp_path, enabled=True, **input_kw):
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.input.scan_dpi = input_kw.get("scan_dpi", 150)
    st.crop.mode = "booklet"
    st.output.folder = str(tmp_path / "scans")
    st.cases.enabled = enabled
    st.cases.folder = str(tmp_path / "cases")
    return Session(st)


def _bottom_out(b, mm=4.0):
    return replace(b, h=b.h + mm, cy=b.cy + mm / 2, manual=True, meta=dict(b.meta))


def test_auto_boxes_get_unique_ids(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    ids = [b.meta["auto_id"] for b in s.boxes_mm]
    assert ids and len(set(ids)) == len(ids)
    assert [a.meta["auto_id"] for a in s.auto_mm] == ids
    s.run_preview()
    assert not set(ids) & {b.meta["auto_id"] for b in s.boxes_mm}  # never reused


def test_a_corrected_scan_writes_one_case(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    notes = []
    outcomes = s.run_scan(status=notes.append)
    assert outcomes[0].paths  # the scan itself saved as usual
    [path] = cases.list_cases(tmp_path / "cases")
    doc = cases.read_case(path)
    assert doc["reason"] == "corrected" and doc["scanner"]["id"] == "demo:booklet"
    moved = [e for e in doc["edits"] if e["kind"] == "moved"]
    assert len(moved) == 1 and moved[0]["sides_mm"]["bottom"] == pytest.approx(4.0, abs=0.01)
    assert doc["preview"]["dpi"] == 50 and doc["scan"]["dpi"] == 150
    assert s.last_case_path == path and s.case_note.startswith(f"Saved crop case {path.name} (bottom edge 4.0 mm out")
    assert s.case_note in notes


def test_an_uncorrected_scan_writes_nothing_until_saved(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    s.run_scan()
    assert cases.list_cases(tmp_path / "cases") == [] and s.case_note == ""
    p = s.save_last_case()
    assert cases.read_case(p)["reason"] == "saved"
    assert s.save_last_case() == p  # once
    assert cases.list_cases(tmp_path / "cases") == [p]


def test_setting_off_keeps_and_writes_nothing(tmp_path):
    s = _session(tmp_path, enabled=False)
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    s.run_scan()
    assert s.last_case is None and s.save_last_case() is None
    assert not (tmp_path / "cases").exists()


def test_a_case_that_cannot_be_written_does_not_break_the_scan(tmp_path):
    s = _session(tmp_path)
    (tmp_path / "blocked").write_text("a file, not a folder")
    s.settings.cases.folder = str(tmp_path / "blocked" / "cases")
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    notes = []
    outcomes = s.run_scan(status=notes.append)
    assert outcomes[0].paths and all(Path(p).exists() for p in outcomes[0].paths)
    assert any(n.startswith("Could not save the crop case:") for n in notes)


def test_the_scan_is_kept_at_400_dpi_at_most(tmp_path):
    from hasami import cases

    s = _session(tmp_path, scan_dpi=600)
    s.run_preview()
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    s.run_scan()
    [path] = cases.list_cases(tmp_path / "cases")
    doc = cases.read_case(path)
    assert doc["scan"]["scan_dpi"] == 600 and doc["scan"]["dpi"] == 400
    with Image.open(path / "scan.jpg") as im:
        assert round(im.info["dpi"][0]) == 400
        w_mm = im.size[0] / 400 * 25.4
    assert w_mm == pytest.approx(doc["scan"]["area_mm"][2], abs=0.5)


def test_a_feeder_batch_writes_one_case(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    dev = s.open()
    one = dev.scan
    dev.scan_batch = lambda req, progress=None, cancel=None: iter([one(req), one(req)])
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    assert len(s.run_scan()) == 2
    assert len(cases.list_cases(tmp_path / "cases")) == 1


def test_off_mode_never_writes_a_case_by_itself(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.settings.crop.mode = "off"
    s.run_preview()
    drawn = CropBox.from_xywh(20, 20, 60, 80, manual=True)
    drawn.meta["drawn"] = True
    s.boxes_mm = [drawn]
    s.run_scan()
    assert cases.list_cases(tmp_path / "cases") == []
    assert s.save_last_case() is not None  # still possible by hand


def test_a_drawn_box_carried_to_the_next_page_is_not_a_new_correction(tmp_path):
    from hasami import cases

    s = _session(tmp_path)
    s.run_preview()
    drawn = CropBox.from_xywh(5, 5, 20, 20, manual=True)
    drawn.meta["drawn"] = True
    s.boxes_mm.append(drawn)
    s.run_scan()
    assert len(cases.list_cases(tmp_path / "cases")) == 1  # drawn on this page: a correction
    s.run_preview()  # next page: the drawn box stays
    assert any(b.meta.get("carried") for b in s.boxes_mm)
    s.run_scan()
    assert len(cases.list_cases(tmp_path / "cases")) == 1  # carried over: not a new one


def test_redetect_keeps_the_edited_boxs_auto_original(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    first = s.auto_mm[0]
    s.boxes_mm[0] = _bottom_out(s.boxes_mm[0])
    s.detect_on_preview()  # a Crop setting changed / Ctrl+R: edited boxes are kept
    assert first.meta["auto_id"] in [a.meta["auto_id"] for a in s.auto_mm]
    kept = next(a for a in s.auto_mm if a.meta["auto_id"] == first.meta["auto_id"])
    assert (kept.cy, kept.h) == (first.cy, first.h)  # the auto original, not the edit
    ids = [a.meta["auto_id"] for a in s.auto_mm]
    assert len(set(ids)) == len(ids)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest -q tests/test_cases.py`
Expected: Task 1-2 tests pass; the new ones FAIL (`KeyError: 'auto_id'`, `AttributeError: 'Session' object has no attribute 'auto_mm'` and similar).

- [ ] **Step 3: Implement in `hasami/workflow.py`**

Imports: add `from . import cases` with the other package imports, and `from pathlib import Path` if not imported.

In `Session.__init__`, after `self.boxes_mm`:

```python
        #: The boxes auto-crop proposed (bed mm, with meta["auto_id"]): what a crop case compares against.
        self.auto_mm: list[CropBox] = []
        self._auto_seq = 0
        #: The last scan's crop case (crop cases on), its folder once written, and its status line.
        self.last_case: cases.CaseData | None = None
        self.last_case_path: Path | None = None
        self.case_note = ""
```

In `run_preview`, replace

```python
        self.boxes_mm = [b for b in self.boxes_mm if b.meta.get("drawn")]
```

with

```python
        self.boxes_mm = [b for b in self.boxes_mm if b.meta.get("drawn")]
        for b in self.boxes_mm:
            b.meta["carried"] = True  # from an earlier page: not a correction of this one
```

Replace the end of `detect_on_preview`:

```python
        origin = (p.area_mm[0], p.area_mm[1])
        manual = [b for b in self.boxes_mm if b.manual]
        self.boxes_mm = manual + [box_px_to_mm(b, p.dpi, origin) for b in found]
        return self.boxes_mm
```

with

```python
        origin = (p.area_mm[0], p.area_mm[1])
        manual = [b for b in self.boxes_mm if b.manual]
        found_mm = [box_px_to_mm(b, p.dpi, origin) for b in found]
        for b in found_mm:
            self._auto_seq += 1
            b.meta["auto_id"] = f"a{self._auto_seq}"
        kept = {b.meta.get("auto_id") for b in manual} - {None}
        self.auto_mm = ([a for a in self.auto_mm if a.meta.get("auto_id") in kept]
                        + [replace(b, meta=dict(b.meta)) for b in found_mm])
        self.boxes_mm = manual + found_mm
        return self.boxes_mm
```

In `run_scan`, insert before `outcomes.append(self._finish(result, boxes, save, status))`:

```python
            if page_no == 0:
                self._capture(result, crops, status)
```

Add the two methods after `run_scan`:

```python
    def _capture(self, result: ScanResult, final_mm: list[CropBox], status: StatusFn | None) -> None:
        """Crop cases: keep this scan's case and write it when an auto box was corrected.  Never raises."""
        self.last_case, self.last_case_path, self.case_note = None, None, ""
        if not self.settings.cases.enabled or self.preview is None:
            return
        try:
            scan, saved_dpi = cases.downscale(result.image, result.dpi)
            info = self.device.info if self.device is not None else None
            c = self.settings.crop
            self.last_case = cases.CaseData(
                preview=self.preview.image, preview_dpi=self.preview.dpi,
                preview_area_mm=tuple(self.preview.area_mm), scan=scan, scan_dpi_saved=saved_dpi,
                scan_dpi=result.dpi, scan_area_mm=tuple(result.area_mm),
                auto=[replace(b, meta=dict(b.meta)) for b in self.auto_mm],
                final=[replace(b, meta=dict(b.meta)) for b in final_mm],
                scanner_id=self.device_id, scanner_name=info.name if info else "", profile=self.profile().name,
                crop={"mode": c.mode, "booklet_output": c.booklet_output, "booklet_rtl": c.booklet_rtl,
                      "margin_mm": c.margin_mm, "edge_refine": c.edge_refine, "rotate": c.rotate})
            edits = cases.diff_boxes(self.last_case.auto, self.last_case.final)
            if c.mode != "off" and cases.corrected(edits):  # Off: drawing boxes is how it works
                self.last_case_path = cases.write_case(cases.folder_of(self.settings.cases), self.last_case)
                self.case_note = f"Saved crop case {self.last_case_path.name} ({cases.summary(edits)})"
        except Exception as e:  # noqa: BLE001 - a crop case must never break a scan
            self.case_note = f"Could not save the crop case: {e}"
        if status and self.case_note:
            status(self.case_note)

    def save_last_case(self) -> Path | None:
        """"Save this case": write the last scan's case (reason "saved"); its path when already written,
        None without one (crop cases off, no scan yet)."""
        if self.last_case is None:
            return None
        if self.last_case_path is None:
            self.last_case.reason = "saved"
            self.last_case_path = cases.write_case(cases.folder_of(self.settings.cases), self.last_case)
        return self.last_case_path
```

(`ScanResult`, `StatusFn` and `replace` are already imported in `workflow.py`; check with `grep -n "^from\|^import" hasami/workflow.py` and add what is missing.)

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest -q tests/test_cases.py`
Expected: 24 passed.

- [ ] **Step 5: Run the existing Session tests**

Run: `$PY -m pytest -q tests/test_test_scan.py tests/test_cli.py`
Expected: all pass (the `carried` flag and auto ids live in `meta`, which `CropBox` equality ignores).

- [ ] **Step 6: Commit**

```bash
git add hasami/workflow.py tests/test_cases.py
git commit -m "Crop cases: the Session keeps each scan's case and writes corrected ones

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: GUI: settings, "Save this case", the share dialog

**Files:**
- Modify: `hasami/gui/settings_panel.py` (two `Field`s in the Crop tab)
- Modify: `hasami/gui/app.py` (actions, menus, `_scan_done`, `_busy`, `on_setting_changed`, `save_case`, `share_cases`)
- Create: `hasami/gui/cases_dialog.py`
- Test: `tests/test_cases_gui.py`

**Interfaces:**
- Consumes: `cases.list_cases`, `read_case`, `pack`, `summary`, `folder_of`, `SHARE_DIR`; `Session.last_case`, `case_note`, `save_last_case()`.
- Produces: `CasesDialog(folder: Path, parent=None)` with `reload()`, `checked() -> list[Path]`, `create_zips() -> list[Path]`, `delete_checked() -> list[Path]`, `_confirm(n: int) -> bool`, `_open(path: Path) -> None` (the last two replaced in tests); `MainWindow.a_save_case`, `MainWindow.a_share_cases`, `MainWindow.save_case()`, `MainWindow.share_cases()`.

- [ ] **Step 1: Write the failing tests** (`tests/test_cases_gui.py`)

```python
import time
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        pytest.skip(str(e))
    return QApplication.instance() or QApplication([])


def _case(folder, minute=0):
    from hasami import cases
    from hasami.core.geometry import CropBox

    a = CropBox(60.0, 80.0, 50.0, 70.0)
    a.meta["auto_id"] = "a1"
    f = replace(a, h=a.h + 4, cy=a.cy + 2, manual=True, meta=dict(a.meta))
    return cases.write_case(folder, cases.CaseData(
        preview=np.full((300, 220, 3), 200, np.uint8), preview_dpi=50.0, preview_area_mm=(0, 0, 111.8, 152.4),
        scan=np.full((200, 150, 3), 180, np.uint8), scan_dpi_saved=100.0, scan_dpi=100.0,
        scan_area_mm=(30, 40, 38.1, 50.8), auto=[a], final=[f], scanner_id="demo:booklet", scanner_name="Demo",
        created=datetime(2026, 10, 15, 14, minute, 0)))


def test_dialog_lists_zips_and_deletes_cases(app, tmp_path):
    from PySide6.QtCore import Qt

    from hasami.gui.cases_dialog import NOTE, CasesDialog

    folder = tmp_path / "cases"
    c1, c2 = _case(folder, 0), _case(folder, 1)
    broken = folder / "2026-10-15_150000"
    broken.mkdir()
    (broken / "case.json").write_text("{broken", encoding="utf-8")
    dlg = CasesDialog(folder)
    opened = []
    dlg._open = opened.append
    dlg._confirm = lambda n: True
    try:
        assert dlg.list.count() == 3
        texts = [dlg.list.item(i).text() for i in range(3)]
        assert texts[0].startswith("2026-10-15_150000") and "unreadable" in texts[0]
        assert "bottom edge 4.0 mm out" in texts[1] and "Demo" in texts[1]
        assert "Share only what you are happy to share" in NOTE and "discord.gg/TAAV8B9WWg" in NOTE
        dlg.list.item(1).setCheckState(Qt.Checked)
        assert dlg.checked() == [c2]
        zips = dlg.create_zips()
        assert [z.name for z in zips] == [f"hasami-case-{c2.name}.zip"] and zips[0].exists()
        assert opened == [folder / "To share"]
        dlg.reload()
        assert dlg.list.count() == 3  # the share folder is not a case
        for i in (0, 2):
            dlg.list.item(i).setCheckState(Qt.Checked)
        assert sorted(dlg.delete_checked()) == sorted([broken, c1])
        assert not broken.exists() and not c1.exists() and c2.exists()
        assert dlg.list.count() == 1
    finally:
        dlg.close()


def test_delete_asks_first(app, tmp_path):
    from PySide6.QtCore import Qt

    from hasami.gui.cases_dialog import CasesDialog

    c = _case(tmp_path / "cases")
    dlg = CasesDialog(tmp_path / "cases")
    dlg._confirm = lambda n: False
    try:
        dlg.list.item(0).setCheckState(Qt.Checked)
        assert dlg.delete_checked() == [] and c.exists()
    finally:
        dlg.close()


def test_the_app_saves_a_corrected_scan_and_offers_save_this_case(app, tmp_path):
    from PySide6.QtCore import QThreadPool

    from hasami import cases
    from hasami.gui.app import MainWindow
    from hasami.settings import AppSettings

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.input.scan_dpi = 150
    st.crop.mode = "booklet"
    st.output.folder = str(tmp_path / "scans")
    st.cases.enabled = True
    st.cases.folder = str(tmp_path / "cases")
    win = MainWindow(st)

    def wait():
        end = time.monotonic() + 30
        while win.task is not None and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert win.task is None

    try:
        assert win.a_share_cases.text() == "Share crop cases..." and not win.a_save_case.isEnabled()
        win.do_preview()
        wait()
        it = win.preview.items_[0]
        b = it.box
        it.box = replace(b, h=b.h + 20, cy=b.cy + 10, manual=True, meta=dict(b.meta))  # bottom out (preview px)
        win.do_scan()
        wait()
        [path] = cases.list_cases(tmp_path / "cases")
        assert f"Saved crop case {path.name}" in win.status_label.text()
        assert win.a_save_case.isEnabled()
        win.save_case()  # already saved as a correction: nothing new
        assert cases.list_cases(tmp_path / "cases") == [path]
    finally:
        QThreadPool.globalInstance().waitForDone(10000)
        app.processEvents()
        win.close()


def test_crop_tab_has_the_case_settings(app):
    from hasami.gui.settings_panel import SettingsPanel
    from hasami.settings import AppSettings

    st = AppSettings()
    panel = SettingsPanel(st)
    assert ("cases", "enabled") in panel._widgets and ("cases", "folder") in panel._widgets
    assert not panel._widgets[("cases", "folder")].isEnabled()  # greyed out until the setting is on
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest -q tests/test_cases_gui.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'hasami.gui.cases_dialog'`, `AttributeError: 'MainWindow' object has no attribute 'a_share_cases'`, and the panel test's KeyError.

- [ ] **Step 3: Crop-tab settings** (`hasami/gui/settings_panel.py`, at the end of the `("Crop", [...])` list, after the `border_search_mm` field)

```python
        Field("cases", "enabled", "Save my corrections as crop cases (to help tune auto-crop)",
              tip="When you correct an auto-crop box, keep the preview, the scanned area and the boxes in the "
                  "case folder (File > Share crop cases... to send some on Discord).  Off by default; nothing "
                  "is ever uploaded."),
        Field("cases", "folder", "Crop case folder", kind="folder", enabled_if=lambda st: st.cases.enabled,
              tip="Empty: Documents/Hasami Scan/Crop cases"),
```

- [ ] **Step 4: Create `hasami/gui/cases_dialog.py`**

```python
"""File > Share crop cases...: pick saved crop cases, zip them for the Discord server, or delete them."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QVBoxLayout)

from .. import cases
from ..core.geometry import CropBox
from .preview import to_qimage

DISCORD = "https://discord.gg/TAAV8B9WWg"
NOTE = ("Each case contains scans of your item (the preview and the scanned area). Share only what you are happy "
        f"to share. Post the zips in the tuning channel on the Hasami Scan Discord: {DISCORD}")
THUMB = 160


class CasesDialog(QDialog):
    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Share crop cases")
        self.resize(640, 560)
        self.folder = Path(folder)
        self.list = QListWidget()
        self.list.setIconSize(QSize(THUMB, THUMB))
        note = QLabel(NOTE.replace(DISCORD, f'<a href="{DISCORD}">{DISCORD}</a>'))
        note.setWordWrap(True)
        note.setTextFormat(Qt.RichText)
        note.setOpenExternalLinks(True)
        zip_btn, del_btn, open_btn, close_btn = (QPushButton(t) for t in
                                                 ("Create zip", "Delete", "Open folder", "Close"))
        zip_btn.clicked.connect(lambda *_: self.create_zips())
        del_btn.clicked.connect(lambda *_: self.delete_checked())
        open_btn.clicked.connect(lambda *_: self._open(self.folder))
        close_btn.clicked.connect(lambda *_: self.accept())
        row = QHBoxLayout()
        for b in (zip_btn, del_btn, open_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close_btn)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addWidget(note)
        lay.addLayout(row)
        self.reload()

    def reload(self) -> None:
        self.list.clear()
        for path in cases.list_cases(self.folder):
            try:
                doc = cases.read_case(path)
                scanner = doc["scanner"].get("name") or doc["scanner"].get("id") or "?"
                text = f"{path.name}  ·  {scanner}  ·  {doc['reason']}\n{cases.summary(doc['edits'])}"
                icon = QIcon(self._thumb(path, doc))
            except (ValueError, OSError, KeyError) as e:
                text, icon = f"{path.name}  ·  unreadable ({e})", QIcon()
            it = QListWidgetItem(icon, text)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, str(path))
            self.list.addItem(it)

    @staticmethod
    def _thumb(path: Path, doc: dict) -> QPixmap:
        """The preview with the boxes scanned drawn on it."""
        with Image.open(path / doc["preview"]["file"]) as im:
            img = np.asarray(im.convert("RGB"))
        pm = QPixmap.fromImage(to_qimage(img))
        f = float(doc["preview"]["dpi"]) / 25.4
        x0, y0 = doc["preview"]["area_mm"][:2]
        p = QPainter(pm)
        p.setPen(QPen(QColor(255, 140, 0), max(2, pm.width() // 150)))
        for b in doc["final"]:
            pts = (CropBox(*b["box_mm"]).corners() - [x0, y0]) * f - 0.5
            p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in pts]))
        p.end()
        return pm.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def checked(self) -> list[Path]:
        return [Path(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def create_zips(self) -> list[Path]:
        sel = self.checked()
        if not sel:
            return []
        out = self.folder / cases.SHARE_DIR
        zips = cases.pack(sel, out)
        self._open(out)
        return zips

    def delete_checked(self) -> list[Path]:
        sel = self.checked()
        if not sel or not self._confirm(len(sel)):
            return []
        for p in sel:
            shutil.rmtree(p, ignore_errors=True)
        self.reload()
        return sel

    def _confirm(self, n: int) -> bool:
        return QMessageBox.question(self, "Delete crop cases",
                                    f"Delete {n} crop case{'s' if n > 1 else ''}? This cannot be undone.") \
            == QMessageBox.Yes

    def _open(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
```

- [ ] **Step 5: Wire the app** (`hasami/gui/app.py`)

Import: `from .. import __version__, cases, updates` (add `cases`).

In `_build_actions`, after `self.a_test_folder = ...`:

```python
        self.a_save_case = act("Save this case", self.save_case,
                               tip="Keep this scan's preview, scanned area and boxes as a crop case (Crop tab), "
                                   "also when no box was corrected")
        self.a_share_cases = act("Share crop cases...", self.share_cases,
                                 tip="Zip saved crop cases to post on the Discord server, or delete them")
```

File menu: after `m.addAction(self.a_test_folder)` add `m.addAction(self.a_share_cases)`. Scan menu: change the loop tuple to `(self.a_preview, self.a_scan, self.a_test, self.a_save_case, self.a_autocrop, self.a_delete, self.a_clear)`.

At the end of `_build_actions` add `self._update_case_actions()`.

New methods (next to `open_test_folder`):

```python
    def _update_case_actions(self, busy: bool = False) -> None:
        self.a_save_case.setEnabled(not busy and self.settings.cases.enabled and self.session.last_case is not None)

    def save_case(self) -> None:
        try:
            path = self.session.save_last_case()
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Hasami Scan", f"Could not save the crop case: {e}")
            return
        if path is not None:
            self.set_status(f"Saved crop case {path.name} to {path.parent}")

    def share_cases(self) -> None:
        from .cases_dialog import CasesDialog

        CasesDialog(cases.folder_of(self.settings.cases), self).exec()
```

In `_busy`, after the `for a in (...): a.setEnabled(not busy)` loop add `self._update_case_actions(busy)`.

In `_scan_done`, replace

```python
        self.set_status(f"Saved {n} item(s) to {self.settings.output.path} (scanned {areas})")
```

with

```python
        msg = f"Saved {n} item(s) to {self.settings.output.path} (scanned {areas})"
        self.set_status(f"{msg}. {self.session.case_note}" if self.session.case_note else msg)
```

and at its end add `self._update_case_actions()`.

In `on_setting_changed`, after the first two lines (`_save_timer` / `_cropped_timer`):

```python
        if section == "cases":
            self._update_case_actions()
            return
```

(`a_share_cases` stays enabled: the dialog shows an empty list when there are no cases.)

- [ ] **Step 6: Run the tests**

Run: `$PY -m pytest -q tests/test_cases_gui.py tests/test_cases.py`
Expected: 28 passed.

- [ ] **Step 7: Run the GUI tests that already exist**

Run: `$PY -m pytest -q tests/test_gui.py tests/test_test_scan.py`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add hasami/gui/settings_panel.py hasami/gui/app.py hasami/gui/cases_dialog.py tests/test_cases_gui.py
git commit -m "Crop cases: Crop-tab settings, Save this case, Share crop cases dialog

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: version, docs, full checks, PR

**Files:**
- Modify: `hasami/__init__.py`, `pyproject.toml` (0.5.16 -> 0.5.17)
- Modify: `README.md` (section "Helping tune auto-crop"), `CLAUDE.md` (note "Crop cases (0.5.17)")

**Interfaces:**
- Consumes: everything above.
- Produces: PR `feat/crop-cases`.

- [ ] **Step 1: Version**

In `hasami/__init__.py`: `__version__ = "0.5.17"`; in `pyproject.toml`: `version = "0.5.17"`.

- [ ] **Step 2: README** (after the paragraph that links the Discord server for tuning, around line 77)

```markdown
### Helping tune auto-crop

Turn on **Crop > Save my corrections as crop cases**. Whenever you fix an auto-crop box before scanning, Hasami
Scan keeps the preview, the scanned area (at most 400 dpi) and the boxes in a "crop case" (Documents/Hasami
Scan/Crop cases). **Scan > Save this case** keeps one when the crop was right. Nothing is uploaded: use **File >
Share crop cases...** to zip the cases you choose and post them on the Discord server.
```

- [ ] **Step 3: CLAUDE.md** (a new section before "## Next: learned cropping")

```markdown
## Crop cases (0.5.17, owner)
Opt-in (`AppSettings.cases.enabled`, Crop tab; folder `cases.folder`, "" = Documents/Hasami Scan/Crop cases).
`hasami/cases.py` (no Qt): a case folder = `preview.jpg` (whole bed, preview dpi) + `scan.jpg` (the first page's
uncropped scanned area, <= 400 dpi, both JPEG q95, scanner frame) + `case.json` (format 1: scanner, profile, crop
settings, auto and final boxes in bed mm, `diff_boxes` edits: moved per side + = outwards / deleted / drawn).
`Session.detect_on_preview` tags auto boxes `meta["auto_id"]` (unique per session) and keeps `auto_mm`;
`run_scan` -> `_capture` after page 1 keeps `last_case` and writes it when corrected (never in Off mode; boxes
carried from an earlier page, `meta["carried"]`, are not edits); errors become a status line, never a failed scan.
Scan > "Save this case" (`save_last_case`), File > "Share crop cases..." (`gui/cases_dialog.py`: one zip per case
in `To share`). Spec `docs/specs/2026-10-15-crop-cases-design.md`. Next: part 2, snapping box edges to the engine's
candidate lines and "check this side" flags (own spec); a hasami-data tool to turn cases into truth records.
```

- [ ] **Step 4: Full test suite**

Run: `$PY -m pytest -q > %TEMP%/crop-cases-tests.txt 2>&1; tail -3` (Git Bash: `$PY -m pytest -q 2>&1 | tail -3`)
Expected: all pass (163 + the 28 new = 191 passed, 1 skipped, 3 xfailed; the exact total is what the run prints, with no failures).

- [ ] **Step 5: Synthetic benchmarks**

Run: `for f in benchmarks/synthetic_*.py; do $PY "$f"; done 2>&1 | tail -20`
Expected: the same numbers as on main (no crop-engine change); report them in the PR.

- [ ] **Step 6: Commit, push, PR**

```bash
git add hasami/__init__.py pyproject.toml README.md CLAUDE.md
git commit -m "0.5.17: crop cases (save and share auto-crop corrections)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/crop-cases
gh pr create --repo PetMyRock/hasami-scan --base main --title "0.5.17: crop cases (save and share auto-crop corrections)" --body-file <body>
```

PR body: what a case is, the setting (off by default), Save this case, the share dialog and its privacy note, the tests, the synthetic numbers (identical), "no crop-engine change"; ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. No closing keywords.
