# Edge snapping, pinned sides and flags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Booklet box sides snap to the engine's candidate lines while dragged (Shift = free), sides moved by hand stay put at scan time while the others are still refined, and - if a measured rule passes its gate - sides the rules were unsure about are flagged.

**Architecture:** A new read-only module `hasami/core/sidelines.py` recomputes candidate lines around each final booklet box with toppage's own line finding (never called by `detect` / `refine`). `Session.detect_on_preview` keeps them per auto box in bed mm (`Session.snap_lines`); the preview's `CropItem` snaps a dragged side to them and records `meta["pinned_sides"]`. `autocrop._refine` refines a partly pinned auto box like an auto box and `geometry.pin_sides` puts the pinned sides back. A private measurement in hasami-data decides whether flags ship and with which rule.

**Tech Stack:** Python 3.14, numpy, OpenCV, PySide6, pytest (`F:/HasamiDev/.venv/Scripts/python.exe`, written `$PY`).

**Spec:** `docs/specs/2026-10-15-edge-snapping-design.md` (approved Oct 15 2026). Read it first.

**Deviations from the spec (decided while planning; the owner reviews this plan):**
- The new code lives in its own module `hasami/core/sidelines.py` (spec: `toppage.py`): it makes "read-only, never in the engine" visible and leaves `toppage.py` byte-identical. `SnapLine` lives there too (spec: `workflow.py`) because both the Session and the GUI use it.
- Snap lines are stored moved inwards by the crop margin (`crop.margin_mm`, 0.25 mm): auto-crop puts a side that far inside the edge it finds, so a snapped side lands where auto-crop would have put it from that line. `side_lines` itself reports offsets from the box's side as the spec says.
- `pin_sides` takes `meta` from the given (hand-edited) box, not the refined one: ids and pins must survive whatever a refine path does with `meta`.
- The measurement uses the V600 records in booklet mode only (48 V600 pages records, the 9 "V600 misc" items are multi mode, which snapping does not cover) + Thick books 26 + KLK6 9 + New data 33 = 116 records, `--turn 0` only (each side counted once).

## Global Constraints

- Snapping: booklet mode only; side drags only (not corners, moves, drawn boxes); Shift drags freely; reach 8 screen px; lines tilted > 1.5 deg against the side are ignored; the side moves parallel (box angle kept). No new setting.
- Lines: paper sides 14 mm inside to 8 mm outside; spine sides +-4 mm (kind "crease"); merged within 0.15 mm (higher support wins, then layer > strong > faint > crease); columns off the image ignored; budget <= 100 ms per page at 200 dpi.
- `detect`, `refine` on unpinned boxes, `top_page`, the candidate collector: unchanged. Synthetic and real-scan benchmarks must be identical to main.
- Pinning: side drag pins that side; corner drag the two sides at that corner; whole-box move, rotate handle and Skew all four; pins only accumulate. Refine touches a manual box only when it has `meta["auto_id"]` and 1-3 pinned sides; the angle is kept.
- Flags only if the measured rule passes on held-out books: flagged <= 1 side in 8 AND >= half of the sides > 0.5 mm off flagged. Snap gate: >= 60% of the sides > 0.5 mm off have the truth within 0.3 mm of a candidate; below it STOP and report to the owner.
- `case.json` stays format 1; new optional keys `check_sides` (auto), `pinned_sides` / `snapped_sides` (final), omitted when empty.
- Scans and truth stay private: nothing from hasami-data or scan folders enters this repo. Tests here use generated scans only. Real-scan runs with `--jobs 2`.
- Never run `MainWindow` with the real APPDATA (tests: `tests/conftest.py` already isolates it).
- Branch `feat/edge-snapping` (hasami-scan) and `feat/snap-check` (hasami-data); PRs the owner merges; never `fix:` / `closes` / `fixes` + `#N` in PR text.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR bodies with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Version 0.5.18 -> 0.5.19 (`hasami/__init__.py`, `pyproject.toml`).

## Review Focus

- A turned preview (`crop.rotate` 1-3, the owner's usual setup): snapping works in the box's own frame, so it must snap the same in a turned view (Task 6: `test_snapping_works_in_a_turned_view`).
- Auto-crop run again after an edit (a Crop setting changed, Ctrl+R): the edited box keeps its snap lines, the replaced boxes' lines go (Task 5: `test_redetect_keeps_the_lines_of_edited_boxes`).
- Grey or 16-bit previews: `side_lines` must still find the edges (Task 1: `test_grey_16_bit_images_work`).
- A page near the bed's edge (the 8 mm outer strip runs off the image): lines from the on-image columns only, none at the image's own edge (Task 1: `test_columns_off_the_image_are_ignored`).
- Boxes edited without pins (made `manual` by older code paths, tests, a box carried over a re-detect from before this version) stay untouched by refine exactly as today (Task 4: `test_drawn_unpinned_and_fully_pinned_boxes_are_untouched`).

---

### Task 1: `side_lines` - candidate lines around a final box

**Files:**
- Modify: `hasami/core/geometry.py` (add `SIDE_NAMES`)
- Create: `hasami/core/sidelines.py`
- Test: `tests/test_sidelines.py`

**Interfaces:**
- Produces: `hasami.core.geometry.SIDE_NAMES = ("top", "right", "bottom", "left")`; `hasami.core.sidelines.SideLine(offset_mm: float, slope: float, support: float, kind: str)` (frozen dataclass); `side_lines(image: np.ndarray, box: CropBox, dpi: float) -> dict[str, list[SideLine]]`; `_merge(lines: list[SideLine]) -> list[SideLine]`; constants `IN_MM = 14.0`, `OUT_MM = 8.0`, `SPINE_MM = 4.0`, `MERGE_MM = 0.15`, `KINDS`, `KEY` (name -> toppage side key).

- [ ] **Step 1: Branch**

```bash
cd F:/HasamiDev && git checkout main && git pull -q && git checkout -b feat/edge-snapping
```

- [ ] **Step 2: Write the failing tests** (`tests/test_sidelines.py`)

```python
"""Candidate lines around a final crop box (generated scans only)."""
import numpy as np
import pytest

from hasami.core.autocrop import AutoCropSettings, detect
from hasami.core.geometry import CropBox
from hasami.demo import BookletOptions, synth_booklet


def _stack_scan(dpi=300):
    """A white page on a grey lid with a page-stack band beyond its right edge: 6 page edges,
    0.4 mm apart (as tests/test_booklet.py)."""
    mm = dpi / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 205, np.uint8)
    x0, y0, x1, y1 = (int(20 * mm), int(20 * mm), int(130 * mm), int(180 * mm))
    img[y0:y1, x0:x1] = 246
    for i in range(6):
        a = x1 + int(i * 0.4 * mm)
        img[y0 + 2:y1 - 2, a:a + int(0.4 * mm)] = 238 - 3 * i
        img[y0 + 2:y1 - 2, a:a + 1] = 200
    img[y0:y1, x1:x1 + 1] = 215
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, (x0, y0, x1, y1), mm


def _page_box(x0, y0, x1, y1):
    return CropBox.from_xywh(x0, y0, x1 - x0, y1 - y0)


def test_paper_side_lines_include_the_top_page_edge_and_the_stack():
    from hasami.core.sidelines import SideLine, side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    lines = side_lines(img, _page_box(x0, y0, x1, y1), 300)
    assert set(lines) == {"top", "right", "bottom", "left"}
    right = [ln.offset_mm for ln in lines["right"]]
    assert right == sorted(right)
    assert any(abs(x) < 0.3 for x in right)  # the top page's own edge, under the side
    assert sum(0.2 < x < 3.0 for x in right) >= 2  # stack sheets outside it
    assert any(abs(ln.offset_mm) < 0.3 for ln in lines["left"])  # the page's edge on the lid
    assert all(isinstance(ln, SideLine) and 0.0 < ln.support <= 1.0 and ln.kind in ("layer", "faint", "strong")
               for ln in lines["right"])


def test_columns_off_the_image_are_ignored():
    from hasami.core.sidelines import side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    img = np.ascontiguousarray(img[:, : x1 + int(3 * mm)])  # the image ends 3 mm right of the top page
    right = [ln.offset_mm for ln in side_lines(img, _page_box(x0, y0, x1, y1), 300)["right"]]
    assert any(abs(x) < 0.3 for x in right)
    assert all(x < 2.9 for x in right)  # nothing at the image's own edge


def test_a_side_off_the_image_has_no_lines():
    from hasami.core.sidelines import side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    box = _page_box(x0, y0, img.shape[1] + 30, y1)  # the right side lies beyond the image
    assert side_lines(img, box, 300)["right"] == []


def test_spine_side_gets_crease_lines():
    from hasami.core.sidelines import side_lines

    mm = 300 / 25.4
    img = np.full((int(200 * mm), int(160 * mm), 3), 245, np.uint8)
    crease = int(100 * mm)
    img[:, crease - 2:crease + 3] = 170  # the fold's dark crease line
    box = CropBox.from_xywh(int(20 * mm), int(20 * mm), crease + 2 * mm - int(20 * mm), int(160 * mm))
    box.meta["keep_sides"] = [(0, 1)]  # the right side is the spine
    right = side_lines(img, box, 300)["right"]
    assert right and all(ln.kind == "crease" for ln in right)
    assert any(abs(ln.offset_mm + 2.0) < 0.3 for ln in right)  # the crease 2 mm inside the side


def test_grey_16_bit_images_work():
    from hasami.core.sidelines import side_lines

    img, (x0, y0, x1, y1), mm = _stack_scan()
    grey16 = img[..., 1].astype(np.uint16) * 257
    right = side_lines(grey16, _page_box(x0, y0, x1, y1), 300)["right"]
    assert any(abs(ln.offset_mm) < 0.3 for ln in right)


def test_lines_closer_than_the_merge_distance_keep_the_best_supported():
    from hasami.core.sidelines import SideLine, _merge

    out = _merge([SideLine(1.0, 0.0, 0.7, "faint"), SideLine(1.1, 0.0, 0.9, "layer"),
                  SideLine(2.0, 0.0, 0.6, "strong"), SideLine(2.1, 0.0, 0.6, "layer")])
    assert [(ln.offset_mm, ln.kind) for ln in out] == [(1.1, "layer"), (2.1, "layer")]


def test_detect_is_unchanged_by_side_lines():
    from hasami.core.sidelines import side_lines

    sc = synth_booklet(BookletOptions(), seed=0)
    st = AutoCropSettings(mode="booklet")
    before = detect(sc.image, st, dpi=200)
    for b in before:
        side_lines(sc.image, b, 200)
    after = detect(sc.image, st, dpi=200)
    assert [(b.cx, b.cy, b.w, b.h, b.angle) for b in before] == [(b.cx, b.cy, b.w, b.h, b.angle) for b in after]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `$PY -m pytest tests/test_sidelines.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'hasami.core.sidelines'` in every test.

- [ ] **Step 4: Add `SIDE_NAMES` to `hasami/core/geometry.py`** (module level, after the imports, before `normalize_angle`)

```python
#: A box's sides by name, in its own frame (top = towards -y at angle 0), in the order of ``corners()``' pairs.
SIDE_NAMES = ("top", "right", "bottom", "left")
```

- [ ] **Step 5: Write `hasami/core/sidelines.py`**

```python
"""Candidate edge lines around a final crop box: what a dragged box side snaps to, and what
"check this side" flags are measured on.

Read-only: computed on the preview after ``detect`` (``Session.detect_on_preview``, booklet mode),
never inside the engine, so no crop changes.  Uses toppage's own line finding (same segments,
thresholds and support) on a wider strip than the rules see: 14 mm inside to 8 mm outside a paper
side, and dark valleys within 4 mm of a spine.  Spec: docs/specs/2026-10-15-edge-snapping-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import SIDE_NAMES, CropBox
from .profiles import current as _profile
from .toppage import MM_PER_INCH, SIDES, _layer_lines, _rgb8, _segment_peaks, _side, _to_lab

IN_MM, OUT_MM = 14.0, 8.0  # strip across a paper side
SPINE_MM = 4.0  # valleys this far either side of a spine
MERGE_MM = 0.15  # lines closer than this are one
KINDS = ("layer", "strong", "faint", "crease")  # preferred in this order when support ties
KEY = dict(zip(SIDE_NAMES, SIDES))  # name -> toppage's (axis, sign)


@dataclass(frozen=True)
class SideLine:
    offset_mm: float  # at the side's middle, + = outwards from the box's side
    slope: float  # tilt against the side: x = offset + slope * y (y along the side)
    support: float  # share of the segments that show it, 0..1
    kind: str  # "layer" | "faint" | "strong" | "crease"


def side_lines(image: np.ndarray, box: CropBox, dpi: float) -> dict[str, list[SideLine]]:
    """Every candidate line near each side of a final page box, keyed "top" / "right" / "bottom" /
    "left" (the box's own frame), each list sorted by offset.  Spine sides (``box.meta["keep_sides"]``)
    get the fold's dark valleys, the other sides the edges toppage's rules choose from."""
    rgb8 = _rgb8(image)
    px_mm = dpi / MM_PER_INCH
    step = max(1.0, px_mm / 16.0)  # as top_page: sample at <= 16 px/mm
    keep = {tuple(k) for k in box.meta.get("keep_sides", ())}
    out = {}
    for name, key in KEY.items():
        a, e, d, n = _side(box, key)
        find = _valleys if key in keep else _edges
        out[name] = find(rgb8, a, e, d, n, px_mm, step)
    return out


def _grid(a, d, n, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    return (a[0] + n[0] * X + d[0] * Y).astype(np.float32), (a[1] + n[1] * X + d[1] * Y).astype(np.float32)


def _usable(mx, my, shape, xs, px_mm):
    """The run of columns on the image (in >= 90% of the rows) around the side itself, or None."""
    H, W = shape[:2]
    ok = ((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)).mean(axis=0) >= 0.9
    i = int(np.argmin(np.abs(xs)))
    if not ok[i]:
        return None
    lo, hi = i, i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return slice(lo, hi + 1) if xs[hi] - xs[lo] >= 2.0 * px_mm else None


def _strip(rgb8, a, d, n, xs, ys, px_mm):
    """RGB samples of the usable columns and their positions, or (None, None)."""
    mx, my = _grid(a, d, n, xs, ys)
    cols = _usable(mx, my, rgb8.shape, xs, px_mm)
    if cols is None:
        return None, None
    mx, my = np.ascontiguousarray(mx[:, cols]), np.ascontiguousarray(my[:, cols])
    return cv2.remap(rgb8, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), xs[cols]


def _edges(rgb8, a, e, d, n, px_mm, step):
    """Layer lines across a paper side at toppage's three thresholds."""
    side_len = float(np.linalg.norm(e - a))
    ys = np.arange(0.06 * side_len, 0.94 * side_len, step)  # as toppage._strip
    rgb, xs = _strip(rgb8, a, d, n, np.arange(-IN_MM * px_mm, OUT_MM * px_mm, step), ys, px_mm)
    if rgb is None:
        return []
    s = _to_lab(rgb)
    prof = _profile()
    lines = []
    for kind, pct in (("layer", 50.0), ("faint", prof.fine_stack_percentile),
                      ("strong", prof.fine_strong_percentile)):
        for x0, b, sup in _layer_lines(_segment_peaks(s, xs, ys, step / px_mm, pct), px_mm, side_len):
            lines.append(SideLine(x0 / px_mm, b, sup / prof.segments, kind))
    return _merge(lines)


def _valleys(rgb8, a, e, d, n, px_mm, step):
    """Dark valleys along a spine side (each segment's local minima, where toppage._crease takes only
    its deepest), pooled into lines."""
    side_len = float(np.linalg.norm(e - a))
    ys = np.arange(0.15 * side_len, 0.85 * side_len, step)  # as toppage._crease
    rgb, xs = _strip(rgb8, a, d, n, np.arange(-SPINE_MM * px_mm, SPINE_MM * px_mm, step), ys, px_mm)
    if rgb is None:
        return []
    s = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    step_mm = step / px_mm
    w = max(1, int(round(0.08 / step_mm)))
    pts = []
    for k, rows in enumerate(np.array_split(np.arange(len(ys)), _profile().segments)):
        if len(rows) < 3:
            continue
        pr = np.median(s[rows], axis=0)
        hp = pr - cv2.GaussianBlur(pr[None], (0, 0), 0.8 / step_mm)[0]
        hp = cv2.GaussianBlur(hp[None], (0, 0), max(0.05 / step_mm, 0.5))[0]
        thr = -max(2.0, 4.5 * float(np.median(np.abs(hp - np.median(hp)))))  # deeper than the paper's texture
        for i in range(w, len(hp) - w):
            if hp[i] <= thr and hp[i] == hp[i - w:i + w + 1].min():
                pts.append((xs[i], float(ys[rows].mean()), float(-hp[i]), k))
    return _merge([SideLine(x0 / px_mm, b, sup / _profile().segments, "crease")
                   for x0, b, sup in _layer_lines(pts, px_mm, side_len)])


def _merge(lines: list[SideLine]) -> list[SideLine]:
    """One line per place (within MERGE_MM): the best supported, then by KINDS; sorted by offset."""
    kept: list[SideLine] = []
    for ln in sorted(lines, key=lambda l: (-l.support, KINDS.index(l.kind))):
        if all(abs(ln.offset_mm - k.offset_mm) > MERGE_MM for k in kept):
            kept.append(ln)
    return sorted(kept, key=lambda l: l.offset_mm)
```

Note on `_merge`'s test: (2.0, strong, 0.6) and (2.1, layer, 0.6) tie on support, "layer" comes first in `KINDS`, so 2.1 is kept.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_sidelines.py -q`
Expected: 7 passed. If `test_paper_side_lines_include_the_top_page_edge_and_the_stack` finds the top page's edge but fewer than 2 stack lines, print `lines["right"]` and debug (systematic-debugging): the strip and thresholds are toppage's, which finds them (`tests/test_line_candidates.py`).

- [ ] **Step 7: Commit**

```bash
git add hasami/core/geometry.py hasami/core/sidelines.py tests/test_sidelines.py
git commit -m "side_lines: candidate lines around a final box (read-only, booklet snapping)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: spread boxes, line geometry, flag signals

**Files:**
- Modify: `hasami/core/sidelines.py`
- Test: `tests/test_sidelines.py`

**Interfaces:**
- Consumes: Task 1's `SideLine`, `side_lines`, `_merge`, `KEY`, `SIDE_NAMES`.
- Produces (all in `hasami.core.sidelines`): `normal_hit(p, dd, m, nrm) -> float | None`; `to_image(line: SideLine, box: CropBox, name: str, px_mm: float, inset_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray]` (point, unit direction, image px); `from_image(p, dd, box, name, px_mm) -> tuple[float, float] | None` (offset_mm, slope); `box_lines(image, box, dpi) -> dict[str, list[SideLine]]` (handles `meta["spread_pages"]`); `spine_names(box) -> set[str]`; `signals(lines: list[SideLine], margin_mm: float, spine: bool) -> dict[str, float]` with keys `near_strong`, `near_strong_at`, `own_support`, `no_crease`, `crowd`; `check_reason(sig: dict, rule) -> str | None`; `CHECK_RULE: tuple[tuple[str, float], ...] = ()`; constants `FAR_MM = 99.0`, `OWN_MM = 0.3`, `CREASE_MM = 0.5`, `CROWD_MM = 3.0`, `STRONG = 0.85`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sidelines.py`)

```python
def test_a_line_survives_the_trip_through_the_image():
    from hasami.core.sidelines import SideLine, from_image, to_image

    box = CropBox(400, 300, 300, 200, 3.0)
    for name in ("top", "right", "bottom", "left"):
        p, dd = to_image(SideLine(-1.7, 0.004, 0.9, "layer"), box, name, 7.87)
        off, slope = from_image(p, dd, box, name, 7.87)
        assert off == pytest.approx(-1.7, abs=1e-9) and slope == pytest.approx(0.004, abs=1e-9)
    p, dd = to_image(SideLine(0.5, 0.0, 0.9, "layer"), box, "right", 7.87, inset_mm=0.25)
    assert from_image(p, dd, box, "right", 7.87)[0] == pytest.approx(0.25)  # moved in by the inset


def test_normal_hit():
    from hasami.core.sidelines import normal_hit

    assert normal_hit((10.0, 0.0), (0.0, 1.0), (0.0, 5.0), (1.0, 0.0)) == pytest.approx(10.0)
    assert normal_hit((10.0, 0.0), (1.0, 0.0), (0.0, 5.0), (1.0, 0.0)) is None  # parallel


def test_a_spread_box_gets_its_pages_lines_on_its_own_sides():
    from hasami.core.sidelines import box_lines

    sc = synth_booklet(BookletOptions(fully_visible=True), seed=0)
    [spread] = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0, booklet_output="spread"), dpi=200)
    assert "spread_pages" in spread.meta
    lines = box_lines(sc.image, spread, 200)
    for name in ("top", "right", "bottom", "left"):
        assert any(abs(ln.offset_mm) < 0.3 for ln in lines[name]), name  # the paper edge under each side
    assert all(ln.kind != "crease" for side in lines.values() for ln in side)  # the spine is no spread side


def test_box_lines_of_a_page_are_its_side_lines():
    from hasami.core.sidelines import box_lines, side_lines

    sc = synth_booklet(BookletOptions(), seed=0)
    [page] = detect(sc.image, AutoCropSettings(mode="booklet"), dpi=200)
    assert box_lines(sc.image, page, 200) == side_lines(sc.image, page, 200)


def test_signals_are_measured_from_the_paper_edge_under_the_side():
    from hasami.core.sidelines import SideLine, signals

    lines = [SideLine(0.25, 0.0, 0.9, "layer"), SideLine(-1.35, 0.0, 0.95, "strong"), SideLine(2.25, 0.0, 0.7, "faint")]
    sig = signals(lines, margin_mm=0.25, spine=False)
    assert sig["own_support"] == 0.9
    assert sig["near_strong"] == pytest.approx(1.6) and sig["near_strong_at"] == pytest.approx(-1.6)
    assert sig["crowd"] == 3.0 and sig["no_crease"] == 0.0
    none = signals([], 0.25, spine=False)
    assert none["near_strong"] == 99.0 and none["own_support"] == 0.0 and none["crowd"] == 0.0


def test_a_spine_without_a_crease_line_at_it():
    from hasami.core.sidelines import SideLine, signals

    assert signals([SideLine(1.25, 0.0, 0.8, "crease")], 0.25, spine=True)["no_crease"] == 1.0
    assert signals([SideLine(0.45, 0.0, 0.8, "crease")], 0.25, spine=True)["no_crease"] == 0.0


def test_check_reason_names_the_first_condition_that_fires():
    from hasami.core.sidelines import SideLine, check_reason, signals

    sig = signals([SideLine(0.25, 0.0, 0.9, "layer"), SideLine(-1.35, 0.0, 0.95, "strong")], 0.25, spine=False)
    assert check_reason(sig, ()) is None
    assert check_reason(sig, (("near_strong", 2.0),)) == "another edge 1.6 mm further in"
    assert check_reason(sig, (("own_support", 0.85),)) is None
    assert check_reason(sig, (("own_support", 0.95), ("near_strong", 2.0))) == "the edge here is faint"
    assert check_reason(sig, (("crowd", 2),)) == "2 edges within 3 mm"
    spine = signals([], 0.25, spine=True)
    assert check_reason(spine, (("no_crease", 0.5),)) == "no clear crease at the fold"


def test_no_flag_rule_until_the_measurement_sets_one():
    from hasami.core import sidelines

    assert sidelines.CHECK_RULE == ()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_sidelines.py -q`
Expected: the 7 Task 1 tests pass; the 8 new ones FAIL with `ImportError: cannot import name 'from_image'` (etc.).

- [ ] **Step 3: Implement** (append to `hasami/core/sidelines.py`; add `import math` to the imports)

```python
FAR_MM = 99.0  # signals: no such line
OWN_MM = 0.3  # a line this close to the paper edge under the side is that edge
CREASE_MM = 0.5
CROWD_MM = 3.0
STRONG = 0.85  # a page edge runs along (nearly) the whole side

#: The "check this side" rule: an OR of (signal, threshold) conditions, fitted by the snap check in the
#: private data repo (spec Section 3).  Empty: no flags.
CHECK_RULE: tuple[tuple[str, float], ...] = ()


def normal_hit(p, dd, m, nrm) -> float | None:
    """Where the line through ``p`` along ``dd`` crosses the line from ``m`` along ``nrm``: the distance
    from ``m`` in ``nrm``'s units, or None when the two run parallel."""
    den = nrm[0] * dd[1] - nrm[1] * dd[0]
    if abs(den) < 1e-9:
        return None
    return float(((p[0] - m[0]) * dd[1] - (p[1] - m[1]) * dd[0]) / den)


def to_image(line: SideLine, box: CropBox, name: str, px_mm: float, inset_mm: float = 0.0):
    """A side's line as a point (at the side's middle) and a unit direction in the image (px);
    ``inset_mm`` moves it inwards."""
    a, e, d, n = _side(box, KEY[name])
    p = (a + e) / 2 + n * (line.offset_mm - inset_mm) * px_mm
    dd = d + n * line.slope
    return p, dd / np.linalg.norm(dd)


def from_image(p, dd, box: CropBox, name: str, px_mm: float) -> tuple[float, float] | None:
    """(offset_mm, slope) of an image line against side ``name`` of ``box``; None when it runs across it."""
    a, e, d, n = _side(box, KEY[name])
    along = float(np.dot(dd, d))
    t = normal_hit(p, dd, (a + e) / 2, n)
    if t is None or abs(along) < 1e-9:
        return None
    return t / px_mm, float(np.dot(dd, n)) / along


def box_lines(image: np.ndarray, box: CropBox, dpi: float) -> dict[str, list[SideLine]]:
    """``side_lines`` of a final box as ``detect`` returns it.  A spread box (``meta["spread_pages"]``)
    gets its pages' lines on the spread sides they lie along (outward normal within 10 deg, side's
    middle within 1 mm of the spread side); the pages' spines are no spread side and give none."""
    if "spread_pages" not in box.meta:
        return side_lines(image, box, dpi)
    from .booklet import _from_frame

    px_mm = dpi / MM_PER_INCH
    out: dict[str, list[SideLine]] = {name: [] for name in SIDE_NAMES}
    for i, corners in enumerate(box.meta["spread_pages"]):
        page = _from_frame(box, corners)
        page.meta = {"keep_sides": box.meta.get("spread_keep", [[], []])[i]}
        for pname, lines in side_lines(image, page, dpi).items():
            pa, pe, _pd, pn = _side(page, KEY[pname])
            for sname in SIDE_NAMES:
                sa, se, _sd, sn = _side(box, KEY[sname])
                if float(pn @ sn) < math.cos(math.radians(10)) or abs(float(((pa + pe) / 2 - (sa + se) / 2) @ sn)) > px_mm:
                    continue
                for ln in lines:
                    got = from_image(*to_image(ln, page, pname, px_mm), box, sname, px_mm)
                    if got is not None:
                        out[sname].append(SideLine(got[0], got[1], ln.support, ln.kind))
    return {name: _merge(lines) for name, lines in out.items()}


def spine_names(box: CropBox) -> set[str]:
    """The names of a box's spine sides (``meta["keep_sides"]``)."""
    keep = {tuple(k) for k in box.meta.get("keep_sides", ())}
    return {name for name, key in KEY.items() if key in keep}


def signals(lines: list[SideLine], margin_mm: float, spine: bool) -> dict[str, float]:
    """What a flag rule looks at on one side, from the paper edge under it (a box side lies
    ``margin_mm`` inside the edge auto-crop found): ``near_strong`` the distance to the nearest other
    line with support >= STRONG (FAR_MM: none) and ``near_strong_at`` its signed offset (+ = out);
    ``own_support`` the support of the line at the edge (0: none within OWN_MM); ``no_crease`` 1 for a
    spine without a valley line within CREASE_MM; ``crowd`` the lines within CROWD_MM."""
    rel = [(ln.offset_mm - margin_mm, ln) for ln in lines]
    strong = [x for x, ln in rel if ln.support >= STRONG and abs(x) > OWN_MM]
    at = min(strong, key=abs) if strong else 0.0
    return {"near_strong": abs(at) if strong else FAR_MM, "near_strong_at": at,
            "own_support": max((ln.support for x, ln in rel if abs(x) <= OWN_MM), default=0.0),
            "no_crease": float(spine and not any(abs(x) <= CREASE_MM for x, ln in rel if ln.kind == "crease")),
            "crowd": float(sum(abs(x) <= CROWD_MM for x, _ in rel))}


_FIRES = {"near_strong": lambda v, t: v <= t, "own_support": lambda v, t: v < t,
          "no_crease": lambda v, t: v >= t, "crowd": lambda v, t: v >= t}


def check_reason(sig: dict, rule) -> str | None:
    """Why a side should be checked under ``rule`` (its first condition that fires), or None."""
    for name, t in rule:
        if not _FIRES[name](sig[name], t):
            continue
        if name == "near_strong":
            at = sig["near_strong_at"]
            return f"another edge {abs(at):.1f} mm further {'out' if at > 0 else 'in'}"
        if name == "own_support":
            return "the edge here is faint"
        if name == "no_crease":
            return "no clear crease at the fold"
        return f"{int(sig['crowd'])} edges within 3 mm"
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_sidelines.py -q`
Expected: 15 passed.

- [ ] **Step 5: Time it** (budget <= 100 ms per page at 200 dpi; record the number in the ledger for the PR)

```bash
$PY -c "import time; from hasami.demo import synth_booklet, BookletOptions; from hasami.core.autocrop import detect, AutoCropSettings; from hasami.core.sidelines import box_lines; sc = synth_booklet(BookletOptions(fully_visible=True), seed=0); bs = detect(sc.image, AutoCropSettings(mode='booklet'), dpi=200); t = time.perf_counter(); [box_lines(sc.image, b, 200) for _ in range(5) for b in bs]; print(round((time.perf_counter() - t) * 1000 / 5 / len(bs), 1), 'ms per page')"
```

Expected: a number <= 100. Over it: ledger a ruling with the number (the spec's budget is a target, not a gate); profile `_layer_lines` only if it is over 300 ms.

- [ ] **Step 6: Commit**

```bash
git add hasami/core/sidelines.py tests/test_sidelines.py
git commit -m "sidelines: spread boxes, line geometry, flag signals (no rule yet)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: the snap and flag check (hasami-data, private) - GATE

Everything in this task lives in `F:/HasamiData` (private repo). Nothing from it is copied into hasami-scan except the fitted rule's conditions (Task 7) and pooled numbers in the hasami-scan PR text.

**Files (all in `F:/HasamiData`):**
- Create: `benchmarks/learned/snap/snapcheck.py`
- Create: `benchmarks/learned/snap/test_snapcheck.py`
- Create: `benchmarks/learned/snapcheck_<YYYY-MM-DD>.md` (the report)

**Interfaces:**
- Consumes: hasami-scan branch `feat/edge-snapping` via `PYTHONPATH=F:/HasamiDev`: `sidelines.box_lines`, `signals`, `check_reason`, `spine_names`; `hasami.cases.side_shifts`; `hasami.core.geometry.SIDE_NAMES`, `match_boxes`, `CropBox`; `hasami.core.booklet.spread_box`; `benchmarks/real_booklets.py`'s `load_scan`.
- Produces: the gate verdicts and, if the flag gate passes, the rule (a tuple of `(signal, threshold)`) for Task 7.

- [ ] **Step 1: Branch** (hasami-data)

```bash
cd F:/HasamiData && git checkout main && git pull -q && git checkout -b feat/snap-check
```

- [ ] **Step 2: Write the failing tests** (`benchmarks/learned/snap/test_snapcheck.py`)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapcheck import MAX_RATE, all_rules, best_rule, coverage, lobo, score  # noqa: E402


def _row(book, err, covered=False, **sig):
    base = {"near_strong": 99.0, "near_strong_at": 0.0, "own_support": 1.0, "no_crease": 0.0, "crowd": 1.0}
    base.update(sig)
    return {"book": book, "err_mm": err, "covered": covered, "n_lines": 3, "sig": base}


def test_coverage_counts_wrong_sides_with_the_truth_among_the_lines():
    c = coverage([_row("a", 2.0, covered=True), _row("a", -0.8), _row("a", 0.1, covered=True)])
    assert c["wrong"] == 2 and c["covered"] == 1 and c["share"] == 0.5 and c["median_lines"] == 3.0


def test_best_rule_keeps_the_flag_rate_under_the_limit():
    rows = [_row("a", 2.0, near_strong=1.0)] + [_row("a", 0.0, near_strong=2.5) for _ in range(9)]
    rule = best_rule(rows, all_rules())
    s = score(rows, rule)
    assert s["recall"] == 1.0 and s["rate"] <= MAX_RATE and len(rule) == 1


def test_lobo_scores_each_book_with_a_rule_chosen_without_it():
    rows = ([_row("a", 2.0, near_strong=1.0)] + [_row("a", 0.0, near_strong=2.5) for _ in range(9)]
            + [_row("b", 2.0, crowd=8.0)] + [_row("b", 0.0) for _ in range(9)])
    res = lobo(rows, all_rules())
    assert res["per_book"]["a"]["recall"] == 0.0 and res["per_book"]["b"]["recall"] == 0.0  # no leak
    assert res["recall"] == 0.0 and res["rate"] <= MAX_RATE


def test_rules_are_singles_and_pairs_of_different_signals():
    rules = all_rules()
    assert all(1 <= len(r) <= 2 for r in rules)
    assert all(r[0][0] != r[1][0] for r in rules if len(r) == 2)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cd F:/HasamiData && PYTHONPATH=F:/HasamiDev $PY -m pytest benchmarks/learned/snap -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'snapcheck'`.

- [ ] **Step 4: Write `benchmarks/learned/snap/snapcheck.py`**

```python
"""Snap and flag check (hasami-scan spec docs/specs/2026-10-15-edge-snapping-design.md, Section 3).

On the bare 200 dpi preview box of every booklet record of the tuned V600 (V600 Scans pages, Thick books,
KLK6, New data): are the right lines among ``sidelines.box_lines``' candidates (snap gate), and does a flag
rule pass on held-out books (flag gate)?

    cd F:/HasamiData && PYTHONPATH=F:/HasamiDev python benchmarks/learned/snap/snapcheck.py \
        "F:/ScannerTraining/Training booklets" --jobs 2 --out F:/HasamiData/learned/snap_run1
"""
import argparse
import datetime
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH))
from real_booklets import load_scan  # noqa: E402

from hasami.cases import side_shifts  # noqa: E402
from hasami.core import profiles  # noqa: E402
from hasami.core.autocrop import AutoCropSettings, detect  # noqa: E402
from hasami.core.booklet import spread_box  # noqa: E402
from hasami.core.geometry import SIDE_NAMES, CropBox, match_boxes  # noqa: E402
from hasami.core.sidelines import box_lines, check_reason, signals, spine_names  # noqa: E402

GROUPS = ("V600 Scans/", "Thick books/", "Kill la kill Vol.6 artboard book/", "New data/")
PREVIEW_DPI = 200.0
PROFILE = "epson-v600"
WRONG_MM = 0.5  # a side this far off truth is wrong
COVER_MM = 0.3  # a candidate this close to the truth would fix it
MAX_RATE = 1 / 8  # flag gate: at most 1 side in 8 flagged
MIN_RECALL = 0.5  # and at least half of the wrong sides
MIN_COVER = 0.6  # snap gate
CONDITIONS = {"near_strong": (0.5, 0.75, 1.0, 1.5, 2.0, 3.0), "own_support": (0.65, 0.75, 0.85, 0.95),
              "no_crease": (0.5,), "crowd": (3, 4, 5, 6, 8)}


def records(truth_path: str) -> dict:
    d = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    return {k: r for k, r in d.items() if r.get("kind") == "pages" and k.startswith(GROUPS)}


def side_rows(args) -> list[dict]:
    """One row per side of each final preview box matched to a truth box."""
    key, rec, root, book = args
    img, rec = load_scan(key, rec, root, 0)
    f = min(1.0, PREVIEW_DPI / rec["dpi"])
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    dpi = rec["dpi"] * f
    st = AutoCropSettings(mode="booklet", booklet_output=rec.get("output", "pages"), profile=PROFILE)
    found = detect(small, st, dpi=dpi)
    truth = [CropBox(*p).scaled(f) for p in rec["pages"]]
    if st.booklet_output == "spread" and len(truth) == 2:
        truth = [spread_box(truth)]  # a whole spread saved as one image
    rows = []
    with profiles.use(PROFILE):
        for i, j, _iou in match_boxes(found, truth):
            box = found[i]
            lines, spines = box_lines(small, box, dpi), spine_names(box)
            shifts = side_shifts(box, truth[j])  # truth side, mm outside the box side
            for name in SIDE_NAMES:
                t, ls = shifts[name], lines[name]
                rows.append({"key": key, "book": book, "page": int(i), "side": name, "spine": name in spines,
                             "err_mm": round(st.margin_mm - t, 3),  # + = the box (without margin) too far out
                             "covered": any(abs(ln.offset_mm - t) <= COVER_MM for ln in ls),
                             "n_lines": len(ls), "sig": signals(ls, st.margin_mm, name in spines)})
    return rows


def wrong(row) -> bool:
    return abs(row["err_mm"]) > WRONG_MM


def flagged(row, rule) -> bool:
    return check_reason(row["sig"], rule) is not None


def coverage(rows: list[dict]) -> dict:
    w = [r for r in rows if wrong(r)]
    covered = sum(bool(r["covered"]) for r in w)
    return {"wrong": len(w), "covered": covered, "share": covered / len(w) if w else 1.0,
            "median_lines": float(np.median([r["n_lines"] for r in rows])) if rows else 0.0}


def all_rules() -> list[tuple]:
    singles = [((name, t),) for name, ts in CONDITIONS.items() for t in ts]
    return singles + [a + b for a, b in itertools.combinations(singles, 2) if a[0][0] != b[0][0]]


def score(rows: list[dict], rule) -> dict:
    f = [flagged(r, rule or ()) for r in rows]
    w = [wrong(r) for r in rows]
    hits = sum(a and b for a, b in zip(f, w))
    return {"n": len(rows), "flagged": sum(f), "wrong": sum(w), "hits": hits,
            "rate": sum(f) / len(rows) if rows else 0.0, "recall": hits / sum(w) if any(w) else 0.0}


def best_rule(rows: list[dict], rules: list[tuple]):
    """The rule with the highest recall among those flagging <= MAX_RATE of ``rows`` (ties: lower rate,
    then fewer conditions); None when none does."""
    ok = [(score(rows, r), r) for r in rules]
    ok = [(s, r) for s, r in ok if s["rate"] <= MAX_RATE]
    if not ok:
        return None
    return max(ok, key=lambda t: (t[0]["recall"], -t[0]["rate"], -len(t[1])))[1]


def lobo(rows: list[dict], rules: list[tuple]) -> dict:
    """Leave one book out: each book scored by the rule chosen on the others; pooled held-out numbers."""
    per_book, f, w = {}, [], []
    for b in sorted({r["book"] for r in rows}):
        rule = best_rule([r for r in rows if r["book"] != b], rules)
        test = [r for r in rows if r["book"] == b]
        per_book[b] = {"rule": rule, **score(test, rule)}
        f += [flagged(r, rule or ()) for r in test]
        w += [wrong(r) for r in test]
    hits = sum(a and b for a, b in zip(f, w))
    return {"per_book": per_book, "rate": sum(f) / len(f) if f else 0.0,
            "recall": hits / sum(w) if any(w) else 0.0, "flagged": sum(f), "wrong": sum(w), "hits": hits}


def report(rows: list[dict], out: Path) -> str:
    cov, rules = coverage(rows), all_rules()
    held = lobo(rows, rules)
    final = best_rule(rows, rules)
    snap_ok = cov["share"] >= MIN_COVER
    flag_ok = held["rate"] <= MAX_RATE and held["recall"] >= MIN_RECALL
    lines = [f"# Snap and flag check ({datetime.date.today().isoformat()})", "",
             "hasami-scan spec `docs/specs/2026-10-15-edge-snapping-design.md`; script "
             "`benchmarks/learned/snap/snapcheck.py`; bare 200 dpi preview box, profile epson-v600, turn 0.", "",
             f"Sides: {len(rows)}; wrong (> {WRONG_MM} mm): {cov['wrong']}.", "",
             "## Snap check", "",
             f"Wrong sides with the truth within {COVER_MM} mm of a candidate: {cov['covered']} / {cov['wrong']} "
             f"= {cov['share']:.0%} (gate >= {MIN_COVER:.0%}: {'PASS' if snap_ok else 'FAIL'}); "
             f"median lines per side {cov['median_lines']:.0f}.", "",
             "| book | sides | wrong | covered | median lines |", "|---|---|---|---|---|"]
    for b in sorted({r["book"] for r in rows}):
        c = coverage([r for r in rows if r["book"] == b])
        lines.append(f"| {b} | {sum(r['book'] == b for r in rows)} | {c['wrong']} | {c['covered']} | "
                     f"{c['median_lines']:.0f} |")
    lines += ["", "## Flag check (leave one book out)", "",
              f"Pooled held-out: flagged {held['flagged']} / {len(rows)} = {held['rate']:.1%} (gate <= "
              f"{MAX_RATE:.1%}), wrong sides flagged {held['hits']} / {held['wrong']} = {held['recall']:.0%} "
              f"(gate >= {MIN_RECALL:.0%}): {'PASS' if flag_ok else 'FAIL'}.", "",
              "| held-out book | rule (chosen on the others) | flagged | wrong | caught |", "|---|---|---|---|---|"]
    for b, s in held["per_book"].items():
        lines.append(f"| {b} | {s['rule']} | {s['flagged']} / {s['n']} | {s['wrong']} | {s['hits']} |")
    fs = score(rows, final)
    lines += ["", f"Rule fitted on all books: `{final}` (in-sample: flagged {fs['rate']:.1%}, recall "
              f"{fs['recall']:.0%}).", "",
              f"Verdict: snapping {'goes ahead' if snap_ok else 'STOPS (owner decides)'}; flags "
              f"{'ship with the fitted rule' if (snap_ok and flag_ok) else 'are dropped'}."]
    text = "\n".join(lines) + "\n"
    (out / "verdict.json").write_text(json.dumps({"snap_ok": snap_ok, "flag_ok": flag_ok, "rule": final,
                                                  "coverage": cov, "held_out": {k: v for k, v in held.items()
                                                                                if k != "per_book"}}, indent=1))
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--truth", default=str(BENCH / "real_booklets_proposals.json"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    books = json.loads((BENCH / "learned" / "books.json").read_text(encoding="utf-8"))
    recs = records(a.truth)
    jobs = [(k, r, a.root, books.get(k, k.split("/")[0])) for k, r in sorted(recs.items())]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [row for part in ex.map(side_rows, jobs) for row in part]
    (out / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    text = report(rows, out)
    (BENCH / "learned" / f"snapcheck_{datetime.date.today().isoformat()}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd F:/HasamiData && PYTHONPATH=F:/HasamiDev $PY -m pytest benchmarks/learned/snap -q`
Expected: 4 passed.

- [ ] **Step 6: Run the check** (background; ~116 records at 200 dpi, `--jobs 2`)

Run: `cd F:/HasamiData && PYTHONPATH=F:/HasamiDev $PY benchmarks/learned/snap/snapcheck.py "F:/ScannerTraining/Training booklets" --jobs 2 --out F:/HasamiData/learned/snap_run1`
Expected: the report printed and written to `benchmarks/learned/snapcheck_<date>.md`; `rows.json` and `verdict.json` in the out folder; 116 records' sides (about 116 x 1.5 pages x 4 sides). If a record fails to load, the run stops: fix the cause (paths: `load_scan` reads `HASAMI_BENCH_EXTRA`, default `F:/JPG;E:/`), do not skip records silently.

- [ ] **Step 7: Read the verdict and act**
  - **Snap gate FAIL** (< 60%): STOP the plan. Commit the report (Step 8), and tell the owner the numbers (coverage per book, median lines) with what the uncovered wrong sides look like (offsets of the nearest candidates vs the truth from `rows.json`). Wait for the owner's decision.
  - **Snap PASS, flag FAIL**: ledger `Task 3: flags dropped (held-out rate X, recall Y)`; Task 7 is skipped.
  - **Both PASS**: ledger the fitted rule exactly as printed (e.g. `(("near_strong", 1.5), ("no_crease", 0.5))`) for Task 7.

- [ ] **Step 8: Commit, push, PR** (hasami-data; the owner merges)

```bash
cd F:/HasamiData && git add benchmarks/learned/snap benchmarks/learned/snapcheck_*.md
git commit -m "Snap and flag check for edge snapping (hasami-scan 0.5.19 spec)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/snap-check
gh pr create --title "Snap and flag check (edge snapping)" --body "Measures whether side_lines' candidates cover the wrong bare-preview sides and whether a flag rule passes on held-out books. Report: benchmarks/learned/snapcheck_<date>.md. Verdict: <snap PASS/FAIL, flags PASS/FAIL, rule>.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
cd F:/HasamiDev
```

---

### Task 4: pinned sides survive the scan-time refine

**Files:**
- Modify: `hasami/core/geometry.py` (add `side_frames`, `pin_sides`)
- Modify: `hasami/core/autocrop.py:1174-1218` (`refine` docstring, `_refine` loop; extract `_refine_box`)
- Test: `tests/test_pinning.py`

**Interfaces:**
- Consumes: `SIDE_NAMES` (Task 1); `sidelines.spine_names` (Task 2, tests only).
- Produces: `geometry.side_frames(box) -> dict[str, tuple[np.ndarray, np.ndarray]]` (middle, outward unit normal); `geometry.pin_sides(given: CropBox, refined: CropBox, pinned: Iterable[str]) -> CropBox`; `refine` behaviour: a manual box with `meta["auto_id"]` and 1-3 `meta["pinned_sides"]` is refined, then pinned.

- [ ] **Step 1: Write the failing tests** (`tests/test_pinning.py`)

```python
"""Sides moved by hand stay where they are at scan time; the others are still refined (generated scans)."""
from dataclasses import replace

import numpy as np
import pytest

from hasami.core.autocrop import AutoCropSettings, detect, refine
from hasami.core.geometry import SIDE_NAMES, CropBox
from hasami.demo import BookletOptions, synth_booklet


def _pos(box, ref, name):
    """Where ``box``'s side ``name`` lies along ``ref``'s normal, from ``ref``'s side (px, + = out)."""
    from hasami.core.geometry import side_frames

    (m, _), (mr, nr) = side_frames(box)[name], side_frames(ref)[name]
    return float((m - mr) @ nr)


def _moved(box, name, d):
    """``box`` with side ``name`` moved ``d`` px outwards (a hand edit)."""
    ux, uy = box.axes
    axis = {"top": -uy, "bottom": uy, "left": -ux, "right": ux}[name]
    wide = name in ("left", "right")
    c = np.array([box.cx, box.cy]) + axis * d / 2
    return replace(box, cx=c[0], cy=c[1], w=box.w + d * wide, h=box.h + d * (not wide), meta=dict(box.meta))


def _edited(box, name, d, pinned=None):
    b = _moved(box, name, d)
    return replace(b, manual=True, meta={**box.meta, "auto_id": "a1", "pinned_sides": pinned or [name]})


def test_pinned_sides_stay_and_the_others_follow_the_refined_box():
    from hasami.core.geometry import pin_sides

    given = CropBox(100, 80, 60, 40, 0.0, manual=True, meta={"auto_id": "a1"})
    refined = CropBox(101, 79, 64, 40, 0.0, meta={"top_page": {}})
    out = pin_sides(given, refined, ["left", "top"])
    x0, y0, x1, y1 = out.bounding_rect()
    assert (x0, y0) == (70.0, 60.0)  # pinned: exactly where they were
    assert x1 == pytest.approx(133.0) and y1 == pytest.approx(99.0)  # refined
    assert out.angle == 0.0 and out.manual and out.meta == {"auto_id": "a1"}


def test_a_turned_box_keeps_its_angle_and_its_pinned_side():
    from hasami.core.geometry import pin_sides, side_frames

    given = CropBox(200, 150, 80, 50, 2.0)
    refined = CropBox(201, 150.5, 82, 51, 2.3)
    out = pin_sides(given, refined, ["left"])
    assert out.angle == pytest.approx(2.0)
    assert _pos(out, given, "left") == pytest.approx(0.0, abs=1e-9)
    for name in ("top", "right", "bottom"):
        m_r = side_frames(refined)[name][0]
        m_o, n_o = side_frames(out)[name]
        assert float((m_r - m_o) @ n_o) == pytest.approx(0.0, abs=1e-9), name


def test_all_four_pinned_is_the_given_box():
    from hasami.core.geometry import pin_sides

    given = CropBox(200, 150, 80, 50, 2.0)
    out = pin_sides(given, CropBox(203, 149, 90, 44, -1.0), SIDE_NAMES)
    assert (out.cx, out.cy, out.w, out.h, out.angle) == pytest.approx((200, 150, 80, 50, 2.0))


def _card(dpi=150):
    mm = dpi / 25.4
    img = np.full((int(150 * mm), int(200 * mm), 3), 235, np.uint8)
    x0, y0, x1, y1 = int(50 * mm), int(40 * mm), int(140 * mm), int(100 * mm)
    img[y0:y1, x0:x1] = (90, 60, 140)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img, CropBox.from_xywh(x0 - 0.5, y0 - 0.5, x1 - x0, y1 - y0), mm


def test_a_pinned_side_stays_while_the_others_are_refined():
    img, card, mm = _card()
    st = AutoCropSettings(mode="multi", margin_mm=0)
    auto = card.translated(2.0, 1.5)  # auto-crop's box, a little off
    [ref] = refine(img, [auto], st, dpi=150)
    edited = _edited(auto, "left", 15.0)  # the left side dragged 15 px out
    [out] = refine(img, [edited], st, dpi=150)
    assert _pos(out, edited, "left") == pytest.approx(0.0, abs=1e-6)
    for name in ("top", "right", "bottom"):
        assert abs(_pos(out, ref, name)) <= 0.05 * mm, name
    assert out.angle == edited.angle and out.manual and out.meta["pinned_sides"] == ["left"]


def test_a_pinned_booklet_side_stays_while_the_others_are_refined():
    from hasami.core.sidelines import spine_names

    sc = synth_booklet(BookletOptions(), seed=0)
    st = AutoCropSettings(mode="booklet", margin_mm=0)
    [page] = detect(sc.image, st, dpi=200)
    name = next(n for n in SIDE_NAMES if n not in spine_names(page))
    mm = 200 / 25.4
    [ref] = refine(sc.image, [page], st, dpi=200)
    edited = _edited(page, name, 1.0 * mm)
    [out] = refine(sc.image, [edited], st, dpi=200)
    assert _pos(out, edited, name) == pytest.approx(0.0, abs=1e-6)
    for other in SIDE_NAMES:
        if other != name:
            assert abs(_pos(out, ref, other)) <= 0.05 * mm, other
    assert out.angle == edited.angle


def test_a_pinned_spread_side_stays():
    sc = synth_booklet(BookletOptions(fully_visible=True), seed=1)
    st = AutoCropSettings(mode="booklet", margin_mm=0, booklet_output="spread")
    [spread] = detect(sc.image, st, dpi=200)
    mm = 200 / 25.4
    [ref] = refine(sc.image, [spread], st, dpi=200)
    edited = _edited(spread, "left", 1.0 * mm)
    [out] = refine(sc.image, [edited], st, dpi=200)
    assert "spread_pages" in out.meta
    assert _pos(out, edited, "left") == pytest.approx(0.0, abs=1e-6)
    assert abs(_pos(out, ref, "right")) <= 0.05 * mm


def test_drawn_unpinned_and_fully_pinned_boxes_are_untouched():
    img, card, mm = _card()
    st = AutoCropSettings(mode="multi", margin_mm=0)
    off = card.translated(2.0, 1.5)
    drawn = replace(off, manual=True, meta={"drawn": True, "pinned_sides": ["left"]})
    unpinned = replace(off, manual=True, meta={"auto_id": "a1"})  # edited before pins existed
    whole = replace(off, manual=True, meta={"auto_id": "a2", "pinned_sides": list(SIDE_NAMES)})
    out = refine(img, [drawn, unpinned, whole], st, dpi=150)
    assert [(b.cx, b.cy, b.w, b.h, b.angle) for b in out] == [(off.cx, off.cy, off.w, off.h, off.angle)] * 3
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_pinning.py -q`
Expected: FAIL: `ImportError: cannot import name 'side_frames'` / `'pin_sides'`; the refine tests fail on the pinned side (the manual box comes back unrefined, so its other sides are off by the 1.5-2 px offset) - except `test_drawn_unpinned_and_fully_pinned_boxes_are_untouched`, which already passes (today's behaviour; it guards it).

- [ ] **Step 3: Add `side_frames` and `pin_sides` to `hasami/core/geometry.py`** (after `class CropBox`; `Iterable` is already imported for `reading_order`)

```python
def side_frames(box: CropBox) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each side's middle and outward unit normal, by name (the box's own frame)."""
    c = box.corners()
    ux, uy = box.axes
    return {"top": ((c[0] + c[1]) / 2, -uy), "right": ((c[1] + c[2]) / 2, ux),
            "bottom": ((c[2] + c[3]) / 2, uy), "left": ((c[3] + c[0]) / 2, -ux)}


def pin_sides(given: CropBox, refined: CropBox, pinned: Iterable[str]) -> CropBox:
    """``given`` with its angle kept, each pinned side where it is and each other side moved parallel to
    where ``refined`` put it (the refined side's middle projected on ``given``'s normal).  Confidence from
    ``refined``; ``manual`` and ``meta`` from ``given`` (a hand-edited box keeps its ids and pins)."""
    pinned = set(pinned)
    mine, theirs = side_frames(given), side_frames(refined)
    move = {}
    for name, (m, n) in mine.items():
        if name in pinned:
            move[name] = 0.0
            continue
        rm = max(theirs.values(), key=lambda t: float(t[1] @ n))[0]  # the refined side facing the same way
        move[name] = float((rm - m) @ n)
    ux, uy = given.axes
    c = (np.array([given.cx, given.cy]) + ux * (move["right"] - move["left"]) / 2
         + uy * (move["bottom"] - move["top"]) / 2)
    return CropBox(c[0], c[1], given.w + move["left"] + move["right"], given.h + move["top"] + move["bottom"],
                   given.angle, refined.confidence, manual=given.manual, meta=dict(given.meta))
```

- [ ] **Step 4: Change `autocrop._refine`** (replace the loop at `hasami/core/autocrop.py:1198-1218`; add `pin_sides` to the `from .geometry import ...` line)

```python
    for b in boxes:
        pinned = set(b.meta.get("pinned_sides", ()))
        if b.manual and not (b.meta.get("auto_id") and 0 < len(pinned) < 4):
            out.append(b)  # drawn, edited as a whole, or edited before pins existed
            continue
        nb = _refine_box(image, L, C, b, st, dpi, r, margin_px, W, H)
        out.append(pin_sides(b, nb, pinned).clamp_to(W, H) if b.manual else nb)
    return out


def _refine_box(image: np.ndarray, L: np.ndarray, C: np.ndarray | None, b: CropBox, st: AutoCropSettings,
                dpi: float, r: float, margin_px: float, W: int, H: int) -> CropBox:
    """One auto-crop box re-refined (see ``refine``)."""
    if st.mode == "booklet" and ("top_page" in b.meta or "spread_pages" in b.meta):
        # Booklet pages end at the top page's edge: find it again (the plain edge snap
        # would jump to a page-stack line next to it).
        from .booklet import refine_booklet

        return refine_booklet(image, b, st, dpi)
    # Undo the margin so that refinement starts on the edge itself.
    start = b.inset(-margin_px)
    keep = {tuple(k) for k in b.meta.get("keep_sides", [])}
    nb = _refine_edges(L, start, r, shadow_px=1.0 * dpi / MM_PER_INCH, skip=keep or None, C=C)
    if not st.deskew:
        x0, y0, x1, y1 = nb.bounding_rect()
        nb = replace(nb, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0, angle=0.0)
    elif abs(nb.angle) < st.snap_angle:
        nb = replace(nb, angle=0.0)
    return nb.inset(margin_px).clamp_to(W, H)
```

And the `refine` docstring's last sentence becomes:

```python
    re-refined, the pinned ones and the angle stay (``geometry.pin_sides``), and sides listed in a
```

i.e. the docstring reads: "... Manual boxes are returned untouched, except auto-crop boxes with one to three sides moved by hand (``meta["pinned_sides"]``): their other sides are re-refined, the pinned ones and the angle stay (``geometry.pin_sides``); sides listed in a box's ``meta["keep_sides"]`` (a booklet page's spine) are not moved."

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_pinning.py tests/test_booklet.py tests/test_autocrop.py -q`
Expected: all pass (test_booklet's known xfails stay xfail). If a booklet "other side" differs from the unedited refine by more than 0.05 mm, it is the strip length changing with the moved side: print the difference, ledger a ruling with the number if <= 0.1 mm (spec: equal "within 0.05 mm" was a design target), else debug.

- [ ] **Step 6: Commit**

```bash
git add hasami/core/geometry.py hasami/core/autocrop.py tests/test_pinning.py
git commit -m "Refine keeps the sides moved by hand (pinned) and still refines the others

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: the Session keeps snap lines; crop cases record pins and snaps

**Files:**
- Modify: `hasami/core/sidelines.py` (add `SnapLine`)
- Modify: `hasami/workflow.py` (`Session.__init__`, `run_preview`, `detect_on_preview`, new `_snap_lines`, `snap_lines_px`)
- Modify: `hasami/cases.py` (`write_case`: optional keys)
- Test: `tests/test_snapping.py`, `tests/test_cases.py`

**Interfaces:**
- Consumes: `sidelines.box_lines`, `to_image` (Task 2); `profiles.use`.
- Produces: `sidelines.SnapLine(p: tuple[float, float], d: tuple[float, float], support: float, kind: str)` (frozen); `Session.snap_lines: dict[str, dict[str, list[SnapLine]]]` (bed mm, moved in by `crop.margin_mm`); `Session.snap_lines_px() -> dict[str, dict[str, list[SnapLine]]]` (preview px); case.json `check_sides` / `pinned_sides` / `snapped_sides` keys.

- [ ] **Step 1: Write the failing tests** (`tests/test_snapping.py`)

```python
"""Snap lines in the Session (demo scanner, generated bed)."""
from dataclasses import replace

import numpy as np
import pytest

from hasami.core.geometry import SIDE_NAMES, side_frames


def _session(tmp_path, mode="booklet"):
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.crop.mode = mode
    st.output.folder = str(tmp_path / "scans")
    return Session(st)


def test_snap_lines_are_kept_per_auto_box_in_booklet_mode(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    assert s.boxes_mm and set(s.snap_lines) == {b.meta["auto_id"] for b in s.boxes_mm}
    for sides in s.snap_lines.values():
        assert set(sides) == set(SIDE_NAMES)
    assert sum(len(ls) for sides in s.snap_lines.values() for ls in sides.values()) > 0


def test_the_line_under_a_side_is_where_the_side_is(tmp_path):
    from hasami.core.sidelines import normal_hit

    s = _session(tmp_path)
    s.run_preview()
    b = s.boxes_mm[0]
    near = 0
    for name, (m, n) in side_frames(b).items():
        hits = [normal_hit(np.array(ln.p), np.array(ln.d), m, n) for ln in s.snap_lines[b.meta["auto_id"]][name]]
        near += any(h is not None and abs(h) <= 0.3 for h in hits)  # mm: the margin is already applied
    assert near >= 2


def test_no_snap_lines_outside_booklet_mode(tmp_path):
    s = _session(tmp_path, "multi")
    s.run_preview()
    assert s.snap_lines == {}


def test_a_new_preview_drops_the_old_lines(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    old = set(s.snap_lines)
    s.run_preview()
    assert s.snap_lines and not old & set(s.snap_lines)


def test_redetect_keeps_the_lines_of_edited_boxes(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    b = s.boxes_mm[0]
    aid = b.meta["auto_id"]
    s.boxes_mm[0] = replace(b, manual=True, meta={**b.meta, "pinned_sides": ["top"]})
    s.detect_on_preview()
    assert s.snap_lines[aid] and set(s.snap_lines) == {x.meta["auto_id"] for x in s.boxes_mm}


def test_snap_lines_px_are_in_the_preview_frame(tmp_path):
    s = _session(tmp_path)
    s.run_preview()
    p = s.preview
    f = p.dpi / 25.4
    px = s.snap_lines_px()
    for aid, sides in s.snap_lines.items():
        for name, ls in sides.items():
            for a, b in zip(ls, px[aid][name]):
                assert b.p == pytest.approx(((a.p[0] - p.area_mm[0]) * f - 0.5, (a.p[1] - p.area_mm[1]) * f - 0.5))
                assert b.d == a.d and b.support == a.support and b.kind == a.kind
```

And append to `tests/test_cases.py`:

```python
def test_a_case_records_flags_pins_and_snaps(tmp_path):
    from hasami import cases

    d = _data()
    d.auto[0].meta["check_sides"] = {"bottom": "another edge 1.6 mm further in"}
    d.final[0].meta.update(pinned_sides=["bottom"], snapped_sides=["bottom"])
    doc = cases.read_case(cases.write_case(tmp_path, d))
    assert doc["auto"][0]["check_sides"] == {"bottom": "another edge 1.6 mm further in"}
    assert doc["final"][0]["pinned_sides"] == ["bottom"] and doc["final"][0]["snapped_sides"] == ["bottom"]
    plain = cases.read_case(cases.write_case(tmp_path, _data()))
    assert "check_sides" not in plain["auto"][0] and "pinned_sides" not in plain["final"][0]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_snapping.py tests/test_cases.py -q`
Expected: the snapping tests FAIL with `AttributeError: 'Session' object has no attribute 'snap_lines'` (`test_no_snap_lines_outside_booklet_mode` too); `test_a_case_records_flags_pins_and_snaps` FAILS with `KeyError: 'check_sides'`; the other case tests pass.

- [ ] **Step 3: Add `SnapLine` to `hasami/core/sidelines.py`** (after `SideLine`)

```python
@dataclass(frozen=True)
class SnapLine:
    """A candidate line in absolute coordinates (bed mm in the Session, preview px in the GUI), already
    moved in by the crop margin: a side snapped to it lands where auto-crop would have put it."""

    p: tuple[float, float]  # a point on the line
    d: tuple[float, float]  # unit direction
    support: float
    kind: str
```

- [ ] **Step 4: Session changes** (`hasami/workflow.py`)

In the imports: `from .core import autocrop, profiles, sidelines` (replacing `from .core import autocrop`).

In `Session.__init__`, after `self._auto_seq = 0`:

```python
        #: Booklet mode: the candidate lines around each auto box (by auto id, then side name; bed mm),
        #: what a dragged side snaps to.
        self.snap_lines: dict[str, dict[str, list[sidelines.SnapLine]]] = {}
```

In `run_preview`, before `self.detect_on_preview()`:

```python
        self.snap_lines = {}  # a new page: none of the old lines apply
```

In `detect_on_preview`, after the `for b in found_mm:` id loop and the `kept = ...` line, before `self.auto_mm = ...`:

```python
        self.snap_lines = {k: v for k, v in self.snap_lines.items() if k in kept}
        if self.settings.crop.mode == "booklet" and found:
            try:
                with profiles.use(self.profile()):
                    for b_px, b_mm in zip(found, found_mm):
                        self.snap_lines[b_mm.meta["auto_id"]] = self._snap_lines(b_px, sidelines.box_lines(p.image, b_px, p.dpi))
            except Exception:  # snapping is a help: never let it break the preview
                pass
```

New methods (after `detect_on_preview`):

```python
    def _snap_lines(self, box_px: CropBox, lines: dict) -> dict[str, list[sidelines.SnapLine]]:
        """A box's candidate lines (preview px, relative to its sides) as absolute bed-mm lines, moved in
        by the crop margin."""
        p = self.preview
        px_mm, f = p.dpi / MM_PER_INCH, MM_PER_INCH / p.dpi
        out = {}
        for name, ls in lines.items():
            out[name] = []
            for ln in ls:
                q, dd = sidelines.to_image(ln, box_px, name, px_mm, inset_mm=self.settings.crop.margin_mm)
                out[name].append(sidelines.SnapLine(((q[0] + 0.5) * f + p.area_mm[0], (q[1] + 0.5) * f + p.area_mm[1]),
                                                    (float(dd[0]), float(dd[1])), ln.support, ln.kind))
        return out

    def snap_lines_px(self) -> dict[str, dict[str, list[sidelines.SnapLine]]]:
        """``snap_lines`` in preview pixels (as ``preview_boxes_px``)."""
        if self.preview is None:
            return {}
        p = self.preview
        f = p.dpi / MM_PER_INCH
        return {aid: {name: [replace(ln, p=((ln.p[0] - p.area_mm[0]) * f - 0.5, (ln.p[1] - p.area_mm[1]) * f - 0.5))
                             for ln in ls] for name, ls in sides.items()}
                for aid, sides in self.snap_lines.items()}
```

- [ ] **Step 5: Crop cases** (`hasami/cases.py`, in `write_case`; add the helper above `write_case`)

```python
def _marks(b: CropBox, *keys: str) -> dict:
    """The box's optional case keys (flags, pinned / snapped sides), omitted when empty."""
    return {k: b.meta[k] for k in keys if b.meta.get(k)}
```

and the two list entries become:

```python
               "auto": [{"id": b.meta.get("auto_id"), "box_mm": _box(b), "confidence": round(float(b.confidence), 3),
                         **_marks(b, "check_sides")} for b in data.auto],
               "final": [{"id": b.meta.get("auto_id"), "box_mm": _box(b), **_marks(b, "pinned_sides", "snapped_sides")}
                         for b in data.final],
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_snapping.py tests/test_cases.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add hasami/core/sidelines.py hasami/workflow.py hasami/cases.py tests/test_snapping.py tests/test_cases.py
git commit -m "Session keeps each booklet box's snap lines; crop cases record pinned and snapped sides

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: snapping and pinning in the preview

**Files:**
- Modify: `hasami/gui/preview.py` (`CropItem`, `PreviewView.set_snap_lines`, `_add_item`, `set_box_angle`)
- Modify: `hasami/gui/app.py:510-512` (`_show_boxes`)
- Modify: `tests/test_gui.py:100-106` (the `Ev` helper gets `modifiers()`)
- Test: `tests/test_snapping_gui.py`

**Interfaces:**
- Consumes: `SnapLine`, `normal_hit` (Tasks 2, 5); `Session.snap_lines_px()` (Task 5); `SIDE_NAMES`.
- Produces: `PreviewView.set_snap_lines(lines: dict) -> None`; `CropItem.snap: dict[str, list[SnapLine]]`; `CropItem.guide_lines() -> list[tuple[QPointF, QPointF, float]]`; box meta `pinned_sides` / `snapped_sides` (lists in `SIDE_NAMES` order); constants `SNAP_PX = 8`, `SNAP_TILT_DEG = 1.5`.

- [ ] **Step 1: Give the existing test event `modifiers()`** (`tests/test_gui.py`, class `Ev` inside `test_side_handle_moves_one_side`)

```python
    class Ev:
        def __init__(self, x, y):
            self._p = QPointF(x, y)

        def pos(self):
            return self._p

        def modifiers(self):
            from PySide6.QtCore import Qt
            return Qt.NoModifier
```

- [ ] **Step 2: Write the failing tests** (`tests/test_snapping_gui.py`)

```python
"""Snapping a dragged side to candidate lines, and which sides an edit pins (offscreen Qt)."""
import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402

from hasami.core.geometry import CropBox  # noqa: E402
from hasami.core.sidelines import SnapLine  # noqa: E402


class Ev:
    def __init__(self, x, y, mods=Qt.NoModifier):
        self._p, self._m = QPointF(x, y), mods

    def pos(self):
        return self._p

    def modifiers(self):
        return self._m


def _view(lines=None):
    from PySide6.QtWidgets import QApplication
    from hasami.gui.preview import PreviewView

    app = QApplication.instance() or QApplication([])
    v = PreviewView()
    v.set_image(np.full((200, 300, 3), 200, np.uint8))
    v.resetTransform()  # 1 scene px = 1 screen px: the 8 px reach is 8 scene px
    v.set_snap_lines({"a1": lines if lines is not None else {"right": [SnapLine((230.0, 100.0), (0.0, 1.0), 0.9, "layer")]}})
    v.set_boxes([CropBox(150, 100, 100, 60, meta={"auto_id": "a1"})])
    return app, v, v.items_[0]


def _drag(it, kind, start, end, mods=Qt.NoModifier):
    it._drag = (kind, QPointF(*start), it.box)
    it.mouseMoveEvent(Ev(*end, mods))
    it._drag = None  # (mouseReleaseEvent needs a real Qt event)


def test_a_dragged_side_snaps_to_a_line_within_reach():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (225, 100))  # the right side, 5 px short of the line at x = 230
    assert it.box.bounding_rect()[2] == pytest.approx(230.0)
    assert it.box.meta["pinned_sides"] == ["right"] and it.box.meta["snapped_sides"] == ["right"]
    assert it.box.manual and it.box.meta["auto_id"] == "a1"


def test_shift_drags_freely():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (225, 100), Qt.ShiftModifier)
    assert it.box.bounding_rect()[2] == pytest.approx(225.0)
    assert it.box.meta["pinned_sides"] == ["right"] and it.box.meta["snapped_sides"] == []


def test_a_line_out_of_reach_does_not_snap():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (215, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(215.0)


def test_a_line_tilted_against_the_side_is_ignored():
    t = math.radians(3.0)
    app, v, it = _view({"right": [SnapLine((230.0, 100.0), (math.sin(t), math.cos(t)), 0.9, "layer")]})
    _drag(it, "s1", (200, 100), (228, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(228.0)


def test_a_free_drag_after_a_snap_clears_snapped_but_stays_pinned():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (225, 100))
    _drag(it, "s1", (230, 100), (212, 100), Qt.ShiftModifier)
    assert it.box.meta["pinned_sides"] == ["right"] and it.box.meta["snapped_sides"] == []


def test_corners_moves_and_skew_pin_their_sides():
    app, v, it = _view()
    _drag(it, "c1", (200, 70), (205, 65))  # top-right corner
    assert it.box.meta["pinned_sides"] == ["top", "right"]
    app, v, it = _view()
    _drag(it, "move", (150, 100), (152, 101))
    assert it.box.meta["pinned_sides"] == ["top", "right", "bottom", "left"]
    app, v, it = _view()
    v.set_box_angle(0, 1.5)
    assert v.items_[0].box.meta["pinned_sides"] == ["top", "right", "bottom", "left"]


def test_pins_accumulate():
    app, v, it = _view()
    _drag(it, "s1", (200, 100), (212, 100))
    _drag(it, "s0", (150, 70), (150, 60))
    assert it.box.meta["pinned_sides"] == ["top", "right"]


def test_guide_lines_show_only_while_a_side_is_dragged():
    app, v, it = _view()
    assert it.guide_lines() == []
    it._drag = ("s1", QPointF(200, 100), it.box)
    it.mouseMoveEvent(Ev(212, 100))
    [(p0, p1, support)] = it.guide_lines()
    assert p0.x() == pytest.approx(230.0) and p1.x() == pytest.approx(230.0) and support == 0.9
    assert it.boundingRect().contains(p0) and it.boundingRect().contains(p1)
    it.mouseMoveEvent(Ev(212, 100, Qt.ShiftModifier))
    assert it.guide_lines() == []  # Shift: free drag, no guides
    it._drag, it._free = ("s0", QPointF(150, 70), it.box), False
    assert it.guide_lines() == []  # the top side has no lines


def test_snapping_works_in_a_turned_view():
    app, v, it = _view()
    v.set_rotation(1)
    v.resetTransform()
    v.rotate(90)  # turned, 1 scene px = 1 screen px
    _drag(it, "s1", (200, 100), (225, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(230.0)
    v.set_rotation(0)


def test_drawn_boxes_do_not_snap():
    app, v, it = _view()
    v.set_boxes([CropBox(150, 100, 100, 60, manual=True, meta={"drawn": True})])
    it = v.items_[0]
    _drag(it, "s1", (200, 100), (225, 100))
    assert it.box.bounding_rect()[2] == pytest.approx(225.0)


def test_the_window_passes_the_sessions_lines_to_the_preview(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from hasami.gui.app import MainWindow
    from hasami.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    st = AppSettings()
    st.output.folder = str(tmp_path)
    win = MainWindow(st)
    lines = {"a1": {"right": [SnapLine((230.0, 100.0), (0.0, 1.0), 0.9, "layer")]}}
    monkeypatch.setattr(win.session, "snap_lines_px", lambda: lines)
    monkeypatch.setattr(win.session, "preview_boxes_px", lambda: [CropBox(150, 100, 100, 60, meta={"auto_id": "a1"})])
    win._show_boxes()
    assert win.preview.items_[0].snap == lines["a1"]
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    win.close()
```

- [ ] **Step 3: Run them to verify they fail**

Run: `$PY -m pytest tests/test_snapping_gui.py tests/test_gui.py -q`
Expected: test_gui passes; the snapping tests FAIL (`AttributeError: 'PreviewView' object has no attribute 'set_snap_lines'`).

- [ ] **Step 4: Implement in `hasami/gui/preview.py`**

Imports: `from ..core.geometry import SIDE_NAMES, CropBox` and `from ..core.sidelines import normal_hit`. Constants next to `HANDLE` / `EDGE`:

```python
SNAP_PX = 8  # a dragged side jumps to a candidate line this close (screen px)
SNAP_TILT_DEG = 1.5  # lines tilted more against the side are not offered
_CORNER_SIDES = (("top", "left"), ("top", "right"), ("bottom", "right"), ("bottom", "left"))  # c0..c3
```

Module helper:

```python
def _ordered(names) -> list[str]:
    return [n for n in SIDE_NAMES if n in set(names)]
```

`CropItem.__init__` gets:

```python
        #: Candidate lines per side name (preview px, ``SnapLine``) a dragged side snaps to.
        self.snap: dict = {}
        self._free = False  # Shift held during this drag
```

New `CropItem` methods (after `_hit`):

```python
    def _side_axis(self, i: int, b: CropBox):
        """Side ``i``'s box axis and sign (outward normal = axis * sign)."""
        ux, uy = b.axes
        return ((uy, -1), (ux, 1), (uy, 1), (ux, -1))[i]

    def _snap_hits(self, i: int, b: CropBox) -> list:
        """(distance from ``b``'s centre along side ``i``'s outward normal, line) of each usable line."""
        axis, sign = self._side_axis(i, b)
        nrm, c = axis * sign, np.array([b.cx, b.cy])
        out = []
        for ln in self.snap.get(SIDE_NAMES[i], ()):
            dd = np.array(ln.d)
            if abs(float(dd @ nrm)) > math.sin(math.radians(SNAP_TILT_DEG)):
                continue
            t = normal_hit(np.array(ln.p), dd, c, nrm)
            if t is not None:
                out.append((t, ln))
        return out

    def guide_lines(self) -> list[tuple[QPointF, QPointF, float]]:
        """The dragged side's lines (while a side is dragged without Shift): segments as long as the side."""
        if self._drag is None or self._free or self._drag[0][0] != "s":
            return []
        i, b = int(self._drag[0][1]), self.box
        axis, sign = self._side_axis(i, b)
        half = (b.w if i % 2 == 0 else b.h) / 2  # top / bottom (s0, s2) run along the width
        out = []
        for t, ln in self._snap_hits(i, b):
            q, dd = np.array([b.cx, b.cy]) + axis * sign * t, np.array(ln.d)
            out.append((QPointF(*(q - dd * half)), QPointF(*(q + dd * half)), float(ln.support)))
        return out
```

`boundingRect` becomes:

```python
    def boundingRect(self) -> QRectF:
        x0, y0, x1, y1 = self.box.bounding_rect()
        pad = 40 * self._scale()
        r = QRectF(x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad)
        for p0, p1, _ in self.guide_lines():
            r = r.united(QRectF(p0, p1).normalized().adjusted(-2, -2, 2, 2))
        return r
```

In `paint`, after `painter.drawPolygon(self._poly())`:

```python
        for p0, p1, support in self.guide_lines():
            c = QColor(ACCENT)
            c.setAlphaF(0.35 + 0.65 * min(1.0, max(0.0, (support - 0.6) / 0.4)))
            guide = QPen(c, 0)  # cosmetic: 1 screen px
            painter.setPen(guide)
            painter.drawLine(p0, p1)
        painter.setPen(pen)
```

`mousePressEvent`: add `self._free = False` before `super()...`.

`mouseMoveEvent`'s side branch becomes:

```python
        elif kind[0] == "s":
            # One side: the opposite side stays where it is.
            i = int(kind[1])
            axis, sign = self._side_axis(i, b0)
            d = float((np.array([pos.x(), pos.y()]) - np.array([b0.cx, b0.cy])) @ axis) * sign
            self._free = bool(event.modifiers() & Qt.ShiftModifier)
            reach = SNAP_PX * self._scale()
            best = None if self._free else min(self._snap_hits(i, b0), key=lambda h: abs(h[0] - d), default=None)
            snapped = best is not None and abs(best[0] - d) <= reach
            if snapped:
                d = best[0]
            half = (b0.h if i % 2 == 0 else b0.w) / 2
            size = max(4.0, d + half)  # from the fixed side to the dragged one
            shift = (size - 2 * half) / 2 * sign
            c = np.array([b0.cx, b0.cy]) + axis * shift
            w, h = (b0.w, size) if i % 2 == 0 else (size, b0.h)
            nb = CropBox(c[0], c[1], w, h, b0.angle, b0.confidence)
```

(`i % 2 == 0` = top / bottom = axis `uy`, the same as the old `axis is uy`.)

And replace the tail (`nb.manual = True` / `nb.meta = dict(b0.meta)`) with:

```python
        nb.manual = True
        nb.meta = dict(b0.meta)  # keep e.g. the crop's own rotation
        moved = (set(SIDE_NAMES) if kind in ("move", "rot") else {SIDE_NAMES[int(kind[1])]} if kind[0] == "s"
                 else set(_CORNER_SIDES[int(kind[1])]))
        snapped_sides = set(b0.meta.get("snapped_sides", ())) - moved
        if kind[0] == "s" and snapped:
            snapped_sides.add(SIDE_NAMES[int(kind[1])])
        nb.meta["pinned_sides"] = _ordered(set(b0.meta.get("pinned_sides", ())) | moved)
        nb.meta["snapped_sides"] = _ordered(snapped_sides)
```

`mouseReleaseEvent`:

```python
    def mouseReleaseEvent(self, event) -> None:
        self.prepareGeometryChange()  # the guide lines go
        self._drag = None
        self._free = False
        self.update()
        super().mouseReleaseEvent(event)
```

`PreviewView.__init__`: `self.snap_lines: dict = {}`. New method and changes:

```python
    def set_snap_lines(self, lines: dict) -> None:
        """Candidate lines per auto id and side (preview px), for snapping dragged sides."""
        self.snap_lines = lines or {}
        for it in self.items_:
            it.snap = self.snap_lines.get(it.box.meta.get("auto_id"), {})
```

In `_add_item`, after creating the item: `it.snap = self.snap_lines.get(box.meta.get("auto_id"), {})`.

In `set_box_angle`, after `nb.meta = dict(b0.meta)`:

```python
        nb.meta["pinned_sides"] = list(SIDE_NAMES)  # a turned box is frozen as a whole
        nb.meta["snapped_sides"] = []
```

- [ ] **Step 5: `hasami/gui/app.py` `_show_boxes`** - before `self.preview.set_boxes(boxes)`:

```python
        self.preview.set_snap_lines(self.session.snap_lines_px())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_snapping_gui.py tests/test_gui.py tests/test_cases_gui.py -q`
Expected: all pass. `test_snapping_works_in_a_turned_view` checks the item works in its own coordinates whatever the view's turn; if `set_rotation` refits the view, keep the `resetTransform(); rotate(90)` after it as written.

- [ ] **Step 7: Commit**

```bash
git add hasami/gui/preview.py hasami/gui/app.py tests/test_gui.py tests/test_snapping_gui.py
git commit -m "Preview: a dragged booklet side snaps to auto-crop's candidate lines (Shift: free); edits pin their sides

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: "check this side" flags (ONLY if Task 3's flag gate passed)

Skip this whole task when the ledger says `Task 3: flags dropped`; ledger `Task 7: skipped (flag gate failed)`.

**Files:**
- Modify: `hasami/core/sidelines.py` (`CHECK_RULE`)
- Modify: `hasami/workflow.py` (`detect_on_preview`)
- Modify: `hasami/gui/preview.py` (`CHECK` colour, `paint`, `hoverMoveEvent`, `mouseMoveEvent`)
- Modify: `hasami/gui/app.py` (`_show_boxes` status)
- Modify: `tests/test_sidelines.py` (`test_no_flag_rule_until_the_measurement_sets_one` -> the fitted rule)
- Test: `tests/test_snapping.py`, `tests/test_snapping_gui.py`

**Interfaces:**
- Consumes: Task 3's fitted rule; `signals`, `check_reason`, `spine_names` (Task 2).
- Produces: box `meta["check_sides"]: dict[str, str]` (side name -> reason) on auto boxes; `CHECK` colour.

- [ ] **Step 1: Write the failing tests**

In `tests/test_sidelines.py`, replace `test_no_flag_rule_until_the_measurement_sets_one` with (RULE = the rule ledgered in Task 3, written out literally):

```python
def test_the_flag_rule_is_the_one_the_snap_check_fitted():
    from hasami.core import sidelines

    assert sidelines.CHECK_RULE == RULE  # e.g. (("near_strong", 1.5),): copy the ledgered rule here
```

Append to `tests/test_snapping.py`:

```python
def test_flags_name_the_sides_the_rule_fires_on(tmp_path, monkeypatch):
    from hasami.core import sidelines

    monkeypatch.setattr(sidelines, "CHECK_RULE", (("crowd", 0),))  # fires on every side
    s = _session(tmp_path)
    s.run_preview()
    b = s.boxes_mm[0]
    assert set(b.meta["check_sides"]) == set(SIDE_NAMES)
    assert all(r.endswith("edges within 3 mm") for r in b.meta["check_sides"].values())
    assert s.auto_mm[0].meta["check_sides"] == b.meta["check_sides"]  # crop cases see the flags


def test_no_flags_when_the_rule_never_fires(tmp_path, monkeypatch):
    from hasami.core import sidelines

    monkeypatch.setattr(sidelines, "CHECK_RULE", (("crowd", 999),))
    s = _session(tmp_path)
    s.run_preview()
    assert all("check_sides" not in b.meta for b in s.boxes_mm)
```

Append to `tests/test_snapping_gui.py`:

```python
def test_a_flagged_side_shows_its_reason_and_loses_the_flag_when_moved():
    app, v, it = _view()
    it.box.meta["check_sides"] = {"right": "another edge 1.6 mm further in", "top": "the edge here is faint"}

    it.hoverMoveEvent(Ev(200, 100))
    assert it.toolTip() == "Check this side: another edge 1.6 mm further in"
    it.hoverMoveEvent(Ev(150, 100))  # inside: no side
    assert it.toolTip() == ""
    _drag(it, "s1", (200, 100), (212, 100))
    assert it.box.meta["check_sides"] == {"top": "the edge here is faint"}


def test_the_status_counts_the_sides_to_check(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from hasami.gui.app import MainWindow
    from hasami.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    st = AppSettings()
    st.output.folder = str(tmp_path)
    st.crop.mode = "booklet"
    win = MainWindow(st)
    box = CropBox(150, 100, 100, 60, meta={"auto_id": "a1", "check_sides": {"right": "x", "top": "y"}})
    monkeypatch.setattr(win.session, "preview_boxes_px", lambda: [box])
    win._show_boxes()
    assert "2 sides to check" in win.status_label.text()
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    win.close()
```

`hoverMoveEvent` calls `super().hoverMoveEvent(event)`, which needs a real event: in the implementation, call it only `if isinstance(event, QGraphicsSceneHoverEvent)` (import it from `PySide6.QtWidgets`).

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_sidelines.py tests/test_snapping.py tests/test_snapping_gui.py -q`
Expected: the five new tests FAIL (`CHECK_RULE` is `()`, `KeyError: 'check_sides'`, tooltip empty, status without the count).

- [ ] **Step 3: Implement**

`hasami/core/sidelines.py`: set `CHECK_RULE` to the ledgered rule, with a comment naming the report:

```python
#: Fitted by the snap check (hasami-data benchmarks/learned/snapcheck_<date>.md: held-out flagged <X>% of sides,
#: caught <Y>% of the sides > 0.5 mm off).
CHECK_RULE: tuple[tuple[str, float], ...] = RULE
```

`hasami/workflow.py` `detect_on_preview`, inside the `for b_px, b_mm in zip(found, found_mm):` loop (replace the single assignment with):

```python
                        lines = sidelines.box_lines(p.image, b_px, p.dpi)
                        self.snap_lines[b_mm.meta["auto_id"]] = self._snap_lines(b_px, lines)
                        spines = sidelines.spine_names(b_px)
                        check = {name: r for name in SIDE_NAMES
                                 if (r := sidelines.check_reason(sidelines.signals(
                                     lines[name], self.settings.crop.margin_mm, name in spines),
                                     sidelines.CHECK_RULE))}
                        if check:
                            b_mm.meta["check_sides"] = check
```

(import `SIDE_NAMES` from `.core.geometry`). This runs before `self.auto_mm = ...` copies the boxes, so crop cases see the flags.

`hasami/gui/preview.py`:

```python
CHECK = QColor(230, 150, 0)  # "check this side"
```

In `paint`, after the guide lines (before the label font is set):

```python
        corners = self.box.corners()
        for name, _reason in self.box.meta.get("check_sides", {}).items():
            a, b = ((0, 1), (1, 2), (2, 3), (3, 0))[SIDE_NAMES.index(name)]
            painter.setPen(QPen(CHECK, 3.0 * s))
            painter.drawLine(QPointF(*corners[a]), QPointF(*corners[b]))
            mid = (corners[a] + corners[b]) / 2
            painter.drawText(QPointF(mid[0] - 3 * s, mid[1] + 4 * s), "?")
        painter.setPen(pen)
```

In `hoverMoveEvent`, after `kind = self._hit(event.pos())`:

```python
        reason = (self.box.meta.get("check_sides", {}).get(SIDE_NAMES[int(kind[1])])
                  if kind[0] == "s" else None)
        self.setToolTip(f"Check this side: {reason}" if reason else "")
```

and its last line becomes `if isinstance(event, QGraphicsSceneHoverEvent): super().hoverMoveEvent(event)`.

In `mouseMoveEvent`'s tail, after the pinned / snapped lines:

```python
        check = {k: v for k, v in b0.meta.get("check_sides", {}).items() if k not in moved}
        if check:
            nb.meta["check_sides"] = check
        else:
            nb.meta.pop("check_sides", None)
```

`hasami/gui/app.py` `_show_boxes`, inside `elif mode == "booklet":` after its status line is set:

```python
            n = sum(len(b.meta.get("check_sides") or {}) for b in boxes)
            if n:
                self.set_status(f"{self.status_label.text()}  ({n} side{'s' if n != 1 else ''} to check)")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_sidelines.py tests/test_snapping.py tests/test_snapping_gui.py tests/test_gui.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hasami/core/sidelines.py hasami/workflow.py hasami/gui/preview.py hasami/gui/app.py tests/test_sidelines.py tests/test_snapping.py tests/test_snapping_gui.py
git commit -m "Check-this-side flags with the rule the snap check fitted

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: version, docs, benchmarks, PR

**Files:**
- Modify: `hasami/__init__.py`, `pyproject.toml` (0.5.18 -> 0.5.19)
- Modify: `README.md` (Use > Editing crop boxes)
- Modify: `CLAUDE.md` (new section after "Crop cases (0.5.17)"; the Crop cases section's "Next:" line)

- [ ] **Step 1: Version** - `__version__ = "0.5.19"` and `version = "0.5.19"`.

- [ ] **Step 2: README** - under "Editing crop boxes:", after the "Drag inside a box..." bullet:

```markdown
- Booklet mode: while you drag an edge, the edges auto-crop found nearby show as thin lines, and the
  edge snaps to one when it comes within a few pixels. Hold Shift to drag freely. An edge you move
  stays exactly where you put it in the saved scan; the box's other edges are still fitted on the
  full-resolution scan.
```

Only if Task 7 ran, add:

```markdown
- An amber edge with a **?** is one auto-crop was unsure about: hover over it to see why, and check it.
```

- [ ] **Step 3: CLAUDE.md** - after the "Crop cases (0.5.17, owner)" section:

```markdown
## Edge snapping and pinned sides (0.5.19, owner)
`hasami/core/sidelines.py` (read-only, never called by `detect` / `refine`): `side_lines` = toppage's line finding
on a wider strip around each FINAL box (paper sides 14 mm in / 8 mm out at the three thresholds; spine sides the
fold's valleys within 4 mm), `box_lines` maps a spread's pages onto its sides. `Session.detect_on_preview` keeps
them per auto id in bed mm (`snap_lines`, booklet mode only, moved in by the crop margin); `CropItem` snaps a
dragged side within 8 screen px (lines tilted > 1.5 deg ignored; Shift = free). Every edit records
`meta["pinned_sides"]` (side drag: that side; corner: two; move / rotate / Skew: all four; `snapped_sides` too);
`autocrop._refine` refines an auto box with 1-3 pinned sides and `geometry.pin_sides` puts them back (angle
kept); drawn, unpinned-manual and fully pinned boxes stay untouched. Measurement: hasami-data
`benchmarks/learned/snapcheck_<date>.md` (<coverage>% of the wrong bare-preview sides had the truth among the
candidates; flags: <verdict and rule, or "dropped: held-out rate X / recall Y">). Spec
`docs/specs/2026-10-15-edge-snapping-design.md`.
```

Fill the `<...>` values from the report. In the Crop cases section, change its last sentence to: "Next: a hasami-data tool to turn cases into truth records."

- [ ] **Step 4: Full suite**

Run: `$PY -m pytest -q > .superpowers/sdd/2026-10-15-edge-snapping-plan/suite.txt 2>&1; tail -3 .superpowers/sdd/2026-10-15-edge-snapping-plan/suite.txt`
Expected: all pass (the 1 skip and 3 xfails as on main; the count is main's 197 + the new tests).

- [ ] **Step 5: Synthetic benchmarks, main vs branch** (must be identical; timings aside)

```bash
git worktree add F:/HasamiDev-base main
cd F:/HasamiDev-base && $PY benchmarks/synthetic_autocrop.py > F:/HasamiDev/.superpowers/sdd/2026-10-15-edge-snapping-plan/syn_auto_main.txt && $PY benchmarks/synthetic_booklet.py > F:/HasamiDev/.superpowers/sdd/2026-10-15-edge-snapping-plan/syn_book_main.txt
cd F:/HasamiDev && $PY benchmarks/synthetic_autocrop.py > .superpowers/sdd/2026-10-15-edge-snapping-plan/syn_auto.txt && $PY benchmarks/synthetic_booklet.py > .superpowers/sdd/2026-10-15-edge-snapping-plan/syn_book.txt
```

Expected: the files differ only in timing columns (compare with `diff`; read any other difference before going on).

- [ ] **Step 6: Real-scan benchmarks, main vs branch** (`--jobs 2`; background; both trees, same commands with `PYTHONPATH` = `F:/HasamiDev-base` then `F:/HasamiDev`, OUT = `F:/HasamiData/learned/bench_snap_{base,branch}/<group>`)

```bash
PYTHONPATH=<tree> $PY F:/HasamiData/benchmarks/real_booklets.py "F:/ScannerTraining/Training booklets" --only "V600 Scans" --scenarios real,preview --turn 0 --profile epson-v600 --jobs 2 --out <OUT>/v600
PYTHONPATH=<tree> $PY F:/HasamiData/benchmarks/real_booklets.py "F:/ScannerTraining/Training booklets" --only "Thick books" --scenarios real,preview --turn 0 --profile epson-v600 --jobs 2 --out <OUT>/thick
PYTHONPATH=<tree> $PY F:/HasamiData/benchmarks/real_booklets.py "F:/ScannerTraining/Training booklets" --only "Kill la kill" --scenarios real,preview --turn 0 --profile epson-v600 --jobs 2 --out <OUT>/klk
PYTHONPATH=<tree> $PY F:/HasamiData/benchmarks/real_booklets.py "F:/ScannerTraining/Training booklets" --only "New data" --scenarios real,preview,app --turn 0 --profile epson-v600 --jobs 2 --out <OUT>/newdata
```

Then per group: `PYTHONPATH=F:/HasamiData/benchmarks/klk6 $PY F:/HasamiData/benchmarks/klk6/compare.py <base>/<g>/results.json <branch>/<g>/results.json`
Expected: identical (no side moved). Any difference: STOP and debug - the engine must not have changed. Remove the base worktree only after every run has finished: `git worktree remove F:/HasamiDev-base`.

- [ ] **Step 7: Commit, push, PR**

```bash
git add hasami/__init__.py pyproject.toml README.md CLAUDE.md
git commit -m "0.5.19: edge snapping and pinned sides

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/edge-snapping
gh pr create --title "0.5.19: snap box edges to auto-crop's candidate lines; keep hand-moved sides at scan time" --body "<what changed; the snap check's pooled numbers (no scans); flags shipped or dropped and why; side_lines time per page; suite result; synthetic and real-scan benchmarks identical to main>

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

The PR text names no scan files' contents and uses no closing keywords.
