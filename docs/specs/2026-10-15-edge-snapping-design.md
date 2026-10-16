# Edge snapping, pinned sides and "check this side" flags (design)

Status: design, approved in chat Oct 15 2026 (sections 1-3); waiting for the owner's review of this file.

**Update (owner, Oct 15 2026): Section 2 (pinned sides) dropped after it was built.** A real-scan check (hasami-data
`benchmarks/learned/snap/pincheck.py`) showed the pinned refine matches the unedited refine and, against freezing
the edited box, improves no side and moves three inward 0.55-0.8 mm. Edited boxes stay frozen at scan time, as
before; `pinned_sides` is not recorded. Snapping and flags ship as specified.
Part (2) of the Oct 15 brainstorm; part (3), crop cases, shipped in 0.5.17 / 0.5.18
(`docs/specs/2026-10-15-crop-cases-design.md`).

## Goal

Most remaining booklet errors are one judgement: the right line (the top page's edge) was among the engine's
candidates and the rules picked another (a stack sheet, a cover rim, a lower page's art column, the wrong valley at
a fold). Fixing such a side should take one rough drag, not careful dragging, and the user should know which sides
are worth a look.

Success:
- Dragging a booklet page's side towards the right line lands on it exactly (snaps), unless Shift is held.
- A side the user moved stays exactly where it was put in the saved scan, while the box's other sides are still
  refined on the full-resolution scan (today any edit freezes the whole box at preview precision).
- When a flag rule passes its measured gate: the few sides the rules were unsure about are marked on the preview.
- Nothing changes for unedited boxes: `detect` and `refine` on unpinned boxes are untouched, so every synthetic and
  real-scan benchmark is identical.

## Owner's decisions (Oct 15 2026)

- Snapping covers **booklet mode only** (the candidate pools exist there; multi / single errors are mostly ramps,
  where snapping would not help).
- **Magnetic dragging**: the dragged side jumps to a candidate line within reach; Shift drags freely. No stepping
  clicks.
- **Freeze only the sides the user moved**: the other sides are still refined at scan time.
- **Flags are measured first** against the private truth, with a gate fixed in advance; no flags if no rule passes.
- Candidates are **recomputed on the final boxes** (approach 1), not mapped back from the engine's records and not
  carried through the booklet code.

## Section 1: the candidate lines

New read-only function in `hasami/core/toppage.py`:

```python
@dataclass(frozen=True)
class SideLine:
    offset_mm: float   # at the side's middle, + = outwards from the box's side
    slope: float       # tilt against the side (mm per mm along it, x = offset + slope * y)
    support: float     # share of the segments that show it, 0..1
    kind: str          # "layer" | "faint" | "strong" | "crease"

def side_lines(image: np.ndarray, box: CropBox, dpi: float) -> dict[str, list[SideLine]]:
    """Every candidate line near each side of a final page box, keyed "top" / "right" / "bottom" / "left"
    (the box's own frame, as ``cases.side_shifts``), each list sorted by offset."""
```

- **Paper sides** (not in `box.meta["keep_sides"]`): a strip from 14 mm inside to 8 mm outside the side (wider
  outwards than the engine's 1 mm, so a box that cuts print gets outward lines), built with the same `_strip`,
  `_segment_peaks` and `_layer_lines` the rules use, at the three thresholds the collector records: the usual one
  (kind "layer"), `fine_stack_percentile` ("faint") and `fine_strong_percentile` ("strong"). Columns off the image
  are ignored (a side whose strip is < 90% on the image gets no lines, as `top_page` skips it). Lines within
  0.15 mm of each other merge, keeping the one with the higher support (on a tie the earlier kind in the order
  layer, strong, faint).
- **Spine sides** (in `keep_sides`): dark-valley lines within +-4 mm: every local minimum of each segment's
  high-passed profile (as `_crease` computes it, not only its deepest), pooled into lines with `_layer_lines`; kind
  "crease". Columns off the image are ignored, as in `_crease`.
- The box is the final page box as `detect` returns it (margin included); offsets are measured from its sides.
- Runs only on the preview, from `Session.detect_on_preview`, in booklet mode, once per final page box; a spread box
  (`booklet_output` "spread") once per page of `meta["spread_pages"]`, the lines kept for the spread box's sides they
  face (a page side whose outward normal is within 10 deg of a spread side's and lies within 1 mm of it; the inner
  spine sides of a spread are not spread sides and get no lines). Thick-book pages (`meta["spine"] == "thick"`) get
  lines too.
- `detect`, `refine`, `top_page` and the candidate collector are unchanged. Budget: <= 100 ms per page at 200 dpi on
  the owner's PC (measured on a generated booklet, reported in the PR).

**Storage.** `Session.snap_lines: dict[str, dict[str, list[SnapLine]]]`, keyed by `auto_id`, then side name. A
`SnapLine` (in `hasami/workflow.py`) is absolute: `p` (a point on the line, bed mm), `d` (unit direction), `support`,
`kind`. Converted from `SideLine` with the auto box's geometry at detection. Absolute lines stay valid after the
side is dragged, the box moved or its Skew changed. Not stored in `CropBox.meta`: they never travel through refine,
box copies or `case.json`. A new preview clears them; `detect_on_preview` replaces the entries of the boxes it
replaces and keeps those of manual boxes it keeps. `Session.snap_lines_px()` returns them in preview pixels (same
transform as `box_mm_to_px`) for the GUI.

## Section 2: the scan keeps the sides the user moved

**Marking.** Edited boxes stay `manual`, as today (dashed outline, kept on re-detect, crop cases see the edit).
`CropItem.mouseMoveEvent` also records `meta["pinned_sides"]` (sorted list of side names):

- dragging one side, snapped or free: that side;
- dragging a corner: the two sides meeting there;
- moving the whole box, the rotate handle, or the Skew field (`PreviewView.set_box_angle`): all four.

Pinned sides only accumulate (union with what is already pinned).

**Refine.** `autocrop._refine` returns manual boxes untouched today. New rule: a manual box with `meta["auto_id"]`
and fewer than four `pinned_sides` is refined as an auto box would be (`refine_booklet` for booklet pages and spreads,
`_refine_edges` otherwise, with the usual margin handling), and then

```python
def pin_sides(given: CropBox, refined: CropBox, pinned: Iterable[str]) -> CropBox:
    """``given`` with its angle kept, each pinned side where it is, and each other side moved parallel to where
    ``refined`` put it (the refined side's middle projected on ``given``'s normal)."""
```

in `hasami/core/geometry.py` rebuilds the result (meta from the refined box, `manual` kept True). Drawn boxes (no
`auto_id`) and boxes with all four sides pinned stay untouched. `top_page` never sees pinned sides as spines (the
refine runs on the box with its own `keep_sides` only), so they cannot jump to a crease. The angle is kept because a
re-tilted box would pivot a pinned side about its middle (~0.2 mm at the ends of a 150 mm side per 0.1 deg); today
the whole box is frozen, angle included, so this is never worse.

This is the scan-time refine, so it applies in **every auto-crop mode** (booklet, multi, single). With
`crop.edge_refine` off nothing is refined, as today.

## Section 3: snapping and flags in the preview, the measurement, the release

**Snapping** (`hasami/gui/preview.py`). `PreviewView.set_snap_lines(lines)` takes `Session.snap_lines_px()`; the app
calls it whenever it sets boxes from the session. While a side ("s0".."s3", top / right / bottom / left of the box's
own frame) of a box with lines (looked up by `meta["auto_id"]`) is dragged without Shift:

- that side's lines are drawn as thin cosmetic 1 px lines along it (the side's length), more opaque with higher
  support; lines tilted more than 1.5 deg against the side are ignored;
- the side's position (along its normal, at its middle) jumps to the nearest line within 8 screen pixels
  (`8 * _scale()` scene px); the side moves parallel, so the box keeps its angle and stays a rectangle;
- a snapped side is added to `meta["snapped_sides"]` (and pinned); a free drag of a side removes it from
  `snapped_sides`.

Corner drags, whole-box moves and drawn boxes do not snap. Nothing is drawn unless a side is being dragged. No new
setting.

**Flags.** `Session.detect_on_preview` applies the measured rule (below) to each booklet page's `side_lines` and sets
`meta["check_sides"] = {side: reason}` (reason: a short sentence, e.g. "another edge 1.6 mm further in").
`CropItem.paint` draws a flagged side in amber (`CHECK` colour) with a small "?" at its middle; hovering the side
shows "Check this side: <reason>" as the tooltip. The status bar's auto-crop message adds "N sides to check". A flag
is removed when its side is moved (pinned).

**Crop cases.** `case.json` gains optional keys, `format` stays 1: `check_sides` on auto boxes, `pinned_sides` and
`snapped_sides` on final boxes (taken from `meta`, omitted when empty). They show later how flags do on scans the
measurement never saw. `read_case` accepts cases with or without them.

**The measurement** (hasami-data `benchmarks/learned/snap/`, private, its own PR there). On V600 Scans 57, Thick
books 26, Kill la kill 9 and New data 33, the bare 200 dpi preview box (the `preview` scenario's detect), `side_lines`
from the hasami-scan branch (via `PYTHONPATH`), truth mapped into the preview frame with `label.py`. Runs with
`--jobs 2`.

- **Snap check:** of the sides > 0.5 mm off truth, the share with a candidate line within 0.3 mm of the truth (at
  the side's middle); plus the median number of lines per side. **Gate: >= 60%.** Below it, work stops and the
  numbers go to the owner.
- **Flag check:** signals per side, computable from `side_lines` and the box alone: (a) distance to the nearest other
  line with support >= 0.85 (inside or outside); (b) the line under the side has support < a threshold or there is
  none within 0.3 mm; (c) a spine side with no crease line within 0.5 mm; (d) the number of lines within 3 mm. A rule
  is one signal or the OR of two, each with one threshold. Thresholds are chosen leave-one-book-out (books as in
  `books.json`): pick on the other books, score on the held-out book, pool the held-out scores. **Gate (pooled
  held-out): at most 1 side in 8 flagged, and at least half of the sides > 0.5 mm off flagged.** The rule with the
  highest recall that passes is fitted on all books and goes into the app (its thresholds as named constants in
  `toppage`, not a profile switch: flags change no crop). If none passes: no flags (snapping and pinning ship alone).
  The rules' choice (`rule`) is not available to approach 1; if only a record-based signal could pass, that is
  reported and flags wait (approach 2 would be its own decision).
- **Report:** `benchmarks/learned/snapcheck_<date>.md` (numbers per book; no scans).

**Order.**
1. hasami-scan branch `feat/edge-snapping`: `side_lines` with its tests.
2. The measurement (hasami-data PR); its result decides flags.
3. The branch: `pin_sides` and the refine rule, `Session.snap_lines`, snapping in the preview, flags (only if they
   passed), the case keys.
4. Version 0.5.19 (`hasami/__init__.py`, `pyproject.toml`); README "Fixing a box" (snapping, Shift, pinned sides,
   flags if any); CLAUDE.md note. One PR.

## Testing (generated scans only)

- `side_lines`: on a synthetic booklet with a page stack (`synth_booklet` or `demo_anime`) the stack's sheet edges and
  the top page's edge are lines on the stack side (offsets within 0.3 mm of the generator's truth); the fold side
  gets a "crease" line within 0.3 mm of the generated fold; a side running off the image gets no lines; `detect`'s
  output is identical with and without a `side_lines` call.
- `pin_sides`: pinned sides stay bit-exact, the angle is kept, unpinned sides follow the refined box (axis-aligned and
  turned boxes; all four pinned = `given`).
- Refine: a pinned auto box on a generated scan keeps its pinned side exactly and has its other sides refined (equal
  to refining the unedited box, within 0.05 mm); a drawn box and an all-pinned box are untouched; booklet, spread and
  multi mode.
- `Session`: `snap_lines` filled in booklet mode only, cleared on a new preview, kept for manual boxes on re-detect;
  `snap_lines_px` round-trips.
- GUI (offscreen, temporary APPDATA): a side drag ending 5 screen px from a line snaps onto it and pins + marks the
  side snapped; with Shift it does not snap; a corner drag pins two sides, a move all four; flagged sides paint the
  amber pen, show the tooltip, and lose the flag when moved.
- Crop cases: the new keys round-trip; a format-1 case without them still reads.
- Benchmarks: `benchmarks/synthetic_*.py` and the real-scan runs (`--jobs 2`; V600, Thick books, Kill la kill,
  New data with `real,preview,app`) identical to main.

## Out of scope

Snapping in multi / single mode; stepping between candidates by click or key; snapping corners; flags from the
rules' own choice (approach 2); a setting to turn snapping off; the hasami-data tool that turns crop cases into truth
records.
