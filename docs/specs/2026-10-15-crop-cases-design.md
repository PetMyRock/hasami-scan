# Crop cases: saving and sharing auto-crop corrections (design)

Status: design, approved in chat Oct 15 2026 (sections 1-3); waiting for the owner's review of this file.
Part (3) of the Oct 15 brainstorm; part (2), snapping box edges to the engine's candidate lines and "check this
side" flags, gets its own spec after this one.

## Goal

Every learned-cropping attempt so far stopped on the same thing: too few kinds of real scans with truth (a held-out
book whose page stack looks like no other book's broke both models). The app is where the failures happen and where
the user already fixes them by hand. A **crop case** keeps what a fix shows: the whole-bed preview, the uncropped
scanned area, the boxes auto-crop proposed and the boxes the user scanned. Cases are saved locally and only when the
user turns this on; sharing them (zips posted on the Discord server) is always a separate, deliberate step.

Success:
- With the setting on, every scan whose auto-crop boxes were corrected leaves one case; "Save this case" keeps any
  other scan's case.
- A case is enough to (a) mark its truth precisely later, as the owner's test scans are marked, and (b) replay it
  through the benchmark's app path (preview -> crop-area scan -> refine).
- A case fits Discord's upload limit (~10 MB without Nitro) as one zip.
- With the setting off nothing changes: no extra memory, time or files.
- Saving can never break or slow down a scan noticeably (one JPEG pair, ~1 s, on the worker thread).

## Owner's decisions (Oct 15 2026)

- For all users, opt-in (off by default), with a share flow: the owner first, Discord users when they want to help
  tune their scanner.
- A case holds the preview and the crop-area scan at 400 dpi, both JPEG q95 (3-6 MB). The owner's V600 truth was
  marked on JPEG scans, so q95 is good enough for labels.
- Cases are saved for corrections, plus a "Save this case" command for scans nobody had to fix.
- Sharing through a "Share crop cases..." dialog that zips the cases the user ticks.

## Section 1: what a case is

**Setting.** New `AppSettings.cases` (`hasami/settings.py`):

```python
@dataclass
class CaseSettings:
    #: Save crop cases (preview + scanned area + boxes) when auto-crop boxes were corrected.
    enabled: bool = False
    #: Where cases go; "" = Documents/Hasami Scan/Crop cases.
    folder: str = ""
```

GUI: Crop tab, "Save my corrections as crop cases (to help tune auto-crop)" and a folder picker (empty = default).

**Folder.** One subfolder per case, named by local time `YYYY-MM-DD_HHMMSS` (a `_2`, `_3` suffix when two land in
the same second).

**Files.** All images in the scanner's own frame, unturned (as test scans; `crop.rotate` is recorded, not applied).

- `preview.jpg`: the whole-bed preview as scanned (its own dpi, normally 200), JPEG quality 95.
- `scan.jpg`: the uncropped image the first page's scan delivered (the crop area + 3 mm, or the whole bed when
  `scan_crop_area_only` is off), resized with `INTER_AREA` to min(scan dpi, 400) dpi, JPEG quality 95. 16-bit scans
  are reduced to 8 bits for it.
- `case.json`:

```json
{
  "format": 1,
  "app_version": "0.5.17",
  "created": "2026-10-15T14:32:10",
  "reason": "corrected",
  "scanner": {"id": "twain:EPSON Perfection V600", "name": "EPSON Perfection V600"},
  "profile": "epson-v600",
  "crop": {"mode": "booklet", "booklet_output": "pages", "booklet_rtl": false, "margin_mm": 0.25,
           "edge_refine": true, "rotate": 0},
  "preview": {"dpi": 200.0, "area_mm": [0.0, 0.0, 216.0, 297.0], "file": "preview.jpg"},
  "scan": {"scan_dpi": 800.0, "dpi": 400.0, "area_mm": [12.1, 20.4, 150.3, 201.7], "file": "scan.jpg"},
  "auto": [{"id": "a1", "box_mm": [cx, cy, w, h, angle], "confidence": 0.93}],
  "final": [{"id": "a1", "box_mm": [cx, cy, w, h, angle]}, {"id": null, "box_mm": [...]}],
  "edits": [{"id": "a1", "kind": "moved", "sides_mm": {"top": 0.0, "right": 0.0, "bottom": -4.2, "left": 0.0},
             "angle_deg": 0.0},
            {"id": null, "kind": "drawn"}]
}
```

Boxes are `CropBox` values in bed mm (`[cx, cy, w, h, angle]`, the same frame for preview and scan). Side names
follow the box's own frame (top = the side towards -y at angle 0). `sides_mm` is signed, + = outwards (the
corrected side lies further out than auto-crop's).

**Edits.** At detection each auto box gets `meta["auto_id"]` ("a1", "a2", ... per preview). Edits keep `meta`
(`CropItem.mouseMoveEvent` and `PreviewView.set_box_angle` copy it), so the id survives dragging and Skew.
`cases.diff_boxes(auto, final)`:
- an auto id found in `final`: `moved` when any side shifted by > 0.1 mm or the angle by > 0.05 deg (per-side
  shift and the angle change recorded), else `unchanged`;
- an auto id missing from `final`: `deleted`;
- a final box without an auto id: `drawn`.

A scan has a correction when any edit is not `unchanged`.

**Not saved:** processed output, raw files, pages after the first of a feeder batch, images opened from disk (File >
Open image: no preview, no scan area), test scans (already whole-bed and uncropped), the CLI (no box editing).

## Section 2: capture flow and code layout

**`hasami/cases.py`** (no Qt):
- `diff_boxes(auto: list[CropBox], final: list[CropBox]) -> list[dict]` (Section 1).
- `CaseData` dataclass: `preview` (np.ndarray), `preview_dpi`, `preview_area_mm`, `scan` (np.ndarray, already
  <= 400 dpi), `scan_dpi_saved`, `scan_dpi`, `scan_area_mm`, `auto` / `final` (CropBox lists in bed mm), `scanner_id`,
  `scanner_name`, `profile`, `crop` (dict), `reason`, `created` (datetime).
- `write_case(folder: Path, data: CaseData) -> Path`: writes into `<name>.partial`, renames to `<name>` at the end.
- `list_cases(folder: Path) -> list[Path]` (newest first; `.partial` folders never listed); `read_case(path) -> dict`
  (raises `ValueError` for a broken case or `format` > 1).
- `pack(cases: list[Path], out_dir: Path, limit_mb: float = 9.5) -> list[Path]`: one zip per case
  (`hasami-case-<name>.zip`: its files + `README.txt`); a case over the limit is split into a `scan.jpg`-only zip and
  the rest.
- `summary(case: dict) -> str`: "bottom edge 4.2 mm in; 1 box drawn" (for the status bar and the dialog).
- `default_folder() -> Path`: Documents/Hasami Scan/Crop cases (`Path.home() / "Documents"` on Windows and macOS,
  `$XDG_DOCUMENTS_DIR` or `~/Documents` on Linux).

**Session (`hasami/workflow.py`):**
- `detect_on_preview` tags the boxes it adds with `meta["auto_id"]` and keeps `self.auto_mm` (copies, bed mm). A new
  preview replaces it; with auto-crop off it is empty.
- `run_scan`, after page 1 is cropped (before processing / saving the outputs): when `settings.cases.enabled`, build a
  `CaseData` (the preview, the delivered image downscaled to <= 400 dpi, `auto_mm`, the boxes scanned) and keep it as
  `self.last_case`; when `diff_boxes` shows a correction, write it (`reason` "corrected") and remember its path.
- `save_last_case() -> Path | None`: writes `last_case` with `reason` "saved"; when that scan was already written,
  returns the existing path and writes nothing.
- Any error while building or writing a case is caught and reported through the status callback ("Could not save the
  crop case: ..."); the scan's own outputs are never affected.
- Setting off: `last_case` stays None, nothing is computed.

**GUI (`hasami/gui/app.py`, `settings_panel.py`):** the Crop tab setting and folder; Scan menu "Save this case"
(enabled after a scan while the setting is on and `last_case` exists); status bar after a capture: "Saved crop case
2026-10-15_143210 (bottom edge 4.2 mm in)".

## Section 3: the share dialog

File menu, "Share crop cases..." (enabled when the case folder holds cases), `hasami/gui/cases_dialog.py`:

- A list, newest first, one row per case with a checkbox: a thumbnail of `preview.jpg` with the final boxes drawn
  (made when the dialog opens), date, scanner, reason, `summary`. An unreadable case (broken, newer format) shows as
  "unreadable" and can still be deleted.
- Buttons: **Create zip** (`pack` the ticked cases into `<folder>/To share`, then open that folder in the file
  manager), **Delete** (ticked cases, after a confirmation), **Open folder**.
- Text above the buttons: "Each case contains scans of your item (the preview and the scanned area). Share only what
  you are happy to share. Post the zips in the tuning channel on the Hasami Scan Discord: https://discord.gg/TAAV8B9WWg".
- `README.txt` in each zip: what the files are, the case format version, and that cases are used to tune auto-crop,
  the scans kept private (never published).

## Testing (generated scans only)

- `tests/test_cases.py`: `diff_boxes` (unchanged; moved with the sign per side; turned; deleted; drawn; the 0.1 mm /
  0.05 deg thresholds); `write_case` -> `read_case` round trip (images readable at the stated dpi, boxes equal);
  a `.partial` folder is not listed; `read_case` refuses `format` 2 and broken JSON; `pack` writes one zip per case
  with `README.txt` and splits a case over the limit; `summary` wording.
- Session tests with the demo backend and a generated bed: preview -> edit an auto box -> scan writes one case with the
  right edit; no edit -> nothing written, then `save_last_case` writes it once (a second call writes nothing);
  setting off -> `last_case` None and no folder created; an unwritable folder -> the scan's outputs still saved and
  the status names the error; `scan.jpg` is <= 400 dpi; a feeder batch writes one case (page 1).
- GUI (offscreen, temporary APPDATA, temporary case folder): the dialog lists cases, zips the ticked ones, deletes
  after confirmation, shows a broken case as unreadable.

## Release

Version 0.5.17 (`hasami/__init__.py`, `pyproject.toml`). README: a short "Helping tune auto-crop" section (turn the
setting on, fix boxes as usual, share from File > Share crop cases). CLAUDE.md: a "Crop cases (0.5.17)" note. No
change to the crop engine: synthetic and real-scan benchmarks are not affected (run the synthetic ones anyway, as
for any release).

## Out of scope

Snapping edges to candidate lines and "check this side" flags (part 2, its own spec); uploading anywhere; turning
cases into hasami-data records (a tool in hasami-data, later); cases from the CLI or from opened images.
