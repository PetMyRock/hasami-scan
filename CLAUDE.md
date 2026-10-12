# Hasami Scan — project notes

A VueScan-style scanning app (Python, PySide6) whose selling point is better
auto-crop, for anime material: artbooks, Blu-ray/DVD booklets, prints,
shikishi, trading cards (not manga). Windows first (TWAIN/WIA); SANE and eSCL
also work. Licence MIT (GPL-3.0 until 0.5.2). Python package `hasami`, distribution
`hasami-scan`, commands `hasami` (CLI) and `hasami-scan` (GUI).

## Working rules
- Branch from `main`, open a PR; the maintainer merges. Bump the version
  (`hasami/__init__.py` + `pyproject.toml`) with each merged feature: the
  window title shows it.
- Tests: `python -m pytest -q` (~3 min; `tests/conftest.py` sets Qt offscreen
  and a temporary APPDATA/XDG config). Run `benchmarks/synthetic_*.py` after
  any change to the crop engine and report the numbers.
- NEVER run the GUI (`MainWindow`) from an ad-hoc script with the real
  APPDATA: it saves its settings on close and overwrites the user's. Set
  `APPDATA` (and `XDG_CONFIG_HOME`) to a temporary folder first.
- Ask before a real scan on someone's scanner (item on the glass, lid down).
- **Real scans and their ground truth are private** (copyrighted artbooks;
  users' scans are shared for testing only). They live in the private data
  repo, never in this repo. Tests here use generated scans only.
- Auto-crop changes: accuracy on the real scans of the tuned scanners comes
  first; synthetic results may drop a little for it (report them anyway).
  Keep booklet logic separate so it cannot degrade multi/single.

## Layout
- `hasami/core/autocrop.py`: detection engine (modes multi / single / booklet
  / off). `_working_image` -> `_coarse_detect` (lid model `_analyze`, fg mask,
  split touching, `fit_rectangle`, `_find_borders` white-border search) ->
  `_finish` (scale up, `_refine_edges` sub-pixel edges, deskew, margin).
  `refine()` re-refines preview boxes on the full-resolution scan (respects
  `box.meta["keep_sides"]`, skips manual boxes). `DARK_FLOOR=12` (Lab L below
  it counts as black). `_edge_falloff`: light lids darken towards the glass
  edges; measured inwards from each image edge and flattened first.
- `hasami/core/booklet.py`: booklet mode (only pages fully on the glass, outer
  edge to spine). Orientation (`_spine_across`) -> visible booklet
  (`_visible_booklet`: `_resquare`, `_trim_protrusions`,
  `_shrink_to_paper_edges` for lid shadows when a thick booklet props the lid)
  -> `_cut_sides` (touch / `_edge_reach`) -> `_find_spine` (cues: outline breaks
  weight 8/mm, gutter shading, seam, crease, text gap) -> pages -> `_top_pages`
  -> thick-book fit -> spread / page order. `booklet.DEBUG` collects profiles.
- `hasami/core/toppage.py`: crops end at the TOP page's printed edge (page-stack
  bands and cover boards excluded). Per side a strip cut into 16 segments,
  Lab-gradient peaks, Hough vote for layer lines (support >= 60%), chain from
  the outermost line inwards (gaps <= 1.0 mm always, <= 2.2 mm when the band
  looks like paper or the stack continues, <= 6 mm when 2-15 L darker than
  light paper). Spine sides snap to the fold's crease (3 mm; 1 mm for "shade"
  spines).
- `hasami/core/thickbook.py`: `fit_spread` for thick books on a light lid
  (black bed -> None); robust per-page lines, validation (inliers >= 0.5, rms
  <= 0.6 mm, pages >= 2 mm from the border); wins when it answers.
- Booklet options: `crop.booklet_output` "pages" | "spread" (union of both
  pages), `crop.booklet_rtl` (right page first, as seen after `crop.rotate`).
- `hasami/core/geometry.py` (CropBox: angle in (-45, 45], positive = clockwise
  on screen), `processing.py` (levels, WB, negatives, descreen, dust, sharpen),
  `output.py` (JPEG/PNG/TIFF 8/16-bit, WebP, PDF, raw TIFFs).
- `hasami/devices/`: `twain_backend.py` + `twain_helper.py` (helper process,
  see TWAIN), `wia_backend.py`, `escl_backend.py`, `sane_backend.py`,
  `file_backend.py`, `demo_backend.py` (synthetic beds, tests only: hidden in
  the GUI, `hasami list --all`), `choose.py` (best connection first; hides
  VueScan's TWAIN source, the WIA-to-TWAIN bridge and WIA twins unless
  `input.show_all_devices`; two TWAIN drivers with one name -> the 32-bit one
  first, e.g. the old EPSON Scan before Epson Scan 2).
- `hasami/workflow.py` (Session: preview -> boxes in bed mm -> scan the crop
  area -> refine -> process -> save; `input.device` "" = best scanner),
  `cli.py`, `gui/` (`app.py`, `preview.py`, `settings_panel.py`, `worker.py`).
- `hasami/settings.py`: JSON in `%APPDATA%\HasamiScan` (macOS
  `~/Library/Application Support/HasamiScan`, Linux `~/.config/hasami-scan`);
  copies settings once from the old `scanner` folder (the app was called
  "Scanner" before 0.4.2).
- `hasami/demo.py`, `demo_anime.py`: synthetic generators with ground truth.
- `packaging/`: Windows installer (see Installer). `.github/workflows/installer.yml`.

## TWAIN (Windows)
- `twain_helper.py` is a standalone script (stdlib + `pytwain==2.3.0`), one JSON
  request per run, one JSON line back. Many drivers are 32-bit only (e.g. the
  old EPSON Scan), so it runs in a 32-bit process: dev runs use a 32-bit Python
  found through `py -0p` (or `input.twain_python`); the installed app ships
  `twain32\twain_helper32.exe` and `twain64\twain_helper64.exe` (frozen ->
  `bundled_helpers()`; `helper_command` passes the request as the only arg).
- Device ids: `twain:<name>`, `twain:<name> [<family>]` for a second driver of
  the same name (opened by identity).
- Scan setup: transfer chosen BEFORE area/resolution (native transfers are
  limited to ~400 MiB in a 32-bit process; a refused resolution switches to a
  file transfer); frame set before and after the resolution; auto-size/border
  detection/paper sizes switched off; raw -> brightness/contrast 0.
  `probe_dpi` finds resolutions above the listed max (they depend on the area).
- Live view: memory transfer (own ctypes structs), strips written to
  `<out>.raw`, read by `RawStrips` while the helper runs.
- Cancel must be graceful: killing the helper mid-transfer hung an Epson V600
  until its power cable was pulled. A cancel file asks the helper to end the
  transfer the TWAIN way (MSG_ENDXFER / MSG_RESET / WM_QUIT, close source and
  DSM); `<cancel>.busy` while inside a driver call; kill only after 15 s not
  busy or 600 s in all (then a warning with recovery advice).
- The helper's answer goes to temp files, not pipes (a big answer deadlocked).
- Driver window ("TWAIN: show the scanner's own window"): after such a scan the
  helper reads the driver's custom caps (>= 0x8000) and later hidden-window
  scans replay them (`twain_driver_settings.json`).
- TWAIN ProductName stays "Scanner" (drivers may keep settings per program).
- The fake driver in `tests/fake_twain` models TWAIN states, devices, limits
  (`FAKE_TWAIN_*` environment variables).
- Epson notes: the old EPSON Scan applies its saved "Current Setting" (e.g.
  No Color Correction) to hidden-window scans. WIA on the V600: odd motor
  pitch, no live view, max 1200 dpi -> always prefer TWAIN.

## GUI notes
- VueScan's workflow and look: Preview / Scan, crop boxes (drag edges anywhere
  within 6 px, corners, inside; rotate handle; Skew field with arrow keys),
  rotate buttons at the bottom right turn the whole preview and set
  `crop.rotate` for saved scans, Cropped area tab, live view (grey placeholder
  filled strip by strip; one crop box -> shown straightened). Do NOT darken the
  preview outside the crop box.
- Qt `clicked` passes a `checked` bool: slots with default args take `*_` first.
- Empty preview: `START_HINT`, or `NO_SCANNER` when nothing is found.

## Auto-crop off (0.5.4, owner)
Switching Auto-crop to Off removes EVERY crop box (drawn ones too: `App.on_setting_changed`
-> `clear_boxes`), and Off never adds a box by itself (`Session.detect_on_preview` adds none;
before, a whole-bed box). A box drawn by hand in Off is still used (owner chose this);
without one the whole bed is scanned (`_scan_area` None, `detect` returns the full image).

## Installer
`packaging/build.py --python32 <32-bit Python with pytwain + pyinstaller>`:
icon (`make_icon.py`) -> PyInstaller `hasami.spec` (onedir: `HasamiScan.exe`
window + `hasami.exe` console) -> `twain_helper.spec` for 32 and 64-bit ->
licence texts (`licenses\`: the app's MIT licence, the libraries' texts, the LGPL-3.0 notice for Qt
+ `packaging/licenses/GPL-3.0.txt`, which the LGPL builds on) -> Inno Setup
`installer.iss` (per-user, `%LOCALAPPDATA%\Programs\Hasami Scan`, no admin) ->
`dist\HasamiScanSetup-<v>.exe`. CI: a `v*` tag builds it and attaches it to a
release (tag must equal the version). In Git Bash, iscc's `/D` arguments need
`MSYS_NO_PATHCONV=1`. Not done: code signing.

## Update check (0.5.1)
`hasami/updates.py`: GitHub API `repos/PetMyRock/hasami-scan/releases/latest`
(urllib, 10 s timeout) -> `Release(version, page, installer)`; `is_newer`
compares numeric versions; `due()` = `updates.check_on_start` and 24 h since
`updates.last_check`; `should_offer` skips `updates.skipped` on start only.
GUI: automatic check only in the installed (frozen) app, in the background,
silent on errors; Help > Check for updates (always, reports errors / "latest")
and the checkable "Check for updates on start". Dialog: Download (opens the
installer link, else the release page) / Skip this version / Later; "What's
new" links the release page. Releases must keep attaching
`HasamiScanSetup-<v>.exe` (the CI does).

## Scanner profiles (0.5.2)
`hasami/core/profiles.py`: frozen `Profile` dataclass = the tuned numbers and
switches (lid `dark_floor_l`; booklet `shrink_to_paper_edges` (V600 propped
lid), `shrink_margin`, `outline_weight`, `spread_confidence`, `middle_window`,
`cover_split_score`, `lid_all_round_split_score`; all of toppage's numbers;
thickbook `thick_books` + its validation limits). `PROFILES`: "default" and
"epson-v600" (match `perfection\s*v600`, `gt-x820`), identical values at the
start. Active profile = a ContextVar: `autocrop.detect` / `refine` wrap the
engine in `profiles.use(settings.profile)`; engine code reads
`profiles.current()` (toppage via `_profile()`). `crop.profile` "auto" =
`Session.profile()` matches the open scanner's name/vendor/model (else
default); `Session.crop_settings()` passes the resolved name. GUI: Crop tab
"Scanner profile" (Automatic / each profile), status bar shows
"[profile: ...]"; CLI `--profile`. Tuning a new scanner: add a profile (and a
switch for new behaviour, default off for the others), never change another
profile's values; still run the real-scan benchmarks of all tuned scanners
(shared code). More inline numbers can move into `Profile` as tuning needs
them (keep the old value as the default so nothing changes).

V600-only switches (0.5.5, artboard book with a thin page stack: very thin sheets 0.5 mm apart,
lower pages' art columns, green cover, propped lid; 9 scans, all pass; the 57 + 26 unchanged
case for case): `min_edge_step` 0.6 (the shrink treats a side as an existing paper edge only for a step of
>= 0.6 L*/px in the working image: the lid's soft brightness hump, 0.1-0.3, kept the box 3 cm
out in the lid); `fine_stack` (`toppage._through_fine_stack`: after the chain, faint lines from
a lower threshold, 25th percentile of the gradient, continue it inwards when they form a run of
>= 5 lines <= 0.8 mm apart; the run stops at 3 half-supported lines in a row = texture in dark
print); `crease_step` (`toppage._step_out`: a broad one-sided valley - dark print up to the
fold, facing page rising slowly - is replaced by the half-way point of the step up to the
facing page, used only when most segments show it, then they may disagree by 0.5 mm; a sharp
crease line rises at once and is left alone).

## Per-scanner tuning
Users ask for tuning on the Discord server (https://discord.gg/TAAV8B9WWg) and share scans (an empty bed and
real items). Truth is marked by eye at full resolution (5 points per side),
stored with the scans in the private data repo; auto-crop is tuned until the
scans pass (count right and IoU >= 0.95), checking the other tuned scanners
do not get worse. Plan: per-scanner profiles picked from the driver's model
name, with a drop-down to override. Tuned so far: Epson Perfection V600 (57
scans + 26 thick-book spreads, all pass), Epson V39 (older scans).

## Ideas not done yet
- Code signing (unsigned installers show SmartScreen's warning).
- Synthetic horizontal booklets: a blank page margin at the image edge is taken
  for lid shadow (known xfail in `tests/test_booklet.py`).
- Dark paper with a lighter page-stack band keeps the band.
- White paper running off a white lid with a blank strip at the border.
