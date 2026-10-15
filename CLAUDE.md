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
- Docs-only changes (CLAUDE.md, README, other notes; no code, tests or packaging) go straight to
  `main`, no PR (maintainer, Oct 2026).
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
- Small regressions are acceptable for a clear real-scan win (owner, Oct 2026; truth is marked by
  eye, ~0.1 mm is its noise) when every worse side: moves <= ~0.2 mm; crosses no threshold (nothing
  within 0.5 / 1 mm goes outside it, no scan stops passing); does not cut print (an inward move
  > 0.3 mm stops the change: ask the owner); and the bare 200 dpi preview box gets no visibly worse.
  Name every regression in the PR anyway.

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
- `hasami/core/candidates.py`: context-local collector (`collecting()`); `toppage.top_page` records each
  side (frame, line pools layer / faint / strong, the rules' choice and rule, crease, off-image) and
  `_refine_edges` its profile peaks. Off = no cost, no change. `linescore.py`: mm-based strip patches
  (16 segments x 64 x 0.1 mm, coarse 32 x 0.5 mm, Lab + inside flag) and features of a candidate: the
  learned line classifier's input (training data built in the private data repo).
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
0.5.6: `stack_by_segment` (`toppage._stack_gap`): a lower page sticking out 1.7 mm showed its
own art column along ~half the side, plain paper elsewhere; the strip's median sat on the
`max_darker_l` limit and a 1 px shift of the crop-area scan decided (owner's app crop kept the
column). Now also a stack gap when the inner line has >= 85% support and >= 25% of the segments
pass the colour test alone (print on the top page leaves no edge where it stops). Case: owner's
F:\JPG\Wrong crop (0023 whole bed, 0024 the bad crop); 57 + 26 + 9 unchanged case for case.
0.5.7: `stack_jump` (`toppage._jump_stack`): 0.5.6 fixed only the scan-time refine; the 200 dpi
PREVIEW box still sat on the green cover (owner saw it). At 200 dpi the 0.45 mm sheets blur into
one band and the top page's edge is a layer line only next to the dark art column (9/16 < 60%).
Now the chain's end moves in to the innermost faint (25th-percentile) line with >= 85% support
within 8 mm when its outside 1.3 mm is darker than the paper inside in >= 75% of the segments and
stack paper in >= 25% (a line printed on the top page has its own margin outside). Lesson: check
the preview box AND the scan before calling a crop fixed.
0.5.10 (owner's test scans `New data`, 10 pages of the same book, stacks ~10-12 mm deep):
`depth_mm` 14 with `layer_depth_mm` 10 (the usual rules see exactly the old 10 mm columns - a
deeper strip shifted the Hough bins and the noise level and moved unrelated sides by ~1 mm;
`_strip` adds the deeper columns in front) and `fine_strong_lines` (`_through_strong_lines` on
the 14 mm strip: lines with >= 85% support from a 10th-percentile threshold; a gap is crossed
when its strip passes `_stack_gap` (wide gaps need only `strong_darker_l` 1 L darker: a plain
lower sheet), or when <= near_mm and one more line follows within gap_mm (a lower page's narrow
art column; at 200 dpi the top page's edge came 1.8 mm after the last one),
never because lines follow across a wider gap (masked streaks in art); only with a strong line
outside the start (a stack, not a cover on the lid), and it must end in front of light paper
(L >= 70) or it backs up). Crease (all profiles): samples outside the image are ignored - the
app's crop-area scan ends 3 mm past the box, at the end of the crease search, and the black
beyond looked like a valley (0004: fold 2.4 mm into the facing page). Benchmark: scenario `app`
(preview -> crop-area scan + 3 mm -> refine) and `max_dpi` per record (test scans scored at
400 dpi). New data 30/30 within 1 mm (was 76% within 0.5 mm, worst 5.2 mm), and the bare 200 dpi
preview box within 1 mm too (was up to 10.4 mm out on the stack side); 57 + 26 + 9 only
better (V600 0003, 0008; KLK 0017). Leftovers: folds 0.5-0.9 mm short on soft/broad creases
(0002, 0004 via the preview, 0007).
0.5.11 (owner's second batch, `Test Data_0011`-`0035`: two more KLK Vol.6 pages, four covers alone,
19 whole spreads of the thick KLK Vol.7 Key Art Collection; 0016/0028 skipped, owner): five V600
switches. `shrink_lid_hue` (the lid-shadow shrink gives up only lid-coloured strips: a plain green
cover running off the bed passed as shadow, its sheen line ended the booklet, no spine, the box ran
8.7 mm into the cover). `walk_after_passes` (after the
faint-line passes the chain goes on over edges <= `near_mm` apart: a lower page's 0.8 mm art column
after 7 mm of white sheets; wider gaps not, see below). `cover_rim` (`toppage._past_cover_rim`: a
uniform coloured band <= `gap_mm` between lid and page, neither paper- nor lid-coloured, with a
full-length line inside and light paper: the hard cover's rim, 1.2 mm). `paper_colour_edges` 15
(`autocrop._past_printed_line` in the edge refinement: coloured paper on both sides of the chosen step
means a printed line, the edge is the steepest chroma drop to neutral lid further out; orange cover on
the grey lid, ~9 L* step, the drawing's line 1.1 mm inside won). `lid_all_round_split_score` 5 (a
cover whose shadow made two sides fail the paper-edge test was split at a printed line scoring 3.0;
V600 whole spreads score >= 13). Tried and dropped: chroma added to the refine's edge image (helped 5
V600 cases, but the bluish lid shadow beside white covers became the edge: 4 covers ~1 mm out), and to
the paper-edge test (no effect); letting the shrink pass steps too close to the border to check (0016/
0028: a lid step 2.6 mm from the bed's edge kept the box on the whole bed; fixed V600 0008 full-scan
+1.5 mm but broke 0014's preview path +0.7 mm). Known risk: once a chain reaches the top page it crosses a
paper-coloured gap <= `gap_mm`, so a frame line printed that close inside is taken; `cover_rim` starts
chains on more pages (`test_v600_profile_crops_a_thick_stack_off_in_a_preview` keeps it off for that
reason). The thick spreads: one rectangle cannot follow the two pages (turned ~0.45 deg against each
other): stack wedges in the corners stay, by the owner's spread rule (no print cut).
Benchmark (scenario `app` = the app's path, plus `preview` and `real`): the 23 new scans all within 1 mm
(85% within 0.5 mm; on main worst 8.7 mm: 0012's fold, 0013 -1.4, 0011 +0.85); 57 + 26 + 9 only
better (V600 0041, 0042 by up to 0.9 mm); synthetic identical. Leftovers: 0014's top +0.77 mm on the
app path (cover on the black bed, a light grey line along its edge: the scan-time top-page search
starts at its far side; as on main); 0012's bare 200 dpi preview box 2.7 mm out on the stack side (as
on main; the sheets do not resolve at 200 dpi; the scan is right).
0.5.13 (step 2 of learned cropping: the line classifier's held-out fixes pointed at one rule):
`stack_shadow_l` 20 (`toppage._paper_gap`). The chain stopped 1-2 mm out on V600 stacks where the sheet
just below the top page sticks out 1-2.2 mm: neutral (a*b* 1-3) but 16-19 L* darker (propped lid,
further from the glass), over `max_darker_l` 15 = "print". Gaps <= `gap_mm` now accept up to 20 L*
darker; wider gaps keep 15. Fixed: V600 0001 (all paths), 0005, 0008, 0015, 0054 and 0056 (thick
stacks, +5-7 mm; 0056's bare preview box still +5.3), New data 0002 and KLK6 0019 / 0021 / 0023
(preview). Worse: 0056's preview bottom -0.68 -> -0.73, 0054 / 0056 app sides by <= 0.1 mm. Thick
books and the other New data crops identical; synthetic identical. Left: stops where the top page has
dark print at its edge (the paper sample inside is print: V600 0009, 0010, 0043; KLK 0014) and bluish
cover bands (0044, 0046).
0.5.14: `dark_edge_l` 15 (`toppage._dark_edge`). Dark covers with the propped lid's grey shadow beside them
(V600 0009, 0010, 0043) and top pages printed dark to the edge with the sheets below showing (0011, KLK
0014): a neutral band 1-1.7 mm wide, L* 67-74, lighter than the print inside, so it never passed as stack
paper. A gap <= `gap_mm` to a line with >= 85% support is now crossed when the band is neutral and >= 2 L*
darker than the lid outside the chain's first line (a white margin on the glass is brighter) and the
inside is >= 15 L* darker than the band; the walk ends there. Fixed from +0.8-1.5 mm to about -0.2 on the
real scan and the bare 200 dpi preview box. Accepted regression (owner): 0043's other long side on the
preview -> refine path -0.23 -> -0.87 mm. Not this rule: fixing 0043's other sides moves the preview box,
and in the refine frame a line 0.6 mm inside the blue cover (its rim) becomes a layer line that the
always-chained <= `near_mm` walk takes. Checked whether that walk could get a colour test: of 666
<= 1 mm steps on the stack sides, 654 are needed, and 0043's looks like V600 0057's needed steps into a
lower page's art (neutral L* 66-72 outside, dark bluish L* 40-45 inside); support does not separate
them either. Also 0010 / 0043 sides by <= 0.1 mm. Thick books, New data, synthetic identical.
0.5.15: `rim_print_inside` (`toppage._past_cover_rim`). A navy hard cover's rim (1.3-1.45 mm, L* ~43, b* -13)
beyond a top page printed dark to its edge (V600 0044) or beyond a lower page's art column (0046) was kept:
`cover_rim` wanted light paper (L* >= 70) inside the rim, and found L* 36-60. Now the rim's other tests
decide (uniform along the side, hue unlike the lid's and the inside's, full-length inner line, <= `gap_mm`).
Fixed: 0044 real +1.20 -> -0.45 (truth line -0.20), 0046 preview -> refine +2.02 -> -0.24 and its bare 200 dpi
preview box (+2.2 mm). Nothing worse; Thick books, KLK, New data, synthetic identical. Left: 0044 preview (the
rim's inner line 10/16 at 200 dpi) and refine / app (the chain starts on a shadow line beyond the rim, so the
rim test never runs; testing the rim after the walk too would cut coloured frames printed on top pages).
Wider risk: a uniform coloured frame <= 2.2 mm along a top page lying on the lid, with art inside, is now
taken for a rim.
0.5.16 (re-measure on 0.5.15: hasami-data `benchmarks/learned/remeasure_2026-10-14.md`, 79 of 1952 sides over
0.5 mm, 45 of them folds): `crease_retry_mm` (1, 2). When no crease agrees within the spine's reach (3 mm), other
dark lines a few mm away split the segments' votes; the crease is looked for again within 1 mm, then 2 mm.
Fixed: V600 0002 fold +-0.9 -> +-0.2 (every path), 0003 +1.03 -> +0.17, 0031 +1.38 -> +0.38, 0018 +0.41 -> +0.16,
0039 refine / app +2.02 -> +1.30, New data 0003 +0.79 -> +0.38, 0008 -0.56 -> -0.33, KLK6 0017 +0.73 -> +0.14;
folds within 0.5 mm 359 -> 373 of 404. Worse (<= 0.11 mm): New data 0004's preview fold -0.34 -> -0.45, KLK7 thick
spreads' union sides by up to 0.1 (their folds now snap), KLK Vol.5 folds by <= 0.05, V600 0005 +0.06. Left:
saddle-stitched centre spreads (V600 0040 -2.5, 0039 preview +2.0): flat cream paper, no crease along about half
the fold; the only strong valley is where the facing page's art begins, 0.3-0.5 mm off. A staple-line cue was
considered and dropped (owner): only these 2 of the 10 stapled booklets (0029-0040; the others show a crease and are
right), the staples are not the signal there, and it would risk every other fold; creases found on the wrong valley (V600 0016 ~0.9, KLK6 0019 / 0022 / 0023
0.5-0.8, New data 0002 / 0004 soft folds ~0.65). Looked at and left (owner): on all 22 such sides the chosen valley is lopsided
(one side rises >= 2x the other) and on 18 the truth is 0.7-1.0 mm towards the steeper side (the junction with the
facing page, not the valley's bottom), but 39 of the 277 right folds are as lopsided and New data 0002's truth is on
the other side: no cue tells which convention a fold follows.
Thick books cut 1.1-1.8 mm inward (Key Art Collection 0006 right, 0025 / 0026 left; looked at and left, owner):
`thickbook.fit_spread` starts a page where paper stays above P - 5 (P = the spread's middle, 75th percentile);
there the top page's outer 1-3 mm reads 6-8 L* darker (79-81) before the sharp drop to the stack or board. The cut
part is blank cream margin, no print. Following the paper outwards to the drop (within 6 L*) fixed 0006 / 0026 but
walked onto the stack on 9 Key Art Collection and KLK7 sides (up to +1.25 mm; 11 better, 31 worse): the darker
margin and the stack band are equally bright.
Multi mode, V600 misc (looked at and left, owner, Oct 2026; mm without the 0.25 crop margin): 0022, a dark grey
print on the black bed, ends on its two short sides in a soft ramp (L* 29 -> 5 over ~1.5 mm) before the black lid;
the truth is the ramp's foot, the edge refinement takes its steepest part: cut -0.54 / -0.70 (real), -0.98 / -0.41
(app). The cut part is the card's dim edge, no print. Following the ramp out to the lid would act on every item on
the black bed, and the sharp-edged ones (0024-0026, 0041, 0042) also tail off 0.3-0.4 mm past their truth. 0021's
second print +0.91 is the real scenario only (a grey lid-shadow band beside it); the app path is +0.38.

## Per-scanner tuning
Users ask for tuning on the Discord server (https://discord.gg/TAAV8B9WWg) and share scans (an empty bed and
real items). Truth is marked by eye at full resolution (5 points per side),
stored with the scans in the private data repo; auto-crop is tuned until the
scans pass (count right and IoU >= 0.95), checking the other tuned scanners
do not get worse. Plan: per-scanner profiles picked from the driver's model
name, with a drop-down to override. Tuned so far: Epson Perfection V600 (57
scans + 26 thick-book spreads + 9 artboard pages + 33 test scans, all pass), Epson V39 (older scans).

## Test scans (0.5.8, owner)
Toolbar/Scan menu "Test scan" (Ctrl+T, `MainWindow.do_test_scan` -> `Session.run_test_scan`):
whole bed at `output.test_dpi` (400), crops and scan dpi ignored, colour, 8-bit, the same raw
flag as normal scans; saved UNCROPPED and UNTURNED (scanner frame: no `--turn` needed in the
benchmark) as TIFF `output.test_prefix` + number in `output.test_folder`. Preview and boxes
are kept (live view fills in the whole bed, `end_live` restores them). File menu "Open test
scan folder". Owner sends these when a crop is wrong.
0.5.9: a new Preview keeps only boxes DRAWN from scratch (`meta["drawn"]`, set in
`PreviewView` when a box is dragged out); auto-crop boxes are replaced even when edited
(drag / Skew made them `manual`, and before they came back on the next page - owner, after
fixing a box and pressing Test scan). Auto-crop on the same preview still keeps edits.

## Crop cases (0.5.17, owner)
Opt-in (`AppSettings.cases.enabled`, Crop tab; folder `cases.folder`, "" = Documents/Hasami Scan/Crop cases).
`hasami/cases.py` (no Qt): a case folder = `preview.jpg` (whole bed, preview dpi) + `scan.jpg` (the first page's
uncropped scanned area, <= 400 dpi, both JPEG q95, scanner frame) + `case.json` (format 1: scanner, profile, crop
settings, auto and final boxes in bed mm, `diff_boxes` edits: moved per side + = outwards / deleted / drawn).
`Session.detect_on_preview` tags auto boxes `meta["auto_id"]` (unique per session) and keeps `auto_mm`;
`run_scan` -> `_capture` after page 1 keeps `last_case` and writes it when corrected (never in Off mode; boxes
carried from an earlier page, `meta["carried"]`, are not edits); errors become a status line, never a failed scan.
Scan > "Save this case" (`save_last_case`), File > "Share crop cases..." (`gui/cases_dialog.py`: one zip per case
in `To share`). Spec `docs/specs/2026-10-15-crop-cases-design.md`. Next: a hasami-data tool to turn cases into truth
records.

## Edge snapping, pinned sides and flags (0.5.19, owner)
`hasami/core/sidelines.py` (read-only, never called by `detect` / `refine`): `side_lines` = toppage's line finding on a
wider strip around each FINAL box (paper sides 14 mm in / 8 mm out at the three thresholds; spine sides the fold's
valleys within 4 mm); `box_lines` maps a spread's pages onto its sides. `Session.detect_on_preview` keeps them per auto
id in bed mm (`snap_lines`, booklet mode only, moved in by the crop margin); `CropItem` snaps a dragged side within
8 screen px (lines tilted > 1.5 deg ignored; Shift = free). Every edit records `meta["pinned_sides"]` (side drag: that
side; corner: two; move / rotate / Skew: all four) and `snapped_sides`; `autocrop._refine` refines an auto box with 1-3
pinned sides and `geometry.pin_sides` puts them back (angle kept); drawn, unpinned-manual and fully pinned boxes stay
untouched. A pinned booklet box's other sides can sit up to ~0.15 mm from an unedited refine's (top_page refits all
sides from its start box). Flags: `sidelines.CHECK_RULE` = no crease line within 0.5 mm of a spine, or >= 8 lines
within 3 mm; amber side + "?" + tooltip, "N sides to check" in the status bar, gone once the side is moved.
Measurement (hasami-data `benchmarks/learned/snapcheck_2026-10-15.md`, bare 200 dpi preview box, 116 V600 booklet
records, 624 sides, 18 > 0.5 mm off): candidates cover 11 / 12 wrong paper sides but 0 / 6 wrong folds (saddle-stitched
and wrong-valley folds are not at a valley; 21 / 129 spines get no valley line at all); flags pass the preset gate on
held-out books (11.2% flagged, 12 / 18 caught) but only 12 of 70 flags are on wrong sides (`crowd` fires on right stack
sides). Thin data: revisit both rules with crop cases (`case.json` keeps `check_sides`, `pinned_sides`,
`snapped_sides`). `side_lines` takes ~170 ms per page at 200 dpi (toppage's `_layer_lines`). Spec
`docs/specs/2026-10-15-edge-snapping-design.md`.

## Next: learned cropping (planned, owner approved Oct 13 2026)
0.5.12: candidate collector and patch/feature extraction (no crop changes); dataset + baseline in
hasami-data benchmarks/learned.
Start here after 0.5.11. Two steps, each compared against the current rules on the same benchmarks
before anything replaces them:
1. **Line classifier** (first). Most real-scan failures are one judgement: which candidate line is
   the top page's edge: a stack sheet, a lower page's art column, a cover rim, a printed line, a lid
   shadow. Hand rules are at their limit: in 0.5.11 two V600 switches each fixed one case and broke
   another and were dropped. The engine already produces the candidates: `toppage.top_page` (layer
   lines per side: `_layer_lines`, then `_chain` / `_walk` and the faint-line passes) and the edge
   refinement (`autocrop._fit_side`, `_pick_edge`). Learn a scorer on a small strip patch around each
   candidate (e.g. ~6 mm across x the side's length, resampled), keeping the rules as a fallback.
   Training data: every marked side gives one positive (the truth line) and several negatives (the
   other candidates): about 500 sides from the real scans below, plus the synthetic generators
   (`hasami/demo.py`, `demo_anime.py`, `synth_booklet`) with heavy augmentation.
2. **Segmentation model** (then). At preview resolution (200 dpi): top page / page stack / cover /
   lid, the classical code fitting the edges. The final edge stays with the sub-pixel refinement:
   0.5 mm is ~3 px at 200 dpi, beyond a whole-bed network's resolution.

Status (Oct 14 2026): both PARKED; the rules (0.5.13-0.5.16) took the wins instead. Step 1: the classifier's
leave-one-book-out gates failed (a held-out book whose stack type no other book shows broke it); its held-out
fixes pointed at the rule fixes `stack_shadow_l` / `dark_edge_l` / `rim_print_inside`; branch feat/line-classifier
stays local, unmerged. Step 2: data check only (hasami-data `benchmarks/learned/seg`, report
`segcheck_2026-10-14.md`): a ~119k-parameter U-Net on 200 dpi tiles, nudging the bare preview box to the rules'
nearest candidate within 1 mm, fixed 4 of the 5 deep-stack preview boxes (New data 0012, V600 0008 / 0015 / 0016) but
broke stacks that look like the page (pages printed green to the edge over same-green sheets, cream sheets under a
cream page): 11 sides better, 14 worse; FAIL. Adding the thick-book stacks to training cut the worse sides 23 -> 14,
so more stack kinds help. Restart only with deep stacks (>= 3 mm) from 5-6 more books, especially page-like ones,
whole-bed scans; then rerun the check unchanged before any app work (spec Section 2: V600 switch `seg_preview`,
ONNX Runtime, preview only).

Deployment: ONNX Runtime on CPU (~15-20 MB more in the installer; check its licence and the
model's for MIT compatibility). Training needs no GPU at this size. The scans and their truth stay
private (hasami-data); whether to ship weights trained on them is the owner's call. Tests in this
repo must keep using generated scans only. Do it behind a profile switch first (V600), like the rules.

Data and benchmarks (all on the maintainer's machine):
- Private data repo `F:\HasamiData` (github PetMyRock/hasami-data): truth in
  `benchmarks/real_booklets_proposals.json` (records: `outline`, `pages` as `[cx, cy, w, h, angle]`,
  `dpi`, `group`, optional `max_dpi`, `output`), tools per batch in `benchmarks/klk6`, `newdata`,
  `newdata2` (READMEs there explain how truth was read).
- Scans: `F:\ScannerTraining\Training booklets` (read only: V600 Scans 57, `Thick books` 26 beside it),
  `F:\JPG` (KLK Vol.6 artboard, 9), `E:\New data` (owner's test scans 0001-0035, unturned).
- Runner: `python F:/HasamiData/benchmarks/real_booklets.py "F:/ScannerTraining/Training booklets"
  --only "V600 Scans" --scenarios real,preview --turn 0 --profile epson-v600 --jobs 2 --out OUT`
  (code under test via `PYTHONPATH`; groups "V600 Scans", "Thick books", "Kill la kill" at `--turn 0`
  and `-1`; the test scans with `--only "New data" --scenarios real,preview,app --turn 0`: `app` is
  the app's own path, preview -> crop-area scan + 3 mm -> refine). Always `--jobs 2`.
- Compare: `benchmarks/klk6/side_mm.py OUT/results.json` (signed mm per side), `side_diff.py OLD NEW`
  and `compare.py OLD NEW` (run with `PYTHONPATH=F:/HasamiData/benchmarks/klk6`). Judge sides in mm
  (within 0.5 / 1 mm), not only IoU >= 0.95; check the bare 200 dpi preview box too (the owner sees it).
- Current V600 state (0.5.16): all 57 + 26 + 9 + 33 pass; test scans 85% of sides within 0.5 mm, all
  within 1 mm. Known leftovers to beat: see the 0.5.10 and 0.5.11 notes in "Scanner profiles".

## Ideas not done yet
- Code signing (unsigned installers show SmartScreen's warning).
- Synthetic horizontal booklets: a blank page margin at the image edge is taken
  for lid shadow (known xfail in `tests/test_booklet.py`).
- Dark paper with a lighter page-stack band keeps the band.
- White paper running off a white lid with a blank strip at the border.
