"""Scanner profiles: the auto-crop's tuned numbers and switches, per scanner model.

How well auto-crop works depends on the scanner (lid colour and shape, how the
lid darkens towards the edges of the glass, the light, the edges it draws), so
each tuned scanner model has its own :class:`Profile`.  The app picks the
profile from the scanner's model name (``crop.profile = "auto"``); the Crop tab
can choose one by hand.  Scanners without their own profile use ``default``.

Tuning a new scanner = a new entry in ``PROFILES`` (and, when it needs new
behaviour, a switch here that only its profile turns on), so the crops of the
other scanners stay exactly as they are.  Check them anyway with the real-scan
benchmark: shared code still changes.

The engine reads the active profile with :func:`current`; ``autocrop.detect``
and ``autocrop.refine`` make ``crop.profile`` active while they run.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Profile:
    name: str = "default"
    #: Shown in the Crop tab.
    label: str = "Default (untuned scanners)"
    #: Regular expressions matched (case-insensitive) against the scanner's name and model.
    match: tuple[str, ...] = ()
    #: Real scans this profile was tuned on (for the README / About; informative only).
    tuned_on: str = ""

    # ---- lid model (autocrop) ----
    #: Lab L below this counts as black (dark fabric or a jacket on a black bed is not an item).
    dark_floor_l: float = 12.0
    #: Edge refinement takes the strongest side-averaged step; a printed line along a coloured
    #: paper's edge can beat its faint step to the lid (orange cover on the grey lid: the drawing's
    #: line 1.1 mm inside won).  With this limit (> 0, a*b* chroma), when there is coloured paper
    #: on both sides of the chosen step, the edge is the steepest drop to neutral lid further out
    #: (white paper and grey shadows never trigger it).
    paper_colour_edges: float = 0.0

    # ---- booklet mode (booklet.py) ----
    #: Light lids: pull the booklet's sides in to the outermost paper edge (a thick
    #: booklet props the lid up; its shadows and the black band at the glass edge
    #: would join the booklet).  Tuned on the Epson V600.
    shrink_to_paper_edges: bool = True
    #: Border margin of that shrink, as a share of the smaller box side (px at least 4).
    shrink_margin: float = 0.012
    #: Weight of the paper-outline spine cue (notches/jumps of the outline), per mm.
    outline_weight: float = 8.0
    #: A booklet with nothing cut off is taken as a whole spread in the landscape frame
    #: only when both pages reach this confidence (spine score / 5).
    spread_confidence: float = 0.99
    #: Where a whole spread's spine is looked for first (share of the booklet's length).
    middle_window: tuple[float, float] = (0.455, 0.545)
    #: A square-ish booklet with four verified paper edges is split only by a spine this clear.
    cover_split_score: float = 4.0
    #: A booklet with lid all round (nothing cut or touching) is split only by a spine this clear.
    lid_all_round_split_score: float = 2.5
    #: Light lids: a side that already lies on a paper edge (lid beyond it) is not pulled in; that
    #: edge must be a brightness step of at least this many L* per pixel of the working image
    #: (about 1000 px long; 0 = any step above the noise).  On a smooth lid the noise is tiny, and
    #: the soft brightness hump of a propped-up lid (0.1-0.3) passed as a paper edge: the box
    #: stayed ~3 cm out in the lid (KLK artboard).
    min_edge_step: float = 0.0
    #: That shrink gives up only lid or its shadow: with this (> 0) a strip whose median colour is
    #: further than this from the lid's in a*b* (shadows on a grey lid stay grey) is kept (a green
    #: cover running off the bed is uniform enough to pass as shadow; its sheen line became the
    #: booklet's end and the box ran 8.7 mm into the cover).
    shrink_lid_hue: float = 0.0

    # ---- top page (toppage.py): crops end at the top page's printed edge ----
    depth_mm: float = 10.0  # how far inside the side a page-stack band may reach
    out_mm: float = 1.0  # and outside it
    start_out_mm: float = 0.6  # the chain starts at a layer line between this far outside the side
    start_in_mm: float = 4.5  # and this far inside it
    wide_gap_mm: float = 6.0  # a wider gap counts only when a little darker than light paper
    near_mm: float = 1.0  # page edges this close together are always one stack
    gap_mm: float = 2.2  # largest gap between two layer lines of one stack (beyond near_mm)
    darker_l: float = 2.0  # a lower page sticking out is at least this much darker ...
    max_darker_l: float = 15.0  # ... and at most this much (more = print or dark paper)
    lighter_l: float = 3.0  # a stepped-out lower page is not lighter than the top page
    max_hue_diff: float = 8.0  # a*b* distance: more = coloured print, not paper
    shadow_l: float = 4.0  # lid shadow beyond the paper is at least this much darker
    light_paper_l: float = 70.0  # the "darker lower page" rule only on light paper
    segments: int = 16  # each side's strip is cut into this many segments
    support: float = 0.6  # share of the segments a layer line needs
    max_tilt: float = 0.02  # layer lines may turn this much (rad) against the side
    crease_mm: float = 3.0  # the spine side snaps to the fold's crease within this distance
    crease_support: float = 0.6
    #: Dark print running into the fold next to a light facing page shows no crease line: the
    #: darkest point is the gutter shading on the page itself (KLK artboard: ~1 mm inside the
    #: fold).  Then the fold is the step up to the facing page (half-way up), when that side
    #: rises clearly more than the page's own side.
    crease_step: bool = False
    crease_step_tol_mm: float = 0.5  # such a soft fold's segments agree within this distance
    #: Thin page stacks (many sheets of very thin paper, ~0.5 mm apart: KLK artboard) have faint
    #: page edges that the usual peak threshold misses next to dark textured art or a green
    #: cover.  With ``fine_stack`` a second, lower threshold (this percentile of the strip's
    #: gradient instead of the median) looks for them; its lines count only as a run of at least
    #: ``fine_stack_lines`` lines at most ``fine_stack_gap_mm`` apart (print rarely makes that).
    fine_stack: bool = False
    fine_stack_percentile: float = 25.0
    fine_stack_gap_mm: float = 0.8
    fine_stack_lines: int = 5
    #: A lower page sticking out further than its neighbours shows a strip of its own art (full
    #: bleed) in parts of the side and plain paper elsewhere; the strip's median colour then sits
    #: on the "too dark: print" limit and a pixel decides (KLK artboard, a crop from the app).
    #: With ``stack_by_segment`` such a gap is also chained when its inner line is a page edge
    #: along (nearly) the whole side and at least ``stack_segment_share`` of the segments look like
    #: stack paper on their own: print on the top page leaves no edge where it stops.
    stack_by_segment: bool = False
    stack_segment_share: float = 0.25
    #: A thick thin-paper stack in a 200 dpi preview: the sheets blur into one band with lower
    #: pages' art columns, and the top page's edge is a clear line only next to those columns.
    #: With ``stack_jump`` the chain's end moves in to a faint line along (nearly) the whole side
    #: within ``stack_jump_mm`` when the strip outside it is darker than the paper inside it in
    #: 75% of the segments and plain stack paper in ``stack_segment_share`` of them.
    stack_jump: bool = False
    stack_jump_mm: float = 8.0
    #: Thick stacks of thin sheets: the top page's edge lay 10.4 mm inside a preview box, beyond
    #: ``depth_mm``.  With ``layer_depth_mm`` (> 0) the usual rules keep seeing the strip only to
    #: that depth (exactly the same columns, so nothing else moves), and ``depth_mm`` deeper is
    #: seen only by the strong-line pass (``fine_strong_lines``).
    layer_depth_mm: float = 0.0
    #: ``fine_stack``'s faint lines along (nearly) the whole side (>= 85% of the segments) join
    #: the chain on their own, not only as a run of close sheets (a single white sheet 2 mm wide
    #: before the top page, its edge a clear line that the raised threshold missed).
    fine_strong_lines: bool = False
    fine_strong_percentile: float = 10.0
    strong_darker_l: float = 1.0  # a wider gap to such a line: its strip this much darker than the paper
    #: The chain runs before the faint-line passes; when they carry its end further in, it goes on
    #: from there over page edges <= ``near_mm`` apart (always one stack: a lower page's 0.8 mm art
    #: column before the top page, after 7 mm of white sheets).  Wider gaps are not crossed there
    #: (a line printed on the top page inside a paper-coloured margin).
    walk_after_passes: bool = False
    #: A hard cover's rim showing beyond the top page (a uniform coloured band <= ``gap_mm`` wide
    #: between the lid and the page; KLK artboard's green cover, 1.2 mm): with this switch the
    #: chain starts inside it when its inner line runs along (nearly) the whole side, light paper
    #: lies inside that line, and the band's colour is neither the paper's nor the lid's.
    cover_rim: bool = False
    #: Under a propped lid the sheet just below the top page, sticking out 1-2 mm, is 16-19 L*
    #: darker than the top page and neutral (further from the glass): over ``max_darker_l`` it
    #: passed as print and the crop kept it.  With this (> ``max_darker_l``) a gap <= ``gap_mm``
    #: is stack paper up to this much darker; wider gaps keep ``max_darker_l``.
    stack_shadow_l: float = 0.0
    #: A dark cover (or a top page printed dark to its edge) with a neutral band 1-2.2 mm wide
    #: beside it: the propped lid's shadow or the edges of the sheets below.  The band is lighter
    #: than the print inside, so it never passes as stack paper.  With this (> 0) a gap <=
    #: ``gap_mm`` to a line along (nearly) the whole side is crossed when the band is neutral and
    #: darker than the lid (a white margin on the glass is brighter) and the inside is at least
    #: this much darker than the band; the walk ends on that line (the item's edge).
    dark_edge_l: float = 0.0
    #: ``cover_rim`` asks for light paper inside the rim; a navy cover's rim beyond a top page printed
    #: dark to its edge, or beyond a lower page's art column, has none and was kept (V600 0044, 0046:
    #: +1.4 / +2.3 mm).  With this the rim's other tests decide (uniform along the side, a hue unlike
    #: both the lid's and the inside's, a full-length line inside, <= ``gap_mm`` wide).
    rim_print_inside: bool = False
    #: When no crease agrees within the spine's full reach (other dark lines a few mm away split the
    #: segments' votes: V600 0002, 0003, 0031), look again within these shorter reaches (mm), the
    #: nearest first; the first crease found is the fold.
    crease_retry_mm: tuple[float, ...] = ()

    # ---- thick books on a light lid (thickbook.py) ----
    #: Fit a whole spread of a thick book (pages = the brightest plain surface).
    thick_books: bool = True
    thick_inliers: float = 0.5  # share of edge points on each fitted line
    thick_rms_mm: float = 0.6  # largest rms distance of those points
    thick_height_tol: float = 0.03  # the two pages' heights within this share
    thick_width_tol: float = 0.15  # and widths
    thick_border_mm: float = 2.0  # pages at least this far from the image border
    #: A lower sheet showing as a plain band beyond the top page's outer edge (KLK Vol.7 Key Art Collection lying
    #: close to the bed's edges, the owner's test scans 0036-0038: the sheets below fan out by 1.8-3.5 mm).  The fit
    #: takes the first 2 mm of plain paper coming from outside as the page, so a band wider than 2 mm became the
    #: page's edge (+2.6 / +3.3 mm) and one about 2 mm wide mixed both lines until the fit gave up (0036: one box
    #: over the whole bed).  With this the outer edge is the top page's step line inside such a band
    #: (``thickbook._top_sheet_edge``).
    thick_sheet_band: bool = False


DEFAULT = Profile()

#: Every profile by name.  The default starts as a copy of the Epson V600's
#: values (the scanner the auto-crop was first tuned on).
PROFILES: dict[str, Profile] = {
    "default": DEFAULT,
    "epson-v600": replace(
        DEFAULT, name="epson-v600", label="Epson Perfection V600",
        match=(r"perfection\s*v600", r"\bgt-x820\b"),
        min_edge_step=0.6, crease_step=True, fine_stack=True, stack_by_segment=True, stack_jump=True, depth_mm=14.0, layer_depth_mm=10.0,
        fine_strong_lines=True, shrink_lid_hue=12.0, walk_after_passes=True,
        paper_colour_edges=15.0, cover_rim=True, lid_all_round_split_score=5.0,
        stack_shadow_l=20.0, dark_edge_l=15.0, rim_print_inside=True, crease_retry_mm=(1.0, 2.0),
        thick_sheet_band=True,
        tuned_on="57 scans (booklet spreads, booklets too big for the bed, covers, prints), "
                 "26 thick-book spreads, 9 artboard pages with a thin page stack and 33 test scans "
                 "(artboard pages, covers, thick-book spreads), Oct 2026"),
}

_ACTIVE: ContextVar[Profile] = ContextVar("hasami_profile", default=DEFAULT)


def current() -> Profile:
    """The profile the engine uses right now."""
    return _ACTIVE.get()


@contextmanager
def use(profile: Profile | str | None):
    """Make ``profile`` (or the profile of that name) active inside the block (per thread)."""
    p = get(profile) if not isinstance(profile, Profile) else profile
    token = _ACTIVE.set(p)
    try:
        yield p
    finally:
        _ACTIVE.reset(token)


def get(name: str | None) -> Profile:
    """The profile called ``name`` ("auto", "" or unknown: default)."""
    return PROFILES.get(name or "default", DEFAULT)


def for_scanner(*texts: str) -> Profile:
    """The profile whose patterns match the scanner's name / model, else default."""
    text = " ".join(t for t in texts if t)
    for p in PROFILES.values():
        if any(re.search(pat, text, re.I) for pat in p.match):
            return p
    return DEFAULT


def choices() -> list[tuple[str, str]]:
    """(label, name) for the Crop tab, "Automatic" first."""
    return [("Automatic (from the scanner)", "auto")] + [(p.label, p.name) for p in PROFILES.values()]
