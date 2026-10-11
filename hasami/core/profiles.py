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

    # ---- thick books on a light lid (thickbook.py) ----
    #: Fit a whole spread of a thick book (pages = the brightest plain surface).
    thick_books: bool = True
    thick_inliers: float = 0.5  # share of edge points on each fitted line
    thick_rms_mm: float = 0.6  # largest rms distance of those points
    thick_height_tol: float = 0.03  # the two pages' heights within this share
    thick_width_tol: float = 0.15  # and widths
    thick_border_mm: float = 2.0  # pages at least this far from the image border


DEFAULT = Profile()

#: Every profile by name.  The default starts as a copy of the Epson V600's
#: values (the scanner the auto-crop was first tuned on).
PROFILES: dict[str, Profile] = {
    "default": DEFAULT,
    "epson-v600": replace(
        DEFAULT, name="epson-v600", label="Epson Perfection V600",
        match=(r"perfection\s*v600", r"\bgt-x820\b"),
        tuned_on="57 scans (booklet spreads, booklets too big for the bed, covers, prints) "
                 "and 26 thick-book spreads, Oct 2026"),
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
