"""Is a newer Hasami Scan released?  (GitHub's "latest release" of the public repo.)

The installed app asks at most once a day on start (``updates.check_on_start``,
Help menu); Help > Check for updates asks at once.  Only the release's version,
page and installer link are read; nothing is sent besides the request itself.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

REPO = "PetMyRock/hasami-scan"
LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES = f"https://github.com/{REPO}/releases"
#: Seconds between automatic checks.
INTERVAL = 24 * 3600


@dataclass
class Release:
    version: str  # "0.5.1" (the tag without its "v")
    page: str  # the release's web page (notes + download)
    installer: str = ""  # direct link to HasamiScanSetup-<v>.exe, when attached


def parse_version(text: str) -> tuple[int, ...]:
    """ "v0.5.10" -> (0, 5, 10); anything after the numbers (e.g. "-beta") is ignored."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", text or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def is_newer(latest: str, current: str) -> bool:
    a, b = parse_version(latest), parse_version(current)
    n = max(len(a), len(b))
    return bool(a) and a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def parse_release(data: dict) -> Release:
    tag = str(data.get("tag_name") or "")
    installer = next((a.get("browser_download_url", "") for a in data.get("assets") or []
                      if str(a.get("name", "")).lower().endswith(".exe")), "")
    return Release(tag.lstrip("vV"), str(data.get("html_url") or RELEASES), installer)


def fetch_latest(timeout: float = 10.0) -> Release:
    """The latest release (raises OSError / ValueError when GitHub cannot be reached)."""
    from . import __version__

    req = urllib.request.Request(LATEST, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": f"HasamiScan/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_release(json.loads(r.read().decode("utf-8")))


def due(settings, now: float | None = None) -> bool:
    """Time for an automatic check (turned on, and none in the last day)."""
    st = settings.updates
    now = time.time() if now is None else now
    return bool(st.check_on_start) and now - float(st.last_check) >= INTERVAL


def should_offer(rel: Release, current: str, skipped: str = "", manual: bool = False) -> bool:
    """Offer ``rel``: it is newer, and (on start) not the version the user skipped."""
    return is_newer(rel.version, current) and (manual or rel.version != skipped)
