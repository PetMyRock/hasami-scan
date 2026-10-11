import dataclasses

import numpy as np

from hasami.core import profiles
from hasami.core.autocrop import AutoCropSettings, detect
from hasami.demo import BookletOptions, synth_booklet


def test_v600_profile_starts_as_the_default():
    # The default starts as a copy of the V600's tuned values; only the identity differs.
    ident = {"name", "label", "match", "tuned_on"}
    v600, default = profiles.get("epson-v600"), profiles.DEFAULT
    for f in dataclasses.fields(profiles.Profile):
        if f.name not in ident:
            assert getattr(v600, f.name) == getattr(default, f.name), f.name


def test_profile_from_the_scanner_name():
    assert profiles.for_scanner("EPSON Perfection V600", "EPSON Perfection V600 - EPSON Flatbed Scanner, 32-bit").name == "epson-v600"
    assert profiles.for_scanner("EPSON GT-X820").name == "epson-v600"  # the V600's Japanese name
    assert profiles.for_scanner("Canon CanoScan LiDE 300").name == "default"
    assert profiles.get("auto") is profiles.DEFAULT and profiles.get("nonsense") is profiles.DEFAULT
    assert profiles.choices()[0] == ("Automatic (from the scanner)", "auto")


def test_detect_runs_with_the_chosen_profile(monkeypatch):
    # A profile's numbers reach the engine (here: the top page's layer lines need more support).
    seen = []
    from hasami.core import toppage
    real = toppage._layer_lines

    def spy(*a, **k):
        seen.append(profiles.current().name)
        return real(*a, **k)

    monkeypatch.setattr(toppage, "_layer_lines", spy)
    monkeypatch.setitem(profiles.PROFILES, "test-scanner",
                        dataclasses.replace(profiles.DEFAULT, name="test-scanner", support=0.95))
    img = synth_booklet(BookletOptions(), seed=1).image
    detect(img, AutoCropSettings(mode="booklet", profile="test-scanner"), dpi=300)
    assert seen and set(seen) == {"test-scanner"}
    assert profiles.current() is profiles.DEFAULT  # only while detect runs


def test_session_resolves_auto_from_the_open_scanner():
    from hasami.devices.base import DeviceInfo
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    s = Session(AppSettings())
    assert s.crop_settings().profile == "default"  # no scanner open
    s.device = type("Dev", (), {"info": DeviceInfo("twain:EPSON Perfection V600", "EPSON Perfection V600")})()
    assert s.crop_settings().profile == "epson-v600"
    s.settings.crop.profile = "default"  # chosen by hand
    assert s.crop_settings().profile == "default"
    s.device = None
