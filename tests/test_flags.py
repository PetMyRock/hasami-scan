"""Check-this-side flags in the Session (demo scanner, generated bed)."""
from hasami.core.geometry import SIDE_NAMES


def _session(tmp_path, mode="booklet"):
    from hasami.settings import AppSettings
    from hasami.workflow import Session

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.crop.mode = mode
    st.output.folder = str(tmp_path / "scans")
    return Session(st)


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


def test_no_flags_outside_booklet_mode(tmp_path, monkeypatch):
    from hasami.core import sidelines

    monkeypatch.setattr(sidelines, "CHECK_RULE", (("crowd", 0),))
    s = _session(tmp_path, "multi")
    s.run_preview()
    assert all("check_sides" not in b.meta for b in s.boxes_mm)
