"""TWAIN backend: helper process + backend, against a fake pytwain (no Windows here)."""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from hasami.devices import twain_backend as tb
from hasami.devices.base import ScanCancelled, ScanRequest, ScannerError

FAKE = Path(__file__).with_name("fake_twain")


@pytest.fixture
def fake_twain(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(FAKE))
    monkeypatch.setattr(tb, "candidate_pythons", lambda: [sys.executable])
    monkeypatch.setattr(tb, "_settings", lambda: None)
    monkeypatch.setattr(tb, "_SOURCES", {})
    log = tmp_path / "twain_log.json"
    monkeypatch.setenv("FAKE_TWAIN_LOG", str(log))
    return log


def test_py_launcher_output():
    text = (" -V:3.12 *        C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python312\\python.exe\n"
            " -V:3.12-32       C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python312-32\\python.exe\n"
            " -3.9-64          C:\\Python39\\python.exe\n")
    tags = tb.parse_py_launcher(text)
    assert [t for t, _ in tags] == ["3.12", "3.12-32", "3.9-64"]
    assert tags[1][1].endswith("Python312-32\\python.exe")


def test_helper_reports_a_missing_pytwain(monkeypatch, tmp_path):
    # A "twain" that cannot be imported, ahead of a real pytwain if one is installed
    # (the maintainer's PC has it; the cloud machine did not).
    (tmp_path / "twain.py").write_text("raise ImportError('no pytwain here')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(ScannerError, match="pip install pytwain"):
        tb.run_helper(sys.executable, {"cmd": "sources"})


def test_list_and_capabilities(fake_twain):
    devs = tb.TwainBackend().list_devices()
    assert [d.id for d in devs] == ["twain:EPSON Perfection V600"]
    sc = tb.TwainBackend().open("EPSON Perfection V600")
    caps = sc.capabilities()
    assert caps.resolutions[0] == 50 and caps.resolutions[-1] == 6400 and 3200 in caps.resolutions
    assert caps.bit_depths == [8, 16]
    assert caps.max_area_mm == pytest.approx((215.9, 297.18), abs=0.1)


def test_resolutions_the_driver_takes_beyond_its_list(fake_twain, monkeypatch):
    # Old EPSON Scan: its TWAIN range stops at 800, its own window offers up to 12800.
    monkeypatch.setenv("FAKE_TWAIN_LISTED_MAX", "800")
    monkeypatch.setenv("FAKE_TWAIN_REAL_MAX", "12800")
    caps = tb.TwainBackend().open("EPSON Perfection V600").capabilities()
    assert caps.resolutions == [50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200,
                                4800, 6400, 9600, 12800]


def test_high_resolutions_that_need_a_small_area(fake_twain, monkeypatch):
    # Old EPSON Scan (real diag): 50-1184 dpi listed for the whole bed, 1200 refused
    # (CapSeqError) until the scan area is small enough.
    monkeypatch.setenv("FAKE_TWAIN_MAX_WIDTH_PX", "10064")
    sc = tb.TwainBackend().open("EPSON Perfection V600")
    caps = sc.capabilities()
    assert caps.resolutions[-1] == 6400 and 2400 in caps.resolutions
    assert caps.bit_depths == [8, 16]
    res = sc.scan(ScanRequest(dpi=2400, area_mm=(25.4, 25.4, 25.4, 12.7)))
    assert res.dpi == 2400 and res.image.shape[:2] == (1200, 2400)
    res = sc.scan(ScanRequest(dpi=2400))  # the whole bed: refused, and the note says why
    assert res.dpi != 2400 and "smaller area" in res.note


def test_big_scan_goes_through_a_file_when_memory_is_refused(fake_twain, monkeypatch):
    # Maintainer: Epson Scan saves 9.3 x 5.3 in at 2400 dpi (800 MB), our app only a tiny
    # crop: the old driver keeps native/memory transfers under ~400 MB (whole bed:
    # 1184 dpi = 418 MB), a file transfer may be bigger.  It offers only BMP files.
    monkeypatch.setenv("FAKE_TWAIN_MAX_BYTES", "3000000")
    monkeypatch.setenv("FAKE_TWAIN_FILE_FORMATS", "2")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    seen = []
    res = sc.scan(ScanRequest(dpi=400, area_mm=(0, 0, 4 * 25.4, 3 * 25.4), live=seen.append))
    assert res.dpi == 400 and res.image.shape == (1200, 1600, 3) and res.image.dtype == np.uint8
    assert any("trying a file" in line for line in sc.last_log), sc.last_log
    small = sc.scan(ScanRequest(dpi=100, area_mm=(0, 0, 25.4, 25.4), live=seen.append))
    assert small.image.shape == (100, 100, 3) and seen  # small ones still come in live


def test_24_bit_raw_scan(fake_twain):
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    res = sc.scan(ScanRequest(dpi=100, area_mm=(25.4, 25.4, 50.8, 25.4), bit_depth=8, raw=True))
    assert res.image.dtype == np.uint8 and res.image.shape == (100, 200, 3)
    assert res.dpi == 100 and res.area_mm == pytest.approx((25.4, 25.4, 50.8, 25.4))
    sent = json.loads(fake_twain.read_text())
    caps = {c: v for c, v in sent["set"]}
    assert caps[4395] == 24  # ICAP_BITDEPTH
    assert caps[4376] == 100  # ICAP_XRESOLUTION
    assert caps[4353] == 0 and caps[4355] == 0  # raw: brightness, contrast neutral
    assert caps[259] == 0  # native transfer
    assert sent["show_ui"] is False


def test_48_bit_scan_uses_a_tiff_file(fake_twain):
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    res = sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 50.8, 50.8), bit_depth=16))
    assert res.image.dtype == np.uint16 and res.image.shape == (100, 100, 3)
    caps = {c: v for c, v in json.loads(fake_twain.read_text())["set"]}
    assert caps[4395] == 48 and caps[259] == 1  # file transfer


def test_driver_that_needs_its_window(fake_twain, monkeypatch):
    monkeypatch.setenv("FAKE_TWAIN_NEEDS_UI", "1")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    res = sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4)))
    assert res.image.shape[:2] == (50, 50)
    assert json.loads(fake_twain.read_text())["show_ui"] is True
    assert any("retrying with it" in x for x in sc.last_log)


def test_scan_area_reaches_a_driver_with_a_paper_size_preset(fake_twain, monkeypatch):
    # The driver ignores the layout until its paper-size preset is switched off.
    monkeypatch.setenv("FAKE_TWAIN_IGNORES_FRAME", "1")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    res = sc.scan(ScanRequest(dpi=50, area_mm=(25.4, 50.8, 50.8, 25.4)))
    assert res.image.shape[:2] == (50, 100) and res.note == ""
    assert res.area_mm == pytest.approx((25.4, 50.8, 50.8, 25.4))
    caps = {c: v for c, v in json.loads(fake_twain.read_text())["set"]}
    assert caps[4386] == 0 and caps[4432] is False  # no paper size, no border detection


def test_driver_that_scans_the_whole_bed_anyway(fake_twain, monkeypatch):
    monkeypatch.setenv("FAKE_TWAIN_IGNORES_FRAME", "always")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    res = sc.scan(ScanRequest(dpi=50, area_mm=(25.4, 50.8, 50.8, 25.4)))
    assert res.area_mm[:2] == (0.0, 0.0)  # crops are placed on the whole bed
    assert res.area_mm[2] == pytest.approx(215.9, abs=1)
    assert "whole bed" in res.note


def test_check_area_catches_a_driver_reporting_the_wrong_area():
    bed = (215.9, 297.2)
    req = (50.0, 50.0, 60.0, 40.0)
    dpi = 100
    whole = (round(bed[0] / 25.4 * dpi), round(bed[1] / 25.4 * dpi))
    area, note = tb.check_area(*whole, dpi, req, req, bed)  # claims the request, sends the bed
    assert area[:2] == (0.0, 0.0) and "whole bed" in note
    part = (round(60 / 25.4 * dpi), round(40 / 25.4 * dpi))
    assert tb.check_area(*part, dpi, req, req, bed) == (req, "")


def test_diag_never_opens_another_scanning_program(monkeypatch):
    srcs = ["VueScan", "WIA-EPSON Perfection V600", "EPSON Perfection V600"]
    monkeypatch.setattr(tb, "_settings", lambda: None)
    assert tb.pick_source(srcs) == "EPSON Perfection V600"
    assert tb.pick_source(["VueScan", "WIA-EPSON Perfection V600"]) == "WIA-EPSON Perfection V600"
    assert tb.pick_source(["VueScan 9"]) is None
    assert tb.pick_source(srcs, "VueScan") == "VueScan"  # asked for by name
    saved = type("S", (), {"device": "twain:WIA-EPSON Perfection V600"})()
    monkeypatch.setattr(tb, "_settings", lambda: saved)
    assert tb.pick_source(srcs) == "WIA-EPSON Perfection V600"  # the scanner chosen in the app


def test_settings_of_the_drivers_window_are_reused(fake_twain, monkeypatch, tmp_path):
    # Epson Scan 2's "Scanning Quality" exists only in its window (a custom
    # capability); scans without the window must use what was chosen there.
    monkeypatch.setattr(tb, "_driver_settings_file", lambda: tmp_path / "driver.json")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4)))
    assert json.loads(fake_twain.read_text())["quality"] == 0  # nothing remembered yet
    ui = type("S", (), {"twain_show_ui": True, "twain_python": "", "device": ""})()
    monkeypatch.setattr(tb, "_settings", lambda: ui)
    sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4)))
    assert tb.load_driver_caps("EPSON Perfection V600") == {str(0x8030): [4, 1]}
    assert any("remembered 1 settings" in x for x in sc.last_log)
    monkeypatch.setattr(tb, "_settings", lambda: None)
    sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4)))
    sent = json.loads(fake_twain.read_text())
    assert sent["show_ui"] is False and sent["quality"] == 1
    assert sent["set"][0] == [0x8030, 1]  # before the app's own settings


@pytest.mark.parametrize("bits", [8, 16])
def test_live_scan_comes_in_strips(fake_twain, monkeypatch, bits):
    monkeypatch.setenv("FAKE_TWAIN_STRIP_DELAY", "0.15")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    area = (0, 0, 50.8, 50.8)
    plain = sc.scan(ScanRequest(dpi=60, area_mm=area, bit_depth=bits)).image  # native / file transfer
    seen = []
    live = sc.scan(ScanRequest(dpi=60, area_mm=area, bit_depth=bits,
                               live=lambda li: seen.append((li.rows, li.done, li.image.shape))))
    assert json.loads(fake_twain.read_text())["xfermech"] == 2  # TWSX_MEMORY
    assert np.array_equal(live.image, plain)  # same picture, RGB, same depth
    rows = [r for r, _, _ in seen]
    assert rows == sorted(rows) and 0 < rows[0] < 120 and seen[-1][:2] == (120, True)
    assert all(shape == (120, 120, 3) for _, _, shape in seen)


def test_live_scan_falls_back_when_memory_transfer_fails(fake_twain, monkeypatch):
    monkeypatch.setenv("FAKE_TWAIN_MEMORY_FAILS", "1")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    seen = []
    res = sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4), live=seen.append))
    assert res.image.shape == (50, 50, 3) and seen[-1].done
    assert any("retrying without it" in x for x in sc.last_log)
    assert json.loads(fake_twain.read_text())["show_ui"] is False  # not the driver's window


def test_every_scan_is_logged(fake_twain, monkeypatch, tmp_path):
    from hasami import settings
    monkeypatch.setattr(settings, "config_dir", lambda: tmp_path)
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4), raw=True, live=lambda li: None))
    text = (tmp_path / "hasami.log").read_text()
    assert "'live': True" in text and "'raw': True" in text and "using memory (strips)" in text
    assert "memory transfer:" in text


def test_helper_runs_on_python_exe_not_pythonw(tmp_path):
    (tmp_path / "python.exe").write_text("")
    (tmp_path / "pythonw.exe").write_text("")
    assert tb.console_python(str(tmp_path / "pythonw.exe")) == str(tmp_path / "python.exe")
    assert tb.console_python(str(tmp_path / "python.exe")) == str(tmp_path / "python.exe")
    alone = tmp_path / "x"
    alone.mkdir()
    (alone / "pythonw.exe").write_text("")
    assert tb.console_python(str(alone / "pythonw.exe")) == str(alone / "pythonw.exe")  # nothing better


def test_a_long_answer_does_not_stall_the_scan(monkeypatch, tmp_path):
    # The answer of a large scan (every strip's time) was bigger than a pipe
    # holds; the scan then hung at 99%.
    script = tmp_path / "helper.py"
    script.write_text("import json, sys\nsys.stderr.write('x' * 300000)\n"
                      "print(json.dumps({'ok': True, 'strips': list(range(100000))}))\n")
    monkeypatch.setattr(tb, "HELPER", script)
    answer = tb.run_helper(sys.executable, {"cmd": "scan"}, timeout=20)
    assert len(answer["strips"]) == 100000


def test_driver_window_setting_counts_without_a_restart(monkeypatch, tmp_path):
    # Maintainer: ticking "TWAIN: show the scanner's own window" did nothing - the TWAIN
    # code read the settings file, which the app saved only on exit.
    from hasami.settings import AppSettings
    from hasami.workflow import Session
    monkeypatch.setenv("PYTHONPATH", str(FAKE))
    monkeypatch.setattr(tb, "candidate_pythons", lambda: [sys.executable])
    monkeypatch.setattr(tb, "_SOURCES", {})
    monkeypatch.setattr(tb, "_driver_settings_file", lambda: tmp_path / "driver.json")
    log = tmp_path / "twain_log.json"
    monkeypatch.setenv("FAKE_TWAIN_LOG", str(log))
    st = AppSettings()
    Session(st)  # what the app does at start
    st.input.twain_show_ui = True  # ticked in the window, not saved yet
    tb.TwainScanner("EPSON Perfection V600", sys.executable).scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4)))
    assert json.loads(log.read_text())["show_ui"] is True
    tb.use_settings(None)


def test_two_drivers_with_the_same_name(fake_twain, monkeypatch):
    # The V600 has Epson Scan 2 (32 and 64-bit) and the old Epson Scan (32-bit only),
    # all called "EPSON Perfection V600"; the maintainer wants to choose the old one.
    monkeypatch.setenv("FAKE_TWAIN_FAMILIES", "EPSON Scan 2,EPSON Scan")
    monkeypatch.setattr(tb, "candidate_pythons", lambda: [sys.executable, sys.executable])  # 64 + 32-bit
    monkeypatch.setattr(tb, "_IDENT", {})
    devs = tb.TwainBackend().list_devices()
    assert [d.id for d in devs] == ["twain:EPSON Perfection V600", "twain:EPSON Perfection V600 [EPSON Scan]"]
    assert "EPSON Scan 2" in devs[0].model and devs[1].model.startswith("EPSON Perfection V600 - EPSON Scan,")
    sc = tb.TwainBackend().open("EPSON Perfection V600 [EPSON Scan]")
    sc.scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4)))
    assert json.loads(fake_twain.read_text())["family"] == "EPSON Scan"
    tb.TwainBackend().open("EPSON Perfection V600").scan(ScanRequest(dpi=50, area_mm=(0, 0, 25.4, 25.4)))
    assert json.loads(fake_twain.read_text())["family"] == "EPSON Scan 2"


# --- Cancel (maintainer, Oct 10 2026): Cancel killed the helper in the middle of a
# transfer and the V600 hung until its power cable was pulled.  The helper must
# stop the TWAIN way and the next scan must work without a power cycle.

@pytest.fixture
def fake_device(fake_twain, monkeypatch, tmp_path):
    """The fake scanner's state ("busy" = hung) and the TWAIN calls made."""
    dev, calls = tmp_path / "device", tmp_path / "calls"
    monkeypatch.setenv("FAKE_TWAIN_DEVICE", str(dev))
    monkeypatch.setenv("FAKE_TWAIN_CALLS", str(calls))
    return dev, calls


def _cancel_when(cond, cancel, timeout=20.0):
    def run():
        end = time.monotonic() + timeout
        while not cond() and time.monotonic() < end:
            time.sleep(0.02)
        cancel.set()
    threading.Thread(target=run, daemon=True).start()


def test_cancel_in_the_middle_of_a_strip_transfer(fake_device, monkeypatch):
    dev, calls = fake_device
    monkeypatch.setenv("FAKE_TWAIN_STRIP_DELAY", "0.15")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    cancel = threading.Event()
    seen = []

    def live(li):
        seen.append(li.rows)
        if li.rows > 0:
            cancel.set()

    t0 = time.monotonic()
    with pytest.raises(ScanCancelled) as err:
        sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 50.8, 101.6), live=live), cancel=cancel)
    assert not err.value.forced and err.value.warning == ""
    assert time.monotonic() - t0 < 10
    seq = calls.read_text().split()
    strips = seq.count("STRIP")
    assert 0 < strips < 240 // 7  # stopped part way through the image
    assert seq[seq.index("ENABLEDS"):] == ["ENABLEDS"] + ["STRIP"] * strips + [
        "ENDXFER", "DISABLEDS", "CLOSEDS", "CLOSEDSM"]
    assert dev.read_text() == "idle"
    assert any(x.startswith("cancel: stopped after") for x in sc.last_log)
    assert "cancel: source manager closed" in sc.last_log
    # The next scan works straight away (no power cycle).
    res = sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 50.8, 50.8), live=lambda li: None))
    assert res.image.shape == (120, 120, 3)
    assert dev.read_text() == "idle"


@pytest.mark.skipif(sys.platform != "win32", reason="ends the TWAIN message loop with WM_QUIT")
def test_cancel_while_the_driver_gets_ready(fake_device, monkeypatch):
    # Before the first data (state 5, pytwain waits in GetMessage): Cancel ends the
    # message loop, the source is disabled and closed.
    dev, calls = fake_device
    monkeypatch.setenv("FAKE_TWAIN_WARMUP", "60")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    sc.capabilities()
    cancel = threading.Event()
    _cancel_when(lambda: calls.exists() and "ENABLEDS" in calls.read_text(), cancel)
    t0 = time.monotonic()
    with pytest.raises(ScanCancelled) as err:
        sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 25.4, 25.4), live=lambda li: None), cancel=cancel)
    assert not err.value.forced and time.monotonic() - t0 < 10
    assert calls.read_text().split()[-4:] == ["ENABLEDS", "DISABLEDS", "CLOSEDS", "CLOSEDSM"]
    assert dev.read_text() == "idle"
    monkeypatch.setenv("FAKE_TWAIN_WARMUP", "0")
    assert sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 25.4, 25.4))).image.shape == (60, 60, 3)


def test_cancel_waits_for_a_transfer_in_one_block(fake_device, monkeypatch):
    # A native transfer cannot be stopped halfway: the app waits for it (the helper
    # says it is busy in the driver) instead of killing the helper after the grace time.
    dev, calls = fake_device
    monkeypatch.setattr(tb, "CANCEL_GRACE", 0.5)
    monkeypatch.setenv("FAKE_TWAIN_XFER_DELAY", "3")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    sc.capabilities()
    cancel = threading.Event()
    _cancel_when(lambda: calls.exists() and "ENABLEDS" in calls.read_text(), cancel)
    with pytest.raises(ScanCancelled) as err:
        sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 25.4, 25.4)), cancel=cancel)
    assert not err.value.forced
    assert calls.read_text().split()[-5:] == ["ENABLEDS", "ENDXFER", "DISABLEDS", "CLOSEDS", "CLOSEDSM"]
    assert dev.read_text() == "idle"


def test_a_helper_that_does_not_stop_is_killed_in_the_end(fake_device, monkeypatch):
    dev, calls = fake_device
    monkeypatch.setattr(tb, "CANCEL_GRACE", 0.5)
    monkeypatch.setattr(tb, "CANCEL_LIMIT", 2.0)
    monkeypatch.setenv("FAKE_TWAIN_XFER_DELAY", "60")
    sc = tb.TwainScanner("EPSON Perfection V600", sys.executable)
    sc.capabilities()
    cancel = threading.Event()
    _cancel_when(lambda: calls.exists() and "ENABLEDS" in calls.read_text(), cancel)
    t0 = time.monotonic()
    with pytest.raises(ScanCancelled) as err:
        sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 25.4, 25.4)), cancel=cancel)
    assert err.value.forced and "power cable" in err.value.warning
    assert time.monotonic() - t0 < 15
    assert "the helper did not stop and was killed" in sc.last_log
    # What killing does to the scanner: it is left in the middle of the scan.
    assert dev.read_text() == "busy"
    monkeypatch.setenv("FAKE_TWAIN_XFER_DELAY", "0")
    with pytest.raises(ScannerError, match="does not respond"):
        sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 25.4, 25.4)))
    dev.write_text("idle")  # the power cable pulled
    assert sc.scan(ScanRequest(dpi=60, area_mm=(0, 0, 25.4, 25.4))).image.shape == (60, 60, 3)


def test_gui_shows_the_advice_after_a_forced_stop():
    from hasami.devices.base import ScanCancelled as SC
    from hasami.gui import worker

    def fn(progress, cancel, status):
        e = SC("scan cancelled")
        e.warning = tb.FORCED_STOP
        raise e

    for f, expected in ((fn, tb.FORCED_STOP), (lambda *a: (_ for _ in ()).throw(SC("x")), "Cancelled")):
        t = worker.Task(f)
        got = []
        t.signals.failed.connect(got.append)
        t.run()
        assert got == [expected]


def test_installed_app_uses_the_bundled_helper_programs(monkeypatch, tmp_path):
    # The installer ships twain64\twain_helper64.exe and twain32\twain_helper32.exe
    # (no Python): the frozen app runs those, with the request as their argument.
    for sub, name in (("twain64", "twain_helper64.exe"), ("twain32", "twain_helper32.exe")):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / name).write_bytes(b"")
    monkeypatch.setattr(tb, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(tb, "_settings", lambda: None)
    found = tb.candidate_pythons()
    assert found == [str(tmp_path / "twain64" / "twain_helper64.exe"),
                     str(tmp_path / "twain32" / "twain_helper32.exe")]
    assert tb.helper_command(found[1], {"cmd": "sources"}) == [found[1], '{"cmd": "sources"}']
    # A Python still runs the script.
    assert tb.helper_command("C:/py/python.exe", {"cmd": "sources"})[1] == str(tb.HELPER)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(tb, "app_dir", lambda: None)
    assert tb.bundled_helpers() == []
