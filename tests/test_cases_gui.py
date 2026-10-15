import time
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        pytest.skip(str(e))
    return QApplication.instance() or QApplication([])


def _case(folder, minute=0):
    from hasami import cases
    from hasami.core.geometry import CropBox

    a = CropBox(60.0, 80.0, 50.0, 70.0)
    a.meta["auto_id"] = "a1"
    f = replace(a, h=a.h + 4, cy=a.cy + 2, manual=True, meta=dict(a.meta))
    return cases.write_case(folder, cases.CaseData(
        preview=np.full((300, 220, 3), 200, np.uint8), preview_dpi=50.0, preview_area_mm=(0, 0, 111.8, 152.4),
        scan=np.full((200, 150, 3), 180, np.uint8), scan_dpi_saved=100.0, scan_dpi=100.0,
        scan_area_mm=(30, 40, 38.1, 50.8), auto=[a], final=[f], scanner_id="demo:booklet", scanner_name="Demo",
        created=datetime(2026, 10, 15, 14, minute, 0)))


def test_dialog_lists_zips_and_deletes_cases(app, tmp_path):
    from PySide6.QtCore import Qt

    from hasami.gui.cases_dialog import NOTE, CasesDialog

    folder = tmp_path / "cases"
    c1, c2 = _case(folder, 0), _case(folder, 1)
    broken = folder / "2026-10-15_150000"
    broken.mkdir()
    (broken / "case.json").write_text("{broken", encoding="utf-8")
    dlg = CasesDialog(folder)
    opened = []
    dlg._open = opened.append
    dlg._confirm = lambda n: True
    try:
        assert dlg.list.count() == 3
        texts = [dlg.list.item(i).text() for i in range(3)]
        assert texts[0].startswith("2026-10-15_150000") and "unreadable" in texts[0]
        assert "bottom edge 4.0 mm out" in texts[1] and "Demo" in texts[1]
        assert "Share only what you are happy to share" in NOTE and "discord.gg/TAAV8B9WWg" in NOTE
        dlg.list.item(1).setCheckState(Qt.Checked)
        assert dlg.checked() == [c2]
        zips = dlg.create_zips()
        assert [z.name for z in zips] == [f"hasami-case-{c2.name}.zip"] and zips[0].exists()
        assert opened == [folder / "To share"]
        dlg.reload()
        assert dlg.list.count() == 3  # the share folder is not a case
        for i in (0, 2):
            dlg.list.item(i).setCheckState(Qt.Checked)
        assert sorted(dlg.delete_checked()) == sorted([broken, c1])
        assert not broken.exists() and not c1.exists() and c2.exists()
        assert dlg.list.count() == 1
    finally:
        dlg.close()


def test_delete_asks_first(app, tmp_path):
    from PySide6.QtCore import Qt

    from hasami.gui.cases_dialog import CasesDialog

    c = _case(tmp_path / "cases")
    dlg = CasesDialog(tmp_path / "cases")
    dlg._confirm = lambda n: False
    try:
        dlg.list.item(0).setCheckState(Qt.Checked)
        assert dlg.delete_checked() == [] and c.exists()
    finally:
        dlg.close()


def test_the_app_saves_a_corrected_scan_and_offers_save_this_case(app, tmp_path):
    from PySide6.QtCore import QThreadPool

    from hasami import cases
    from hasami.gui.app import MainWindow
    from hasami.settings import AppSettings

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.input.scan_dpi = 150
    st.crop.mode = "booklet"
    st.output.folder = str(tmp_path / "scans")
    st.cases.enabled = True
    st.cases.folder = str(tmp_path / "cases")
    win = MainWindow(st)

    def wait():
        end = time.monotonic() + 30
        while win.task is not None and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert win.task is None

    try:
        assert win.a_share_cases.text() == "Share crop cases..." and not win.a_save_case.isEnabled()
        win.do_preview()
        wait()
        it = win.preview.items_[0]
        b = it.box
        it.box = replace(b, h=b.h + 20, cy=b.cy + 10, manual=True, meta=dict(b.meta))  # bottom out (preview px)
        win.do_scan()
        wait()
        [path] = cases.list_cases(tmp_path / "cases")
        assert f"Saved crop case {path.name}" in win.status_label.text()
        assert win.a_save_case.isEnabled()
        win.save_case()  # already saved as a correction: nothing new
        assert cases.list_cases(tmp_path / "cases") == [path]
    finally:
        QThreadPool.globalInstance().waitForDone(10000)
        app.processEvents()
        win.close()


def test_crop_tab_has_the_case_settings(app):
    from hasami.gui.settings_panel import SettingsPanel
    from hasami.settings import AppSettings

    st = AppSettings()
    panel = SettingsPanel(st)
    assert ("cases", "enabled") in panel._widgets and ("cases", "folder") in panel._widgets
    assert not panel._widgets[("cases", "folder")].isEnabled()  # greyed out until the setting is on


def test_turning_crop_cases_off_drops_the_kept_case(app, tmp_path):
    from PySide6.QtCore import QThreadPool

    from hasami.gui.app import MainWindow
    from hasami.settings import AppSettings

    st = AppSettings()
    st.input.device = "demo:booklet"
    st.input.preview_dpi = 50
    st.input.scan_dpi = 150
    st.crop.mode = "booklet"
    st.output.folder = str(tmp_path / "scans")
    st.cases.enabled = True
    st.cases.folder = str(tmp_path / "cases")
    win = MainWindow(st)

    def wait():
        end = time.monotonic() + 30
        while win.task is not None and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)

    try:
        win.do_preview()
        wait()
        win.do_scan()
        wait()
        assert win.session.last_case is not None and win.a_save_case.isEnabled()
        st.cases.enabled = False
        win.on_setting_changed("cases", "enabled")
        assert win.session.last_case is None and not win.a_save_case.isEnabled()
    finally:
        QThreadPool.globalInstance().waitForDone(10000)
        app.processEvents()
        win.close()


def test_the_share_note_says_who_can_see_posted_zips():
    from hasami.gui.cases_dialog import NOTE

    assert "members of that channel can see what you post" in NOTE
    assert "Share only what you are happy to share" in NOTE
