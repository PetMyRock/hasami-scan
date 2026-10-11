from hasami import updates
from hasami.settings import AppSettings
from hasami.updates import Release, is_newer, parse_release, should_offer


def test_versions():
    assert is_newer("v0.5.1", "0.5.0")
    assert is_newer("0.10.0", "0.9.9")
    assert is_newer("1.0", "0.9.12")
    assert not is_newer("v0.5.0", "0.5.0")
    assert not is_newer("0.5", "0.5.0")
    assert not is_newer("0.4.9", "0.5.0")
    assert not is_newer("", "0.5.0")


def test_release_from_github_answer():
    rel = parse_release({"tag_name": "v0.5.1", "html_url": "https://github.com/x/releases/tag/v0.5.1",
                         "assets": [{"name": "notes.txt", "browser_download_url": "n"},
                                    {"name": "HasamiScanSetup-0.5.1.exe", "browser_download_url": "https://dl/s.exe"}]})
    assert (rel.version, rel.page, rel.installer) == ("0.5.1", "https://github.com/x/releases/tag/v0.5.1",
                                                      "https://dl/s.exe")


def test_once_a_day_and_skipped_versions():
    st = AppSettings()
    assert updates.due(st, now=1_000_000)
    st.updates.last_check = 1_000_000
    assert not updates.due(st, now=1_000_000 + 3600)
    assert updates.due(st, now=1_000_000 + 25 * 3600)
    st.updates.check_on_start = False
    assert not updates.due(st, now=1_000_000 + 25 * 3600)
    rel = Release("0.5.1", "page")
    assert should_offer(rel, "0.5.0")
    assert not should_offer(rel, "0.5.0", skipped="0.5.1")  # on start
    assert should_offer(rel, "0.5.0", skipped="0.5.1", manual=True)  # Help > Check for updates
    assert not should_offer(rel, "0.5.1")


def test_help_menu_check(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from hasami.gui import app as gui_app
    app = QApplication.instance() or QApplication([])
    st = AppSettings()
    st.output.folder = str(tmp_path)
    monkeypatch.setattr(gui_app, "list_devices", lambda: [])
    win = gui_app.MainWindow(st)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a[2]))
    win._update_checked(Release(gui_app.__version__, "page"), manual=True)
    assert shown and shown[0].startswith("You have the latest version")
    assert st.updates.last_check > 0
    shown.clear()
    win._update_checked(Release("0.0.1", "page"), manual=False)  # older: nothing shown
    assert not shown
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    win.close()
