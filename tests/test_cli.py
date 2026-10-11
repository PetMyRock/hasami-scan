import cv2

from hasami.cli import main
from hasami.demo import SynthOptions, synth_scan


def test_cli_crop(tmp_path, capsys):
    sc = synth_scan(SynthOptions(n_photos=3), seed=3)
    src = tmp_path / "bed.png"
    cv2.imwrite(str(src), cv2.cvtColor(sc.image, cv2.COLOR_RGB2BGR))
    out = tmp_path / "out"
    rc = main(["crop", str(src), "--dpi", "200", "--out", str(out), "--format", "png"])
    assert rc == 0
    files = sorted(out.glob("*.png"))
    assert len(files) == len(sc.truth)


def test_cli_list(capsys):
    assert main(["list"]) == 0
    assert "demo:" not in capsys.readouterr().out  # test scanners only with --all
    assert main(["list", "--all"]) == 0
    assert "demo:" in capsys.readouterr().out


def test_settings_of_the_old_name_are_carried_over(tmp_path, monkeypatch):
    # 0.4.2: "Scanner" became "Hasami Scan"; the settings folder moved with it.
    import sys
    from hasami import settings
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    old = tmp_path / "scanner"
    old.mkdir()
    (old / "settings.json").write_text('{"input": {"scan_dpi": 800}}')
    (old / "twain_driver_settings.json").write_text("{}")
    (old / "hasami.log").write_text("old log")
    new = settings.config_dir()
    assert new.name == ("hasami-scan" if sys.platform not in ("win32", "darwin") else "HasamiScan")
    assert settings.load_settings().input.scan_dpi == 800
    assert (new / "twain_driver_settings.json").exists() and not (new / "hasami.log").exists()
    assert (old / "settings.json").exists()  # left as it was
    (new / "settings.json").write_text('{"input": {"scan_dpi": 300}}')
    assert settings.load_settings().input.scan_dpi == 300  # copied only once
