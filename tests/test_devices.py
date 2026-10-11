import numpy as np
import pytest

from hasami.devices.base import ScanRequest, list_devices, open_device


def test_demo_scanner_scans_area():
    dev = open_device("demo:anime")
    caps = dev.capabilities()
    full = dev.scan(ScanRequest(dpi=50))
    assert full.image.ndim == 3
    part = dev.scan(ScanRequest(dpi=50, area_mm=(10, 10, 50, 40)))
    assert abs(part.image.shape[1] - 50 / 25.4 * 50) <= 2
    assert caps.max_area_mm[0] > 100


def test_file_scanner(tmp_path):
    import cv2
    img = np.full((300, 200, 3), 128, np.uint8)
    path = tmp_path / "bed.png"
    cv2.imwrite(str(path), img)
    dev = open_device(f"file:{path}")
    res = dev.scan(ScanRequest(dpi=150))
    assert res.image.shape[2] == 3


def test_escl_mock_roundtrip():
    from escl_mock import MockEscl
    with MockEscl() as mock:
        dev = open_device(f"escl:{mock.url}")
        caps = dev.capabilities()
        assert "color" in caps.modes
        res = dev.scan(ScanRequest(dpi=75, area_mm=(0, 0, 100, 80)))
        assert res.image.shape[0] > 50


def test_sane_test_device():
    pytest.importorskip("sane")
    devs = [d for d in list_devices() if d.id.startswith("sane:test")]
    if not devs:
        pytest.skip("SANE test backend not enabled")
    dev = open_device(devs[0].id)
    res = dev.scan(ScanRequest(dpi=50, area_mm=(0, 0, 40, 30)))
    assert res.image.size > 0
