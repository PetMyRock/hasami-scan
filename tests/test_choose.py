from hasami.devices.base import DeviceInfo
from hasami.devices.choose import best, offered


def _d(dev_id, name, model="", backend="twain"):
    return DeviceInfo(dev_id, name, "", model, backend)


def test_old_epson_scan_before_epson_scan_2():
    # Both drivers are called "EPSON Perfection V600"; Epson Scan 2 is found first
    # (64-bit helper), the old EPSON Scan is 32-bit only.
    devs = [_d("twain:EPSON Perfection V600", "EPSON Perfection V600", "EPSON Perfection V600 - ES2, 64-bit"),
            _d("twain:EPSON Perfection V600 [EPSON Flatbed Scanner]", "EPSON Perfection V600",
               "EPSON Perfection V600 - EPSON Flatbed Scanner, 32-bit")]
    assert best(devs).id == "twain:EPSON Perfection V600 [EPSON Flatbed Scanner]"
    assert len(offered(devs)) == 2  # both stay listed


def test_wia_only_scanner_is_offered():
    devs = [_d("wia:{A}", "Canon LiDE 300", backend="wia"), _d("twain:VueScan TWAIN", "VueScan TWAIN"),
            _d("demo:anime", "Demo scanner", backend="demo")]
    assert [d.id for d in offered(devs)] == ["wia:{A}"]
    assert best(devs).id == "wia:{A}"


def test_no_scanner():
    assert best([_d("demo:anime", "Demo scanner", backend="demo")]) is None
    assert best([]) is None
