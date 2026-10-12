"""Raw files: the cropped scan as the scanner delivered it, no colour processing."""
import numpy as np
import pytest

from hasami.core.geometry import extract
from hasami.core.output import OutputSettings, load_image, save_image, save_raw_items
from hasami.devices import wia_backend as wia
from hasami.devices.base import ScanRequest
from hasami.settings import AppSettings
from hasami.workflow import Session


def test_raw_file_names(tmp_path):
    st = OutputSettings(folder=str(tmp_path))
    img = np.zeros((10, 12, 3), np.uint8)
    beside = [tmp_path / "scan0003.jpg", tmp_path / "scan0004.jpg"]
    assert [p.name for p in save_raw_items([img, img], st, 300, beside)] == ["scan0003_raw.tif", "scan0004_raw.tif"]
    # Several items in one PDF.
    assert [p.name for p in save_raw_items([img, img], st, 300, [tmp_path / "scan0005.pdf"])] == \
        ["scan0005_raw1.tif", "scan0005_raw2.tif"]
    # On their own: numbered like processed files (after the highest scanNNNN.*).
    (tmp_path / "scan0006.jpg").write_bytes(b"")
    assert [p.name for p in save_raw_items([img], st, 300)] == ["scan0007.tif"]


@pytest.fixture
def bed(tmp_path):
    """A dull, low-contrast print on a light lid: auto levels would change it."""
    img = np.full((600, 800, 3), 200, np.uint8)
    yy, xx = np.mgrid[0:300, 0:400]
    img[150:450, 200:600, 0] = 90 + (xx % 40)
    img[150:450, 200:600, 1] = 110 + (yy % 30)
    img[150:450, 200:600, 2] = 100
    path = tmp_path / "bed.png"
    save_image(img, path, dpi=150)
    return path


def _session(tmp_path, bed, raw: str, bits: int = 8) -> Session:
    st = AppSettings()
    st.input.device = f"file:{bed}"
    st.input.scan_dpi = 150
    st.input.bit_depth = bits
    st.crop.mode = "single"
    st.output.folder = str(tmp_path / "out")
    st.output.format = "jpg"
    st.output.raw_file = raw
    return Session(st)


def test_raw_only_keeps_the_scanned_colours(tmp_path, bed):
    s = _session(tmp_path, bed, "only")
    (oc,) = s.run_scan()
    assert len(oc.paths) == 1 and oc.paths[0].suffix == ".tif" and oc.raw_paths == oc.paths
    raw, dpi = load_image(oc.paths[0])
    assert dpi == pytest.approx(150, abs=1)
    assert np.array_equal(raw, extract(oc.scan.image, oc.boxes[0]))  # no levels, nothing
    assert raw.dtype == np.uint8  # 24-bit colour, as scanned


def test_raw_also_writes_both(tmp_path, bed):
    s = _session(tmp_path, bed, "also")
    (oc,) = s.run_scan()
    assert [p.suffix for p in oc.paths] == [".jpg"]
    assert [p.name for p in oc.raw_paths] == [oc.paths[0].stem + "_raw.tif"]
    raw, _ = load_image(oc.raw_paths[0])
    processed = oc.items[0]
    assert raw.shape == processed.shape
    assert not np.array_equal(raw, processed)  # auto levels stretched the processed copy


def test_raw_48_bit(tmp_path, bed):
    s = _session(tmp_path, bed, "only", bits=16)
    (oc,) = s.run_scan()
    raw, _ = load_image(oc.paths[0])
    assert raw.dtype == np.uint16 and raw.ndim == 3


# ------------------------------------------------------------ WIA (no Windows)
class _Vector:
    def __init__(self, values):
        self.values = list(values)
        self.Count = len(self.values)

    def Item(self, i):
        return self.values[i - 1]


class _Prop:
    def __init__(self, pid, value, values=None, rng=None):
        self.PropertyID, self.Value = pid, value
        self.SubType = 0
        if values is not None:
            self.SubType, self.SubTypeValues = wia.LIST_SUBTYPE, _Vector(values)
        if rng is not None:
            self.SubType = wia.RANGE_SUBTYPE
            self.SubTypeMin, self.SubTypeMax, self.SubTypeStep = rng


class _Props:
    def __init__(self, props):
        self.items = props
        self.Count = len(props)

    def Item(self, i):
        return self.items[i - 1]


class _Image:
    def __init__(self, fmt):
        self.FileData = type("FD", (), {"BinaryData": fmt.encode()})()


class _Item:
    def __init__(self, depths, transfer_ok):
        self.Properties = _Props([
            _Prop(wia.WIA_IPS_CUR_INTENT, 1), _Prop(wia.WIA_IPA_DEPTH, 24, values=depths),
            _Prop(wia.WIA_IPS_XRES, 300, rng=(50, 6400, 1)), _Prop(wia.WIA_IPS_YRES, 300),
            _Prop(wia.WIA_IPS_XPOS, 0), _Prop(wia.WIA_IPS_YPOS, 0), _Prop(wia.WIA_IPS_XEXTENT, 0),
            _Prop(wia.WIA_IPS_YEXTENT, 0), _Prop(wia.WIA_IPS_BRIGHTNESS, 250), _Prop(wia.WIA_IPS_CONTRAST, -100)])
        self.transfer_ok = transfer_ok
        self.tried = []

    def value(self, pid):
        return wia._props(self).get(pid).Value

    def Transfer(self, fmt):
        self.tried.append(fmt)
        if fmt not in self.transfer_ok:
            raise RuntimeError("format not supported")
        return _Image(fmt)


class _Device:
    def __init__(self, item):
        self.item = item
        self.Properties = _Props([_Prop(wia.WIA_DPS_HORIZONTAL_BED_SIZE, 8500),
                                  _Prop(wia.WIA_DPS_VERTICAL_BED_SIZE, 11700)])

    def Items(self, i):
        return self.item


def _wia(depths, transfer_ok=(wia.FORMAT_TIFF, wia.FORMAT_PNG, wia.FORMAT_BMP)):
    sc = object.__new__(wia.WiaScanner)
    sc.dev = _Device(_Item(depths, transfer_ok))
    return sc


def test_wia_capabilities_from_the_driver():
    caps = _wia([24, 48]).capabilities()
    assert caps.bit_depths == [8, 16]
    assert 6400 in caps.resolutions and 1200 in caps.resolutions
    assert _wia([24]).capabilities().bit_depths == [8]


def test_wia_24_bit_raw_scan():
    sc = _wia([24, 48])
    item, _, _ = sc._configure(ScanRequest(dpi=600, bit_depth=8, raw=True))
    assert item.value(wia.WIA_IPA_DEPTH) == 24
    assert item.value(wia.WIA_IPS_BRIGHTNESS) == 0 and item.value(wia.WIA_IPS_CONTRAST) == 0
    assert sc._transfer(item) == wia.FORMAT_PNG.encode()


def test_wia_48_bit_uses_tiff_and_falls_back():
    sc = _wia([24, 48])
    item, _, _ = sc._configure(ScanRequest(dpi=600, bit_depth=16))
    assert item.value(wia.WIA_IPA_DEPTH) == 48
    assert item.value(wia.WIA_IPS_BRIGHTNESS) == 250  # not a raw scan: left alone
    assert sc._transfer(item) == wia.FORMAT_TIFF.encode()
    sc = _wia([24, 48], transfer_ok=(wia.FORMAT_PNG,))
    item, _, _ = sc._configure(ScanRequest(dpi=600, bit_depth=16))
    assert sc._transfer(item) == wia.FORMAT_PNG.encode()
    # A driver without 48-bit colour gets 24.
    sc = _wia([24])
    item, _, _ = sc._configure(ScanRequest(dpi=600, bit_depth=16))
    assert item.value(wia.WIA_IPA_DEPTH) == 24


def test_wia_errors_are_explained():
    e = Exception(-2147352567, "Exception occurred.", (0, None, "Unspecified error\r\n", None, 0, -2147467259), None)
    assert wia.describe(e) == "unspecified error (0x80004005)"
    busy = Exception(-2147352567, "Exception occurred.", (0, None, "", None, 0, -2145320954), None)
    assert "busy" in wia.describe(busy)
    sc = _wia([24])
    sc.dev.item.Transfer = lambda fmt: (_ for _ in ()).throw(e)
    from hasami.devices.base import ScannerError
    with pytest.raises(ScannerError, match="WIA: scanning failed: unspecified error"):
        sc.scan(ScanRequest(dpi=75))


def test_wia_extent_is_clamped_to_the_driver_range():
    sc = _wia([24])
    item = sc.dev.item
    ext = wia._props(item)[wia.WIA_IPS_XEXTENT]
    ext.SubType, ext.SubTypeMin, ext.SubTypeMax = wia.RANGE_SUBTYPE, 1, 500
    log = []
    sc._configure(ScanRequest(dpi=75), log)
    assert item.value(wia.WIA_IPS_XEXTENT) == 500
    assert any("6151 = 500: ok" in x for x in log)


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_rotation_turns_every_saved_crop(tmp_path, bed, turns):
    s = _session(tmp_path, bed, "only")
    s.run_preview()
    plain = s.run_scan(save=False)[0].items[0]
    s.settings.crop.rotate = turns  # what the GUI's Rotate buttons set
    turned = s.run_scan(save=False)[0].items[0]
    assert np.array_equal(turned, np.rot90(plain, -turns))  # clockwise


def test_a_box_drawn_by_hand_counts_with_auto_crop_off(tmp_path, bed):
    # Owner: with auto-crop off and a crop drawn by hand, the whole bed was
    # scanned and saved.
    from hasami.core.geometry import CropBox
    s = _session(tmp_path, bed, "only")
    s.settings.crop.mode = "off"
    s.settings.input.scan_crop_area_only = True
    s.run_preview()
    assert s.boxes_mm == []  # auto-crop off: no box at all (owner, 0.5.4)
    drawn = CropBox.from_xywh(40.0, 30.0, 50.0, 20.0, manual=True)
    s.boxes_mm.append(drawn)
    assert s._scan_area() == pytest.approx((37.0, 27.0, 56.0, 26.0), abs=0.6)  # the drawn box + 3 mm
    (oc,) = s.run_scan()
    h, w = oc.items[0].shape[:2]
    assert len(oc.items) == 1  # only the drawn box, not also the whole bed
    assert abs(w - 50 / 25.4 * 150) <= 2 and abs(h - 20 / 25.4 * 150) <= 2


def test_auto_crop_off_scans_the_whole_bed_without_boxes(tmp_path, bed):
    s = _session(tmp_path, bed, "only")
    s.settings.crop.mode = "off"
    s.run_preview()
    assert s.boxes_mm == [] and s._scan_area() is None  # whole bed
    (oc,) = s.run_scan(save=False)
    (item,) = oc.items  # one image: the whole bed
    bed_w, bed_h = s.open().capabilities().max_area_mm
    assert abs(item.shape[1] - bed_w / 25.4 * 150) <= 3 and abs(item.shape[0] - bed_h / 25.4 * 150) <= 3
