import numpy as np
import pytest

from hasami.core.output import OutputSettings, load_image, next_filenames, save_image, save_items


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)


@pytest.mark.parametrize("fmt", ["png", "tif", "jpg", "webp"])
def test_roundtrip_keeps_size_and_dpi(tmp_path, img, fmt):
    st = OutputSettings(folder=str(tmp_path), format=fmt)
    p = save_image(img, tmp_path / f"a.{fmt}", st, dpi=600)
    back, dpi = load_image(p)
    assert back.shape[:2] == img.shape[:2]
    if fmt in ("png", "tif"):
        assert np.array_equal(back, img)
    if fmt != "webp":
        assert dpi == pytest.approx(600, abs=1)


def test_16_bit_tiff(tmp_path):
    img = (np.arange(40 * 50 * 3).reshape(40, 50, 3) * 37 % 65535).astype(np.uint16)
    p = save_image(img, tmp_path / "a.tif", OutputSettings(format="tif"), dpi=300)
    back, _ = load_image(p)
    assert back.dtype == np.uint16 and np.array_equal(back, img)


def test_pdf_and_numbering(tmp_path, img):
    st = OutputSettings(folder=str(tmp_path), format="pdf", multipage_pdf=True)
    paths = save_items([img, img], st, dpi=300)
    assert len(paths) == 1 and paths[0].suffix == ".pdf" and paths[0].stat().st_size > 0
    # Numbers continue after existing files of any format with the prefix.
    names = next_filenames(tmp_path, "scan", "jpg", 2)
    assert [n.name for n in names] == ["scan0002.jpg", "scan0003.jpg"]
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert [n.name for n in next_filenames(fresh, "scan", "png", 2)] == ["scan0001.png", "scan0002.png"]
