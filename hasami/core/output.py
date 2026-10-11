"""Writing scans to disk: JPEG, PNG, TIFF (8/16-bit), WebP and PDF.

Resolution (DPI) is stored in every format so that prints come out at the
right size.  Images are RGB (or single-channel grey) numpy arrays.
"""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FORMATS = ("jpg", "png", "tif", "webp", "pdf")


@dataclass
class OutputSettings:
    folder: str = "~/Pictures/Scans"
    prefix: str = "scan"
    #: "jpg", "png", "tif", "webp" or "pdf".
    format: str = "jpg"
    jpeg_quality: int = 92
    #: "lzw", "deflate" or "none".
    tiff_compression: str = "lzw"
    #: Put every item of one scan into a single multi-page PDF.
    multipage_pdf: bool = True
    digits: int = 4
    #: Raw file: the cropped scan exactly as the scanner delivered it, with no
    #: colour processing, as a TIFF at the scanner's bit depth.  "off", "only"
    #: (instead of the processed file) or "also" (next to it, ``*_raw.tif``).
    raw_file: str = "off"

    @property
    def path(self) -> Path:
        return Path(self.folder).expanduser()


def normalise_format(fmt: str) -> str:
    fmt = fmt.lower().lstrip(".")
    return {"jpeg": "jpg", "tiff": "tif"}.get(fmt, fmt)


def next_filenames(folder: Path, prefix: str, ext: str, count: int, digits: int = 4) -> list[Path]:
    """Return ``count`` unused, consecutive names like ``scan0007.jpg``.

    Numbering continues after the highest existing number for this prefix.
    """
    folder.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.[A-Za-z0-9]+$")
    highest = 0
    for f in folder.iterdir():
        m = pat.match(f.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return [folder / f"{prefix}{highest + i + 1:0{digits}d}.{ext}" for i in range(count)]


def _png_with_dpi(data: bytes, dpi: float) -> bytes:
    """Insert a pHYs chunk (pixels per metre) right after IHDR."""
    ppm = int(round(dpi / 0.0254))
    payload = struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(payload)) + b"pHYs" + payload
    chunk += struct.pack(">I", zlib.crc32(b"pHYs" + payload) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature + IHDR length/type/data/crc
    return data[:ihdr_end] + chunk + data[ihdr_end:]


def _to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
    if img.ndim == 2:
        return Image.fromarray(img, "L")
    return Image.fromarray(img, "RGB")


def save_image(img: np.ndarray, path: Path, settings: OutputSettings | None = None, dpi: float = 300.0) -> Path:
    """Save one image; the format follows the file extension."""
    st = settings or OutputSettings()
    path = Path(path)
    fmt = normalise_format(path.suffix)
    sixteen = img.dtype == np.uint16
    if fmt == "tif":
        if sixteen:
            # Pillow cannot write 16-bit RGB; OpenCV can, with LZW and DPI.
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
            comp = {"lzw": 5, "deflate": 8, "none": 1}[st.tiff_compression]
            ok = cv2.imwrite(
                str(path),
                bgr,
                [cv2.IMWRITE_TIFF_COMPRESSION, comp, cv2.IMWRITE_TIFF_RESUNIT, 2,
                 cv2.IMWRITE_TIFF_XDPI, int(round(dpi)), cv2.IMWRITE_TIFF_YDPI, int(round(dpi))],
            )
            if not ok:
                raise OSError(f"could not write {path}")
            return path
        comp = {"lzw": "tiff_lzw", "deflate": "tiff_adobe_deflate", "none": None}[st.tiff_compression]
        _to_pil(img).save(path, compression=comp, dpi=(dpi, dpi))
        return path
    if fmt == "png":
        if sixteen:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
            ok, enc = cv2.imencode(".png", bgr)
            if not ok:
                raise OSError(f"could not encode {path}")
            path.write_bytes(_png_with_dpi(enc.tobytes(), dpi))
            return path
        _to_pil(img).save(path, dpi=(dpi, dpi), optimize=False)
        return path
    if fmt == "jpg":
        _to_pil(img).save(path, quality=int(st.jpeg_quality), dpi=(dpi, dpi), subsampling=0 if st.jpeg_quality >= 90 else 2)
        return path
    if fmt == "webp":
        _to_pil(img).save(path, quality=int(st.jpeg_quality))
        return path
    if fmt == "pdf":
        _to_pil(img).save(path, resolution=dpi)
        return path
    raise ValueError(f"unsupported format: {path.suffix}")


def save_pdf(pages: list[np.ndarray], path: Path, dpi: float = 300.0, quality: int = 92) -> Path:
    """Save several images as one multi-page PDF (JPEG-compressed pages)."""
    if not pages:
        raise ValueError("no pages")
    ims = [_to_pil(p).convert("RGB") for p in pages]
    ims[0].save(path, save_all=True, append_images=ims[1:], resolution=dpi, quality=quality)
    return path


def save_items(items: list[np.ndarray], settings: OutputSettings, dpi: float) -> list[Path]:
    """Save the cropped items of one scan following the output settings."""
    fmt = normalise_format(settings.format)
    if fmt not in FORMATS:
        raise ValueError(f"unsupported format: {settings.format}")
    if fmt == "pdf" and settings.multipage_pdf and len(items) > 1:
        (path,) = next_filenames(settings.path, settings.prefix, "pdf", 1, settings.digits)
        return [save_pdf(items, path, dpi, settings.jpeg_quality)]
    paths = next_filenames(settings.path, settings.prefix, fmt, len(items), settings.digits)
    return [save_image(img, p, settings, dpi) for img, p in zip(items, paths)]


def save_raw_items(items: list[np.ndarray], settings: OutputSettings, dpi: float,
                   beside: list[Path] | None = None) -> list[Path]:
    """Save raw items as TIFF (8 or 16-bit, as delivered by the scanner).

    With ``beside`` (the processed files of the same items) each raw file is
    named after its processed file: ``scan0007.jpg`` -> ``scan0007_raw.tif``;
    several items in one PDF get ``scan0007_raw1.tif``, ``scan0007_raw2.tif``...
    Without it they are numbered like processed files.
    """
    if not items:
        return []
    if beside is None:
        paths = next_filenames(settings.path, settings.prefix, "tif", len(items), settings.digits)
    elif len(beside) == len(items):
        paths = [p.with_name(p.stem + "_raw.tif") for p in beside]
    else:
        base = beside[0]
        paths = [base.with_name(f"{base.stem}_raw{i + 1}.tif") for i in range(len(items))]
    return [save_image(img, p, settings, dpi) for img, p in zip(items, paths)]


def load_image(path: str | Path) -> tuple[np.ndarray, float | None]:
    """Read an image file as RGB (8 or 16-bit) plus its DPI if recorded."""
    path = Path(path)
    dpi = None
    try:
        with Image.open(path) as im:
            info = im.info.get("dpi")
            if info and info[0] and float(info[0]) > 1:
                dpi = round(float(info[0]), 1)  # PNG stores pixels per metre
    except Exception:
        pass
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))
        return img, dpi
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, dpi
