"""Command line interface.

    hasami                       start the GUI (if PySide6 is installed)
    hasami list                  list scanners
    hasami scan [options]        preview, auto-crop, scan, save
    hasami diag [-d DEVICE]      what the scanner offers, and a small test scan
    hasami crop FILE... [opts]   auto-crop existing scans (with --crop off: just
                                 apply the colour settings, e.g. to raw files)

Options default to the settings saved by the GUI; command line flags override
them for that run only.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from . import __version__
from .core import autocrop
from .core.output import load_image, save_image
from .settings import AppSettings, load_settings


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("auto-crop")
    g.add_argument("--crop", choices=["multi", "single", "booklet", "off"],
                   help="multi: every item; single: one item; booklet: only the page fully on the glass")
    g.add_argument("--booklet-output", choices=["pages", "spread"],
                   help="booklet mode with the whole spread on the glass: each page (default) or one image")
    g.add_argument("--booklet-rtl", action="store_true",
                   help="booklet mode, two pages: right page first (books read right to left)")
    g.add_argument("--fold-clear", action="store_true",
                   help="booklet mode, two pages: crop each page's fold side clear of the dark crease")
    g.add_argument("--profile", help="scanner profile (tuned numbers per scanner model): auto (from the "
                        "scanner), default, epson-v600, ...")
    g.add_argument("--sensitivity", type=float, help="0..1, higher finds fainter items")
    g.add_argument("--margin-mm", type=float, help="crop inside (+) or outside (-) the detected edge")
    g.add_argument("--no-deskew", action="store_true", help="keep crops axis-aligned")
    g.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], help="turn every saved crop clockwise by this many degrees")
    g.add_argument("--min-area", type=float, help="smallest item as a fraction of the scan area")
    c = p.add_argument_group("colour and filters")
    c.add_argument("--media", choices=["photo", "document", "color_negative", "bw_negative", "slide"])
    c.add_argument("--no-auto-levels", action="store_true")
    c.add_argument("--restore-fading", action="store_true")
    c.add_argument("--white-balance", choices=["none", "auto", "gray_world"])
    c.add_argument("--brightness", type=float)
    c.add_argument("--contrast", type=float)
    c.add_argument("--saturation", type=float)
    c.add_argument("--sharpen", type=float, help="unsharp mask amount, e.g. 0.5")
    c.add_argument("--descreen", choices=["off", "auto", "on"], help="remove halftone print patterns (magazines, artbooks)")
    c.add_argument("--dust", action="store_true", help="remove dust specks")
    c.add_argument("--output-mode", choices=["color", "gray", "bw"])
    o = p.add_argument_group("output")
    o.add_argument("-o", "--out", help="output folder")
    o.add_argument("--prefix", help="file name prefix (default 'scan')")
    o.add_argument("--format", choices=["jpg", "png", "tif", "webp", "pdf"])
    o.add_argument("--quality", type=int, help="JPEG/WebP quality")
    o.add_argument("--bits", type=int, choices=[8, 16], help="bits per channel in the output (8 = 24-bit colour, 16 = 48-bit)")
    o.add_argument("--raw", choices=["off", "only", "also"],
                   help="raw file: the cropped scan with no colour processing, as TIFF; "
                        "'only' instead of the processed file, 'also' next to it (*_raw.tif)")
    o.add_argument("--debug", action="store_true", help="also save an image showing the detected crops")
    p.add_argument("--settings", help="settings JSON to use instead of the saved GUI settings")
    p.add_argument("--defaults", action="store_true", help="ignore saved settings")


def _apply(args, st: AppSettings) -> AppSettings:
    c, k, o = st.crop, st.color, st.output
    for attr, val in (("mode", args.crop), ("sensitivity", args.sensitivity), ("margin_mm", args.margin_mm),
                      ("min_area", args.min_area), ("booklet_output", args.booklet_output),
                      ("profile", args.profile)):
        if val is not None:
            setattr(c, attr, val)
    if args.booklet_rtl:
        c.booklet_rtl = True
    if args.fold_clear:
        c.fold_clear = True
    if args.no_deskew:
        c.deskew = False
    if args.rotate is not None:
        c.rotate = args.rotate // 90
    for attr, val in (("media", args.media), ("white_balance", args.white_balance), ("brightness", args.brightness),
                      ("contrast", args.contrast), ("saturation", args.saturation), ("sharpen", args.sharpen),
                      ("descreen", args.descreen), ("output_mode", args.output_mode), ("bits", args.bits)):
        if val is not None:
            setattr(k, attr, val)
    if args.no_auto_levels:
        k.auto_levels = False
    if args.restore_fading:
        k.restore_fading = True
    if args.dust:
        k.dust_removal = True
    for attr, val in (("folder", args.out), ("prefix", args.prefix), ("format", args.format),
                      ("jpeg_quality", args.quality), ("raw_file", args.raw)):
        if val is not None:
            setattr(o, attr, val)
    return st


def _settings(args) -> AppSettings:
    if args.defaults:
        st = AppSettings()
    else:
        st = load_settings(Path(args.settings) if args.settings else None)
    return _apply(args, st)


def _progress_bar(label: str):
    def cb(f: float):
        n = int(f * 30)
        sys.stderr.write(f"\r{label} [{'#' * n}{'.' * (30 - n)}] {f * 100:5.1f}%")
        sys.stderr.flush()
        if f >= 1.0:
            sys.stderr.write("\n")
    return cb


def cmd_list(args) -> int:
    from .devices.base import backends

    found = False
    for b in backends():
        if b.name == "demo" and not args.all:
            continue  # test scanners: only with --all
        if not b.available():
            if args.all:
                print(f"[{b.name}] unavailable: {b.unavailable_reason()}")
            continue
        try:
            devs = b.list_devices()
        except Exception as e:
            print(f"[{b.name}] error: {e}")
            continue
        for d in devs:
            found = True
            print(f"{d.id}\t{d.label()}")
    if not found:
        print("No scanners found.  Network scanners can be used directly as escl:http://<address>/eSCL")
    return 0


def cmd_scan(args) -> int:
    from .workflow import Session

    st = _settings(args)
    inp = st.input
    for attr, val in (("device", args.device), ("scan_dpi", args.dpi), ("preview_dpi", args.preview_dpi),
                      ("mode", args.mode), ("source", args.source), ("bit_depth", args.depth)):
        if val is not None:
            setattr(inp, attr, val)
    if args.full_area:
        inp.scan_crop_area_only = False
    session = Session(st)
    cancel = threading.Event()
    try:
        dev = session.open()
        print(f"Scanner: {dev.info.label()}", file=sys.stderr)
        if not args.no_preview and st.crop.mode != "off":
            session.run_preview(_progress_bar("Preview"), cancel)
            print(f"Found {len(session.boxes_mm)} item(s) in the preview", file=sys.stderr)
        outcomes = session.run_scan(_progress_bar("Scan   "), cancel,
                                    status=lambda m: print(m, file=sys.stderr))
    except KeyboardInterrupt:
        cancel.set()
        print("\nCancelled.", file=sys.stderr)
        return 130
    finally:
        session.close()
    for oc in outcomes:
        for p in oc.paths + [r for r in oc.raw_paths if r not in oc.paths]:
            print(p)
        if args.debug and oc.paths:
            dbg = oc.paths[0].with_name(oc.paths[0].stem + "_crops.jpg")
            save_image(autocrop.debug_overlay(oc.scan.image, oc.boxes), dbg, dpi=oc.scan.dpi)
            print(dbg)
    return 0


def cmd_crop(args) -> int:
    from .workflow import Session

    st = _settings(args)
    session = Session(st)
    status = 0
    for f in args.files:
        path = Path(f)
        try:
            img, dpi = load_image(path)
        except Exception as e:
            print(f"{path}: cannot read ({e})", file=sys.stderr)
            status = 1
            continue
        dpi = args.dpi or dpi or 300.0
        if st.output.prefix == "scan" and not args.prefix:
            st.output.prefix = path.stem + "_"
        oc = session.crop_image(img, dpi)
        print(f"{path}: {len(oc.boxes)} item(s)", file=sys.stderr)
        for p in oc.paths + [r for r in oc.raw_paths if r not in oc.paths]:
            print(p)
        if args.debug:
            dbg = Path(st.output.path) / f"{path.stem}_crops.jpg"
            save_image(autocrop.debug_overlay(img, oc.boxes), dbg, dpi=dpi)
            print(dbg)
        st.output.prefix = args.prefix or "scan"
    return status


def cmd_diag(args) -> int:
    """Report what the scanners offer and try a small test scan."""
    import platform
    import traceback

    from . import __version__ as ver
    from .devices.base import ScanRequest, backends, open_device

    print(f"hasami {ver}, Python {platform.python_version()}, {platform.platform()}")
    for b in backends():
        print(f"[{b.name}] {'available' if b.available() else 'unavailable: ' + b.unavailable_reason()}")
    if sys.platform == "win32" and not (args.device and not args.device.startswith(("wia:", "twain:"))):
        from .devices import twain_backend, wia_backend

        if getattr(args, "live_test", False):
            print(twain_backend.live_test(args.device[6:] if args.device else None, args.dpi or 300,
                                          args.height or 100.0))
            return 0
        if not args.device or args.device.startswith("twain:"):
            print("=== TWAIN ===")
            native = args.device[6:] if args.device else None
            print(twain_backend.diagnose(native, args.save))
        if (not args.device or args.device.startswith("wia:")) and wia_backend.WiaBackend().available():
            print("=== WIA ===")
            native = args.device[4:] if args.device else None
            print(wia_backend.diagnose(native, None if not args.device else args.save))
        return 0
    if args.device:
        try:
            dev = open_device(args.device)
            print(f"Opened {dev.info.label()}")
            print(f"Capabilities: {dev.capabilities()}")
            res = dev.scan(ScanRequest(dpi=75, preview=True))
            print(f"Test scan: {res.image.shape} {res.image.dtype} at {res.dpi} dpi")
            if args.save:
                save_image(res.image, Path(args.save), dpi=res.dpi)
                print(f"Saved the test scan to {args.save}")
        except Exception:
            traceback.print_exc(file=sys.stdout)
    return 0


def cmd_gui(args) -> int:
    try:
        from .gui.app import main as gui_main
    except ImportError as e:
        print(f"The GUI needs PySide6 (pip install PySide6-Essentials): {e}", file=sys.stderr)
        return 1
    return gui_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hasami", description="Hasami Scan: scanning with smarter auto-crop.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd")

    lp = sub.add_parser("list", help="list scanners")
    lp.add_argument("--all", action="store_true", help="also show unavailable backends and the demo scanners")
    lp.set_defaults(func=cmd_list)

    sp = sub.add_parser("scan", help="scan, auto-crop and save")
    sp.add_argument("-d", "--device", help="device id from 'scanner list'")
    sp.add_argument("--dpi", type=int, help="scan resolution")
    sp.add_argument("--preview-dpi", type=int)
    sp.add_argument("--mode", choices=["color", "gray", "lineart"])
    sp.add_argument("--source", help="Flatbed, ADF, ADF Duplex, ...")
    sp.add_argument("--depth", type=int, choices=[8, 16],
                    help="bits per channel from the scanner (8 = 24-bit colour, 16 = 48-bit colour)")
    sp.add_argument("--no-preview", action="store_true", help="skip the preview pass; detect on the final scan")
    sp.add_argument("--full-area", action="store_true", help="always scan the whole bed")
    _add_common(sp)
    sp.set_defaults(func=cmd_scan)

    cp = sub.add_parser("crop", help="auto-crop existing scans")
    cp.add_argument("files", nargs="+")
    cp.add_argument("--dpi", type=float, help="resolution of the files if not recorded in them")
    _add_common(cp)
    cp.set_defaults(func=cmd_crop)

    dp = sub.add_parser("diag", help="report what the scanner offers and try a small test scan")
    dp.add_argument("-d", "--device", help="device id from 'scanner list' (default: the first scanner)")
    dp.add_argument("--save", help="save the test scan to this file")
    dp.add_argument("--live-test", action="store_true",
                    help="TWAIN: scan a 100 mm band and report whether the image comes in while scanning")
    dp.add_argument("--dpi", type=int, help="resolution for --live-test (default 300)")
    dp.add_argument("--height", type=float, help="height in mm of the band --live-test scans (default 100)")
    dp.set_defaults(func=cmd_diag)

    gp = sub.add_parser("gui", help="start the graphical interface")
    gp.set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        return cmd_gui(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
