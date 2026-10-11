"""Build the Windows installer: dist/HasamiScanSetup-<version>.exe

    python packaging/build.py --python32 C:/path/to/32-bit/python.exe [--python64 ...] [--iscc ...]

Run it with the 64-bit Python that has the app and PyInstaller installed
(``pip install -e .[gui,wia,twain] pyinstaller``).  ``--python32`` / ``--python64``
are Pythons with ``pytwain==2.3.0`` and ``pyinstaller`` that build the TWAIN
helper programs (default for 64-bit: this Python).  Steps:

1. icon (build/hasami.ico),
2. the app folder dist/HasamiScan (HasamiScan.exe + hasami.exe),
3. the helpers into dist/HasamiScan/twain64 and dist/HasamiScan/twain32,
4. licence texts of the bundled packages into dist/HasamiScan/licenses,
5. Inno Setup (iscc) -> dist/HasamiScanSetup-<version>.exe (skipped with --no-installer
   or when iscc is not found).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"
APP = DIST / "HasamiScan"

#: Packages shipped in the app folder (and the helpers): their licence files go in licenses/.
BUNDLED = ["numpy", "opencv-python-headless", "pillow", "PySide6_Essentials", "shiboken6", "pywin32",
           "zeroconf", "ifaddr", "pyinstaller"]
HELPER_PACKAGES = ["pytwain"]


def version() -> str:
    text = (ROOT / "hasami" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def run(cmd: list[str], **env) -> None:
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, env=dict(os.environ, **env))


def pyinstaller(python: str, spec: str, work: str, **env) -> None:
    run([python, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", DIST,
         "--workpath", BUILD / work, HERE / spec], **env)


def licence_files(dist_name: str, python: str | None = None) -> list[tuple[str, str]]:
    """(file name, text) of a package's licence files, from its installed metadata."""
    if python and Path(python).resolve() != Path(sys.executable).resolve():
        code = ("import json, sys\nfrom importlib import metadata\n"
                "d = metadata.distribution(sys.argv[1])\n"
                "out = [(str(f), f.read_text()) for f in (d.files or []) "
                "if any(k in str(f).upper() for k in ('LICENSE', 'LICENCE', 'COPYING', 'NOTICE'))]\n"
                "print(json.dumps(out))")
        r = subprocess.run([python, "-c", code, dist_name], capture_output=True, text=True)
        import json
        return [tuple(x) for x in json.loads(r.stdout)] if r.returncode == 0 else []
    try:
        d = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return []
    out = []
    for f in d.files or []:
        if any(k in str(f).upper() for k in ("LICENSE", "LICENCE", "COPYING", "NOTICE")):
            try:
                out.append((str(f), Path(f.locate()).read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    return out


def write_licences(python32: str | None) -> None:
    folder = APP / "licenses"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "LICENSE", folder / "HasamiScan-MIT.txt")
    shutil.copy(HERE / "licenses" / "GPL-3.0.txt", folder / "GPL-3.0.txt")  # the LGPL (Qt) builds on it
    py_lic = Path(sys.base_prefix) / "LICENSE.txt"
    if py_lic.is_file():
        shutil.copy(py_lic, folder / "Python.txt")
    found = []
    for name in BUNDLED:
        for fname, text in licence_files(name):
            found.append((name, fname, text))
    for name in HELPER_PACKAGES:
        for fname, text in licence_files(name, python32) or licence_files(name):
            found.append((name, fname, text))
    for name, fname, text in found:
        safe = re.sub(r"[^\w.-]+", "_", f"{name}-{fname}")
        (folder / safe).write_text(text, encoding="utf-8")
    write_qt_notice(folder)
    names = sorted({n for n, _, _ in found})
    print(f"licence texts: {len(found)} files for {', '.join(names)}")
    missing = [n for n in BUNDLED + HELPER_PACKAGES if n not in names]
    if missing:
        print(f"  (no licence file found for: {', '.join(missing)})")


QT_NOTICE = """\
Hasami Scan uses Qt for Python (PySide6, shiboken6) and the Qt libraries under the
GNU Lesser General Public License version 3 (LGPL-3.0).  They are shipped
unchanged as separate files (the PySide6 and shiboken6 folders in _internal)
and may be replaced with other builds of the same version.  Source code:
https://code.qt.io/ and https://download.qt.io/official_releases/QtForPython/
The LGPL-3.0 adds to the GNU General Public License version 3
(GPL-3.0.txt in this folder).

"""


def write_qt_notice(folder: Path) -> None:
    """The LGPL-3.0 notice for Qt / PySide6 (their wheels ship only the commercial licence text).

    The LGPL-3.0 text is taken from OpenCV's third-party notices (it ships it for Qt too)."""
    text = ""
    try:
        src = Path(metadata.distribution("opencv-python-headless").locate_file("cv2/LICENSE-3RD-PARTY.txt"))
        notices = src.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^\s*GNU LESSER GENERAL PUBLIC LICENSE\s*\n\s*Version 3, 29 June 2007.*?(?=^-{20,})",
                      notices, re.S | re.M)
        text = m.group(0) if m else ""
    except Exception:
        pass
    if not text:
        print("  (LGPL-3.0 text not found: only the notice is written)")
    (folder / "Qt-PySide6-LGPL-3.0.txt").write_text(QT_NOTICE + text, encoding="utf-8")


def find_iscc(given: str | None) -> str | None:
    if given:
        return given
    found = shutil.which("iscc")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            p = Path(base) / "Inno Setup 6" / "ISCC.exe"
            if p.is_file():
                return str(p)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python32", help="32-bit Python with pytwain + pyinstaller (twain_helper32.exe)")
    ap.add_argument("--python64", default=sys.executable,
                    help="64-bit Python with pytwain + pyinstaller (twain_helper64.exe); default: this one")
    ap.add_argument("--iscc", help="Inno Setup's ISCC.exe (default: found on PATH or in Program Files)")
    ap.add_argument("--no-installer", action="store_true", help="stop after the app folder")
    args = ap.parse_args()
    for k in ("python32", "python64", "iscc"):  # CreateProcess wants full Windows paths
        if getattr(args, k):
            setattr(args, k, str(Path(getattr(args, k)).resolve()))
    if sys.platform != "win32":
        ap.error("the installer is built on Windows")
    ver = version()
    BUILD.mkdir(exist_ok=True)
    icon = BUILD / "hasami.ico"
    run([sys.executable, HERE / "make_icon.py", icon])
    if APP.exists():
        shutil.rmtree(APP)
    pyinstaller(sys.executable, "hasami.spec", "app", SCANNER_ICON=str(icon))
    helpers = [("twain64", "twain_helper64", args.python64)]
    if args.python32:
        helpers.append(("twain32", "twain_helper32", args.python32))
    else:
        print("WARNING: no --python32: the app will not reach 32-bit TWAIN drivers (old EPSON Scan)")
    for folder, name, python in helpers:
        pyinstaller(python, "twain_helper.spec", name, SCANNER_HELPER_NAME=name)
        shutil.move(str(DIST / name), str(APP / folder))
    write_licences(args.python32)
    print(f"app folder: {APP}")
    if args.no_installer:
        return 0
    iscc = find_iscc(args.iscc)
    if not iscc:
        print("Inno Setup (ISCC.exe) not found: installer skipped")
        return 0
    run([iscc, f"/DAppVersion={ver}", f"/DAppDir={APP}", f"/DOutDir={DIST}", f"/DIconFile={icon}",
         HERE / "installer.iss"])
    print(f"installer: {DIST / f'HasamiScanSetup-{ver}.exe'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
