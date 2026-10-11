# PyInstaller spec: Hasami Scan as one folder (onedir) with two programs,
#   HasamiScan.exe  - the window (no console)
#   hasami.exe      - the command line (hasami list / diag / scan ...)
# Run through build.py, which passes the icon:  pyinstaller packaging/scanner.spec
import os

HERE = os.path.dirname(os.path.abspath(SPEC))
ROOT = os.path.dirname(HERE)
ICON = os.environ.get("SCANNER_ICON") or None

# The TWAIN helper script stays a plain file next to the backend: with a
# Python set in "TWAIN: Python to use" the app still runs it with that Python.
datas = [(os.path.join(ROOT, "hasami", "devices", "twain_helper.py"), os.path.join("hasami", "devices"))]
hidden = [
    "hasami.devices.demo_backend", "hasami.devices.escl_backend", "hasami.devices.file_backend",
    "hasami.devices.sane_backend", "hasami.devices.twain_backend", "hasami.devices.wia_backend",
]
excludes = ["tkinter", "pytest", "IPython", "matplotlib", "scipy", "twain", "PyInstaller"]

gui = Analysis([os.path.join(HERE, "gui_main.py")], pathex=[ROOT], datas=datas, hiddenimports=hidden,
               excludes=excludes)
cli = Analysis([os.path.join(HERE, "cli_main.py")], pathex=[ROOT], datas=datas, hiddenimports=hidden,
               excludes=excludes)

gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="HasamiScan", icon=ICON,
              console=False, upx=False)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="hasami", icon=ICON,
              console=True, upx=False)

COLLECT(gui_exe, cli_exe, gui.binaries, gui.datas, cli.binaries, cli.datas, name="HasamiScan", upx=False)
