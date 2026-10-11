# PyInstaller spec: the TWAIN helper as its own small program, built once with a
# 32-bit Python (twain_helper32.exe, for 32-bit drivers such as the old EPSON
# Scan) and once with a 64-bit one (twain_helper64.exe).  Each needs only the
# standard library and pytwain==2.3.0.  build.py sets SCANNER_HELPER_NAME.
import os

HERE = os.path.dirname(os.path.abspath(SPEC))
ROOT = os.path.dirname(HERE)
NAME = os.environ.get("SCANNER_HELPER_NAME", "twain_helper")

a = Analysis([os.path.join(ROOT, "hasami", "devices", "twain_helper.py")], hiddenimports=["twain"],
             excludes=["tkinter", "numpy", "PIL", "PySide6", "cv2", "pytest"])
exe = EXE(PYZ(a.pure), a.scripts, [], exclude_binaries=True, name=NAME, console=True, upx=False)
COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
