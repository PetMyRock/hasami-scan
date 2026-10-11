@echo off
rem Set up this folder (a git clone of hasami-scan) for working on the code,
rem with Claude Code or by hand: makes .venv, installs the app editable and
rem runs the tests.
cd /d "%~dp0"
if not exist .git (
    echo This is not a git clone. See README: "Working on the code".
    pause
    exit /b 1
)
rem The app needs a 64-bit Python (Qt has no 32-bit packages); a 32-bit one
rem (with pytwain) is only needed for 32-bit TWAIN drivers such as the old
rem EPSON Scan, and is found by the app by itself.
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "import struct, sys; sys.exit(struct.calcsize('P') != 8)"
    if errorlevel 1 (
        echo The existing .venv uses a 32-bit Python: making a new one.
        rmdir /s /q .venv
    )
)
if not exist .venv\Scripts\python.exe (
    echo Making .venv with 64-bit Python ...
    py -3-64 -m venv .venv
    if errorlevel 1 (
        echo No 64-bit Python found by "py -3-64". Install it from python.org.
        pause
        exit /b 1
    )
)
echo Installing the app (editable: code changes count at once) and the test tools ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[gui,twain,dev]" pywin32 zeroconf
if errorlevel 1 (
    echo Installing failed - see the messages above.
    pause
    exit /b 1
)
echo.
echo Running the tests (about 3 minutes) ...
.venv\Scripts\python.exe -m pytest -q
echo.
echo Done.  Start the app from here with:  .venv\Scripts\hasami-scan
echo Start Claude Code in this folder with:  claude
pause
