import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Never read or overwrite the user's real settings (CLI and GUI load them).
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="scanner-test-config-")
os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _own_config_folder(tmp_path_factory, monkeypatch):
    """Each test gets its own settings folder: the GUI saves its settings on close and the CLI loads them, so
    one test's booklet-mode settings made another test's CLI crop of three photos find two."""
    folder = tmp_path_factory.mktemp("config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(folder))
    monkeypatch.setenv("APPDATA", str(folder))
