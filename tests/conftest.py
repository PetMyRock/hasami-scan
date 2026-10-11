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
