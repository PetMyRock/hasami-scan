"""Private backends: packages in <settings dir>/private loaded by path (generated packages only)."""
from pathlib import Path

from hasami.devices import base
from hasami.devices.private import private_backends

GOOD = '''
from hasami.devices.base import Backend, DeviceInfo
from .helper import NAME


class PrivateBackend(Backend):
    name = NAME

    def available(self):
        return True

    def list_devices(self):
        return [DeviceInfo(f"{NAME}:one", "Private scanner", backend=NAME)]

    def open(self, native_id):
        raise NotImplementedError


def backend():
    return PrivateBackend()
'''


def make(folder: Path, name: str, backend_src: str, helper: str = 'NAME = "privatetest"\n') -> None:
    pkg = folder / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "helper.py").write_text(helper)
    (pkg / "backend.py").write_text(backend_src)


def test_private_backend_is_listed(tmp_path, monkeypatch):
    make(tmp_path, "goodpkg", GOOD)
    found = private_backends(tmp_path)
    assert [b.name for b in found] == ["privatetest"]
    monkeypatch.setattr("hasami.devices.private.private_backends", lambda folder=None: found)
    assert "privatetest" in [b.name for b in base.backends()]
    assert any(d.id == "privatetest:one" for d in base.list_devices())


def test_broken_private_module_is_skipped(tmp_path, caplog):
    make(tmp_path, "badsyntax", "def backend(:\n")
    make(tmp_path, "raises", "raise RuntimeError('boom')\n")
    make(tmp_path, "noentry", "X = 1\n")
    make(tmp_path, "goodpkg", GOOD)
    found = private_backends(tmp_path)
    assert [b.name for b in found] == ["privatetest"]
    assert "badsyntax" in caplog.text and "raises" in caplog.text


def test_no_private_folder_no_change(tmp_path):
    assert private_backends(tmp_path / "missing") == []
    assert private_backends(tmp_path) == []  # empty folder
