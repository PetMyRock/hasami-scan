"""Private backends: scanner backends that are not part of this repository.

Each subfolder of ``<settings dir>/private`` that holds a package with a ``backend.py`` defining ``backend()`` is
imported by path (this also works in the installed app) and ``backend()`` is called; the result joins the
backend registry. A package that fails to load is skipped with a log line and never breaks device listing.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from .base import Backend

log = logging.getLogger(__name__)

_loaded: dict[Path, object | None] = {}


def private_folder() -> Path:
    from ..settings import config_dir

    return config_dir() / "private"


def _load(pkg: Path):
    if pkg in _loaded:
        return _loaded[pkg]
    name = f"hasami_private_{pkg.name}"
    module = None
    try:
        spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        module = importlib.import_module(f"{name}.backend")
        if not callable(getattr(module, "backend", None)):
            log.warning("private backend %s: backend.py defines no backend()", pkg.name)
            module = None
    except Exception as e:  # a broken private package must not break the app
        log.warning("private backend %s could not be loaded: %s", pkg.name, e)
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
        module = None
    _loaded[pkg] = module
    return module


def private_backends(folder: Path | None = None) -> list[Backend]:
    folder = private_folder() if folder is None else Path(folder)
    if not folder.is_dir():
        return []
    found: list[Backend] = []
    for pkg in sorted(p for p in folder.iterdir() if (p / "backend.py").is_file() and (p / "__init__.py").is_file()):
        module = _load(pkg.resolve())
        if module is None:
            continue
        try:
            found.append(module.backend())
        except Exception as e:
            log.warning("private backend %s: backend() failed: %s", pkg.name, e)
    return found
