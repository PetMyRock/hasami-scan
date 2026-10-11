"""Run slow work (scanning, detection) off the GUI thread."""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    progress = Signal(float)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Call ``fn(progress, cancel, status)`` in the thread pool.

    ``progress`` takes 0..1, ``status`` takes a message, ``cancel`` is a
    ``threading.Event`` the work should check.  Results and errors come back
    through Qt signals on the GUI thread.
    """

    def __init__(self, fn: Callable[..., Any]):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()
        self.cancel = threading.Event()
        self.setAutoDelete(False)

    def run(self) -> None:
        try:
            result = self.fn(self.signals.progress.emit, self.cancel, self.signals.status.emit)
        except Exception as e:  # report everything; the GUI decides what to show
            from ..devices.base import ScanCancelled

            if isinstance(e, ScanCancelled) or self.cancel.is_set():
                # A driver that had to be ended by force comes with advice (shown in a box).
                self.signals.failed.emit(getattr(e, "warning", "") or "Cancelled")
            else:
                traceback.print_exc()
                msg = str(e) or e.__class__.__name__
                log = _log_error()
                if log:
                    msg += f"\n\nDetails were saved to {log}"
                self.signals.failed.emit(msg)
            return
        self.signals.finished.emit(result)


def _log_error() -> str:
    """Append the current exception's traceback to hasami.log; its path or ''."""
    import datetime

    from ..settings import config_dir

    try:
        path = config_dir() / "hasami.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{traceback.format_exc()}\n")
        return str(path)
    except OSError:
        return ""


def start(task: Task) -> Task:
    QThreadPool.globalInstance().start(task)
    return task
