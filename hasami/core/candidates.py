"""Candidate edge lines the auto-crop considers, collected for the learned line classifier.

Off unless a caller collects (the dataset builder in the private data repo; later the classifier
itself).  Collecting changes no crop: the engine only hands over what it computed anyway, plus
line pools computed only while collecting.  Context-local like the active profile.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_SINK: ContextVar[list | None] = ContextVar("hasami_candidates", default=None)


def active() -> bool:
    """Is someone collecting right now?"""
    return _SINK.get() is not None


def record(**fields) -> None:
    """Append one record (a side the engine looked at) when collecting."""
    sink = _SINK.get()
    if sink is not None:
        sink.append(fields)


@contextmanager
def collecting():
    """``with collecting() as recs:`` every record made inside the block goes to ``recs``."""
    recs: list[dict] = []
    token = _SINK.set(recs)
    try:
        yield recs
    finally:
        _SINK.reset(token)
