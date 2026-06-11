"""Per-frame object association (spec §7 Phase 4 / §14.6.1)."""

from .protocol import TrackerBackend, make_tracker
from .tracker import TrackedObject, Tracker

__all__ = ["Tracker", "TrackedObject", "TrackerBackend", "make_tracker"]
