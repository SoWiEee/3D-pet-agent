"""§14.6.1 — tracker backend factory selection + fallback."""

from __future__ import annotations

from src.config import TrackingThresholds
from src.tracking import Tracker, TrackerBackend, make_tracker


def test_default_backend_returns_greedy_tracker() -> None:
    cfg = TrackingThresholds()  # backend = "simple_iou_then_bytetrack"
    t = make_tracker(cfg)
    assert isinstance(t, Tracker)


def test_unknown_backend_falls_back_to_greedy() -> None:
    cfg = TrackingThresholds(backend="does_not_exist")
    t = make_tracker(cfg)
    assert isinstance(t, Tracker)


def test_env_override_selects_greedy(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_TRACKER", "greedy")
    cfg = TrackingThresholds(backend="supervision_bytetrack")
    t = make_tracker(cfg)
    assert isinstance(t, Tracker)


def test_greedy_tracker_satisfies_protocol() -> None:
    # The runtime-checkable Protocol structurally validates the backend surface
    # (update / reset / active_tracks) the perception loop depends on.
    t = make_tracker(TrackingThresholds())
    assert isinstance(t, TrackerBackend)


def test_bytetrack_backend_builds_adapter_when_available(monkeypatch) -> None:
    import pytest

    pytest.importorskip("supervision")
    monkeypatch.delenv("PET_AGENT_TRACKER", raising=False)
    from src.tracking.bytetrack_adapter import ByteTrackTracker

    cfg = TrackingThresholds(backend="supervision_bytetrack")
    t = make_tracker(cfg)
    assert isinstance(t, ByteTrackTracker)
