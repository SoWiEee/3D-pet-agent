"""§14.6.1 — tracker backend factory selection + fallback."""

from __future__ import annotations

from src.config import TrackingThresholds
from src.tracking import Tracker, make_tracker


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
