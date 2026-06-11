"""Tracker backend protocol + factory (spec §14.6.1).

Every tracker backend exposes the same surface so the perception loop is
backend-agnostic. ``make_tracker`` selects the backend from config, honouring a
``PET_AGENT_TRACKER`` env override, and always degrades to the greedy
``Tracker`` rather than raising — a missing heavy dependency must never gate the
demo.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..spatial.object_lifter import ObjectState3D
from .tracker import Tracker

if TYPE_CHECKING:
    from ..config import TrackingThresholds

log = logging.getLogger("pet_agent.tracking")

# Config/env values that select the real ByteTrack adapter.
_BYTETRACK_NAMES = {"supervision_bytetrack", "bytetrack"}


@runtime_checkable
class TrackerBackend(Protocol):
    """The surface the perception loop depends on."""

    def update(self, detections: list[ObjectState3D], frame_id: int) -> list[ObjectState3D]: ...

    def reset(self) -> None: ...

    @property
    def active_tracks(self) -> dict[str, object]: ...


def make_tracker(cfg: TrackingThresholds) -> TrackerBackend:
    """Build the configured tracker. ``PET_AGENT_TRACKER`` env overrides
    ``cfg.backend``. Unknown names or a failed ByteTrack import → greedy."""
    choice = (os.environ.get("PET_AGENT_TRACKER") or cfg.backend or "").strip().lower()

    if choice in _BYTETRACK_NAMES:
        try:
            from .bytetrack_adapter import ByteTrackTracker

            return ByteTrackTracker(
                # The high/low ByteTrack split threshold. TODO: promote to a
                # cfg.high_confidence field if §14.6.x needs it tunable.
                high_confidence=0.5,
                persistence_frames=cfg.persistence_frames,
                min_iou=cfg.min_iou,
            )
        except Exception as e:  # noqa: BLE001 — heavy dep optional
            log.warning("ByteTrack unavailable (%s); using greedy tracker", e)

    return Tracker(
        min_iou=cfg.min_iou,
        max_center_distance=cfg.max_center_distance,
        persistence_frames=cfg.persistence_frames,
    )
