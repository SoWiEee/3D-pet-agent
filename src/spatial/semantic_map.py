"""Persistent semantic map (spec §3.4 + §7.1 Phase 4).

Holds long-term world model: one ``ObjectState3D`` per tracked object, fused
across frames. The tracker upstream rewrites detection ``object_id`` to a
stable ``track_NNN`` slug; the map keys on that and merges observations.

Fusion rules
------------
- **position**:  ``p ← α·p_obs + (1-α)·p_old``  (configurable via ``position_alpha``)
- **extent**:    same EMA as position
- **median_depth / depth_uncertainty**: EMA
- **confidence**: Bayesian log-odds-style update::

      c ← clip(c_obs + (1 - c_obs) · c_old, 0, 1)

  On a missed frame, the same field decays by ``confidence_decay`` per frame.

Status machine (per spec):

    tracked         seen this frame
    occluded        unseen ≤ persistence_frames frames in a row
    stale           unseen > persistence_frames but ≤ stale_frames
    lost            unseen > stale_frames        (still retained until prune)

Persistence
-----------
``save(path)`` writes JSON with sorted keys + stable ordering so save→load→save
is byte-identical (acceptance criterion). ``load(path)`` returns a fresh map.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .object_lifter import ObjectConfidence, ObjectState3D, TrackingStatus

log = logging.getLogger("pet_agent.semantic_map")

_SCHEMA_VERSION = 1


def _lerp(a: float, b: float, alpha: float) -> float:
    return alpha * a + (1.0 - alpha) * b


def _lerp_tuple3(
    a: tuple[float, float, float], b: tuple[float, float, float], alpha: float
) -> tuple[float, float, float]:
    return (_lerp(a[0], b[0], alpha), _lerp(a[1], b[1], alpha), _lerp(a[2], b[2], alpha))


def _bayes_update(c_old: float, c_obs: float) -> float:
    return max(0.0, min(1.0, c_obs + (1.0 - c_obs) * c_old))


class SemanticMap:
    """Persistent ObjectState3D store with EMA fusion and status decay."""

    def __init__(
        self,
        *,
        map_id: str = "session",
        coordinate_frame: str = "world",
        position_alpha: float = 0.6,
        confidence_decay: float = 0.08,
        persistence_frames: int = 3,
        stale_frames: int = 30,
        lost_frames: int = 120,
    ) -> None:
        self.map_id = map_id
        self.coordinate_frame = coordinate_frame
        self.position_alpha = position_alpha
        self.confidence_decay = confidence_decay
        self.persistence_frames = persistence_frames
        self.stale_frames = stale_frames
        self.lost_frames = lost_frames
        self._objects: dict[str, ObjectState3D] = {}
        self.last_frame_id: int = -1
        self.last_updated: float = 0.0

    # ── public API ──────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._objects.clear()
        self.last_frame_id = -1
        self.last_updated = 0.0

    @property
    def objects(self) -> dict[str, ObjectState3D]:
        return dict(self._objects)

    def get(self, object_id: str) -> ObjectState3D | None:
        return self._objects.get(object_id)

    def values(self) -> list[ObjectState3D]:
        # Sorted by object_id for deterministic output.
        return [self._objects[k] for k in sorted(self._objects)]

    def update(self, observations: list[ObjectState3D], frame_id: int) -> None:
        """Fuse ``observations`` (tracked, with stable ``object_id``) into the map.

        Each observation either updates an existing entry (EMA + Bayes) or is
        inserted fresh. All previously-known objects not present this frame
        decay one step and may transition status.
        """
        seen_now: set[str] = set()
        for obs in observations:
            oid = obs.object_id
            seen_now.add(oid)
            prior = self._objects.get(oid)
            if prior is None:
                self._objects[oid] = obs.model_copy(
                    update={
                        "last_seen_frame": frame_id,
                        "tracking_status": "tracked",
                    }
                )
                continue
            fused = self._fuse(prior, obs, frame_id)
            self._objects[oid] = fused

        # Decay unseen.
        for oid, prior in list(self._objects.items()):
            if oid in seen_now:
                continue
            self._objects[oid] = self._decay(prior, frame_id)

        # Prune fully-lost.
        to_drop = [
            oid
            for oid, o in self._objects.items()
            if (frame_id - o.last_seen_frame) > self.lost_frames
        ]
        for oid in to_drop:
            del self._objects[oid]

        self.last_frame_id = frame_id
        self.last_updated = time.time()
        log.debug(
            "frame %d: map size=%d (seen=%d, pruned=%d)",
            frame_id,
            len(self._objects),
            len(seen_now),
            len(to_drop),
        )

    # ── fusion + decay ──────────────────────────────────────────────────────
    def _fuse(self, prior: ObjectState3D, obs: ObjectState3D, frame_id: int) -> ObjectState3D:
        a = self.position_alpha
        center = _lerp_tuple3(obs.center_3d_world, prior.center_3d_world, a)
        extent = _lerp_tuple3(obs.extent_3d, prior.extent_3d, a)
        median_depth = _lerp(obs.median_depth, prior.median_depth, a)
        depth_uncertainty = _lerp(obs.depth_uncertainty, prior.depth_uncertainty, a)

        # Bayesian-ish confidence update per channel.
        pc, oc = prior.confidence, obs.confidence
        fused_conf = ObjectConfidence(
            detector=_bayes_update(pc.detector, oc.detector),
            mask_quality=_bayes_update(pc.mask_quality, oc.mask_quality),
            depth_quality=_bayes_update(pc.depth_quality, oc.depth_quality),
            tracking=min(1.0, pc.tracking + 0.1),  # slow ramp on success
            overall=_bayes_update(pc.overall, oc.overall),
        )

        return obs.model_copy(
            update={
                "object_id": prior.object_id,  # keep stable id
                "center_3d_world": center,
                "extent_3d": extent,
                "median_depth": median_depth,
                "depth_uncertainty": depth_uncertainty,
                "confidence": fused_conf,
                "last_seen_frame": frame_id,
                "tracking_status": "tracked",
            }
        )

    def _decay(self, prior: ObjectState3D, frame_id: int) -> ObjectState3D:
        missed = frame_id - prior.last_seen_frame
        status: TrackingStatus
        if missed <= self.persistence_frames:
            status = "occluded"
        elif missed <= self.stale_frames:
            status = "stale"
        else:
            status = "lost"
        pc = prior.confidence
        new_overall = max(0.0, pc.overall - self.confidence_decay)
        new_tracking = max(0.0, pc.tracking - self.confidence_decay)
        return prior.model_copy(
            update={
                "tracking_status": status,
                "confidence": pc.model_copy(
                    update={"overall": new_overall, "tracking": new_tracking}
                ),
            }
        )

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "map_id": self.map_id,
            "coordinate_frame": self.coordinate_frame,
            "last_frame_id": self.last_frame_id,
            "last_updated": self.last_updated,
            "params": {
                "position_alpha": self.position_alpha,
                "confidence_decay": self.confidence_decay,
                "persistence_frames": self.persistence_frames,
                "stale_frames": self.stale_frames,
                "lost_frames": self.lost_frames,
            },
            # Sorted by object_id for byte-stable serialization.
            "objects": [self._objects[k].model_dump() for k in sorted(self._objects)],
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> SemanticMap:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        params = data.get("params", {})
        m = cls(
            map_id=data.get("map_id", "session"),
            coordinate_frame=data.get("coordinate_frame", "world"),
            position_alpha=params.get("position_alpha", 0.6),
            confidence_decay=params.get("confidence_decay", 0.08),
            persistence_frames=params.get("persistence_frames", 3),
            stale_frames=params.get("stale_frames", 30),
            lost_frames=params.get("lost_frames", 120),
        )
        m.last_frame_id = int(data.get("last_frame_id", -1))
        m.last_updated = float(data.get("last_updated", 0.0))
        for obj in data.get("objects", []):
            state = ObjectState3D(**obj)
            m._objects[state.object_id] = state
        return m
