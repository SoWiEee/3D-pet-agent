"""Per-frame IoU + class + 3D-distance tracker (spec §7.1).

The tracker holds short-term association memory: it rewrites each detection's
``object_id`` to a stable ``track_NNN`` slug so the SemanticMap downstream can
fuse repeated observations of the same physical thing.

Backends progression (spec):

    simple_iou_then_bytetrack
        Phase 4 ships the simple variant. The ByteTrack upgrade slots in by
        replacing ``_associate`` once detections are stable enough to justify
        the dependency.

Association score per (track, detection) pair, gated by class equality:

    score = iou_2d + w_dist * (1 - clip(||Δcenter_3d|| / max_center_distance, 0, 1))

Greedy matching descending by score; tracks with no match in a frame increment
``miss_count`` and are pruned after ``persistence_frames`` consecutive misses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..spatial.object_lifter import ObjectState3D

log = logging.getLogger("pet_agent.tracker")


@dataclass
class TrackedObject:
    """Tracker-internal state per active track (light, association-only)."""

    track_id: str
    class_label: str
    last_bbox: tuple[float, float, float, float]
    last_center_3d: tuple[float, float, float]
    last_seen_frame: int
    miss_count: int = 0
    history: list[int] = field(default_factory=list)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _l2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


class Tracker:
    """Simple greedy IoU + 3D-distance tracker.

    Stateless across resets; reusable across frames. Not thread-safe — drive
    from one perception loop only.
    """

    def __init__(
        self,
        *,
        min_iou: float = 0.35,
        max_center_distance: float = 0.20,
        persistence_frames: int = 3,
        distance_weight: float = 0.4,
    ) -> None:
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.persistence_frames = persistence_frames
        self.distance_weight = distance_weight
        self._tracks: dict[str, TrackedObject] = {}
        self._next_id = 1

    # ── public ──────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> dict[str, TrackedObject]:
        return dict(self._tracks)

    def update(self, detections: list[ObjectState3D], frame_id: int) -> list[ObjectState3D]:
        """Associate ``detections`` to live tracks and rewrite ``object_id``.

        Returns a new list of ``ObjectState3D`` (originals unmodified) with
        ``object_id`` set to the matched track id (existing or newly minted).
        Track bookkeeping advances: matched tracks reset ``miss_count``,
        unmatched ones increment it and are pruned past ``persistence_frames``.
        """
        matches, unmatched_det_idx = self._associate(detections)

        out: list[ObjectState3D] = []
        matched_track_ids: set[str] = set()

        # Matched detections take over their assigned track id.
        for det_idx, track_id in matches.items():
            det = detections[det_idx]
            track = self._tracks[track_id]
            track.last_bbox = det.bbox_xyxy
            track.last_center_3d = det.center_3d_world
            track.last_seen_frame = frame_id
            track.miss_count = 0
            track.history.append(frame_id)
            matched_track_ids.add(track_id)
            out.append(det.model_copy(update={"object_id": track_id}))

        # Unmatched detections become new tracks (and count as seen-this-frame
        # so the decay pass below doesn't immediately tick their miss_count).
        for det_idx in unmatched_det_idx:
            det = detections[det_idx]
            new_id = self._mint_id()
            self._tracks[new_id] = TrackedObject(
                track_id=new_id,
                class_label=det.class_label,
                last_bbox=det.bbox_xyxy,
                last_center_3d=det.center_3d_world,
                last_seen_frame=frame_id,
                history=[frame_id],
            )
            matched_track_ids.add(new_id)
            out.append(det.model_copy(update={"object_id": new_id}))

        # Decay unmatched tracks; prune if exhausted.
        to_prune: list[str] = []
        for tid, track in self._tracks.items():
            if tid in matched_track_ids:
                continue
            track.miss_count += 1
            if track.miss_count > self.persistence_frames:
                to_prune.append(tid)
        for tid in to_prune:
            del self._tracks[tid]

        log.debug(
            "frame %d: %d det → %d matched, %d new, %d pruned (%d active)",
            frame_id,
            len(detections),
            len(matches),
            len(unmatched_det_idx),
            len(to_prune),
            len(self._tracks),
        )
        return out

    # ── internals ───────────────────────────────────────────────────────────
    def _mint_id(self) -> str:
        tid = f"track_{self._next_id:03d}"
        self._next_id += 1
        return tid

    def _associate(self, detections: list[ObjectState3D]) -> tuple[dict[int, str], list[int]]:
        """Greedy descending-score matching gated by class equality.

        Returns ``(matches[det_idx] = track_id, unmatched_det_indices)``.
        """
        if not self._tracks or not detections:
            return {}, list(range(len(detections)))

        # Build all candidate pairs above gate.
        pairs: list[tuple[float, int, str]] = []
        for di, det in enumerate(detections):
            for tid, track in self._tracks.items():
                if det.class_label != track.class_label:
                    continue
                iou = _iou(det.bbox_xyxy, track.last_bbox)
                dist = _l2(det.center_3d_world, track.last_center_3d)
                dist_norm = min(1.0, dist / max(1e-6, self.max_center_distance))
                # Accept if either bbox IoU is healthy OR 3D distance is tight.
                if iou < self.min_iou and dist > self.max_center_distance:
                    continue
                score = iou + self.distance_weight * (1.0 - dist_norm)
                pairs.append((score, di, tid))

        pairs.sort(reverse=True)
        matches: dict[int, str] = {}
        claimed_tracks: set[str] = set()
        for _, di, tid in pairs:
            if di in matches or tid in claimed_tracks:
                continue
            matches[di] = tid
            claimed_tracks.add(tid)

        unmatched = [i for i in range(len(detections)) if i not in matches]
        return matches, unmatched
