"""Per-frame IoU + class + 3D-distance tracker (spec §7.1).

The tracker holds short-term association memory: it rewrites each detection's
``object_id`` to a stable ``track_NNN`` slug so the SemanticMap downstream can
fuse repeated observations of the same physical thing.

Backends progression (spec):

    simple_iou_then_bytetrack
        The association now follows ByteTrack's two-stage *cascade*: live tracks
        are matched against high-confidence detections first, then any tracks
        still unmatched get a second chance against the leftover low-confidence
        detections. Low-confidence detections only *recover* existing tracks —
        they never spawn new ones (that is the ByteTrack insight: faint boxes are
        usually momentary occlusions/blur, not genuinely new objects).

Per-track motion is smoothed with a constant-velocity predict step (a light
Kalman stand-in): each track carries an EMA velocity for its 3D center and 2D
bbox, and association compares detections against the track's *predicted* pose
for the current frame rather than its last seen pose. This keeps ids stable
through fast, consistent motion.

Association score per (track, detection) pair, gated by class equality:

    score = iou_2d + w_dist * (1 - clip(||Δcenter_3d|| / max_center_distance, 0, 1))

Greedy matching descending by score within each stage; tracks with no match in a
frame increment ``miss_count`` and are pruned after ``persistence_frames``
consecutive misses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..spatial.object_lifter import ObjectState3D

log = logging.getLogger("pet_agent.tracker")

BBox = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]


@dataclass
class TrackedObject:
    """Tracker-internal state per active track (light, association-only)."""

    track_id: str
    class_label: str
    last_bbox: BBox
    last_center_3d: Vec3
    last_seen_frame: int
    miss_count: int = 0
    history: list[int] = field(default_factory=list)
    # Constant-velocity motion model (per-frame deltas, EMA-smoothed).
    vel_center_3d: Vec3 = (0.0, 0.0, 0.0)
    vel_bbox: BBox = (0.0, 0.0, 0.0, 0.0)


def _iou(a: BBox, b: BBox) -> float:
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


def _l2(a: Vec3, b: Vec3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _det_score(det: ObjectState3D) -> float:
    """Detection score for the ByteTrack high/low split.

    Real GroundingDINO detections carry ``confidence.detector`` (the box score);
    objects arriving via ``/perception/lifted`` or the eval runner often set only
    ``confidence.overall``. Take the best available signal so neither path is
    silently dropped.
    """
    return max(det.confidence.detector, det.confidence.overall)


def _ema(new: tuple[float, ...], old: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    return tuple(alpha * n + (1.0 - alpha) * o for n, o in zip(new, old, strict=True))


class Tracker:
    """Greedy IoU + 3D-distance tracker with ByteTrack cascade + velocity model.

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
        high_confidence: float = 0.5,
        velocity_smoothing: float = 0.5,
    ) -> None:
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.persistence_frames = persistence_frames
        self.distance_weight = distance_weight
        # Detections at/above this ``confidence.detector`` are matched first and
        # may start new tracks; below it they can only continue existing tracks.
        self.high_confidence = high_confidence
        self.velocity_smoothing = velocity_smoothing
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
        Track bookkeeping advances: matched tracks reset ``miss_count`` and
        update their velocity model, unmatched ones increment it and are pruned
        past ``persistence_frames``.
        """
        high_idx = [i for i, d in enumerate(detections) if _det_score(d) >= self.high_confidence]
        low_idx = [i for i, d in enumerate(detections) if _det_score(d) < self.high_confidence]

        # Stage 1: all live tracks vs high-confidence detections.
        matches = self._associate(detections, high_idx, set(self._tracks), frame_id)
        # Stage 2: tracks still unmatched vs low-confidence detections.
        remaining_tracks = set(self._tracks) - set(matches.values())
        matches.update(self._associate(detections, low_idx, remaining_tracks, frame_id))

        out: list[ObjectState3D] = []
        seen_this_frame: set[str] = set()

        for det_idx, track_id in matches.items():
            det = detections[det_idx]
            self._advance_track(self._tracks[track_id], det, frame_id)
            seen_this_frame.add(track_id)
            out.append(det.model_copy(update={"object_id": track_id}))

        # Only *high-confidence* unmatched detections become new tracks; faint
        # boxes that found no home are dropped (ByteTrack behaviour).
        for det_idx in high_idx:
            if det_idx in matches:
                continue
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
            seen_this_frame.add(new_id)
            out.append(det.model_copy(update={"object_id": new_id}))

        dropped_low = sum(1 for i in low_idx if i not in matches)

        # Decay unmatched tracks; prune if exhausted.
        to_prune: list[str] = []
        for tid, track in self._tracks.items():
            if tid in seen_this_frame:
                continue
            track.miss_count += 1
            if track.miss_count > self.persistence_frames:
                to_prune.append(tid)
        for tid in to_prune:
            del self._tracks[tid]

        log.debug(
            "frame %d: %d det (%d hi/%d lo) → %d matched, %d new, %d lo-dropped, %d pruned (%d active)",
            frame_id,
            len(detections),
            len(high_idx),
            len(low_idx),
            len(matches),
            len(seen_this_frame) - len(matches),
            dropped_low,
            len(to_prune),
            len(self._tracks),
        )
        return out

    # ── internals ───────────────────────────────────────────────────────────
    def _mint_id(self) -> str:
        tid = f"track_{self._next_id:03d}"
        self._next_id += 1
        return tid

    def _advance_track(self, track: TrackedObject, det: ObjectState3D, frame_id: int) -> None:
        """Fold a matched detection into a track, updating the velocity model."""
        gap = max(1, frame_id - track.last_seen_frame)
        a = self.velocity_smoothing
        inst_v_center = tuple(
            (n - o) / gap for n, o in zip(det.center_3d_world, track.last_center_3d, strict=True)
        )
        inst_v_bbox = tuple(
            (n - o) / gap for n, o in zip(det.bbox_xyxy, track.last_bbox, strict=True)
        )
        track.vel_center_3d = _ema(inst_v_center, track.vel_center_3d, a)  # type: ignore[assignment]
        track.vel_bbox = _ema(inst_v_bbox, track.vel_bbox, a)  # type: ignore[assignment]
        track.last_bbox = det.bbox_xyxy
        track.last_center_3d = det.center_3d_world
        track.last_seen_frame = frame_id
        track.miss_count = 0
        track.history.append(frame_id)

    def _predicted_pose(self, track: TrackedObject, frame_id: int) -> tuple[BBox, Vec3]:
        """Project the track's last pose forward by its velocity to ``frame_id``."""
        gap = min(self.persistence_frames + 1, max(1, frame_id - track.last_seen_frame))
        center = tuple(
            c + v * gap for c, v in zip(track.last_center_3d, track.vel_center_3d, strict=True)
        )
        bbox = tuple(b + v * gap for b, v in zip(track.last_bbox, track.vel_bbox, strict=True))
        return bbox, center  # type: ignore[return-value]

    def _associate(
        self,
        detections: list[ObjectState3D],
        det_indices: list[int],
        track_ids: set[str],
        frame_id: int,
    ) -> dict[int, str]:
        """Greedy descending-score matching gated by class equality.

        Matches the given ``det_indices`` against the given ``track_ids`` using
        each track's velocity-predicted pose. Returns ``matches[det_idx] = track_id``.
        """
        if not track_ids or not det_indices:
            return {}

        predicted = {tid: self._predicted_pose(self._tracks[tid], frame_id) for tid in track_ids}

        pairs: list[tuple[float, int, str]] = []
        for di in det_indices:
            det = detections[di]
            for tid in track_ids:
                track = self._tracks[tid]
                if det.class_label != track.class_label:
                    continue
                pred_bbox, pred_center = predicted[tid]
                iou = _iou(det.bbox_xyxy, pred_bbox)
                dist = _l2(det.center_3d_world, pred_center)
                # Accept if either bbox IoU is healthy OR 3D distance is tight.
                if iou < self.min_iou and dist > self.max_center_distance:
                    continue
                dist_norm = min(1.0, dist / max(1e-6, self.max_center_distance))
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
        return matches
