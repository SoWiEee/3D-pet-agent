"""supervision ByteTrack adapter (spec §14.6.1).

This module wraps the ``supervision`` library's ``ByteTrack`` implementation
behind the same surface as the greedy ``Tracker`` so the perception loop is
backend-agnostic.

Design notes
------------
**Real ByteTrack (Kalman + Hungarian)**: supervision's ByteTrack runs a
Kalman-filter predict/update cycle per frame and uses the Hungarian algorithm
(linear assignment) for track-to-detection matching. This is more robust than
greedy IoU for crowded scenes with temporary occlusions.

**Per-class instances for class gating**: one ``supervision.ByteTrack`` instance
is maintained *per class label*. This mirrors the greedy tracker's class-equality
gate: two objects of different classes that happen to share the same bounding box
region are never associated with each other.

**2-D only association, 3-D rides along**: supervision's ByteTrack operates on
2-D bounding boxes only. The 3-D world coordinates from the original
``ObjectState3D`` are carried through unchanged on the returned copy — no 3-D
prediction is applied here, so the SemanticMap's EMA fusion is the right place
for 3-D smoothing.

**Import is lazy**: the ``supervision`` library is an optional heavy dependency.
All imports happen inside ``__init__`` so that a missing or incompatible
``supervision`` install is caught by ``protocol.make_tracker``'s try/except and
falls back to the greedy tracker transparently.

**active_tracks pruning**: ``active_tracks`` reflects only the slugs emitted
by the *most recent* ``update()`` call.  A slug that ByteTrack no longer
returns for the current frame is immediately removed from the live set.  This
mirrors the greedy ``Tracker``'s behaviour (which prunes after
``persistence_frames`` consecutive misses) and keeps the docstring contract
("currently-live") true.  Note that ByteTrack itself still buffers lost tracks
internally for up to ``persistence_frames`` frames; the pruning here is on the
*adapter's* view of what is live, not inside the ByteTrack core.

**Robust box→input mapping**: instead of a round-then-dict lookup (which
silently drops duplicate same-class boxes that round to the same key), each
ByteTrack-returned box is matched to the *nearest* unmatched input box by L2
distance in xyxy space.  Boxes farther than 5.0 pixels are rejected (see
``_nearest_input``).  Each input index is consumed at most once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..spatial.object_lifter import ObjectState3D


@dataclass
class _LiteTrack:
    """Lightweight per-track record (enough for ``active_tracks``)."""

    track_id: str
    class_label: str
    last_seen_frame: int


class ByteTrackTracker:
    """supervision ByteTrack adapter with per-class gating.

    Exposes the same surface as ``src.tracking.tracker.Tracker``
    (``update``, ``reset``, ``active_tracks``) so it is a drop-in
    backend selectable via ``make_tracker``.

    Parameters
    ----------
    high_confidence:
        Maps to ``track_activation_threshold`` in ByteTrack.
        Detections below this score can only re-activate existing
        (lost) tracks; they never start new ones — the original
        ByteTrack insight for occlusion robustness.
    persistence_frames:
        Maps to ``lost_track_buffer``.  A track may be absent for
        this many frames before ByteTrack promotes it to "removed".
    min_iou:
        Maps to ``minimum_matching_threshold`` (after inversion from
        IoU to cost space: cost = 1 − IoU, so high IoU ↔ low cost).
    frame_rate:
        Nominal frames-per-second of the perception loop.  Used by
        ByteTrack to convert ``lost_track_buffer`` into real time.
    """

    def __init__(
        self,
        *,
        high_confidence: float = 0.5,
        persistence_frames: int = 3,
        min_iou: float = 0.35,
        frame_rate: int = 10,
    ) -> None:
        # Lazy import — keeps the module importable even when supervision is absent.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from supervision.detection.core import Detections as _Dets
            from supervision.tracker.byte_tracker.core import ByteTrack as _BT

        self._BT = _BT
        self._Dets = _Dets

        self._high_confidence = high_confidence
        self._persistence_frames = max(1, persistence_frames)
        # ByteTrack's minimum_matching_threshold is a *cost* threshold (cost = 1 − IoU).
        # The caller passes min_iou in IoU space (the same semantics as the greedy tracker).
        # Convert: if the caller requires IoU ≥ min_iou, cost must be ≤ 1 − min_iou.
        # Note: fuse_score further modifies costs, so we use the higher (more permissive)
        # value to allow real matches that are slightly penalised by score fusion.
        self._matching_threshold = 1.0 - float(min_iou)
        self._frame_rate = frame_rate

        # One ByteTrack instance per class label.
        self._trackers: dict[str, object] = {}

        # slug registry: (class_label, sv_tracker_id) → "track_NNN"
        self._slug_map: dict[tuple[str, int], str] = {}

        # lightweight per-track metadata — reflects only the most-recent frame's
        # live tracks (pruned at the start of each update call).
        self._tracks: dict[str, _LiteTrack] = {}

        # instance-local id counter (reset on .reset())
        self._next_id: int = 1

    # ── public API (mirrors Tracker) ────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracker state; the next update starts fresh from track_001."""
        for bt in self._trackers.values():
            bt.reset()  # type: ignore[union-attr]
        self._trackers.clear()
        self._slug_map.clear()
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> dict[str, _LiteTrack]:
        """Snapshot of currently-live tracks keyed by stable slug.

        Only slugs emitted by the most recent ``update()`` call are present.
        A slug disappears immediately when ByteTrack stops returning it.
        """
        return dict(self._tracks)

    def update(
        self,
        detections: list[ObjectState3D],
        frame_id: int,
    ) -> list[ObjectState3D]:
        """Associate *detections* via supervision ByteTrack and rewrite ``object_id``.

        Detections are grouped by ``class_label``; each class runs through its own
        ByteTrack instance so class-gated association is preserved.  Only
        detections that ByteTrack assigns a valid ``tracker_id`` are returned —
        sub-threshold detections are silently dropped (ByteTrack behaviour).

        The returned ``ObjectState3D`` copies are identical to the input except
        for ``object_id``, which is rewritten to a stable ``track_NNN`` slug.
        The original ``center_3d_world`` is preserved unchanged.

        ``active_tracks`` is rebuilt from scratch on every call so it only
        contains the slugs that ByteTrack returned this frame.
        """
        # Rebuild live-track set from scratch each frame (Fix 1: pruning).
        self._tracks = {}

        if not detections:
            return []

        # Group input indices by class label.
        by_label: dict[str, list[int]] = {}
        for idx, det in enumerate(detections):
            by_label.setdefault(det.class_label, []).append(idx)

        out: list[ObjectState3D] = []

        for label, indices in by_label.items():
            bt = self._get_or_create_tracker(label)

            # Build xyxy / confidence arrays for this label's detections.
            xyxy = np.array([detections[i].bbox_xyxy for i in indices], dtype=np.float32)
            conf = np.array(
                [
                    max(
                        detections[i].confidence.detector,
                        detections[i].confidence.overall,
                    )
                    for i in indices
                ],
                dtype=np.float32,
            )
            # class_id is unused for per-class instances; set to zeros.
            cls_id = np.zeros(len(indices), dtype=int)

            sv_dets = self._Dets(xyxy=xyxy, confidence=conf, class_id=cls_id)

            # No catch_warnings here — suppression belongs only around construction
            # (Fix 3: the FutureWarning fires on ByteTrack.__init__, not on update).
            result = bt.update_with_detections(sv_dets)  # type: ignore[union-attr]

            if len(result) == 0:
                continue

            # Fix 2: robust box→input mapping via nearest-unmatched-box search.
            # Track which input indices have already been claimed so two ByteTrack
            # results cannot map to the same input.
            consumed: set[int] = set()

            for k in range(len(result)):
                sv_tid = int(result.tracker_id[k])

                if sv_tid < 0:
                    continue  # ByteTrack not yet confident about this track.

                input_idx = self._nearest_input(result.xyxy[k], xyxy, indices, consumed)

                if input_idx is None:
                    continue

                consumed.add(input_idx)
                slug = self._mint_slug(label, sv_tid)
                det = detections[input_idx]
                self._tracks[slug] = _LiteTrack(
                    track_id=slug,
                    class_label=label,
                    last_seen_frame=frame_id,
                )
                out.append(det.model_copy(update={"object_id": slug}))

        return out

    # ── internals ────────────────────────────────────────────────────────────

    def _get_or_create_tracker(self, label: str) -> object:
        """Return the ByteTrack instance for *label*, creating it if needed."""
        if label not in self._trackers:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self._trackers[label] = self._BT(
                    track_activation_threshold=self._high_confidence,
                    lost_track_buffer=self._persistence_frames,
                    minimum_matching_threshold=self._matching_threshold,
                    frame_rate=self._frame_rate,
                    minimum_consecutive_frames=1,
                )
        return self._trackers[label]

    def _mint_slug(self, label: str, sv_tid: int) -> str:
        """Return the stable slug for *(label, sv_tid)*, minting one if new."""
        key = (label, sv_tid)
        if key not in self._slug_map:
            slug = f"track_{self._next_id:03d}"
            self._next_id += 1
            self._slug_map[key] = slug
        return self._slug_map[key]

    @staticmethod
    def _nearest_input(
        sv_box: npt.NDArray[np.float64],
        xyxy: npt.NDArray[np.float64],
        indices: list[int],
        consumed: set[int],
    ) -> int | None:
        """Return the closest *unconsumed* input index by L2 distance in xyxy space.

        Boxes whose nearest match exceeds 5.0 px in L2 xyxy distance are rejected
        — this fence tolerates Kalman-filter coordinate nudges while preventing
        cross-object mis-assignment when detections are far apart.
        """
        # 5.0 px L2 rejection fence: tolerates Kalman nudges, blocks mis-assignment.
        _REJECT_DISTANCE = 5.0

        sv_box_arr = np.asarray(sv_box, dtype=np.float32)
        xyxy_arr = np.asarray(xyxy, dtype=np.float32)
        dists = np.linalg.norm(xyxy_arr - sv_box_arr, axis=1)

        # Sort candidates by distance; skip already-consumed indices.
        order = np.argsort(dists)
        for pos in order:
            candidate_idx = indices[int(pos)]
            if candidate_idx not in consumed and dists[pos] < _REJECT_DISTANCE:
                return candidate_idx

        return None
