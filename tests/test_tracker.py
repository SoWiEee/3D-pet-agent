"""Phase 4 — Tracker association tests."""

from __future__ import annotations

import random

from src.spatial.object_lifter import ObjectConfidence, ObjectState3D
from src.tracking import Tracker
from tests.factories import make_object


def _make_obs(
    *,
    object_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    center_3d: tuple[float, float, float],
    frame_id: int,
    detector: float = 0.8,
    overall: float = 0.75,
) -> ObjectState3D:
    return make_object(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=bbox,
        center_3d_world=center_3d,
        last_seen_frame=frame_id,
        detector=detector,
        mask_quality=0.7,
        depth_quality=0.7,
        overall=overall,
    )


def test_first_frame_mints_new_ids() -> None:
    t = Tracker()
    obs = [
        _make_obs(
            object_id="raw_a",
            label="cup",
            bbox=(100, 100, 200, 200),
            center_3d=(0.0, 0.0, -2.0),
            frame_id=0,
        ),
        _make_obs(
            object_id="raw_b",
            label="keyboard",
            bbox=(300, 300, 500, 400),
            center_3d=(1.0, 0.0, -2.0),
            frame_id=0,
        ),
    ]
    out = t.update(obs, frame_id=0)
    ids = sorted(o.object_id for o in out)
    assert ids == ["track_001", "track_002"]
    assert len(t.active_tracks) == 2


def test_high_iou_same_class_keeps_track_id() -> None:
    t = Tracker()
    a0 = _make_obs(
        object_id="raw",
        label="cup",
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=0,
    )
    [tracked0] = t.update([a0], frame_id=0)
    a1 = a0.model_copy(update={"bbox_xyxy": (105, 105, 205, 205), "last_seen_frame": 1})
    [tracked1] = t.update([a1], frame_id=1)
    assert tracked0.object_id == tracked1.object_id == "track_001"


def test_class_mismatch_starts_new_track() -> None:
    t = Tracker()
    a = _make_obs(
        object_id="x",
        label="cup",
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=0,
    )
    t.update([a], frame_id=0)
    b = _make_obs(
        object_id="y",
        label="mouse",  # different class — must NOT match the cup
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=1,
    )
    [out] = t.update([b], frame_id=1)
    assert out.object_id == "track_002"


def test_persists_50_frames_with_jitter() -> None:
    """Acceptance §7.2: same object trackable ≥ 50 consecutive frames."""
    t = Tracker(min_iou=0.4, max_center_distance=0.15, persistence_frames=2)
    rng = random.Random(42)
    seen_ids: set[str] = set()
    bbox = [100.0, 100.0, 200.0, 200.0]
    center = [0.0, 0.0, -2.0]
    for fi in range(50):
        bbox = [b + rng.uniform(-3, 3) for b in bbox]
        center = [
            center[0] + rng.uniform(-0.01, 0.01),
            center[1] + rng.uniform(-0.01, 0.01),
            center[2] + rng.uniform(-0.01, 0.01),
        ]
        obs = _make_obs(
            object_id="raw",
            label="cup",
            bbox=tuple(bbox),  # type: ignore[arg-type]
            center_3d=tuple(center),  # type: ignore[arg-type]
            frame_id=fi,
        )
        [out] = t.update([obs], frame_id=fi)
        seen_ids.add(out.object_id)
    assert seen_ids == {"track_001"}, f"id flipped mid-track: {seen_ids}"


def test_missing_track_pruned_after_persistence_frames() -> None:
    t = Tracker(persistence_frames=2)
    a = _make_obs(
        object_id="x",
        label="cup",
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=0,
    )
    t.update([a], frame_id=0)
    assert "track_001" in t.active_tracks
    t.update([], frame_id=1)
    t.update([], frame_id=2)
    assert "track_001" in t.active_tracks  # still in grace period
    t.update([], frame_id=3)
    assert "track_001" not in t.active_tracks  # pruned past persistence_frames


def test_greedy_avoids_double_assign() -> None:
    """Two existing tracks, two detections: each detection takes its closest
    track, not the highest-IoU one twice."""
    t = Tracker()
    a = _make_obs(
        object_id="a",
        label="cup",
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=0,
    )
    b = _make_obs(
        object_id="b",
        label="cup",
        bbox=(300, 100, 400, 200),
        center_3d=(1.0, 0.0, -2.0),
        frame_id=0,
    )
    t.update([a, b], frame_id=0)
    # Next frame, both shifted slightly right.
    a1 = a.model_copy(update={"bbox_xyxy": (110, 100, 210, 200)})
    b1 = b.model_copy(update={"bbox_xyxy": (310, 100, 410, 200)})
    [tracked_a, tracked_b] = t.update([a1, b1], frame_id=1)
    assert {tracked_a.object_id, tracked_b.object_id} == {"track_001", "track_002"}
    assert tracked_a.object_id != tracked_b.object_id


def test_low_confidence_detection_does_not_spawn_track() -> None:
    """ByteTrack: a faint box with no track to recover is dropped, not minted."""
    t = Tracker(high_confidence=0.5)
    faint = _make_obs(
        object_id="x",
        label="cup",
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=0,
        detector=0.2,  # below high_confidence
        overall=0.2,
    )
    out = t.update([faint], frame_id=0)
    assert out == []
    assert t.active_tracks == {}


def test_low_confidence_detection_recovers_existing_track() -> None:
    """ByteTrack stage 2: a faint box continues an established high-conf track."""
    t = Tracker(high_confidence=0.5)
    strong = _make_obs(
        object_id="x",
        label="cup",
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=0,
        detector=0.9,
    )
    [first] = t.update([strong], frame_id=0)
    assert first.object_id == "track_001"

    faint = strong.model_copy(
        update={
            "bbox_xyxy": (104, 104, 204, 204),
            "last_seen_frame": 1,
            "confidence": ObjectConfidence(detector=0.2, overall=0.2),
        }
    )
    [recovered] = t.update([faint], frame_id=1)
    assert recovered.object_id == "track_001"  # same id, recovered by stage 2


def test_velocity_prediction_holds_id_under_fast_bbox_motion() -> None:
    """A consistently moving object keeps one id across many frames. Per-frame
    bbox displacement (60px on a 100px box → raw IoU 0.25) is well below
    ``min_iou``; once the velocity model warms up, the predicted bbox overlaps
    the next detection and association stays confident instead of flipping ids."""
    t = Tracker(min_iou=0.5, max_center_distance=0.06, distance_weight=0.4)
    center = [0.0, 0.0, -2.0]
    step = 0.05  # within the distance gate so the cold-start frame matches
    bbox = [100.0, 100.0, 200.0, 200.0]
    seen: set[str] = set()
    for fi in range(8):
        obs = _make_obs(
            object_id="raw",
            label="cup",
            bbox=tuple(bbox),  # type: ignore[arg-type]
            center_3d=tuple(center),  # type: ignore[arg-type]
            frame_id=fi,
        )
        [out] = t.update([obs], frame_id=fi)
        seen.add(out.object_id)
        center[0] += step
        bbox[0] += 60
        bbox[2] += 60
    assert seen == {"track_001"}, f"velocity model failed to hold id: {seen}"


def test_reset_clears_state() -> None:
    t = Tracker()
    a = _make_obs(
        object_id="x",
        label="cup",
        bbox=(100, 100, 200, 200),
        center_3d=(0.0, 0.0, -2.0),
        frame_id=0,
    )
    t.update([a], frame_id=0)
    t.reset()
    assert t.active_tracks == {}
    # Id counter restarts.
    [out] = t.update([a], frame_id=1)
    assert out.object_id == "track_001"
