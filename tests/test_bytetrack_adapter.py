"""§14.6.1 — supervision ByteTrack adapter behavioral tests."""

from __future__ import annotations

import pytest

from tests.factories import make_object

sv = pytest.importorskip("supervision")
from src.tracking.bytetrack_adapter import ByteTrackTracker  # noqa: E402


def _obs(object_id, label, bbox, center, frame, detector=0.9, overall=0.85):
    return make_object(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=bbox,
        center_3d_world=center,
        last_seen_frame=frame,
        detector=detector,
        mask_quality=0.7,
        depth_quality=0.7,
        overall=overall,
    )


def test_first_frame_mints_track_ids() -> None:
    t = ByteTrackTracker()
    obs = [
        _obs("raw_a", "cup", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0),
        _obs("raw_b", "keyboard", (300, 300, 500, 400), (1.0, 0.0, -2.0), 0),
    ]
    out = t.update(obs, frame_id=0)
    ids = sorted(o.object_id for o in out)
    assert all(i.startswith("track_") for i in ids)
    assert len(set(ids)) == 2


def test_3d_centre_rides_along_unchanged() -> None:
    t = ByteTrackTracker()
    obs = [_obs("raw_a", "cup", (100, 100, 200, 200), (0.4, 0.1, -2.0), 0)]
    out = t.update(obs, frame_id=0)
    assert len(out) == 1
    assert out[0].center_3d_world == (0.4, 0.1, -2.0)
    assert out[0].object_id.startswith("track_")


def test_same_object_keeps_id_across_frames() -> None:
    t = ByteTrackTracker(min_iou=0.2)
    first = t.update(
        [_obs("raw_a", "cup", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0)],
        frame_id=0,
    )
    assert len(first) == 1
    track_id = first[0].object_id
    for frame, bbox in ((1, (106, 104, 206, 204)), (2, (112, 108, 212, 208))):
        out = t.update(
            [_obs("raw_a", "cup", bbox, (0.0, 0.0, -2.0), frame)],
            frame_id=frame,
        )
        assert len(out) == 1
        assert out[0].object_id == track_id


def test_overlapping_boxes_of_different_classes_get_distinct_ids() -> None:
    t = ByteTrackTracker(min_iou=0.2)
    out = t.update(
        [
            _obs("raw_cup", "cup", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0),
            _obs("raw_book", "book", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0),
        ],
        frame_id=0,
    )
    by_label = {o.class_label: o.object_id for o in out}
    assert set(by_label) == {"cup", "book"}
    assert by_label["cup"] != by_label["book"]


def test_reset_clears_all_state() -> None:
    t = ByteTrackTracker()
    t.update([_obs("raw_a", "cup", (100, 100, 200, 200), (0, 0, -2), 0)], frame_id=0)
    assert len(t.active_tracks) >= 1
    t.reset()
    assert t.active_tracks == {}
    out = t.update([_obs("raw_a", "cup", (100, 100, 200, 200), (0, 0, -2), 0)], 0)
    assert out[0].object_id == "track_001"


def test_low_confidence_only_detection_is_not_emitted_as_new_track() -> None:
    t = ByteTrackTracker(high_confidence=0.5)
    out = t.update(
        [
            _obs(
                "raw_a",
                "cup",
                (100, 100, 200, 200),
                (0, 0, -2),
                0,
                detector=0.2,
                overall=0.2,
            )
        ],
        frame_id=0,
    )
    assert out == []


def test_active_tracks_drops_disappeared_object() -> None:
    """active_tracks must reflect only the most-recent frame's live tracks.

    Pruning rule: ``self._tracks`` is rebuilt from scratch at the top of each
    ``update()`` call. A slug that ByteTrack no longer returns for the current
    frame is immediately absent from ``active_tracks``, regardless of
    ``persistence_frames``. We use ``persistence_frames=1`` so ByteTrack's
    internal buffer expires quickly; the adapter-level pruning is visible from
    frame 1 onward (the very first empty-detection frame).
    """
    t = ByteTrackTracker(high_confidence=0.5, persistence_frames=1)

    # Frame 0: cup is visible and tracked.
    out0 = t.update(
        [_obs("raw_a", "cup", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0)],
        frame_id=0,
    )
    assert len(out0) == 1, "cup should be tracked in frame 0"
    cup_slug = out0[0].object_id
    assert cup_slug in t.active_tracks, "cup slug must be in active_tracks after frame 0"

    # Frames 1–3: no detections at all.  active_tracks must be empty from
    # frame 1 onward because the adapter rebuilds _tracks from scratch each call.
    for fid in range(1, 4):
        t.update([], frame_id=fid)
        assert cup_slug not in t.active_tracks, (
            f"cup slug must not be in active_tracks after empty frame {fid}"
        )
    assert t.active_tracks == {}, "active_tracks must be empty after several empty frames"
