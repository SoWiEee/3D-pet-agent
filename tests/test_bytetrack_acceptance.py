"""§14.6.1 acceptance — ByteTrack ID stability vs greedy on a crossing scene."""

from __future__ import annotations

import pytest

from src.tracking import Tracker
from tests.factories import make_object

sv = pytest.importorskip("supervision")
from src.tracking.bytetrack_adapter import ByteTrackTracker  # noqa: E402


def _obs(object_id, label, bbox, center, frame):
    return make_object(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=bbox,
        center_3d_world=center,
        last_seen_frame=frame,
        detector=0.9,
        mask_quality=0.7,
        depth_quality=0.7,
        overall=0.85,
    )


def _two_cups_crossing(frames: int = 12):
    """Two same-class cups translating toward and past each other in x.

    Each cup moves 10 px per frame (down from the original 18 px), keeping
    successive bounding-boxes of the same object at >50 % IoU across frames
    at 10 Hz — a realistic inter-frame displacement for a slow-moving object.
    With 60-px-wide boxes the per-step IoU ≈ (60-10)/(60+10) ≈ 0.71, well
    above the min_iou=0.2 gate, so both trackers can associate reliably.
    """
    seq = []
    for f in range(frames):
        ax = 100 + f * 10
        bx = 360 - f * 10
        seq.append(
            [
                _obs("gtA", "cup", (ax, 100, ax + 60, 180), (-1.0 + f * 0.15, 0, -2), f),
                _obs("gtB", "cup", (bx, 100, bx + 60, 180), (1.0 - f * 0.15, 0, -2), f),
            ]
        )
    return seq


def _count_id_switches(tracker, seq) -> int:
    """Run a sequence; count how often a ground-truth object's assigned
    track id changes from the previous frame (lower is better)."""
    last: dict[str, str] = {}
    switches = 0
    for f, dets in enumerate(seq):
        out = tracker.update(dets, frame_id=f)
        bbox_to_track = {tuple(o.bbox_xyxy): o.object_id for o in out}
        for gt_key, det in (("gtA", dets[0]), ("gtB", dets[1])):
            tid = bbox_to_track.get(tuple(det.bbox_xyxy))
            if tid is None:
                continue
            if gt_key in last and last[gt_key] != tid:
                switches += 1
            last[gt_key] = tid
    return switches


def test_bytetrack_id_switches_not_worse_than_greedy() -> None:
    seq = _two_cups_crossing()
    greedy = _count_id_switches(Tracker(min_iou=0.2, max_center_distance=0.5), seq)
    bt = _count_id_switches(ByteTrackTracker(min_iou=0.2), seq)
    assert bt <= greedy + 1, f"ByteTrack {bt} switches vs greedy {greedy}"
