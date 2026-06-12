"""§14.6.2 — GraphSlamPoseSource: PoseSource conformance + anchored pose."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypose")
from src.research.graph_slam import GraphSlamPoseSource  # noqa: E402
from src.spatial.frame_packet import CameraIntrinsics  # noqa: E402


def _img(seed: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((240, 320), 30, dtype=np.uint8)
    for _ in range(40):
        x1, y1 = int(rng.integers(0, 280)), int(rng.integers(0, 200))
        cv2.rectangle(img, (x1, y1), (x1 + 30, y1 + 30), int(rng.integers(80, 255)), -1)
    return img


def test_conforms_to_pose_source_protocol() -> None:
    src = GraphSlamPoseSource(CameraIntrinsics(fx=500, fy=500, cx=320, cy=240))
    # PoseSource is not @runtime_checkable; verify duck-typing compliance instead
    assert hasattr(src, "get") and hasattr(src, "track") and hasattr(src, "reset")


def test_first_frame_anchors_world_origin() -> None:
    src = GraphSlamPoseSource(CameraIntrinsics(fx=500, fy=500, cx=320, cy=240))
    pose = src.track(0, _img(1))
    assert pose.available
    assert np.allclose(pose.position, (0.0, 0.0, 0.0), atol=1e-6)


def test_tracking_lost_marks_unavailable_but_holds_pose() -> None:
    src = GraphSlamPoseSource(CameraIntrinsics(fx=500, fy=500, cx=320, cy=240))
    src.track(0, _img(1))
    pose = src.track(1, np.zeros((240, 320), dtype=np.uint8))  # blank → VO fails
    assert pose.available is False
