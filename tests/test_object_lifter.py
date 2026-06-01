"""ObjectLifter: mask + depth + intrinsics → ObjectState3D. Spec §6."""

from pathlib import Path

import cv2
import numpy as np

from src.perception.schema import ObjectCandidate2D
from src.spatial import (
    CameraIntrinsics,
    FixedPoseSource,
    ObjectLifter,
)


def _write_circle_mask(
    path: Path, *, image_size: tuple[int, int], cx: int, cy: int, r: int
) -> None:
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    cv2.imwrite(str(path), mask)


def _make_candidate(
    mask_path: Path, *, bbox: tuple[float, float, float, float]
) -> ObjectCandidate2D:
    return ObjectCandidate2D(
        id="obj_test_001",
        label="cup",
        bbox_xyxy=bbox,
        mask_path=str(mask_path),
        detector_confidence=0.8,
        mask_quality=0.7,
    )


def test_lift_constant_depth_centered(tmp_path: Path):
    """Mask centered at principal point, constant depth → centroid on optical axis."""
    image_size = (480, 640)
    cx_px, cy_px = 320, 240
    radius = 30
    mask_path = tmp_path / "circle.png"
    _write_circle_mask(mask_path, image_size=image_size, cx=cx_px, cy=cy_px, r=radius)

    depth = np.full(image_size, 2.0, dtype=np.float32)  # 2 m everywhere

    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=cx_px, cy=cy_px)
    candidate = _make_candidate(
        mask_path,
        bbox=(cx_px - radius, cy_px - radius, cx_px + radius, cy_px + radius),
    )

    obj = ObjectLifter(min_valid_pixels=10).lift(
        candidate, depth, intr, FixedPoseSource().get(0), frame_id=0
    )
    assert obj is not None
    # Camera-frame X≈0, Y≈0 → graphics-world (X, -Y, -Z) ≈ (0, 0, -2)
    cx_w, cy_w, cz_w = obj.center_3d_world
    assert abs(cx_w) < 0.01
    assert abs(cy_w) < 0.01
    assert abs(cz_w - (-2.0)) < 0.01
    assert abs(obj.median_depth - 2.0) < 1e-3
    assert obj.depth_uncertainty < 1e-3  # constant depth
    assert obj.coordinate_frame == "world"


def test_lift_offset_pixel_recovers_world_xy(tmp_path: Path):
    """Mask offset right of center → camera-frame +X → graphics-world +X."""
    image_size = (480, 640)
    mask_path = tmp_path / "offset.png"
    # circle at (420, 240) — 100 px right of principal point
    _write_circle_mask(mask_path, image_size=image_size, cx=420, cy=240, r=20)
    depth = np.full(image_size, 3.0, dtype=np.float32)

    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    candidate = _make_candidate(mask_path, bbox=(400, 220, 440, 260))

    obj = ObjectLifter(min_valid_pixels=10).lift(
        candidate, depth, intr, FixedPoseSource().get(0), frame_id=0
    )
    assert obj is not None
    # X = (u-cx)·Z/fx = 100 · 3 / 600 = 0.5  → world X same
    assert abs(obj.center_3d_world[0] - 0.5) < 0.01
    # Y ≈ 0  → world Y ≈ 0
    assert abs(obj.center_3d_world[1]) < 0.05
    # depth_world = -depth = -3
    assert abs(obj.center_3d_world[2] - (-3.0)) < 0.01


def test_lift_rejects_too_few_valid_pixels(tmp_path: Path):
    """Single-pixel mask should be rejected."""
    image_size = (100, 100)
    mask = np.zeros(image_size, dtype=np.uint8)
    mask[50, 50] = 255
    mp = tmp_path / "tiny.png"
    cv2.imwrite(str(mp), mask)
    depth = np.full(image_size, 1.5, dtype=np.float32)
    intr = CameraIntrinsics(fx=300, fy=300, cx=50, cy=50)

    candidate = _make_candidate(mp, bbox=(49, 49, 51, 51))
    obj = ObjectLifter(min_valid_pixels=50).lift(
        candidate, depth, intr, FixedPoseSource().get(0), frame_id=0
    )
    assert obj is None


def test_lift_records_depth_uncertainty_for_noisy_depth(tmp_path: Path):
    """Bimodal depth inside mask → non-trivial uncertainty."""
    image_size = (200, 200)
    mp = tmp_path / "block.png"
    mask = np.zeros(image_size, dtype=np.uint8)
    mask[80:120, 80:120] = 255
    cv2.imwrite(str(mp), mask)
    depth = np.zeros(image_size, dtype=np.float32)
    depth[80:100, 80:120] = 2.0
    depth[100:120, 80:120] = 3.0  # half at 3 m, half at 2 m

    intr = CameraIntrinsics(fx=300, fy=300, cx=100, cy=100)
    candidate = _make_candidate(mp, bbox=(80, 80, 120, 120))

    obj = ObjectLifter(min_valid_pixels=50, low_percentile=10, high_percentile=90).lift(
        candidate, depth, intr, FixedPoseSource().get(0), frame_id=0
    )
    assert obj is not None
    assert obj.depth_uncertainty > 0.5


def test_lift_camera_frame_when_pose_unavailable(tmp_path: Path):
    """Pose unavailable → coordinate_frame == 'camera' and no axis flip."""
    image_size = (200, 200)
    mp = tmp_path / "c.png"
    _write_circle_mask(mp, image_size=image_size, cx=100, cy=100, r=20)
    depth = np.full(image_size, 1.0, dtype=np.float32)
    intr = CameraIntrinsics(fx=200, fy=200, cx=100, cy=100)
    candidate = _make_candidate(mp, bbox=(80, 80, 120, 120))

    from src.spatial.frame_packet import CameraPoseWorld

    obj = ObjectLifter(min_valid_pixels=10).lift(
        candidate,
        depth,
        intr,
        CameraPoseWorld(available=False, source="sim"),
        frame_id=0,
    )
    assert obj is not None
    assert obj.coordinate_frame == "camera"
    # In camera frame, Z = +depth (positive, forward).
    assert obj.center_3d_world[2] > 0.99
