"""FramePacket + intrinsics math. Spec §3.1."""

import json
from pathlib import Path

from src.spatial import (
    CameraIntrinsics,
    CameraPoseWorld,
    FixedPoseSource,
    FramePacket,
    SimPoseSource,
)


def test_intrinsics_from_fov_basic():
    intr = CameraIntrinsics.from_fov(image_size=(480, 640), horizontal_fov_deg=60.0)
    # cx, cy at image center
    assert intr.cx == 320.0
    assert intr.cy == 240.0
    # fx for 60° FOV: f = (w/2) / tan(30°) ≈ 554.26
    assert 553.0 < intr.fx < 556.0
    assert intr.fy == intr.fx


def test_intrinsics_narrow_fov_yields_long_focal():
    wide = CameraIntrinsics.from_fov(image_size=(480, 640), horizontal_fov_deg=90.0)
    narrow = CameraIntrinsics.from_fov(image_size=(480, 640), horizontal_fov_deg=30.0)
    assert narrow.fx > wide.fx


def test_frame_packet_serializes_roundtrip():
    pkt = FramePacket(
        frame_id=42,
        timestamp=1700000000.0,
        image_size=(480, 640),
        camera_intrinsics=CameraIntrinsics(fx=600, fy=600, cx=320, cy=240),
    )
    blob = pkt.model_dump_json()
    parsed = FramePacket.model_validate_json(blob)
    assert parsed.frame_id == 42
    assert parsed.camera_intrinsics.fx == 600
    assert parsed.camera_pose_world.available is True
    assert parsed.camera_pose_world.source == "fixed"


def test_fixed_pose_source_returns_identity():
    src = FixedPoseSource()
    pose = src.get(frame_id=10)
    assert pose.available is True
    assert pose.source == "fixed"
    assert pose.position == (0.0, 0.0, 0.0)
    assert pose.quaternion == (0.0, 0.0, 0.0, 1.0)


def test_sim_pose_source_reads_jsonl(tmp_path: Path):
    jsonl = tmp_path / "poses.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {"frame_id": 0, "position": [0.0, 0.0, 0.0], "quaternion": [0, 0, 0, 1]}
                ),
                json.dumps(
                    {"frame_id": 5, "position": [1.2, 0.3, -0.4], "quaternion": [0, 0, 0, 1]}
                ),
            ]
        ),
        encoding="utf-8",
    )
    src = SimPoseSource(jsonl)
    p0 = src.get(0)
    p5 = src.get(5)
    missing = src.get(99)
    assert p0.available is True and p0.position == (0.0, 0.0, 0.0)
    assert p5.position == (1.2, 0.3, -0.4)
    assert missing.available is False


def test_camera_pose_world_quaternion_default_is_identity():
    p = CameraPoseWorld()
    assert p.quaternion == (0.0, 0.0, 0.0, 1.0)
