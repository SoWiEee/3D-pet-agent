"""FramePacket — the unit of work for 3D-aware processing (spec §3.1).

A single sensor frame plus everything downstream needs: image size, intrinsics,
camera pose. Created at the camera-service boundary and threaded through
depth + lifting.
"""

from __future__ import annotations

import math
import time
from typing import Literal

from pydantic import BaseModel, Field

PoseSourceTag = Literal["fixed", "sim", "slam", "manual"]


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics in pixel units. ``fx`` and ``fy`` are focal lengths;
    ``cx`` and ``cy`` are the principal point.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(
        cls,
        *,
        image_size: tuple[int, int],
        horizontal_fov_deg: float = 60.0,
    ) -> CameraIntrinsics:
        """Estimate intrinsics from horizontal FOV when calibration is unknown.

        Used as a fallback for sample images of unknown provenance. 60° is a
        reasonable mid-range webcam FOV.
        """
        h, w = image_size
        fx = (w / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        fy = fx  # square pixels assumption
        return cls(fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0)


class CameraPoseWorld(BaseModel):
    """SE(3) camera pose in the world frame.

    Quaternion uses (x, y, z, w) order — matches scipy / pytorch3d. When
    ``available`` is False, downstream stages should interpret
    ``ObjectState.center_3d_world`` as camera-frame coordinates instead.
    """

    available: bool = True
    source: PoseSourceTag = "fixed"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class FramePacket(BaseModel):
    frame_id: int
    timestamp: float = Field(default_factory=time.time)
    rgb_path: str | None = None
    depth_path: str | None = None
    image_size: tuple[int, int]  # (height, width)
    camera_intrinsics: CameraIntrinsics
    camera_pose_world: CameraPoseWorld = Field(default_factory=CameraPoseWorld)
