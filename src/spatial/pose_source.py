"""Pose sources (spec v2 §1.4, §6.1).

Phase 3 ships ``FixedPoseSource`` (camera at world origin, MVP) and
``SimPoseSource`` (sidecar JSONL). ``SLAMPoseSource`` is the optional
extension in §14.1 — its slot is reserved here so the optional ORB-SLAM
work later does not need to change call sites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .frame_packet import CameraPoseWorld


class PoseSource(Protocol):
    """Anything that can answer ``what is the camera pose at this frame?``"""

    def get(self, frame_id: int, timestamp: float | None = None) -> CameraPoseWorld: ...


class FixedPoseSource:
    """Camera permanently at the world origin, identity rotation.

    The default for snapshot mode and any setup without external tracking.
    The world frame equals the camera frame, so the lifter writes
    ``center_3d_world`` directly from the pinhole projection without
    applying any rotation.
    """

    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self._pose = CameraPoseWorld(
            available=True,
            source="fixed",
            position=position,
            quaternion=quaternion,
        )

    def get(self, frame_id: int, timestamp: float | None = None) -> CameraPoseWorld:  # noqa: ARG002
        return self._pose


class SimPoseSource:
    """Read poses from a sidecar JSONL file (one record per frame).

    Each line:

    ```json
    {"frame_id": 12, "position": [0.1, 0.2, 0.3], "quaternion": [0, 0, 0, 1]}
    ```

    Missing frames return an ``available=False`` pose so downstream code can
    fall through to the camera-frame branch.
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self._poses: dict[int, CameraPoseWorld] = {}
        path = Path(jsonl_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fid = int(record["frame_id"])
            self._poses[fid] = CameraPoseWorld(
                available=True,
                source="sim",
                position=tuple(record.get("position", (0.0, 0.0, 0.0))),
                quaternion=tuple(record.get("quaternion", (0.0, 0.0, 0.0, 1.0))),
            )

    def get(self, frame_id: int, timestamp: float | None = None) -> CameraPoseWorld:  # noqa: ARG002
        if frame_id in self._poses:
            return self._poses[frame_id]
        return CameraPoseWorld(available=False, source="sim")
