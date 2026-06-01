"""Lift 2D masks + depth into 3D ``ObjectState`` (spec v2 §6).

Coordinate conventions
----------------------
- **Pixel coords** (``u``, ``v``): top-left origin, ``u`` →  right, ``v`` ↓
  down. Matches OpenCV / Pillow.
- **Camera frame** (``Xc``, ``Yc``, ``Zc``): OpenCV convention — ``Xc`` right,
  ``Yc`` down, ``Zc`` forward (away from camera).
- **World frame**: graphics convention used by the Three.js renderer —
  ``Xw`` right, ``Yw`` up, ``Zw`` toward the viewer. The conversion is
  ``(Xw, Yw, Zw) = (Xc, -Yc, -Zc)`` when the pose is identity. A non-identity
  pose then translates and rotates on top of that.

The renderer expects ``center_3d_world`` already in graphics-world
coordinates, so the lifter applies the axis flip here. SLAM (optional §14.1)
should publish its pose in graphics-world too, or this convention will need
revisiting.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..perception.schema import ObjectCandidate2D
from .frame_packet import CameraIntrinsics, CameraPoseWorld

log = logging.getLogger("pet_agent.lifter")


TrackingStatus = Literal["tracked", "occluded", "stale", "lost"]
SourceBackend = Literal["mainline_grounding_sam", "openscene"]


class ObjectConfidence(BaseModel):
    detector: float = 0.0
    mask_quality: float = 0.0
    depth_quality: float = 0.0
    tracking: float = 1.0
    overall: float = 0.0


class ObjectState3D(BaseModel):
    """Post-lifting object state. Spec §3.2."""

    object_id: str
    class_label: str
    attributes: list[str] = Field(default_factory=list)

    bbox_xyxy: tuple[float, float, float, float]
    mask_path: str | None = None
    center_2d: tuple[float, float]

    coordinate_frame: Literal["world", "camera"] = "world"
    center_3d_world: tuple[float, float, float]
    extent_3d: tuple[float, float, float]
    median_depth: float
    depth_uncertainty: float

    source_backend: SourceBackend = "mainline_grounding_sam"
    confidence: ObjectConfidence
    last_seen_frame: int
    tracking_status: TrackingStatus = "tracked"

    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _quat_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Quaternion (x, y, z, w) → 3×3 rotation matrix."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# OpenCV camera (X right, Y down, Z forward) → graphics world (X right, Y up,
# Z toward viewer). Flips Y and Z.
_CV_TO_GRAPHICS = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


class ObjectLifter:
    """Lift 2D ``ObjectCandidate2D`` + depth → ``ObjectState3D``.

    The lifter is stateless; one instance is reusable across frames.
    """

    def __init__(
        self,
        *,
        min_valid_pixels: int = 50,
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        depth_min: float = 0.05,
        depth_max: float = 50.0,
    ) -> None:
        self.min_valid_pixels = min_valid_pixels
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.depth_min = depth_min
        self.depth_max = depth_max

    # ── core: single-object lift ────────────────────────────────────────
    def lift(
        self,
        candidate: ObjectCandidate2D,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        pose: CameraPoseWorld,
        *,
        frame_id: int,
    ) -> ObjectState3D | None:
        """Return ``None`` when the object cannot be lifted (mask missing,
        too few valid depth pixels, NaN depth, etc.)."""
        mask = _load_mask(candidate.mask_path)
        if mask is None:
            log.debug("object %s: no mask, skipping lift", candidate.id)
            return None

        # Align mask shape to the depth map (mask may have been written at the
        # native resolution; both should match here, but be defensive).
        if mask.shape != depth.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        depth_in_mask = depth[mask]
        valid = (
            np.isfinite(depth_in_mask)
            & (depth_in_mask > self.depth_min)
            & (depth_in_mask < self.depth_max)
        )
        depth_in_mask = depth_in_mask[valid]
        if depth_in_mask.size < self.min_valid_pixels:
            log.debug(
                "object %s: only %d valid depth pixels (need %d)",
                candidate.id,
                depth_in_mask.size,
                self.min_valid_pixels,
            )
            return None

        lo = float(np.percentile(depth_in_mask, self.low_percentile))
        hi = float(np.percentile(depth_in_mask, self.high_percentile))
        trimmed = depth_in_mask[(depth_in_mask >= lo) & (depth_in_mask <= hi)]
        if trimmed.size < self.min_valid_pixels // 2:
            log.debug("object %s: trimmed window too thin", candidate.id)
            return None

        median_depth = float(np.median(trimmed))
        depth_uncertainty = float(hi - lo)  # IQR-like spread

        # Pixel-level back-projection of every mask pixel for extent.
        ys, xs = np.where(mask)
        # Match the in-mask depth filter set used above.
        ds = depth[ys, xs]
        keep = np.isfinite(ds) & (ds > self.depth_min) & (ds < self.depth_max)
        ys, xs, ds = ys[keep], xs[keep], ds[keep]
        if ys.size < self.min_valid_pixels:
            return None

        cx_px, cy_px = intrinsics.cx, intrinsics.cy
        fx, fy = intrinsics.fx, intrinsics.fy
        Xc = (xs - cx_px) * ds / fx
        Yc = (ys - cy_px) * ds / fy
        Zc = ds
        pts_cam = np.stack([Xc, Yc, Zc], axis=1)  # (N, 3)

        # Centroid + extent in camera frame.
        centroid_cam = np.array(
            [
                (candidate.bbox_xyxy[0] + candidate.bbox_xyxy[2]) / 2.0,
                (candidate.bbox_xyxy[1] + candidate.bbox_xyxy[3]) / 2.0,
            ]
        )
        u_c, v_c = centroid_cam
        Xc_c = (u_c - cx_px) * median_depth / fx
        Yc_c = (v_c - cy_px) * median_depth / fy
        Zc_c = median_depth
        center_cam = np.array([Xc_c, Yc_c, Zc_c])
        extent_cam = pts_cam.max(axis=0) - pts_cam.min(axis=0)

        # Convert into the (graphics) world frame.
        if pose.available:
            R_pose = _quat_to_matrix(pose.quaternion)
            t_pose = np.array(pose.position, dtype=np.float64)
            # camera → graphics-world, then pose transform.
            R_total = R_pose @ _CV_TO_GRAPHICS
            center_world = R_total @ center_cam + t_pose
            extent_world = np.abs(R_total @ extent_cam)
            coord_frame: Literal["world", "camera"] = "world"
        else:
            center_world = center_cam
            extent_world = np.abs(extent_cam)
            coord_frame = "camera"

        # Depth quality proxy: low when uncertainty is large relative to depth.
        depth_quality = float(np.clip(1.0 - depth_uncertainty / max(1e-3, median_depth), 0.0, 1.0))
        overall = float(
            0.4 * candidate.detector_confidence
            + 0.3 * candidate.mask_quality
            + 0.3 * depth_quality
        )

        return ObjectState3D(
            object_id=candidate.id,
            class_label=candidate.label,
            attributes=[],
            bbox_xyxy=candidate.bbox_xyxy,
            mask_path=candidate.mask_path,
            center_2d=(float(u_c), float(v_c)),
            coordinate_frame=coord_frame,
            center_3d_world=(float(center_world[0]), float(center_world[1]), float(center_world[2])),
            extent_3d=(float(extent_world[0]), float(extent_world[1]), float(extent_world[2])),
            median_depth=median_depth,
            depth_uncertainty=depth_uncertainty,
            source_backend="mainline_grounding_sam",
            confidence=ObjectConfidence(
                detector=candidate.detector_confidence,
                mask_quality=candidate.mask_quality,
                depth_quality=depth_quality,
                tracking=1.0,
                overall=overall,
            ),
            last_seen_frame=frame_id,
            tracking_status="tracked",
        )

    # ── batch helper ────────────────────────────────────────────────────
    def lift_many(
        self,
        candidates: list[ObjectCandidate2D],
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        pose: CameraPoseWorld,
        *,
        frame_id: int,
    ) -> list[ObjectState3D]:
        lifted: list[ObjectState3D] = []
        for c in candidates:
            obj = self.lift(c, depth, intrinsics, pose, frame_id=frame_id)
            if obj is not None:
                lifted.append(obj)
        return lifted


def _load_mask(mask_path: str | None) -> np.ndarray | None:
    if not mask_path:
        return None
    p = Path(mask_path)
    if not p.exists():
        return None
    raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    return raw > 0
