"""ORB visual-odometry pose source — optional Visual SLAM sidecar (spec §14.1).

``SLAMPoseSource`` plugs into ``spatial/pose_source.py``'s ``PoseSource``
protocol: frames are pushed in with :meth:`SLAMPoseSource.track` and
:meth:`SLAMPoseSource.get` returns the integrated ``world ← camera`` pose as a
:class:`CameraPoseWorld`, already in the **graphics-world** convention the
object lifter expects (so the lifter applies it transparently, per §14.1's
coordinate handshake).

Backbone: OpenCV ORB features + (RGB-D PnP when depth is supplied, else
monocular essential matrix). This is a lightweight, pip-only stand-in for
ORB-SLAM3 — it does frame-to-frame visual odometry without loop closure or
global bundle adjustment, so it drifts over long loops, but it satisfies the
pose-source slot, runs without a C++ build, and is fully testable. A real
ORB-SLAM3 / DROID-SLAM binding can drop in behind the :class:`VisualOdometry`
protocol without touching call sites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..spatial.frame_packet import CameraIntrinsics, CameraPoseWorld

log = logging.getLogger(__name__)

# OpenCV camera (X right, Y down, Z forward) → graphics world (X right, Y up,
# Z back). The object lifter applies this same flip to points; we apply it to
# the SLAM trajectory so the pose we publish is already graphics-world. It is
# its own inverse (a diagonal ±1 matrix).
_CV_TO_GRAPHICS = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class RelativePose:
    """Camera motion ``T_{curr←prev}`` — maps prev-camera coords into the curr
    camera frame (``x_curr = R x_prev + t``), the convention both
    ``cv2.recoverPose`` and ``cv2.solvePnP`` return."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    n_inliers: int


class VisualOdometry(Protocol):
    """Estimate the relative camera motion between two frames."""

    def estimate(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        k_matrix: np.ndarray,
        prev_depth: np.ndarray | None = None,
    ) -> RelativePose | None: ...


def _se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return t


def _se3_inv(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    inv = np.eye(4)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t[:3, 3]
    return inv


class OrbVisualOdometry:
    """Frame-to-frame ORB visual odometry.

    With ``prev_depth`` it solves PnP (metric scale); without depth it falls
    back to the essential matrix (translation recovered up to scale).
    """

    def __init__(
        self,
        *,
        n_features: int = 1500,
        min_matches: int = 20,
        ratio: float = 0.75,
        ransac_thresh: float = 1.0,
    ) -> None:
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_matches = min_matches
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh

    def estimate(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        k_matrix: np.ndarray,
        prev_depth: np.ndarray | None = None,
    ) -> RelativePose | None:
        kp1, des1 = self._orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self._orb.detectAndCompute(curr_gray, None)
        if (
            des1 is None
            or des2 is None
            or len(kp1) < self.min_matches
            or len(kp2) < self.min_matches
        ):
            return None

        good = self._ratio_matches(des1, des2)
        if len(good) < self.min_matches:
            return None

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
        if prev_depth is not None:
            return self._estimate_pnp(pts1, pts2, prev_depth, k_matrix)
        return self._estimate_essential(pts1, pts2, k_matrix)

    def _ratio_matches(self, des1: np.ndarray, des2: np.ndarray) -> list:
        good = []
        for pair in self._matcher.knnMatch(des1, des2, k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good

    def _estimate_essential(
        self, pts1: np.ndarray, pts2: np.ndarray, k: np.ndarray
    ) -> RelativePose | None:
        e_mat, mask = cv2.findEssentialMat(
            pts1, pts2, k, method=cv2.RANSAC, prob=0.999, threshold=self.ransac_thresh
        )
        if e_mat is None or e_mat.shape != (3, 3):
            return None
        n_in, rotation, t, _ = cv2.recoverPose(e_mat, pts1, pts2, k, mask=mask)
        if n_in < max(6, self.min_matches // 2):
            return None
        return RelativePose(rotation=rotation, translation=t.reshape(3), n_inliers=int(n_in))

    def _estimate_pnp(
        self, pts1: np.ndarray, pts2: np.ndarray, prev_depth: np.ndarray, k: np.ndarray
    ) -> RelativePose | None:
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        h, w = prev_depth.shape[:2]
        obj_pts: list[tuple[float, float, float]] = []
        img_pts: list[tuple[float, float]] = []
        for (u1, v1), (u2, v2) in zip(pts1, pts2, strict=True):
            iu, iv = int(round(u1)), int(round(v1))
            if not (0 <= iu < w and 0 <= iv < h):
                continue
            z = float(prev_depth[iv, iu])
            if not np.isfinite(z) or z <= 0.0:
                continue
            obj_pts.append(((u1 - cx) * z / fx, (v1 - cy) * z / fy, z))
            img_pts.append((u2, v2))
        if len(obj_pts) < max(6, self.min_matches // 2):
            return None

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.float64(obj_pts),
            np.float64(img_pts),
            k,
            None,
            reprojectionError=self.ransac_thresh * 2.0,
        )
        if not ok or inliers is None or len(inliers) < 6:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        return RelativePose(
            rotation=rotation, translation=tvec.reshape(3), n_inliers=int(len(inliers))
        )


class SLAMPoseSource:
    """A streaming ``PoseSource``: push frames via :meth:`track`, query the
    integrated camera pose via :meth:`get`.

    World frame ``= the first tracked camera frame`` (so the initial pose is the
    identity, exactly like :class:`FixedPoseSource`). When tracking is lost the
    last good pose is held but ``available`` flips to ``False``, so the lifter
    falls through to camera-frame coordinates — the same contract
    :class:`SimPoseSource` honours for missing frames.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        vo: VisualOdometry | None = None,
        scale: float = 1.0,
        min_inliers: int = 12,
    ) -> None:
        self._k = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._vo: VisualOdometry = vo or OrbVisualOdometry()
        self._scale = float(scale)
        self._min_inliers = int(min_inliers)
        self.reset()

    def reset(self) -> None:
        self._t_wc = np.eye(4)  # world ← camera, OpenCV-camera convention
        self._prev_gray: np.ndarray | None = None
        self._available = False
        self._last_frame_id = -1

    def track(
        self,
        frame_id: int,
        image: np.ndarray,
        depth: np.ndarray | None = None,
        timestamp: float | None = None,  # noqa: ARG002 — interface symmetry
    ) -> CameraPoseWorld:
        """Ingest one frame and advance the pose. Returns the updated pose."""
        gray = self._to_gray(image)
        self._last_frame_id = frame_id

        if self._prev_gray is None:
            # First frame anchors the world frame at this camera pose.
            self._prev_gray = gray
            self._available = True
            return self._current_pose()

        rel = self._vo.estimate(self._prev_gray, gray, self._k, prev_depth=depth)
        if rel is None or rel.n_inliers < self._min_inliers:
            self._available = False
            log.debug("SLAM tracking lost at frame %d", frame_id)
            return self._current_pose()  # hold last pose, mark unavailable

        motion = _se3(rel.rotation, rel.translation * self._scale)  # T_{curr←prev}
        self._t_wc = self._t_wc @ _se3_inv(motion)  # compose T_{world←curr}
        self._prev_gray = gray
        self._available = True
        return self._current_pose()

    def get(
        self,
        frame_id: int | None = None,  # noqa: ARG002 — pose is streaming, not indexed
        timestamp: float | None = None,  # noqa: ARG002
    ) -> CameraPoseWorld:
        return self._current_pose()

    def _current_pose(self) -> CameraPoseWorld:
        r_wc = self._t_wc[:3, :3]
        t_wc = self._t_wc[:3, 3]
        # Express the camera pose in graphics-world (see _CV_TO_GRAPHICS).
        r_pose = _CV_TO_GRAPHICS @ r_wc @ _CV_TO_GRAPHICS
        pos = _CV_TO_GRAPHICS @ t_wc
        quat = Rotation.from_matrix(r_pose).as_quat()  # (x, y, z, w)
        return CameraPoseWorld(
            available=self._available,
            source="slam",
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            quaternion=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
