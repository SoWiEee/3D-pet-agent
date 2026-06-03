"""Visual SLAM sidecar (spec §14.1).

Two layers:
- ``SLAMPoseSource`` integration logic (pose composition, graphics-world
  handshake, lost-tracking, scale) — tested deterministically with a stub VO.
- ``OrbVisualOdometry`` real OpenCV plumbing — tested on synthetic frames.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.research.slam_adapter import (
    OrbVisualOdometry,
    RelativePose,
    SLAMPoseSource,
)
from src.spatial.frame_packet import CameraIntrinsics

INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)


class _StubVO:
    """Returns a scripted sequence of relative poses (None = tracking lost)."""

    def __init__(self, results: list[RelativePose | None]) -> None:
        self._results = list(results)
        self.calls = 0

    def estimate(self, prev_gray, curr_gray, k_matrix, prev_depth=None):  # noqa: ANN001, ARG002
        out = self._results[self.calls] if self.calls < len(self._results) else None
        self.calls += 1
        return out


def _forward(d: float, inliers: int = 100) -> RelativePose:
    """A pure +Z translation of magnitude d (identity rotation)."""
    return RelativePose(rotation=np.eye(3), translation=np.array([0.0, 0.0, d]), n_inliers=inliers)


def _gray_frame() -> np.ndarray:
    return np.zeros((480, 640), dtype=np.uint8)


def _is_rotation(r: np.ndarray) -> bool:
    return (
        r.shape == (3, 3)
        and np.allclose(r @ r.T, np.eye(3), atol=1e-4)
        and abs(np.linalg.det(r) - 1.0) < 1e-4
    )


# ── SLAMPoseSource integration logic (deterministic) ─────────────────────────
def test_first_frame_anchors_world_at_identity() -> None:
    src = SLAMPoseSource(INTRINSICS, vo=_StubVO([]))
    pose = src.track(0, _gray_frame())

    assert pose.available is True
    assert pose.source == "slam"
    assert pose.position == (0.0, 0.0, 0.0)
    assert pytest.approx(pose.quaternion, abs=1e-9) == (0.0, 0.0, 0.0, 1.0)


def test_forward_motion_accumulates_in_graphics_world() -> None:
    # Pure +Z camera motion composes deterministically; the graphics-world flip
    # (y, z) maps it to a known published position.
    src = SLAMPoseSource(INTRINSICS, vo=_StubVO([_forward(0.5), _forward(0.5)]))
    src.track(0, _gray_frame())
    p1 = src.track(1, _gray_frame())
    p2 = src.track(2, _gray_frame())

    assert pytest.approx(p1.position, abs=1e-9) == (0.0, 0.0, 0.5)
    assert pytest.approx(p2.position, abs=1e-9) == (0.0, 0.0, 1.0)  # accumulates
    # Identity rotation throughout.
    assert pytest.approx(p2.quaternion, abs=1e-9) == (0.0, 0.0, 0.0, 1.0)


def test_scale_multiplies_translation() -> None:
    src = SLAMPoseSource(INTRINSICS, vo=_StubVO([_forward(0.5)]), scale=2.0)
    src.track(0, _gray_frame())
    p1 = src.track(1, _gray_frame())
    assert pytest.approx(p1.position, abs=1e-9) == (0.0, 0.0, 1.0)


def test_lost_tracking_holds_pose_but_marks_unavailable() -> None:
    src = SLAMPoseSource(INTRINSICS, vo=_StubVO([_forward(0.5), None]))
    src.track(0, _gray_frame())
    moved = src.track(1, _gray_frame())
    lost = src.track(2, _gray_frame())

    assert moved.available is True
    assert lost.available is False
    assert lost.position == moved.position  # held, not reset


def test_low_inlier_estimate_treated_as_lost() -> None:
    src = SLAMPoseSource(INTRINSICS, vo=_StubVO([_forward(0.5, inliers=3)]), min_inliers=12)
    src.track(0, _gray_frame())
    pose = src.track(1, _gray_frame())
    assert pose.available is False
    assert pose.position == (0.0, 0.0, 0.0)  # never moved


def test_get_returns_current_pose_and_conforms_to_protocol() -> None:
    src = SLAMPoseSource(INTRINSICS, vo=_StubVO([_forward(0.5)]))
    src.track(0, _gray_frame())
    src.track(1, _gray_frame())
    # PoseSource.get(frame_id) signature.
    assert src.get(1).position == (0.0, 0.0, 0.5)
    assert src.get().source == "slam"


def test_reset_returns_to_origin() -> None:
    src = SLAMPoseSource(INTRINSICS, vo=_StubVO([_forward(0.5)]))
    src.track(0, _gray_frame())
    src.track(1, _gray_frame())
    src.reset()
    src.track(0, _gray_frame())
    assert src.get().position == (0.0, 0.0, 0.0)


# ── OrbVisualOdometry real OpenCV plumbing ───────────────────────────────────
def _textured_image(seed: int = 0) -> np.ndarray:
    """A dense field of random discs — plenty of stable ORB corners."""
    h, w = 480, 640
    img = np.full((h, w), 28, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(260):
        c = (int(rng.integers(8, w - 8)), int(rng.integers(8, h - 8)))
        cv2.circle(img, c, int(rng.integers(5, 16)), int(rng.integers(90, 255)), -1)
    return img


def _k_matrix() -> np.ndarray:
    return np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


def test_orb_vo_essential_path_handles_degenerate_planar_motion() -> None:
    # A pure lateral shift of a fronto-planar scene is geometrically degenerate
    # for the essential matrix (no parallax). The estimator must handle it
    # gracefully: either bail out (None) or return a structurally valid pose —
    # never crash or emit a non-rotation. The metric accuracy check lives in the
    # PnP test below, which is the non-degenerate RGB-D path.
    vo = OrbVisualOdometry()
    img = _textured_image()
    rel = vo.estimate(img, _shift(img, 6, 0), _k_matrix())

    assert rel is None or (_is_rotation(rel.rotation) and np.all(np.isfinite(rel.translation)))


def test_orb_vo_pnp_path_recovers_near_identity_rotation() -> None:
    # Constant-depth + pure lateral shift is non-degenerate for PnP: it should
    # recover a near-identity rotation (the camera only translated).
    vo = OrbVisualOdometry()
    img = _textured_image(seed=1)
    depth = np.full(img.shape, 2.0, dtype=np.float64)
    rel = vo.estimate(img, _shift(img, 8, 0), _k_matrix(), prev_depth=depth)

    assert rel is not None
    assert _is_rotation(rel.rotation)
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel.rotation) - 1.0) / 2.0))))
    assert angle_deg < 10.0


def test_orb_vo_returns_none_on_blank_frames() -> None:
    vo = OrbVisualOdometry()
    blank = np.zeros((480, 640), dtype=np.uint8)
    assert vo.estimate(blank, blank, _k_matrix()) is None
