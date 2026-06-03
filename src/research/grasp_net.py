"""Point-cloud grasp synthesis — Stage D of the mobile-manipulator track (spec §14.5).

Stage C grasps from a *box approximation* (object centre + extent) with a
fixed top-down heuristic. Stage D upgrades that to a **6-DoF grasp synthesised
from the object point cloud**: it reads the actual geometry, so it can grip a
mug by the wall, a pen along its length, or a tilted box on its short face —
poses the axis-aligned heuristic cannot express.

The real Stage-D backend is a learned 6-DoF grasp net (GraspNet /
Contact-GraspNet / AnyGrasp), which needs heavy CUDA models + datasets. As
elsewhere on this track (cf. the ORB-VO stand-in for ORB-SLAM3), we ship a
pip-only **analytic sampler** behind a :class:`GraspSynthesizer` protocol: PCA
of the cloud gives the principal axes, the gripper closes along the thinnest
axis, approaches from the most top-down perpendicular direction, and candidates
are scored on gripper fit + top-down stability + centredness. A real net drops
in behind the protocol without touching call sites.

Grasp poses reuse the Stage-C contracts (:class:`~research.manipulation.GraspGoal`,
:class:`~research.manipulation.Pose`), so the existing
``plan_pick_and_place`` consumes a learned-or-analytic grasp identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .manipulation import ArmConfig, GraspGoal, Pose, _grasp_orientation

# World "up" is +Y; a top-down approach descends along -Y.
_DOWN = np.array([0.0, -1.0, 0.0])


@dataclass(frozen=True)
class GraspSamplerConfig:
    """Scoring weights + sampling density for the analytic sampler."""

    top_k: int = 5
    n_positions: int = 7  # grasp centres sampled along the object's long axis
    slab_fraction: float = 0.15  # half-width of the point slab (× long-axis span)
    w_fit: float = 0.5  # reward narrow (well-fitting) grasps
    w_topdown: float = 0.3  # reward approaching from above (stable)
    w_center: float = 0.2  # reward grasping near the centre of mass


class GraspSynthesizer(Protocol):
    """Synthesise ranked grasps for an object point cloud (world frame)."""

    def synthesize(self, points: np.ndarray, object_id: str) -> list[GraspGoal]: ...


# ── analytic sampler ────────────────────────────────────────────────────────
def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid, axes) where ``axes`` rows are the eigenvectors sorted
    by descending spread (axes[0] = longest, axes[2] = thinnest)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # descending
    return centroid, eigvecs[:, order].T  # rows = axes, [long, mid, thin]


def _span(points: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> float:
    proj = (points - centroid) @ axis
    return float(proj.max() - proj.min())


class AnalyticGraspSampler:
    """PCA + antipodal analytic grasp sampler (pip-only Stage-D stand-in)."""

    def __init__(self, arm: ArmConfig | None = None, cfg: GraspSamplerConfig | None = None) -> None:
        self.arm = arm or ArmConfig()
        self.cfg = cfg or GraspSamplerConfig()

    def synthesize(self, points: np.ndarray, object_id: str) -> list[GraspGoal]:
        points = np.asarray(points, dtype=float)
        if points.shape[0] < 3:
            return []
        centroid, axes = _principal_axes(points)
        long_axis, mid_axis, thin_axis = axes[0], axes[1], axes[2]

        # Gripper closes along the thinnest axis; approach is the perpendicular
        # direction most aligned with straight-down (most top-down access).
        closing = thin_axis
        approach = self._best_approach(long_axis, mid_axis)
        long_span = _span(points, long_axis, centroid)
        slab_half = max(self.cfg.slab_fraction * long_span, 1e-3)

        candidates: list[tuple[float, GraspGoal]] = []
        offsets = np.linspace(-0.5, 0.5, self.cfg.n_positions) * long_span
        proj_long = (points - centroid) @ long_axis
        for i, t in enumerate(offsets):
            slab = points[np.abs(proj_long - t) <= slab_half]
            if slab.shape[0] < 3:
                continue
            width = _span(slab, closing, slab.mean(axis=0))
            if width > self.arm.max_gripper_width:
                continue  # cannot close around this slice
            pos = centroid + t * long_axis
            score = self._score(width, approach, abs(t), long_span)
            candidates.append(
                (
                    score,
                    GraspGoal(
                        grasp_id=f"grasp_{object_id}_{i}",
                        target_object_id=object_id,
                        grasp_pose_world=Pose(
                            position=(float(pos[0]), float(pos[1]), float(pos[2])),
                            orientation=_grasp_orientation(approach, closing),
                        ),
                        approach_vector_world=(
                            float(approach[0]),
                            float(approach[1]),
                            float(approach[2]),
                        ),
                        gripper_width=min(
                            width + self.arm.grip_clearance, self.arm.max_gripper_width
                        ),
                        confidence=score,
                        explanation=(
                            f"6-DoF cloud grasp of {object_id}: width={width:.3f} m, "
                            f"approach·down={float(approach @ _DOWN):.2f}, "
                            f"offset={t:+.3f} m from centroid"
                        ),
                    ),
                )
            )

        candidates.sort(key=lambda sc: sc[0], reverse=True)
        return [g for _, g in candidates[: self.cfg.top_k]]

    def _best_approach(self, long_axis: np.ndarray, mid_axis: np.ndarray) -> np.ndarray:
        """Of the four directions perpendicular to the closing axis, pick the
        one most aligned with straight-down (gripper reaches in from above)."""
        candidates = [long_axis, -long_axis, mid_axis, -mid_axis]
        return max(candidates, key=lambda a: float(a @ _DOWN))

    def _score(self, width: float, approach: np.ndarray, offset: float, long_span: float) -> float:
        fit = 1.0 - width / self.arm.max_gripper_width  # narrower ⇒ better
        topdown = max(0.0, float(approach @ _DOWN))  # 1.0 when straight down
        centred = 1.0 - (offset / (0.5 * long_span + 1e-9))
        raw = self.cfg.w_fit * fit + self.cfg.w_topdown * topdown + self.cfg.w_center * centred
        return float(np.clip(raw, 0.0, 1.0))


class ContactGraspNetSynthesizer:  # pragma: no cover - requires the trained net
    """Live :class:`GraspSynthesizer` backed by a learned 6-DoF net. Imported
    lazily so the package has no heavy CUDA/model dependency."""

    def __init__(self, checkpoint: str) -> None:
        raise NotImplementedError(
            "wire up Contact-GraspNet / AnyGrasp inference here; "
            f"would load checkpoint {checkpoint!r}"
        )

    def synthesize(self, points: np.ndarray, object_id: str) -> list[GraspGoal]:
        raise NotImplementedError


# ── point-cloud helpers (synthetic + depth) ─────────────────────────────────
def points_from_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Back-project masked depth pixels to a camera-frame point cloud (N, 3).

    Mirrors the pinhole model in ``object_lifter`` — the densified per-object
    cloud Stage D consumes instead of the median-depth box. The caller applies
    the camera→world pose (and the graphics-world axis flip) as the lifter does.
    """
    ys, xs = np.nonzero(mask)
    ds = depth[ys, xs].astype(float)
    keep = np.isfinite(ds) & (ds > 0)
    ys, xs, ds = ys[keep], xs[keep], ds[keep]
    xc = (xs - cx) * ds / fx
    yc = (ys - cy) * ds / fy
    return np.column_stack([xc, yc, ds])


def box_point_cloud(
    center: tuple[float, float, float],
    extent: tuple[float, float, float],
    *,
    n: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Uniform random points inside an axis-aligned box (test fixture)."""
    rng = np.random.default_rng(seed)
    half = np.asarray(extent) / 2.0
    pts = rng.uniform(-half, half, size=(n, 3))
    return pts + np.asarray(center)


def cylinder_point_cloud(
    center: tuple[float, float, float],
    radius: float,
    height: float,
    *,
    axis: str = "y",
    n: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Uniform random points inside a cylinder about ``axis`` (test fixture)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * math.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    h = rng.uniform(-height / 2, height / 2, n)
    a, b = r * np.cos(theta), r * np.sin(theta)
    cols = {"x": (h, a, b), "y": (a, h, b), "z": (a, b, h)}[axis]
    return np.column_stack(cols) + np.asarray(center)
