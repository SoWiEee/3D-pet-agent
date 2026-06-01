"""Score spatial relations between ``ObjectState3D`` pairs (spec §8.2).

Coordinate frame
----------------
All scoring operates in the (graphics-)world frame produced by
``ObjectLifter``: X right, Y up, Z toward the viewer (so camera looks down
−Z). This means:

- ``left_of`` / ``right_of``  → compare ``x``
- ``above`` / ``below``       → compare ``y``
- ``in_front_of`` / ``behind`` → camera-relative; an object is *in front of*
  the other when it is **closer to the camera**, i.e. its ``z`` is **larger**
  (less negative) than the other's.

For ``occluding`` we use the **camera-frame ``median_depth``** which the lifter
records — smaller is closer in that frame, regardless of the world axis flip.

Scoring shape
-------------
Each relation returns a score in ``[0, 1]``. Axis-projected relations use a
smooth ramp::

    s = clip((delta − threshold) / (2·threshold), 0, 1)

so the threshold is the **half-confidence point**, and 2× threshold means full
confidence. Below threshold the score is zero (the relation is undefined or
the wrong sign). This is good enough for grounding in Phase 6 — replace with a
calibrated classifier later if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .object_lifter import ObjectState3D


def _ramp(delta: float, threshold: float) -> float:
    """Smooth one-sided ramp: 0 at ``delta ≤ 0``, 1 at ``delta ≥ 2·threshold``."""
    if delta <= 0.0:
        return 0.0
    return min(1.0, delta / max(1e-6, 2.0 * threshold))


def _bbox_iou_2d(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter
    )
    return inter / union if union > 0 else 0.0


@dataclass
class RelationConfig:
    """Tunables. Defaults sourced from spec ``configs/thresholds.yaml::relations``."""

    near_sigma: float = 0.5
    right_left_threshold: float = 0.08
    behind_front_threshold: float = 0.10
    above_below_threshold: float = 0.06
    between_segment_tolerance: float = 0.12
    surface_attach_tolerance: float = 0.03
    occlusion_min_iou: float = 0.05


class RelationScorer:
    """Pairwise + triadic spatial-relation scoring on ObjectState3D.

    Stateless; instantiate once and call ``score_<rel>`` for each pair.
    """

    def __init__(self, cfg: RelationConfig | None = None) -> None:
        self.cfg = cfg or RelationConfig()

    # ── axis-projected pair relations ───────────────────────────────────────
    def left_of(self, a: ObjectState3D, b: ObjectState3D) -> float:
        # a is left of b → a.x < b.x → delta = b.x − a.x
        return _ramp(b.center_3d_world[0] - a.center_3d_world[0], self.cfg.right_left_threshold)

    def right_of(self, a: ObjectState3D, b: ObjectState3D) -> float:
        return _ramp(a.center_3d_world[0] - b.center_3d_world[0], self.cfg.right_left_threshold)

    def above(self, a: ObjectState3D, b: ObjectState3D) -> float:
        return _ramp(a.center_3d_world[1] - b.center_3d_world[1], self.cfg.above_below_threshold)

    def below(self, a: ObjectState3D, b: ObjectState3D) -> float:
        return _ramp(b.center_3d_world[1] - a.center_3d_world[1], self.cfg.above_below_threshold)

    def in_front_of(self, a: ObjectState3D, b: ObjectState3D) -> float:
        # Closer to camera (graphics world: larger z).
        return _ramp(a.center_3d_world[2] - b.center_3d_world[2], self.cfg.behind_front_threshold)

    def behind(self, a: ObjectState3D, b: ObjectState3D) -> float:
        return _ramp(b.center_3d_world[2] - a.center_3d_world[2], self.cfg.behind_front_threshold)

    # ── distance-driven pair relations ──────────────────────────────────────
    def near(self, a: ObjectState3D, b: ObjectState3D) -> float:
        d = self._distance(a, b)
        return math.exp(-0.5 * (d / max(1e-6, self.cfg.near_sigma)) ** 2)

    def far_from(self, a: ObjectState3D, b: ObjectState3D) -> float:
        return 1.0 - self.near(a, b)

    # ── triadic / mask-based relations ──────────────────────────────────────
    def between(self, a: ObjectState3D, b: ObjectState3D, c: ObjectState3D) -> float:
        """Score how much ``a`` lies on the segment from ``b`` to ``c``.

        Returns 0 unless ``a``'s projection lands inside the segment AND its
        perpendicular distance is within ``between_segment_tolerance``.
        """
        ax, ay, az = a.center_3d_world
        bx, by, bz = b.center_3d_world
        cx, cy, cz = c.center_3d_world
        vx, vy, vz = cx - bx, cy - by, cz - bz
        seg_len_sq = vx * vx + vy * vy + vz * vz
        if seg_len_sq < 1e-9:
            return 0.0
        # Projection parameter t along BC.
        t = ((ax - bx) * vx + (ay - by) * vy + (az - bz) * vz) / seg_len_sq
        if t <= 0.0 or t >= 1.0:
            return 0.0
        # Foot of perpendicular.
        fx, fy, fz = bx + t * vx, by + t * vy, bz + t * vz
        perp = math.sqrt((ax - fx) ** 2 + (ay - fy) ** 2 + (az - fz) ** 2)
        tol = self.cfg.between_segment_tolerance
        if perp >= 2.0 * tol:
            return 0.0
        endpoint_score = 1.0 - abs(t - 0.5) * 2.0  # peaks at midpoint
        perp_score = max(0.0, 1.0 - perp / (2.0 * tol))
        return float(max(0.0, min(1.0, 0.5 * endpoint_score + 0.5 * perp_score)))

    def on_surface(self, a: ObjectState3D, s: ObjectState3D) -> float:
        """Score whether ``a`` rests on the top of ``s``.

        Plane-attachment proxy: bottom of ``a`` near top of ``s`` along Y, and
        ``a``'s XZ centroid lies within ``s``'s XZ extent.
        """
        _, ay, _ = a.center_3d_world
        _, sy, _ = s.center_3d_world
        a_half_y = a.extent_3d[1] * 0.5
        s_half_y = s.extent_3d[1] * 0.5
        a_bottom = ay - a_half_y
        s_top = sy + s_half_y
        gap = abs(a_bottom - s_top)
        if gap > 2.0 * self.cfg.surface_attach_tolerance:
            return 0.0
        # XZ overlap.
        if not self._xz_inside(a, s):
            return 0.0
        return float(max(0.0, 1.0 - gap / (2.0 * self.cfg.surface_attach_tolerance)))

    def occluding(self, a: ObjectState3D, b: ObjectState3D) -> float:
        """Score whether ``a`` occludes ``b`` from the camera.

        Combines 2D bbox IoU (must exceed ``occlusion_min_iou``) with
        camera-frame depth ordering — ``a`` must be closer than ``b`` in
        ``median_depth`` (camera Z, smaller = closer).
        """
        iou = _bbox_iou_2d(a.bbox_xyxy, b.bbox_xyxy)
        if iou < self.cfg.occlusion_min_iou:
            return 0.0
        if a.median_depth >= b.median_depth:
            return 0.0
        depth_gap = b.median_depth - a.median_depth
        depth_score = _ramp(depth_gap, 0.05)
        return float(iou * 0.6 + depth_score * 0.4)

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _distance(a: ObjectState3D, b: ObjectState3D) -> float:
        ax, ay, az = a.center_3d_world
        bx, by, bz = b.center_3d_world
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    @staticmethod
    def _xz_inside(a: ObjectState3D, s: ObjectState3D) -> bool:
        ax, _, az = a.center_3d_world
        sx, _, sz = s.center_3d_world
        sx_half, _, sz_half = (e * 0.5 for e in s.extent_3d)
        return (abs(ax - sx) <= sx_half) and (abs(az - sz) <= sz_half)


# Convenient list — Phase 5 acceptance uses these names as JSON labels.
BASE_PAIR_RELATIONS: tuple[str, ...] = (
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "above",
    "below",
    "near",
    "far_from",
    "on_surface",
    "occluding",
)
TRIADIC_RELATIONS: tuple[str, ...] = ("between",)
