"""Pure-pursuit path follower (spec §11.2).

Given a polyline path ``[(x0,y0), (x1,y1), ...]`` and current unicycle
state, the controller picks the first point on the path at arc-distance
``≥ lookahead_distance`` from the cat (or the path endpoint if no such
point exists), computes the heading error to it, and emits ``(v, ω)``.

We split the geometry helpers out as module-level pure functions so they
are individually testable and don't carry hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .kinematic import UnicycleState

Point2 = tuple[float, float]


@dataclass(frozen=True)
class ControlStep:
    """One controller tick — emitted by :class:`PathFollower` for logging.

    Field order matches the spec §11.4 control-log requirement
    (speed, heading error, cross-track error, path progress, final error).
    """

    t: float
    x: float
    y: float
    theta: float
    v: float
    omega: float
    heading_error: float
    cross_track_error: float
    path_progress: float  # 0.0 → 1.0 along the polyline
    distance_to_goal: float


# ── geometry helpers ────────────────────────────────────────────────────────


def _segment_lengths(path: list[Point2]) -> list[float]:
    out: list[float] = []
    for a, b in zip(path[:-1], path[1:], strict=True):
        out.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    return out


def _closest_point_on_segment(p: Point2, a: Point2, b: Point2) -> tuple[Point2, float]:
    """Return ``(projection, t)`` where ``t∈[0,1]`` is the param along ``a→b``."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return a, 0.0
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / seg_len_sq
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return (ax + t * dx, ay + t * dy), t


def closest_path_index(p: Point2, path: list[Point2]) -> tuple[int, float, float]:
    """Find the closest projection onto any segment of ``path``.

    Returns ``(segment_index, t_along_segment, signed_cross_track_distance)``.
    Sign convention: positive cross-track means the cat is to the left of
    the segment's forward direction (right-hand rule with planar normal up).
    """
    if len(path) < 2:
        return 0, 0.0, math.hypot(p[0] - path[0][0], p[1] - path[0][1])
    best_idx = 0
    best_t = 0.0
    best_d_sq = float("inf")
    best_signed: float = 0.0
    for i in range(len(path) - 1):
        proj, t = _closest_point_on_segment(p, path[i], path[i + 1])
        d_sq = (p[0] - proj[0]) ** 2 + (p[1] - proj[1]) ** 2
        if d_sq < best_d_sq:
            best_d_sq = d_sq
            best_idx = i
            best_t = t
            # signed distance using 2D cross product of segment-tangent × (p - a)
            ax, ay = path[i]
            bx, by = path[i + 1]
            tx, ty = bx - ax, by - ay
            seg_len = math.hypot(tx, ty)
            if seg_len > 1e-9:
                nx, ny = -ty / seg_len, tx / seg_len  # left-hand normal
                best_signed = (p[0] - ax) * nx + (p[1] - ay) * ny
            else:
                best_signed = math.sqrt(d_sq)
    return best_idx, best_t, best_signed


def lookahead_point(
    p: Point2,
    path: list[Point2],
    lookahead: float,
    *,
    seg_lengths: list[float] | None = None,
) -> Point2:
    """First point on the polyline at arc-distance ``≥ lookahead`` from ``p``.

    Falls back to the path endpoint when no such point exists — this is the
    spec's "drive to the goal" condition near the end of the path.
    """
    if len(path) == 1:
        return path[0]
    seg_lengths = seg_lengths or _segment_lengths(path)
    start_idx, start_t, _ = closest_path_index(p, path)
    # Start walking arc-length from the projection along segment ``start_idx``.
    ax, ay = path[start_idx]
    bx, by = path[start_idx + 1]
    proj_x = ax + start_t * (bx - ax)
    proj_y = ay + start_t * (by - ay)
    remaining = lookahead
    # First, walk the remainder of the current segment.
    seg_remaining = (1.0 - start_t) * seg_lengths[start_idx]
    if seg_remaining >= remaining:
        ratio = remaining / max(seg_lengths[start_idx], 1e-9)
        return (proj_x + ratio * (bx - ax), proj_y + ratio * (by - ay))
    remaining -= seg_remaining
    # Then consume subsequent segments.
    for i in range(start_idx + 1, len(path) - 1):
        seg_len = seg_lengths[i]
        if seg_len >= remaining:
            ratio = remaining / max(seg_len, 1e-9)
            ax_i, ay_i = path[i]
            bx_i, by_i = path[i + 1]
            return (ax_i + ratio * (bx_i - ax_i), ay_i + ratio * (by_i - ay_i))
        remaining -= seg_len
    return path[-1]


def path_progress(p: Point2, path: list[Point2], seg_lengths: list[float]) -> float:
    """0.0 (at start) → 1.0 (at end) — arc-length fraction along ``path``."""
    total = sum(seg_lengths)
    if total <= 1e-9:
        return 1.0
    idx, t, _ = closest_path_index(p, path)
    travelled = sum(seg_lengths[:idx]) + t * seg_lengths[idx]
    return max(0.0, min(1.0, travelled / total))


# ── controller ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PurePursuitController:
    """Spec §11.2 controller — stateless, parameterised by gains + limits."""

    lookahead_distance: float = 0.30
    base_speed: float = 0.45
    kp_heading: float = 2.40
    v_max: float = 0.80
    v_min: float = 0.05
    omega_max: float = 3.20
    # Allow callers to bypass the v_min floor for the final approach so the
    # cat doesn't overshoot.
    slow_down_radius: float = 0.20
    _seg_cache: dict[int, list[float]] = field(default_factory=dict, compare=False)

    def step(self, state: UnicycleState, path: list[Point2]) -> tuple[float, float, float, float]:
        """Compute one ``(v, ω, heading_error, cross_track_error)`` tuple.

        Returns command-space values; the caller is responsible for the
        actual integration (so PID / preempt logic can wrap us).
        """
        if not path:
            return 0.0, 0.0, 0.0, 0.0
        seg_lengths = _segment_lengths(path) if len(path) >= 2 else [0.0]
        target = lookahead_point(
            state.as_xz(), path, self.lookahead_distance, seg_lengths=seg_lengths
        )
        dx = target[0] - state.x
        dy = target[1] - state.y
        desired_heading = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(desired_heading - state.theta),
            math.cos(desired_heading - state.theta),
        )
        omega = self.kp_heading * heading_error
        # Spec §11.2: v = clamp(base · cos²(heading_error), v_min, v_max)
        cos_sq = math.cos(heading_error) ** 2
        v_raw = self.base_speed * max(0.0, cos_sq)
        # Slow down inside ``slow_down_radius`` of the goal so we can stop on a dime.
        end = path[-1]
        dist_to_goal = math.hypot(end[0] - state.x, end[1] - state.y)
        if dist_to_goal < self.slow_down_radius:
            v_raw *= max(0.0, dist_to_goal / max(self.slow_down_radius, 1e-9))
        v = max(self.v_min, min(v_raw, self.v_max)) if v_raw > 1e-6 else 0.0
        _, _, cross_track = closest_path_index(state.as_xz(), path)
        return v, omega, heading_error, cross_track
