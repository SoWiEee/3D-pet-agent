"""Reeds-Shepp shortest-path planner — spec §14.5 car kinematics.

A Reeds-Shepp path is the shortest curve between two SE(2) poses for a car
with a **minimum turning radius** that may drive **forwards or backwards**.
Unlike a unicycle it cannot pivot in place; unlike Dubins (forward-only) it
can reverse, which is what produces the back-and-forth "倒車喬角度" maneuver
when the goal heading is awkward relative to the approach.

Each path is a short word of segments, each a constant control: a left turn
(``steering=+1``), right turn (``-1``) or straight (``0``), driven forward
(``gear=+1``) or in reverse (``-1``). Lengths are **normalised** (turning
radius = 1); multiply by the real radius for metric distance.

We port the analytic OMPL / Reeds-Shepp 1990 word families (CSC, CCC, CCCC,
CCSC, CCSCC) with the four symmetry expansions (timeflip, reflect, both).
Every generated candidate is then **reconstructed** under the unit-radius
bicycle model and discarded unless it lands on the goal — so a transcription
slip in any one formula drops that candidate instead of returning a wrong
path. The shortest surviving candidate wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi
_ZERO = 1e-9

Pose = tuple[float, float, float]


def _mod2pi(x: float) -> float:
    """Wrap to ``(-π, π]`` (OMPL's ``mod2pi`` convention)."""
    v = math.fmod(x, _TWO_PI)
    if v < -math.pi:
        v += _TWO_PI
    elif v > math.pi:
        v -= _TWO_PI
    return v


def _polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _tau_omega(u: float, v: float, xi: float, eta: float, phi: float) -> tuple[float, float]:
    delta = _mod2pi(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = _mod2pi(t1 + math.pi) if t2 < 0 else _mod2pi(t1)
    omega = _mod2pi(tau - u + v - phi)
    return tau, omega


# ── base word solvers ───────────────────────────────────────────────────────
# Each returns the segment lengths (t, u, v[, ...]) in fixed L/R/S order if the
# word is geometrically valid, else ``None``. Steering patterns live alongside
# in ``_FAMILIES`` so the symmetry expander can swap L↔R on reflection.


def _lp_sp_lp(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CSC
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= -_ZERO:
        v = _mod2pi(phi - t)
        if v >= -_ZERO:
            return (t, u, v)
    return None


def _lp_sp_rp(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CSC
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= -_ZERO and v >= -_ZERO:
            return (t, u, v)
    return None


def _lp_rm_l(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CCC
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = _mod2pi(theta + 0.5 * u + math.pi)
        v = _mod2pi(phi - t + u)
        if t >= -_ZERO and u <= _ZERO:
            return (t, u, v)
    return None


def _lp_rup_lum_rm(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CCCC
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -_ZERO and v <= _ZERO:
            return (t, u, -u, v)
    return None


def _lp_rum_lum_rp(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CCCC
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -_HALF_PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -_ZERO and v >= -_ZERO:
                return (t, u, u, v)
    return None


def _lp_rm_sm_lm(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CCSC
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = _mod2pi(theta + math.atan2(r, -2.0))
        v = _mod2pi(phi - _HALF_PI - t)
        if t >= -_ZERO and u <= _ZERO and v <= _ZERO:
            return (t, -_HALF_PI, u, v)
    return None


def _lp_rm_sm_rm(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CCSC
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = _mod2pi(t + _HALF_PI - phi)
        if t >= -_ZERO and u <= _ZERO and v <= _ZERO:
            return (t, -_HALF_PI, u, v)
    return None


def _lp_rm_sl_m_rp(x: float, y: float, phi: float) -> tuple[float, ...] | None:  # CCSCC
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= _ZERO:
            t = _mod2pi(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = _mod2pi(t - phi)
            if t >= -_ZERO and v >= -_ZERO:
                return (t, -_HALF_PI, u, -_HALF_PI, v)
    return None


# (solver, steering-pattern) — steering: +1 left, -1 right, 0 straight.
_FAMILIES: tuple[tuple, ...] = (
    (_lp_sp_lp, (1, 0, 1)),
    (_lp_sp_rp, (1, 0, -1)),
    (_lp_rm_l, (1, -1, 1)),
    (_lp_rup_lum_rm, (1, -1, 1, -1)),
    (_lp_rum_lum_rp, (1, -1, 1, -1)),
    (_lp_rm_sm_lm, (1, -1, 0, 1)),
    (_lp_rm_sm_rm, (1, -1, 0, -1)),
    (_lp_rm_sl_m_rp, (1, -1, 0, 1, -1)),
)


@dataclass(frozen=True)
class ReedsSheppSegment:
    """One constant-control segment. ``length`` is non-negative and
    **normalised** (turning radius = 1)."""

    steering: int  # +1 left, -1 right, 0 straight
    gear: int  # +1 forward, -1 reverse
    length: float


@dataclass(frozen=True)
class ReedsSheppPath:
    """An immutable Reeds-Shepp word + the radius it was solved for."""

    segments: tuple[ReedsSheppSegment, ...]
    radius: float

    @property
    def length(self) -> float:
        """Total **metric** arc length (radius × normalised lengths)."""
        return self.radius * sum(s.length for s in self.segments)

    def sample(self, start: Pose, radius: float, step: float = 0.05) -> list[Pose]:
        return _sample(start, self, radius, step)


def _signed_word_to_segments(word: list[tuple[int, float]]) -> tuple[ReedsSheppSegment, ...]:
    """Convert (steering, signed-length) pairs into normalised forward-length
    segments with an explicit gear, dropping ~zero-length controls."""
    out: list[ReedsSheppSegment] = []
    for steering, signed_len in word:
        if abs(signed_len) < _ZERO:
            continue
        gear = 1 if signed_len >= 0 else -1
        out.append(ReedsSheppSegment(steering=steering, gear=gear, length=abs(signed_len)))
    return tuple(out)


def _step_pose(pose: Pose, seg: ReedsSheppSegment, radius: float) -> Pose:
    """Exact closed-form advance of one segment under the bicycle model."""
    x, y, theta = pose
    g, s, length = seg.gear, seg.steering, seg.length
    if s == 0:
        nx = x + g * radius * length * math.cos(theta)
        ny = y + g * radius * length * math.sin(theta)
        return (nx, ny, theta)
    theta2 = theta + g * s * length
    nx = x + s * radius * (math.sin(theta2) - math.sin(theta))
    ny = y + s * radius * (math.cos(theta) - math.cos(theta2))
    return (nx, ny, _mod2pi(theta2))


def reconstruct(start: Pose, path: ReedsSheppPath, radius: float) -> Pose:
    """Integrate the whole word from ``start``; returns the final pose."""
    pose = (start[0], start[1], _mod2pi(start[2]))
    for seg in path.segments:
        pose = _step_pose(pose, seg, radius)
    return (pose[0], pose[1], _mod2pi(pose[2]))


def _sample(start: Pose, path: ReedsSheppPath, radius: float, step: float) -> list[Pose]:
    poses: list[Pose] = [(start[0], start[1], _mod2pi(start[2]))]
    pose = poses[0]
    for seg in path.segments:
        metric = radius * seg.length
        n = max(1, int(math.ceil(metric / max(step, 1e-4))))
        sub = seg.length / n
        for _ in range(n):
            pose = _step_pose(pose, ReedsSheppSegment(seg.steering, seg.gear, sub), radius)
            poses.append(pose)
    return poses


def _candidates(x: float, y: float, phi: float):
    """Yield every (steering, signed-length) word from all families × the four
    symmetries (direct, timeflip, reflect, timeflip+reflect)."""
    for solver, pattern in _FAMILIES:
        reflected = tuple(-s for s in pattern)
        # direct
        lens = solver(x, y, phi)
        if lens is not None:
            yield list(zip(pattern, lens, strict=True))
        # timeflip — solve mirrored-in-time problem, negate lengths
        lens = solver(-x, y, -phi)
        if lens is not None:
            yield list(zip(pattern, (-v for v in lens), strict=True))
        # reflect — solve mirrored-in-y problem, swap L↔R
        lens = solver(x, -y, -phi)
        if lens is not None:
            yield list(zip(reflected, lens, strict=True))
        # timeflip + reflect
        lens = solver(-x, -y, phi)
        if lens is not None:
            yield list(zip(reflected, (-v for v in lens), strict=True))


def reeds_shepp_path(start: Pose, goal: Pose, radius: float) -> ReedsSheppPath | None:
    """Shortest verified Reeds-Shepp path from ``start`` to ``goal``.

    Returns ``None`` only if no family produces a path that reconstructs to the
    goal — which, for the complete word set, should not happen for finite poses.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    # Transform goal into the start frame, normalised by the turning radius, so
    # the analytic formulas (which assume start = origin, radius = 1) apply.
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    c = math.cos(start[2])
    s = math.sin(start[2])
    x = (c * dx + s * dy) / radius
    y = (-s * dx + c * dy) / radius
    phi = _mod2pi(goal[2] - start[2])

    best: ReedsSheppPath | None = None
    best_len = math.inf
    for word in _candidates(x, y, phi):
        segments = _signed_word_to_segments(word)
        if not segments:
            continue
        candidate = ReedsSheppPath(segments=segments, radius=radius)
        end = reconstruct(start, candidate, radius)
        if not _pose_close(end, goal):
            continue
        if candidate.length < best_len:
            best, best_len = candidate, candidate.length
    return best


def _pose_close(a: Pose, b: Pose, tol: float = 1e-5) -> bool:
    dth = _mod2pi(a[2] - b[2])
    return math.hypot(a[0] - b[0], a[1] - b[1]) < tol and abs(dth) < tol
