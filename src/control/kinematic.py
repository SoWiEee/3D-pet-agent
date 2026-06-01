"""Planar unicycle kinematic model (spec §11.1).

State ``(x, y, θ)`` — note that here ``y`` is the **planar second axis**,
i.e. we operate in the XZ ground plane of the graphics world. The renderer
treats world-Y as up, so we map (x_world, z_world) ↔ (kinematic x, y).

Pure functions, frozen state — every step returns a new :class:`UnicycleState`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UnicycleState:
    """Immutable planar pose. ``theta`` is the heading angle in radians,
    measured counter-clockwise from the +x axis (so heading 0 means
    "facing world +X" and heading π/2 means "facing world +Z")."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    t: float = 0.0

    def as_xz(self) -> tuple[float, float]:
        return (self.x, self.y)


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def kinematic_step(
    state: UnicycleState,
    v: float,
    omega: float,
    dt: float,
    *,
    v_max: float,
    omega_max: float,
    v_min_nonzero: float = 0.0,
) -> UnicycleState:
    """One Euler step under control ``(v, ω)``.

    Returns a **new** state. ``v`` is clamped to ``[0, v_max]`` (the cat
    never reverses; it would rather rotate in place). ``ω`` is clamped to
    ``[-omega_max, omega_max]``. ``v_min_nonzero`` filters out below-floor
    speeds so a controller that emits ``v = 1e-12`` doesn't accumulate
    numerical drift across hundreds of frames.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    v_cmd = _clamp(v, 0.0, v_max)
    if 0.0 < v_cmd < v_min_nonzero:
        v_cmd = 0.0
    w_cmd = _clamp(omega, -omega_max, omega_max)
    new_theta = state.theta + w_cmd * dt
    # Normalise to (-π, π] so heading errors stay sane over long runs.
    new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
    new_x = state.x + v_cmd * math.cos(state.theta) * dt
    new_y = state.y + v_cmd * math.sin(state.theta) * dt
    return UnicycleState(x=new_x, y=new_y, theta=new_theta, t=state.t + dt)
