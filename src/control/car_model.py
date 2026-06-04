"""Bicycle (car-like) kinematic model — spec §14.5 Stage E (car kinematics).

Where :mod:`control.kinematic` is a *unicycle* (can spin in place, never
reverses — the cat's model), this is a **front-steered car**: it has a finite
wheelbase ``L`` and a steering limit ``δ_max``, so it cannot rotate on the
spot and has a hard **minimum turning radius** ``R = L / tan(δ_max)``. It *can*
drive in reverse (``v < 0``), which is what lets a Reeds-Shepp planner shuffle
the car back and forth to square up its heading ("倒車喬角度").

State ``(x, y, θ)`` lives in the same XZ ground plane as the unicycle
(``y`` is the planar second axis = world Z), so a car path drops straight into
the same ``move_follow_path`` renderer the cat uses.

Pure functions, frozen state — every step returns a new :class:`CarState`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CarState:
    """Immutable planar car pose. ``theta`` is the heading (radians, CCW from
    +x), i.e. the direction the chassis *points* — independent of whether the
    car is driving forward or reversing."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    t: float = 0.0

    def as_xz(self) -> tuple[float, float]:
        return (self.x, self.y)


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def wrap_angle(theta: float) -> float:
    """Normalise to ``(-π, π]``."""
    return math.atan2(math.sin(theta), math.cos(theta))


def min_turning_radius(wheelbase: float, max_steer: float) -> float:
    """``R = L / tan(δ_max)`` — the tightest circle the car can trace.

    ``max_steer`` is clamped below 90° so ``tan`` stays finite; a car that
    could steer to 90° would have zero turning radius (i.e. spin in place),
    which is exactly the non-holonomic constraint we want to forbid.
    """
    if wheelbase <= 0.0:
        raise ValueError("wheelbase must be positive")
    delta = _clamp(abs(max_steer), 1e-3, math.radians(89.0))
    return wheelbase / math.tan(delta)


def bicycle_step(
    state: CarState,
    v: float,
    steer: float,
    dt: float,
    *,
    wheelbase: float,
    v_max: float,
    max_steer: float,
) -> CarState:
    """One Euler step of the kinematic bicycle model.

    ``v`` is the rear-axle speed and **may be negative** (reverse). It is
    clamped to ``[-v_max, v_max]``; ``steer`` (front-wheel angle) to
    ``[-max_steer, max_steer]``. Heading rate is ``θ̇ = v·tan(δ) / L``, so a
    stationary car (``v = 0``) cannot change heading — no in-place pivot.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    v_cmd = _clamp(v, -v_max, v_max)
    delta = _clamp(steer, -max_steer, max_steer)
    new_x = state.x + v_cmd * math.cos(state.theta) * dt
    new_y = state.y + v_cmd * math.sin(state.theta) * dt
    new_theta = wrap_angle(state.theta + (v_cmd * math.tan(delta) / wheelbase) * dt)
    return CarState(x=new_x, y=new_y, theta=new_theta, t=state.t + dt)
