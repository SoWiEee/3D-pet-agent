"""Car path-follower — spec §14.5 car kinematics.

Turns a Reeds-Shepp word into the dense, per-step trajectory the renderer
animates: a list of :class:`MotionSample` carrying the **real** control at
each tick — signed linear speed ``v`` (negative on reverse), yaw rate ``ω``,
front-wheel steering angle, and gear. This is the car analogue of
:class:`control.path_follower.ControlTrace`; it is what lets the browser drive
the robot's wheels from the actual controller output instead of guessing
``v``/``ω`` from frame-to-frame position deltas.

Reeds-Shepp turns are bang-bang: a curve segment runs at full steering lock
(``±δ_max``), a straight at ``δ = 0``. That reads as a real car cranking the
wheel over for a tight maneuver and centring it on the straights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .car_model import CarState, min_turning_radius, wrap_angle
from .reeds_shepp import ReedsSheppSegment, reeds_shepp_path

Pose = tuple[float, float, float]
TraceStatus = Literal["success", "fallback", "empty"]


@dataclass(frozen=True)
class MotionSample:
    """One controller tick, in the renderer's world frame (x, z ground plane)."""

    t: float
    x: float
    z: float
    theta: float
    v: float  # signed linear speed (m/s); < 0 = reverse
    omega: float  # yaw rate (rad/s)
    gear: int  # +1 forward, -1 reverse
    steer: float  # front-wheel angle (rad), signed


@dataclass(frozen=True)
class CarTrace:
    """Densified car trajectory + the path the renderer tweens through."""

    path_world: list[tuple[float, float, float]]  # (x, y_floor, z)
    samples: list[MotionSample]
    length: float
    n_reversals: int
    status: TraceStatus


@dataclass(frozen=True)
class CarFollowerConfig:
    wheelbase: float = 0.44
    max_steer: float = 0.55  # rad (~31°)
    v_max: float = 0.80  # m/s
    speed: float = 0.45  # m/s cruise magnitude
    dt: float = 0.05  # s integrator step
    floor_y: float = 0.0


class CarPathFollower:
    """Plan a Reeds-Shepp path to a goal pose and densify it into a trace."""

    def __init__(self, config: CarFollowerConfig | None = None) -> None:
        self.config = config or CarFollowerConfig()

    @property
    def turning_radius(self) -> float:
        return min_turning_radius(self.config.wheelbase, self.config.max_steer)

    def simulate(self, start: CarState, goal: Pose) -> CarTrace:
        cfg = self.config
        speed = max(1e-3, min(cfg.speed, cfg.v_max))
        step = max(speed * cfg.dt, 1e-3)
        radius = self.turning_radius
        start_pose: Pose = (start.x, start.y, wrap_angle(start.theta))

        path = reeds_shepp_path(start_pose, (goal[0], goal[1], wrap_angle(goal[2])), radius)
        if path is None:
            return self._fallback(start, goal, speed, step)

        samples = self._densify(start, path.segments, radius, speed, step)
        if not samples:
            return CarTrace([], [], 0.0, 0, "empty")
        n_reversals = sum(
            1
            for a, b in zip(path.segments[:-1], path.segments[1:], strict=True)
            if a.gear != b.gear
        )
        return self._finish(samples, goal, path.length, n_reversals, "success")

    # ── internals ────────────────────────────────────────────────────────────
    def _densify(
        self,
        start: CarState,
        segments: tuple[ReedsSheppSegment, ...],
        radius: float,
        speed: float,
        step: float,
    ) -> list[MotionSample]:
        cfg = self.config
        x, y, theta = start.x, start.y, wrap_angle(start.theta)
        t = 0.0
        samples = [MotionSample(t, x, y, theta, 0.0, 0.0, segments[0].gear, 0.0)]
        for seg in segments:
            metric = radius * seg.length
            n = max(1, int(math.ceil(metric / step)))
            sub_len = seg.length / n  # normalised arc length per sub-step
            v_signed = seg.gear * speed
            steer = seg.steering * cfg.max_steer
            # Yaw rate is exact for the bicycle model: θ̇ = gear·steering·|v|/R.
            omega = (seg.gear * seg.steering * speed / radius) if seg.steering != 0 else 0.0
            for _ in range(n):
                if seg.steering == 0:
                    x += seg.gear * radius * sub_len * math.cos(theta)
                    y += seg.gear * radius * sub_len * math.sin(theta)
                else:
                    theta2 = theta + seg.gear * seg.steering * sub_len
                    x += seg.steering * radius * (math.sin(theta2) - math.sin(theta))
                    y += seg.steering * radius * (math.cos(theta) - math.cos(theta2))
                    theta = wrap_angle(theta2)
                t += cfg.dt
                samples.append(
                    MotionSample(t, x, y, wrap_angle(theta), v_signed, omega, seg.gear, steer)
                )
        return samples

    def _finish(
        self,
        samples: list[MotionSample],
        goal: Pose,
        length: float,
        n_reversals: int,
        status: TraceStatus,
    ) -> CarTrace:
        floor = self.config.floor_y
        path_world = [(s.x, floor, s.z) for s in samples]
        # Anchor the final point exactly on the goal so the renderer lands clean.
        path_world[-1] = (goal[0], floor, goal[1])
        return CarTrace(path_world, samples, length, n_reversals, status)

    def _fallback(self, start: CarState, goal: Pose, speed: float, step: float) -> CarTrace:
        """Straight dash to the goal position when no RS word verified — should
        be unreachable for the complete word set, but the demo must never hang."""
        dx, dz = goal[0] - start.x, goal[1] - start.y
        dist = math.hypot(dx, dz)
        heading = math.atan2(dz, dx) if dist > 1e-6 else wrap_angle(start.theta)
        n = max(1, int(math.ceil(dist / step)))
        samples: list[MotionSample] = []
        for i in range(n + 1):
            f = i / n
            samples.append(
                MotionSample(
                    t=i * self.config.dt,
                    x=start.x + dx * f,
                    z=start.y + dz * f,
                    theta=heading,
                    v=speed if i < n else 0.0,
                    omega=0.0,
                    gear=1,
                    steer=0.0,
                )
            )
        return self._finish(samples, goal, dist, 0, "fallback")
