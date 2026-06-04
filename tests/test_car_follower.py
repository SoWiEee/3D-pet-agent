"""Car follower + bicycle model — spec §14.5 car kinematics."""

from __future__ import annotations

import math

import pytest

from src.control.car_follower import CarFollowerConfig, CarPathFollower
from src.control.car_model import CarState, bicycle_step, min_turning_radius, wrap_angle

# ── bicycle model ────────────────────────────────────────────────────────────


def test_stationary_car_cannot_change_heading() -> None:
    s = CarState(0.0, 0.0, 0.0)
    out = bicycle_step(s, v=0.0, steer=0.5, dt=0.1, wheelbase=0.44, v_max=0.8, max_steer=0.55)
    assert out.theta == pytest.approx(0.0)


def test_reverse_moves_backwards_along_heading() -> None:
    s = CarState(0.0, 0.0, 0.0)
    out = bicycle_step(s, v=-0.5, steer=0.0, dt=0.1, wheelbase=0.44, v_max=0.8, max_steer=0.55)
    assert out.x < 0.0
    assert out.y == pytest.approx(0.0)


def test_min_turning_radius_matches_formula() -> None:
    r = min_turning_radius(0.44, 0.55)
    assert r == pytest.approx(0.44 / math.tan(0.55), rel=1e-9)


# ── follower ─────────────────────────────────────────────────────────────────


def _cfg() -> CarFollowerConfig:
    return CarFollowerConfig(wheelbase=0.44, max_steer=0.55, v_max=0.8, speed=0.45, dt=0.05)


def test_straight_goal_drives_forward_only() -> None:
    follower = CarPathFollower(_cfg())
    trace = follower.simulate(CarState(0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert trace.status == "success"
    assert trace.n_reversals == 0
    assert all(s.v >= 0 for s in trace.samples)
    # Lands on the goal.
    last = trace.path_world[-1]
    assert last[0] == pytest.approx(2.0, abs=1e-3)
    assert last[2] == pytest.approx(0.0, abs=1e-3)


def test_goal_behind_uses_reverse() -> None:
    follower = CarPathFollower(_cfg())
    trace = follower.simulate(CarState(0.0, 0.0, 0.0), (-1.5, 0.0, 0.0))
    assert trace.status == "success"
    assert any(s.v < 0 for s in trace.samples), "expected reverse motion"


def test_samples_respect_speed_and_steer_limits() -> None:
    follower = CarPathFollower(_cfg())
    trace = follower.simulate(CarState(0.0, 0.0, 0.0), (1.0, 1.2, math.pi / 2))
    assert trace.samples
    for s in trace.samples:
        assert abs(s.v) <= 0.45 + 1e-9
        assert abs(s.steer) <= 0.55 + 1e-9


def test_time_is_monotonic() -> None:
    follower = CarPathFollower(_cfg())
    trace = follower.simulate(CarState(0.0, 0.0, 0.0), (1.5, -0.8, -0.6))
    ts = [s.t for s in trace.samples]
    assert all(b >= a for a, b in zip(ts[:-1], ts[1:], strict=True))


def test_omega_consistent_with_heading_change() -> None:
    # On a curved segment, the reported ω should match the realised heading rate.
    follower = CarPathFollower(_cfg())
    trace = follower.simulate(CarState(0.0, 0.0, 0.0), (1.0, 1.0, math.pi / 2))
    dt = 0.05
    for a, b in zip(trace.samples[1:-1], trace.samples[2:], strict=True):
        if b.steer != 0.0 and b.gear == a.gear:
            dtheta = wrap_angle(b.theta - a.theta)
            assert dtheta / dt == pytest.approx(b.omega, abs=0.05)


def test_endpoint_matches_awkward_goal() -> None:
    follower = CarPathFollower(_cfg())
    goal = (0.3, -1.4, 2.3)
    trace = follower.simulate(CarState(0.0, 0.0, 0.0), goal)
    last = trace.path_world[-1]
    assert last[0] == pytest.approx(goal[0], abs=1e-3)
    assert last[2] == pytest.approx(goal[1], abs=1e-3)
