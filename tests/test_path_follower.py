"""Phase 8 — path follower acceptance tests (spec §11.4)."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from src.control.kinematic import UnicycleState
from src.control.path_follower import PathFollower
from src.control.pid import PIDController
from src.control.pure_pursuit import PurePursuitController


def _follower(**overrides) -> PathFollower:
    defaults: dict = dict(
        controller=PurePursuitController(
            lookahead_distance=0.30,
            base_speed=0.45,
            kp_heading=2.4,
            v_max=0.8,
            v_min=0.05,
            omega_max=3.2,
            slow_down_radius=0.20,
        ),
        v_max=0.8,
        omega_max=3.2,
        dt=0.05,
        max_steps=400,
        goal_tolerance=0.08,
    )
    defaults.update(overrides)
    return PathFollower(**defaults)


def test_straight_line_reaches_goal_within_tolerance() -> None:
    follower = _follower()
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    initial = UnicycleState(x=0.0, y=0.0, theta=math.pi / 2)
    trace = follower.simulate(path, initial)
    assert trace.summary.status == "success"
    assert trace.summary.final_distance_to_goal <= 0.08
    assert trace.summary.max_cross_track_error < 0.05


def test_l_shaped_path_completes() -> None:
    follower = _follower()
    path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
    initial = UnicycleState(x=0.0, y=0.0, theta=0.0)
    trace = follower.simulate(path, initial)
    assert trace.summary.status == "success"
    assert trace.summary.max_cross_track_error < 0.20
    assert len(trace.path_world) > len(path)


def test_curved_path_no_huge_oscillation() -> None:
    follower = _follower()
    path = [(math.cos(t), 0.0, math.sin(t)) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    initial = UnicycleState(x=1.0, y=0.0, theta=math.pi / 2)
    trace = follower.simulate(path, initial)
    assert trace.summary.status == "success"
    assert trace.summary.max_heading_error < math.pi


def test_path_world_terminates_at_goal_exactly() -> None:
    follower = _follower()
    path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    trace = follower.simulate(path, UnicycleState())
    last = trace.path_world[-1]
    assert last == pytest.approx((1.0, 0.0, 0.0))


def test_max_steps_caps_runaway() -> None:
    follower = _follower(max_steps=5)
    path = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    trace = follower.simulate(path, UnicycleState())
    assert trace.summary.status == "max_steps"
    assert trace.summary.steps == 5


def test_empty_path_returns_empty_trace_without_raising() -> None:
    follower = _follower()
    trace = follower.simulate([], UnicycleState())
    assert trace.summary.status == "empty_path"
    assert trace.path_world == []
    assert trace.steps == []


def test_pid_smoothing_does_not_explode_speed() -> None:
    follower = _follower()
    follower.pid = PIDController(kp=0.5, ki=0.1, kd=0.0, integral_clamp=0.4)
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    trace = follower.simulate(path, UnicycleState(theta=math.pi / 2))
    assert trace.summary.status == "success"
    assert all(0.0 <= s.v <= follower.v_max for s in trace.steps)


def test_summary_records_progress_monotone() -> None:
    follower = _follower()
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.5)]
    trace = follower.simulate(path, UnicycleState(theta=math.pi / 2))
    progresses = [s.path_progress for s in trace.steps]
    for a, b in pairwise(progresses):
        assert b >= a - 1e-6


def test_handles_collocated_first_waypoint_without_nan() -> None:
    follower = _follower()
    initial = UnicycleState(x=0.5, y=0.0, theta=0.0)
    path = [(0.5, 0.0, 0.0), (0.5, 0.0, 1.0)]
    trace = follower.simulate(path, initial)
    assert trace.summary.status == "success"
    # cross-track should stay near zero since cat lies on the path
    assert trace.summary.max_cross_track_error < 0.1


def test_prepends_cat_position_when_path_starts_elsewhere() -> None:
    follower = _follower()
    initial = UnicycleState(x=0.0, y=0.0, theta=0.0)
    # Path starts 0.5m away — follower should still converge from (0,0).
    path = [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    trace = follower.simulate(path, initial)
    assert trace.summary.status == "success"
