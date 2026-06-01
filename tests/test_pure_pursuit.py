"""Phase 8 — pure-pursuit geometry + controller unit tests."""

from __future__ import annotations

import math

import pytest

from src.control.kinematic import UnicycleState
from src.control.pure_pursuit import (
    PurePursuitController,
    closest_path_index,
    lookahead_point,
    path_progress,
)


def test_closest_path_index_picks_correct_segment() -> None:
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    idx, t, _ = closest_path_index((0.4, 0.0), path)
    assert idx == 0
    assert t == pytest.approx(0.4)


def test_closest_path_index_cross_track_sign() -> None:
    _, _, signed = closest_path_index((0.5, 0.5), [(0.0, 0.0), (1.0, 0.0)])
    assert signed == pytest.approx(0.5)
    _, _, signed2 = closest_path_index((0.5, -0.5), [(0.0, 0.0), (1.0, 0.0)])
    assert signed2 == pytest.approx(-0.5)


def test_lookahead_picks_distance_along_straight_path() -> None:
    path = [(0.0, 0.0), (2.0, 0.0)]
    p = lookahead_point((0.0, 0.0), path, lookahead=0.5)
    assert p == pytest.approx((0.5, 0.0))


def test_lookahead_advances_across_segments() -> None:
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    p = lookahead_point((0.0, 0.0), path, lookahead=1.5)
    assert p[0] == pytest.approx(1.0)
    assert p[1] == pytest.approx(0.5)


def test_lookahead_clamps_to_endpoint_when_path_too_short() -> None:
    path = [(0.0, 0.0), (1.0, 0.0)]
    p = lookahead_point((0.0, 0.0), path, lookahead=10.0)
    assert p == pytest.approx((1.0, 0.0))


def test_path_progress_monotone_along_segment() -> None:
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    seg = [2.0, 2.0]
    assert path_progress((0.0, 0.0), path, seg) == pytest.approx(0.0)
    assert path_progress((1.0, 0.0), path, seg) == pytest.approx(0.25)
    assert path_progress((2.0, 1.0), path, seg) == pytest.approx(0.75)
    assert path_progress((2.0, 2.0), path, seg) == pytest.approx(1.0)


def test_controller_zero_heading_error_drives_at_base_speed() -> None:
    ctrl = PurePursuitController(lookahead_distance=0.5, base_speed=0.4, v_min=0.05)
    state = UnicycleState(x=0.0, y=0.0, theta=0.0)
    path = [(0.0, 0.0), (5.0, 0.0)]
    v, omega, he, xte = ctrl.step(state, path)
    assert he == pytest.approx(0.0, abs=1e-9)
    assert omega == pytest.approx(0.0, abs=1e-9)
    assert v == pytest.approx(0.4)
    assert xte == pytest.approx(0.0, abs=1e-9)


def test_controller_heading_error_produces_omega_with_correct_sign() -> None:
    ctrl = PurePursuitController(lookahead_distance=0.5, kp_heading=2.0, base_speed=0.4)
    path = [(0.0, 0.0), (0.0, 5.0)]
    _, omega, he, _ = ctrl.step(UnicycleState(theta=0.0), path)
    assert he == pytest.approx(math.pi / 2)
    assert omega > 0.0


def test_controller_slows_inside_goal_radius() -> None:
    ctrl = PurePursuitController(
        lookahead_distance=0.5, base_speed=0.4, v_min=0.05, slow_down_radius=1.0
    )
    state = UnicycleState(x=0.0, y=0.0, theta=0.0)
    v_close, _, _, _ = ctrl.step(state, [(0.0, 0.0), (0.5, 0.0)])
    v_far, _, _, _ = ctrl.step(state, [(0.0, 0.0), (5.0, 0.0)])
    assert v_close < v_far


def test_controller_empty_path_returns_zero_command() -> None:
    ctrl = PurePursuitController()
    v, omega, he, xte = ctrl.step(UnicycleState(), [])
    assert (v, omega, he, xte) == (0.0, 0.0, 0.0, 0.0)
