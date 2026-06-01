"""Phase 8 — unicycle kinematic model unit tests."""

from __future__ import annotations

import math

import pytest

from src.control.kinematic import UnicycleState, kinematic_step


def test_step_returns_new_state_without_mutating_input() -> None:
    s0 = UnicycleState(x=1.0, y=2.0, theta=0.0)
    s1 = kinematic_step(s0, v=0.5, omega=0.0, dt=0.1, v_max=1.0, omega_max=1.0)
    assert s0 == UnicycleState(x=1.0, y=2.0, theta=0.0)
    assert s1 is not s0
    assert s1.x == pytest.approx(1.05)
    assert s1.y == pytest.approx(2.0)
    assert s1.theta == pytest.approx(0.0)
    assert s1.t == pytest.approx(0.1)


def test_zero_dt_is_noop() -> None:
    s0 = UnicycleState(x=0.0, y=0.0, theta=0.5)
    s1 = kinematic_step(s0, v=1.0, omega=1.0, dt=0.0, v_max=1.0, omega_max=1.0)
    assert (s1.x, s1.y, s1.theta) == (s0.x, s0.y, s0.theta)


def test_negative_dt_rejected() -> None:
    with pytest.raises(ValueError):
        kinematic_step(UnicycleState(), v=0.0, omega=0.0, dt=-0.1, v_max=1.0, omega_max=1.0)


def test_v_max_clamp_prevents_overspeed() -> None:
    s = kinematic_step(UnicycleState(), v=10.0, omega=0.0, dt=1.0, v_max=0.5, omega_max=1.0)
    assert s.x == pytest.approx(0.5)


def test_omega_max_clamp_prevents_oversteer() -> None:
    s = kinematic_step(UnicycleState(), v=0.0, omega=10.0, dt=1.0, v_max=1.0, omega_max=0.4)
    assert s.theta == pytest.approx(0.4)


def test_negative_v_clamped_to_zero() -> None:
    s = kinematic_step(UnicycleState(x=1.0), v=-2.0, omega=0.0, dt=1.0, v_max=1.0, omega_max=1.0)
    assert s.x == pytest.approx(1.0)


def test_theta_normalised_to_unit_circle() -> None:
    s = kinematic_step(
        UnicycleState(theta=3.0), v=0.0, omega=1.0, dt=2.0, v_max=1.0, omega_max=10.0
    )
    assert -math.pi < s.theta <= math.pi


def test_heading_pi_over_2_moves_along_y() -> None:
    s = kinematic_step(
        UnicycleState(theta=math.pi / 2),
        v=1.0,
        omega=0.0,
        dt=0.5,
        v_max=1.0,
        omega_max=1.0,
    )
    assert s.x == pytest.approx(0.0, abs=1e-9)
    assert s.y == pytest.approx(0.5)


def test_below_v_min_nonzero_clamps_to_zero() -> None:
    s = kinematic_step(
        UnicycleState(), v=0.01, omega=0.0, dt=1.0, v_max=1.0, omega_max=1.0, v_min_nonzero=0.05
    )
    assert s.x == pytest.approx(0.0)
