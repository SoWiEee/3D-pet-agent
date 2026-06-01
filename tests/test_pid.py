"""Phase 8 — PID controller unit tests."""

from __future__ import annotations

import pytest

from src.control.pid import PIDController, PIDState


def test_p_only_response_equals_kp_times_error() -> None:
    pid = PIDController(kp=2.0)
    out, _ = pid.step(PIDState(), error=0.5, dt=0.1)
    assert out == pytest.approx(1.0)


def test_first_tick_has_zero_derivative_kick() -> None:
    pid = PIDController(kp=0.0, kd=10.0)
    out, _ = pid.step(PIDState(), error=1.0, dt=0.1)
    assert out == pytest.approx(0.0)


def test_derivative_kicks_in_second_tick() -> None:
    pid = PIDController(kp=0.0, kd=1.0)
    _, s1 = pid.step(PIDState(), error=1.0, dt=0.1)
    out, _ = pid.step(s1, error=2.0, dt=0.1)
    assert out == pytest.approx(10.0)


def test_integral_accumulates_with_error_x_dt() -> None:
    pid = PIDController(kp=0.0, ki=1.0)
    state = PIDState()
    for _ in range(5):
        _, state = pid.step(state, error=0.5, dt=0.1)
    assert state.integral == pytest.approx(0.25)


def test_anti_windup_clamps_integral() -> None:
    pid = PIDController(kp=0.0, ki=1.0, integral_clamp=0.1)
    state = PIDState()
    for _ in range(100):
        _, state = pid.step(state, error=1.0, dt=0.1)
    assert state.integral == pytest.approx(0.1)


def test_reset_returns_zero_state() -> None:
    pid = PIDController(kp=1.0)
    s = PIDState(integral=5.0, prev_error=2.0)
    assert pid.reset() == PIDState()
    assert s.integral == 5.0


def test_negative_dt_rejected() -> None:
    pid = PIDController(kp=1.0)
    with pytest.raises(ValueError):
        pid.step(PIDState(), error=0.1, dt=-0.01)


def test_state_immutability() -> None:
    pid = PIDController(kp=1.0, ki=1.0)
    s0 = PIDState()
    _, s1 = pid.step(s0, error=0.5, dt=0.1)
    assert s0.integral == 0.0
    assert s1.integral == pytest.approx(0.05)
