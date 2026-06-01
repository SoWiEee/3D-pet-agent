"""Classic PID with anti-windup (spec §11.2 — speed smoothing).

Both controller and state are immutable. Each ``step`` returns ``(output,
new_state)`` so a path-follower can keep PID state alongside its main
unicycle state without surprise side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PIDState:
    integral: float = 0.0
    prev_error: float | None = None


@dataclass(frozen=True)
class PIDController:
    """Gains + anti-windup clamp. Stateless — pair with :class:`PIDState`."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_clamp: float = 1.0  # |integral| ≤ clamp → bounded windup

    def step(self, state: PIDState, error: float, dt: float) -> tuple[float, PIDState]:
        """Advance one PID tick.

        ``dt`` may be 0 (gives a P-only response). The derivative uses
        ``(error - prev_error) / dt`` when both are defined; otherwise it
        contributes nothing — this matches the standard "first-tick
        derivative is zero" convention and prevents a spurious kick.
        """
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        integral = state.integral + error * dt
        if integral > self.integral_clamp:
            integral = self.integral_clamp
        elif integral < -self.integral_clamp:
            integral = -self.integral_clamp
        derivative = 0.0
        if state.prev_error is not None and dt > 0.0:
            derivative = (error - state.prev_error) / dt
        output = self.kp * error + self.ki * integral + self.kd * derivative
        return output, replace(state, integral=integral, prev_error=error)

    def reset(self) -> PIDState:
        return PIDState()
