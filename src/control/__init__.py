"""Phase 8 — Pure-pursuit controller + path following.

Public surface:

- :class:`UnicycleState`           planar (x, y, θ) immutable state
- :func:`kinematic_step`           one Euler integration step under (v, ω)
- :class:`PIDController` / :class:`PIDState`   classic anti-windup PID smoother
- :class:`PurePursuitController`   lookahead-based path follower
- :class:`PathFollower`            offline simulation orchestrator
- :class:`ControlStep` / :class:`ControlTrace`   bounded execution log
"""

from __future__ import annotations

from .kinematic import UnicycleState, kinematic_step
from .path_follower import ControlSummary, ControlTrace, PathFollower
from .pid import PIDController, PIDState
from .pure_pursuit import ControlStep, PurePursuitController

__all__ = [
    "ControlStep",
    "ControlSummary",
    "ControlTrace",
    "PIDController",
    "PIDState",
    "PathFollower",
    "PurePursuitController",
    "UnicycleState",
    "kinematic_step",
]
