"""Offline path-follower simulator (spec §11 acceptance criteria).

The backend can't actuate a robot, so we simulate the controller at
``configs/control.yaml::kinematic.dt`` resolution and emit a dense
trajectory the renderer can tween. This same harness is what the
acceptance tests exercise: feed in a planned path + initial state, get
back a :class:`ControlTrace` with goal-stop semantics, cross-track error,
and a summary suitable for ``EvaluationRecord.controller_metrics``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

from .kinematic import UnicycleState, kinematic_step
from .pid import PIDController, PIDState
from .pure_pursuit import ControlStep, PurePursuitController, path_progress

log = logging.getLogger("pet_agent.control")

Point2 = tuple[float, float]
SimStatus = Literal["success", "max_steps", "empty_path"]


@dataclass(frozen=True)
class ControlSummary:
    """Reduced metrics suitable for :class:`EvaluationRecord` (spec §3.8)."""

    status: SimStatus
    steps: int
    duration_s: float
    final_distance_to_goal: float
    max_cross_track_error: float
    max_heading_error: float
    mean_speed: float


@dataclass(frozen=True)
class ControlTrace:
    """Densified controller output. ``path_world`` is what the renderer needs;
    ``steps`` are kept around for `/control/last_trace` and for tests."""

    path_world: list[tuple[float, float, float]]
    steps: list[ControlStep]
    summary: ControlSummary
    final_state: UnicycleState


@dataclass
class PathFollower:
    """Compose unicycle kinematics + pure-pursuit + PID speed smoothing.

    The class itself is intentionally lightweight — most behaviour lives in
    the (pure) ``PurePursuitController`` and ``kinematic_step``. The
    follower's job is just to drive the loop, stamp logs, and decide when
    to stop.
    """

    controller: PurePursuitController
    pid: PIDController | None = None
    v_max: float = 0.80
    omega_max: float = 3.20
    v_min_nonzero: float = 0.02
    dt: float = 0.05
    max_steps: int = 400
    goal_tolerance: float = 0.08
    _path_y: float = field(default=0.0, init=False)

    def simulate(
        self,
        path_world: list[tuple[float, float, float]],
        initial: UnicycleState,
    ) -> ControlTrace:
        if not path_world:
            return ControlTrace(
                path_world=[],
                steps=[],
                summary=ControlSummary(
                    status="empty_path",
                    steps=0,
                    duration_s=0.0,
                    final_distance_to_goal=float("nan"),
                    max_cross_track_error=0.0,
                    max_heading_error=0.0,
                    mean_speed=0.0,
                ),
                final_state=initial,
            )
        # Renderer Y is fixed by the floor plane; we carry it through so
        # ``move_follow_path`` waypoints have the same Y as the planner gave us.
        self._path_y = path_world[0][1]
        path_xz: list[Point2] = [(p[0], p[2]) for p in path_world]
        # Ensure the path begins at the cat's actual position so cross-track
        # error is defined for the whole simulation. A planner that already
        # anchors on the cat will produce a 0-length first segment, which
        # closest_path_index handles correctly.
        if math.hypot(path_xz[0][0] - initial.x, path_xz[0][1] - initial.y) > 1e-6:
            path_xz = [(initial.x, initial.y), *path_xz]

        pid_state: PIDState = self.pid.reset() if self.pid else PIDState()
        state = initial
        steps: list[ControlStep] = []
        max_xte = 0.0
        max_he = 0.0
        total_v = 0.0
        goal = path_xz[-1]

        for _ in range(self.max_steps):
            v_cmd, omega, heading_err, cross_track = self.controller.step(state, path_xz)
            if self.pid is not None:
                error = v_cmd - (steps[-1].v if steps else 0.0)
                delta, pid_state = self.pid.step(pid_state, error, self.dt)
                # Use delta as a soft correction to the raw command (delta in m/s)
                v_cmd = max(0.0, min(self.v_max, v_cmd + delta))

            dist_to_goal = math.hypot(goal[0] - state.x, goal[1] - state.y)
            progress = path_progress(state.as_xz(), path_xz, _segment_lengths(path_xz))
            steps.append(
                ControlStep(
                    t=state.t,
                    x=state.x,
                    y=state.y,
                    theta=state.theta,
                    v=v_cmd,
                    omega=omega,
                    heading_error=heading_err,
                    cross_track_error=cross_track,
                    path_progress=progress,
                    distance_to_goal=dist_to_goal,
                )
            )
            max_xte = max(max_xte, abs(cross_track))
            max_he = max(max_he, abs(heading_err))
            total_v += v_cmd

            if dist_to_goal <= self.goal_tolerance:
                status: SimStatus = "success"
                break

            state = kinematic_step(
                state,
                v_cmd,
                omega,
                self.dt,
                v_max=self.v_max,
                omega_max=self.omega_max,
                v_min_nonzero=self.v_min_nonzero,
            )
        else:
            status = "max_steps"

        densified = [(s.x, self._path_y, s.y) for s in steps]
        # Always anchor on the actual goal so the renderer's last tween lands
        # the cat exactly on the planner-chosen pose.
        densified.append((goal[0], self._path_y, goal[1]))
        summary = ControlSummary(
            status=status,
            steps=len(steps),
            duration_s=(steps[-1].t - steps[0].t) if steps else 0.0,
            final_distance_to_goal=steps[-1].distance_to_goal if steps else float("nan"),
            max_cross_track_error=max_xte,
            max_heading_error=max_he,
            mean_speed=(total_v / len(steps)) if steps else 0.0,
        )
        log.info(
            "follower: status=%s steps=%d xte_max=%.3f he_max=%.3f final_d=%.3f",
            summary.status,
            summary.steps,
            summary.max_cross_track_error,
            summary.max_heading_error,
            summary.final_distance_to_goal,
        )
        return ControlTrace(
            path_world=densified,
            steps=steps,
            summary=summary,
            final_state=state,
        )


def _segment_lengths(path: list[Point2]) -> list[float]:
    out: list[float] = []
    for a, b in zip(path[:-1], path[1:], strict=True):
        out.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    return out
