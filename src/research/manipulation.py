"""Manipulation planning — Stage C of the mobile-manipulator track (spec §14.5).

Adds the arm half of the mobile manipulator. The navigation spine answers
"where does the base go"; this answers "how does the arm pick the object up".
It mirrors the navigation pipeline one-for-one:

* :class:`GraspGoal` is the manipulation analogue of ``NavigationGoal`` — an
  *explainable* grasp target synthesised from a known object pose;
* :class:`ManipulationAction` is the analogue of ``PetAction`` — a single
  arm primitive (``reach`` / ``grasp`` / ``lift`` / ``place`` / ``retract``)
  consumed by a backend;
* :class:`Manipulator` is the analogue of ``Planner`` — it turns a
  SemanticMap object into a feasibility-checked pick-and-place sequence.

The first target (spec §14.5 Stage C) is pick/place at a **known** object pose
from the SemanticMap, so grasp synthesis here is a deterministic top-down
heuristic — no learned grasp net yet (that is Stage D). A real MoveIt2 backend
(collision-aware IK + gripper control) drops in behind
:class:`ManipulationBackend`; :class:`RecordingBackend` drives the tests
without ROS / MoveIt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from ..spatial.object_lifter import ObjectState3D

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (x, y, z, w)
ActionKind = Literal["reach", "grasp", "lift", "place", "retract"]
GripperState = Literal["open", "closed"]


class Pose(BaseModel):
    """A 6-DoF pose in the graphics-world frame."""

    position: Vec3
    orientation: Quat = (0.0, 0.0, 0.0, 1.0)


class GraspGoal(BaseModel):
    """Spec §14.5 — the explainable grasp target (manipulation analogue of
    ``NavigationGoal``)."""

    grasp_id: str
    target_object_id: str
    grasp_pose_world: Pose
    approach_vector_world: Vec3  # unit vector the gripper travels along to the object
    gripper_width: float
    confidence: float
    explanation: str


class ManipulationAction(BaseModel):
    """Spec §14.5 — one arm primitive (manipulation analogue of ``PetAction``)."""

    action: ActionKind
    target_pose_world: Pose
    gripper: GripperState = "open"
    speed: float = 0.3


@dataclass(frozen=True)
class ArmConfig:
    """Workspace + motion tunables. In a real deployment these mirror the URDF
    / MoveIt config; kept inline to keep Stage C self-contained."""

    base_position_world: Vec3 = (0.0, 0.5, 0.0)  # arm shoulder, ~0.5 m up on the base
    reach_min: float = 0.10
    reach_max: float = 0.80
    standoff: float = 0.12  # pre-grasp / retract clearance along the approach
    lift_height: float = 0.15
    max_gripper_width: float = 0.12
    grip_clearance: float = 0.02  # opening margin around the object


@dataclass
class ManipulationPlan:
    """Feasibility-checked output (analogue of ``PlannerResult``)."""

    grasp_goal: GraspGoal
    actions: list[ManipulationAction]
    feasible: bool
    explanation: str


# ── geometry helpers ────────────────────────────────────────────────────────
def is_reachable(point: Vec3, arm: ArmConfig) -> bool:
    """True if ``point`` lies inside the arm's spherical reach shell."""
    base = np.asarray(arm.base_position_world)
    d = float(np.linalg.norm(np.asarray(point) - base))
    return arm.reach_min <= d <= arm.reach_max


def _grasp_orientation(approach: np.ndarray, closing: np.ndarray) -> Quat:
    """Build a grasp quaternion: tool +Z = approach direction, tool +X =
    gripper closing direction, +Y completes a right-handed frame."""
    z = approach / (np.linalg.norm(approach) or 1.0)
    x = closing - np.dot(closing, z) * z  # orthogonalise against approach
    nx = np.linalg.norm(x)
    # Degenerate closing ∥ approach → pick any perpendicular axis.
    x = x / nx if nx > 1e-6 else _any_perpendicular(z)
    y = np.cross(z, x)
    r = np.column_stack([x, y, z])
    q = Rotation.from_matrix(r).as_quat()  # (x, y, z, w)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _any_perpendicular(v: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    p = np.cross(v, ref)
    return p / (np.linalg.norm(p) or 1.0)


# ── grasp synthesis (Stage C: known pose, top-down heuristic) ───────────────
def top_down_grasp_goal(obj: ObjectState3D, arm: ArmConfig) -> GraspGoal:
    """Synthesise a top-down grasp for a known object.

    Grasp at the object centre, descending from above (approach ``-Y``). The
    gripper closes along the object's shorter horizontal axis. Confidence
    folds in detection quality, reachability, and gripper fit so the
    orchestrator can refuse a guess rather than crash an unreachable arm.
    """
    center = np.asarray(obj.center_3d_world, dtype=float)
    ex, _ey, ez = obj.extent_3d
    approach = np.array([0.0, -1.0, 0.0])  # descend from above

    # Close along the shorter horizontal axis (easier, more stable grasp).
    closing = np.array([1.0, 0.0, 0.0]) if ex <= ez else np.array([0.0, 0.0, 1.0])
    min_horiz = min(ex, ez)
    fits = min_horiz <= arm.max_gripper_width
    gripper_width = min(min_horiz + arm.grip_clearance, arm.max_gripper_width)

    reachable = is_reachable(tuple(center), arm)
    fit_factor = 1.0 if fits else 0.2
    reach_factor = 1.0 if reachable else 0.0
    confidence = float(obj.confidence.overall) * fit_factor * reach_factor

    reasons = []
    if not reachable:
        reasons.append("out of arm reach")
    if not fits:
        reasons.append(f"object {min_horiz:.2f} m wider than gripper {arm.max_gripper_width:.2f} m")
    explanation = (
        f"top-down grasp of {obj.class_label} at {tuple(round(c, 2) for c in center)}; "
        + ("; ".join(reasons) if reasons else "reachable and within gripper width")
    )

    return GraspGoal(
        grasp_id=f"grasp_{obj.object_id}",
        target_object_id=obj.object_id,
        grasp_pose_world=Pose(
            position=(float(center[0]), float(center[1]), float(center[2])),
            orientation=_grasp_orientation(approach, closing),
        ),
        approach_vector_world=(0.0, -1.0, 0.0),
        gripper_width=gripper_width,
        confidence=confidence,
        explanation=explanation,
    )


# ── pick-and-place sequencing ───────────────────────────────────────────────
def plan_pick_and_place(
    grasp: GraspGoal,
    place_position_world: Vec3,
    arm: ArmConfig,
) -> list[ManipulationAction]:
    """Sequence the arm primitives for a top-down pick then place.

    reach (pre-grasp, open) → grasp (close) → lift → reach over place
    (closed) → place (open) → retract (open).
    """
    orient = grasp.grasp_pose_world.orientation
    grasp_pos = np.asarray(grasp.grasp_pose_world.position, dtype=float)
    approach = np.asarray(grasp.approach_vector_world, dtype=float)
    up = -approach  # retreat/lift direction (opposite the descent)

    pre_grasp = grasp_pos + up * arm.standoff
    lifted = grasp_pos + up * arm.lift_height
    place = np.asarray(place_position_world, dtype=float)
    over_place = place + up * arm.lift_height
    retract = place + up * arm.standoff

    def pose(p: np.ndarray) -> Pose:
        return Pose(position=(float(p[0]), float(p[1]), float(p[2])), orientation=orient)

    return [
        ManipulationAction(action="reach", target_pose_world=pose(pre_grasp), gripper="open"),
        ManipulationAction(action="grasp", target_pose_world=pose(grasp_pos), gripper="closed"),
        ManipulationAction(action="lift", target_pose_world=pose(lifted), gripper="closed"),
        ManipulationAction(action="reach", target_pose_world=pose(over_place), gripper="closed"),
        ManipulationAction(action="place", target_pose_world=pose(place), gripper="open"),
        ManipulationAction(action="retract", target_pose_world=pose(retract), gripper="open"),
    ]


# ── backend + orchestrator ──────────────────────────────────────────────────
class ManipulationBackend(Protocol):
    """The surface a real arm controller exposes. MoveIt2 and a test double
    both satisfy it."""

    def execute(self, action: ManipulationAction) -> None: ...


@dataclass
class RecordingBackend:
    """In-memory :class:`ManipulationBackend` for tests — records executed
    actions in order."""

    executed: list[ManipulationAction] = field(default_factory=list)

    def execute(self, action: ManipulationAction) -> None:
        self.executed.append(action)


class Manipulator:
    """Ties grasp synthesis + sequencing + a backend (analogue of ``Planner``
    feeding the controller)."""

    def __init__(
        self,
        backend: ManipulationBackend,
        *,
        arm: ArmConfig | None = None,
        min_confidence: float = 0.2,
    ) -> None:
        self.backend = backend
        self.arm = arm or ArmConfig()
        self.min_confidence = min_confidence

    def plan(self, obj: ObjectState3D, place_position_world: Vec3) -> ManipulationPlan:
        """Synthesise + sequence without executing. ``feasible`` is False (and
        ``actions`` empty) when confidence is below threshold."""
        goal = top_down_grasp_goal(obj, self.arm)
        if goal.confidence < self.min_confidence:
            return ManipulationPlan(
                grasp_goal=goal,
                actions=[],
                feasible=False,
                explanation=f"grasp rejected ({goal.confidence:.2f} < {self.min_confidence}): "
                + goal.explanation,
            )
        actions = plan_pick_and_place(goal, place_position_world, self.arm)
        return ManipulationPlan(
            grasp_goal=goal, actions=actions, feasible=True, explanation=goal.explanation
        )

    def pick_and_place(self, obj: ObjectState3D, place_position_world: Vec3) -> ManipulationPlan:
        """Plan and, if feasible, execute the sequence on the backend."""
        plan = self.plan(obj, place_position_world)
        if plan.feasible:
            for action in plan.actions:
                self.backend.execute(action)
        return plan


class MoveItBackend:  # pragma: no cover - requires ROS 2 + MoveIt2
    """Live :class:`ManipulationBackend` backed by MoveIt2. Imported lazily so
    the package has no hard ROS / MoveIt dependency; only on real hardware."""

    def __init__(self, group_name: str = "arm") -> None:
        from moveit.planning import MoveItPy  # type: ignore

        self._moveit = MoveItPy(node_name="pet_manipulator")
        self._arm = self._moveit.get_planning_component(group_name)

    def execute(self, action: ManipulationAction) -> None:
        # Translate the world-frame pose into a MoveIt goal, plan, and execute.
        raise NotImplementedError("wire up MoveItPy goal/plan/execute on hardware")
