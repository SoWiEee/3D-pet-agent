"""Tests for the manipulation layer — spec §14.5 Stage C.

All offline (no ROS / MoveIt): grasp synthesis from a known object pose,
feasibility gating, and the pick-and-place action sequence, executed against a
recording backend.
"""

from __future__ import annotations

import pytest
from scipy.spatial.transform import Rotation

from src.research.manipulation import (
    ArmConfig,
    GraspGoal,
    ManipulationAction,
    Manipulator,
    Pose,
    RecordingBackend,
    is_reachable,
    plan_pick_and_place,
    top_down_grasp_goal,
)
from tests.factories import make_object

# Arm shoulder at (0, 0.5, 0); a small cup ~0.3 m away is reachable + graspable.
ARM = ArmConfig()
_GRASPABLE = (0.3, 0.4, 0.0)


def _cup(center=_GRASPABLE, extent=(0.06, 0.06, 0.06), overall=0.8):
    return make_object(center_3d_world=center, extent_3d=extent, overall=overall)


# ── reachability ────────────────────────────────────────────────────────────


def test_reachable_inside_shell():
    assert is_reachable(_GRASPABLE, ARM)


def test_unreachable_beyond_shell():
    assert not is_reachable((0.0, 0.5, 2.0), ARM)


def test_unreachable_inside_min_radius():
    assert not is_reachable((0.0, 0.5, 0.0), ARM)  # at the shoulder itself


# ── grasp synthesis ─────────────────────────────────────────────────────────


def test_top_down_grasp_descends_from_above():
    g = top_down_grasp_goal(_cup(), ARM)
    assert g.approach_vector_world == (0.0, -1.0, 0.0)
    assert g.grasp_pose_world.position == pytest.approx(_GRASPABLE)
    assert g.target_object_id == "obj_001"


def test_grasp_orientation_aligns_tool_z_to_approach():
    g = top_down_grasp_goal(_cup(), ARM)
    tool_z = Rotation.from_quat(g.grasp_pose_world.orientation).apply([0, 0, 1])
    assert tool_z == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)


def test_grasp_closes_along_shorter_axis():
    # ex < ez ⇒ close along X; tool +X aligns to world X.
    g = top_down_grasp_goal(_cup(extent=(0.04, 0.06, 0.10)), ARM)
    tool_x = Rotation.from_quat(g.grasp_pose_world.orientation).apply([1, 0, 0])
    assert abs(tool_x[0]) == pytest.approx(1.0, abs=1e-6)


def test_grasp_width_clamped_to_gripper():
    g = top_down_grasp_goal(_cup(extent=(0.06, 0.06, 0.06)), ARM)
    assert g.gripper_width == pytest.approx(0.06 + ARM.grip_clearance)


def test_unreachable_object_zero_confidence():
    g = top_down_grasp_goal(_cup(center=(0.0, 0.5, 2.0)), ARM)
    assert g.confidence == 0.0
    assert "out of arm reach" in g.explanation


def test_oversized_object_low_confidence():
    # Both horizontal extents exceed the gripper ⇒ heavy confidence penalty.
    g = top_down_grasp_goal(_cup(extent=(0.30, 0.10, 0.30)), ARM)
    assert g.confidence == pytest.approx(0.8 * 0.2)
    assert "wider than gripper" in g.explanation


# ── pick-and-place sequencing ───────────────────────────────────────────────


def test_pick_and_place_sequence_order_and_gripper():
    g = top_down_grasp_goal(_cup(), ARM)
    actions = plan_pick_and_place(g, place_position_world=(0.2, 0.4, 0.3), arm=ARM)
    assert [a.action for a in actions] == [
        "reach",
        "grasp",
        "lift",
        "reach",
        "place",
        "retract",
    ]
    assert [a.gripper for a in actions] == [
        "open",
        "closed",
        "closed",
        "closed",
        "open",
        "open",
    ]


def test_pre_grasp_is_above_grasp_and_lift_rises():
    g = top_down_grasp_goal(_cup(), ARM)
    actions = plan_pick_and_place(g, (0.2, 0.4, 0.3), ARM)
    reach, grasp, lift = actions[0], actions[1], actions[2]
    gy = grasp.target_pose_world.position[1]
    assert reach.target_pose_world.position[1] == pytest.approx(gy + ARM.standoff)
    assert lift.target_pose_world.position[1] == pytest.approx(gy + ARM.lift_height)


def test_place_action_targets_place_position():
    g = top_down_grasp_goal(_cup(), ARM)
    place = (0.2, 0.4, 0.3)
    actions = plan_pick_and_place(g, place, ARM)
    place_action = next(a for a in actions if a.action == "place")
    assert place_action.target_pose_world.position == pytest.approx(place)


# ── orchestrator ────────────────────────────────────────────────────────────


def test_manipulator_executes_feasible_plan():
    backend = RecordingBackend()
    m = Manipulator(backend, arm=ARM)
    plan = m.pick_and_place(_cup(), (0.2, 0.4, 0.3))
    assert plan.feasible
    assert len(backend.executed) == len(plan.actions) == 6


def test_manipulator_skips_infeasible_plan():
    backend = RecordingBackend()
    m = Manipulator(backend, arm=ARM)
    plan = m.pick_and_place(_cup(center=(0.0, 0.5, 2.0)), (0.2, 0.4, 0.3))
    assert not plan.feasible
    assert plan.actions == []
    assert backend.executed == []
    assert "rejected" in plan.explanation


def test_plan_does_not_touch_backend():
    backend = RecordingBackend()
    m = Manipulator(backend, arm=ARM)
    m.plan(_cup(), (0.2, 0.4, 0.3))
    assert backend.executed == []


# ── contracts ───────────────────────────────────────────────────────────────


def test_contracts_round_trip_json():
    g = top_down_grasp_goal(_cup(), ARM)
    assert GraspGoal.model_validate_json(g.model_dump_json()) == g
    action = ManipulationAction(action="grasp", target_pose_world=Pose(position=(0, 0, 0)))
    assert ManipulationAction.model_validate_json(action.model_dump_json()) == action
