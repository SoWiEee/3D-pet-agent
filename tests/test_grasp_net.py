"""Tests for point-cloud grasp synthesis — spec §14.5 Stage D.

All offline: PCA-driven analytic sampler over synthetic clouds. The thinnest
axis must become the closing direction, the approach must prefer top-down, and
candidates must be ranked + gripper-fit-filtered. Output GraspGoals feed the
Stage-C pick-and-place sequencer unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.research.grasp_net import (
    AnalyticGraspSampler,
    GraspSamplerConfig,
    box_point_cloud,
    cylinder_point_cloud,
    points_from_depth,
)
from src.research.manipulation import ArmConfig, GraspGoal, plan_pick_and_place

ARM = ArmConfig()


def _sampler(**cfg) -> AnalyticGraspSampler:
    return AnalyticGraspSampler(arm=ARM, cfg=GraspSamplerConfig(**cfg))


# ── basic synthesis ─────────────────────────────────────────────────────────


def test_returns_ranked_grasps_for_graspable_box():
    # 0.04 m thin (fits the 0.12 m gripper), long along Z.
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.04, 0.06, 0.20), seed=1)
    grasps = _sampler(top_k=5).synthesize(cloud, "obj_box")
    assert grasps
    assert all(isinstance(g, GraspGoal) for g in grasps)
    # Confidence is the ranking score: non-increasing.
    confs = [g.confidence for g in grasps]
    assert confs == sorted(confs, reverse=True)


def test_closing_axis_is_thinnest_dimension():
    # Thin along X (0.03), so the gripper width should track ~0.03 + clearance.
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.03, 0.10, 0.18), seed=2)
    g = _sampler().synthesize(cloud, "obj")[0]
    assert g.gripper_width == pytest.approx(0.03 + ARM.grip_clearance, abs=0.01)


def test_approach_prefers_top_down():
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.04, 0.06, 0.20), seed=3)
    g = _sampler().synthesize(cloud, "obj")[0]
    # Approach vector should have a downward (−Y) component.
    assert g.approach_vector_world[1] < 0


def test_grasp_orientation_aligns_tool_z_to_approach():
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.04, 0.06, 0.20), seed=4)
    g = _sampler().synthesize(cloud, "obj")[0]
    tool_z = Rotation.from_quat(g.grasp_pose_world.orientation).apply([0, 0, 1])
    assert tool_z == pytest.approx(g.approach_vector_world, abs=1e-6)


# ── fit filtering ───────────────────────────────────────────────────────────


def test_oversized_object_yields_no_grasp():
    # Every axis exceeds the 0.12 m gripper — no orientation can close on it.
    # (A box thin on *any* one axis is graspable by that axis; 6-DoF finds it.)
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.30, 0.30, 0.30), seed=5)
    assert _sampler().synthesize(cloud, "obj") == []


def test_top_k_caps_candidate_count():
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.04, 0.06, 0.30), seed=6)
    grasps = _sampler(top_k=3, n_positions=11).synthesize(cloud, "obj")
    assert len(grasps) <= 3


def test_too_few_points_returns_empty():
    assert _sampler().synthesize(np.zeros((2, 3)), "obj") == []


# ── geometry: cylinder + determinism ────────────────────────────────────────


def test_cylinder_grasp_width_tracks_diameter():
    # Upright cylinder (axis=y): thin axes are X/Z (diameter 0.08).
    cloud = cylinder_point_cloud((0.3, 0.4, 0.0), radius=0.04, height=0.20, axis="y", seed=7)
    g = _sampler().synthesize(cloud, "cup")[0]
    assert g.gripper_width <= ARM.max_gripper_width
    assert g.gripper_width == pytest.approx(0.08 + ARM.grip_clearance, abs=0.03)


def test_synthesis_is_deterministic_for_fixed_cloud():
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.04, 0.06, 0.20), seed=8)
    a = _sampler().synthesize(cloud, "obj")
    b = _sampler().synthesize(cloud, "obj")
    assert [g.grasp_pose_world.position for g in a] == [g.grasp_pose_world.position for g in b]


# ── integration with Stage C ────────────────────────────────────────────────


def test_synthesized_grasp_feeds_pick_and_place():
    cloud = box_point_cloud((0.3, 0.4, 0.0), (0.04, 0.06, 0.20), seed=9)
    g = _sampler().synthesize(cloud, "obj")[0]
    actions = plan_pick_and_place(g, place_position_world=(0.2, 0.4, 0.3), arm=ARM)
    assert [a.action for a in actions] == ["reach", "grasp", "lift", "reach", "place", "retract"]


# ── depth back-projection helper ────────────────────────────────────────────


def test_points_from_depth_back_projects_masked_pixels():
    depth = np.full((4, 4), 2.0, dtype=float)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True  # 4 pixels
    pts = points_from_depth(depth, mask, fx=100.0, fy=100.0, cx=2.0, cy=2.0)
    assert pts.shape == (4, 3)
    assert np.allclose(pts[:, 2], 2.0)  # all at depth 2 m
