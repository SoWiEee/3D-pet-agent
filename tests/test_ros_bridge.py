"""Tests for the Nav2 bridge — spec §14.5 Stage A.

All offline (no live ROS graph): a NavigationGoal must round-trip to a
frame-correct goal pose, and a synthetic ``/cmd_vel`` stream must integrate to
the expected world trajectory (CCW command ⇒ CCW yaw).
"""

from __future__ import annotations

import math

import pytest

from src.control.kinematic import UnicycleState
from src.planning.schema import NavigationGoal
from src.research.ros_bridge import (
    Nav2Bridge,
    Nav2BridgeConfig,
    RecordingTransport,
    integrate_twist,
    navigation_goal_to_pose_stamped,
    quaternion_to_yaw,
    unicycle_to_world,
    yaw_to_quaternion,
)


def _goal(pos=(1.0, 0.0, 2.0)) -> NavigationGoal:
    return NavigationGoal(
        goal_id="g1",
        target_position_world=pos,
        source_command="go to the cup",
        explanation="test",
    )


# ── coordinate handshake ──────────────────────────────────────────────────


def test_quaternion_yaw_round_trips():
    for yaw in (-2.0, -0.3, 0.0, 0.5, 1.57, 3.0):
        assert quaternion_to_yaw(yaw_to_quaternion(yaw)) == pytest.approx(yaw, abs=1e-9)


def test_goal_maps_graphics_ground_to_ros_plane():
    # graphics (world_x, world_z) → ROS (x, y); ground z dropped to 0.
    pose = navigation_goal_to_pose_stamped(_goal((1.0, 5.0, 2.0)), frame_id="map")
    assert pose["header"]["frame_id"] == "map"
    assert pose["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 0.0}


def test_goal_orientation_encodes_yaw():
    pose = navigation_goal_to_pose_stamped(_goal(), yaw=math.pi / 2)
    assert quaternion_to_yaw(pose["pose"]["orientation"]) == pytest.approx(math.pi / 2)


def test_goal_without_position_raises():
    g = NavigationGoal(goal_id="g", source_command="x", explanation="y")
    with pytest.raises(ValueError, match="target_position_world"):
        navigation_goal_to_pose_stamped(g)


# ── twist integration ─────────────────────────────────────────────────────


def test_forward_twist_moves_along_heading():
    s = integrate_twist(UnicycleState(), (0.5, 0.0), dt=1.0, cfg=Nav2BridgeConfig())
    assert s.x == pytest.approx(0.5)  # heading 0 ⇒ +X (0.5 < v_max so unclamped)
    assert s.y == pytest.approx(0.0)
    assert s.theta == pytest.approx(0.0)


def test_positive_omega_is_ccw_yaw():
    # CCW command (ω>0) must produce positive yaw — sign preserved into ROS.
    s = integrate_twist(UnicycleState(), (0.0, 1.0), dt=1.0, cfg=Nav2BridgeConfig())
    assert s.theta == pytest.approx(1.0)
    assert s.theta > 0


def test_reverse_allowed_by_config():
    fwd = Nav2BridgeConfig(allow_reverse=True)
    s = integrate_twist(UnicycleState(), (-0.5, 0.0), dt=1.0, cfg=fwd)
    assert s.x == pytest.approx(-0.5)


def test_reverse_clamped_when_disallowed():
    nofwd = Nav2BridgeConfig(allow_reverse=False)
    s = integrate_twist(UnicycleState(), (-0.5, 0.0), dt=1.0, cfg=nofwd)
    assert s.x == pytest.approx(0.0)


def test_twist_clamped_to_limits():
    cfg = Nav2BridgeConfig(v_max=0.8, omega_max=1.5)
    s = integrate_twist(UnicycleState(), (10.0, 10.0), dt=1.0, cfg=cfg)
    assert s.x == pytest.approx(0.8)
    assert s.theta == pytest.approx(math.atan2(math.sin(1.5), math.cos(1.5)))


def test_negative_dt_rejected():
    with pytest.raises(ValueError, match="dt"):
        integrate_twist(UnicycleState(), (1.0, 0.0), dt=-0.1, cfg=Nav2BridgeConfig())


def test_unicycle_to_world_puts_kinematic_y_on_world_z():
    assert unicycle_to_world(UnicycleState(x=1.0, y=2.0, theta=0.5)) == (1.0, 0.0, 2.0)


# ── bridge wiring ──────────────────────────────────────────────────────────


def test_send_goal_publishes_through_transport():
    t = RecordingTransport()
    bridge = Nav2Bridge(t)
    bridge.send_goal(_goal())
    assert len(t.published_goals) == 1
    assert t.published_goals[0]["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 0.0}


def test_cmd_vel_stream_integrates_and_emits_pose():
    poses: list[tuple[float, float, float]] = []
    t = RecordingTransport()
    # Fixed-dt clock so the trajectory is deterministic.
    clock = iter([0.0, 1.0, 2.0]).__next__
    bridge = Nav2Bridge(t, on_pose=poses.append, clock=clock)

    t.feed_cmd_vel((0.5, 0.0))  # first msg: dt=0 (establishes t0), no motion
    t.feed_cmd_vel((0.5, 0.0))  # dt=1 forward 0.5m along +X

    assert poses[-1] == pytest.approx((0.5, 0.0, 0.0))
    assert bridge.state.x == pytest.approx(0.5)


def test_reset_clears_state():
    t = RecordingTransport()
    bridge = Nav2Bridge(t)
    t.feed_cmd_vel((1.0, 1.0))
    bridge.reset()
    assert bridge.state == UnicycleState()
