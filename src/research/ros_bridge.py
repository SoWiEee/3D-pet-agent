"""ROS 2 Nav2 bridge — Stage A of the mobile-manipulator track (spec §14.5 / §14.4).

Re-targets the virtual-pet navigation output onto a real differential-drive
base. Two directions:

* **outbound** — a :class:`~planning.schema.NavigationGoal` becomes a
  ``geometry_msgs/PoseStamped``-shaped goal published on ``/goal_pose`` for
  Nav2 to plan toward;
* **inbound** — the ``/cmd_vel`` ``Twist`` stream Nav2 emits is integrated
  back into a graphics-world pose the renderer / pet runtime can animate.

All ROS ↔ graphics conversion lives here so no other module sees a ROS type.
The transport is behind the :class:`RosTransport` protocol: tests use
:class:`RecordingTransport` (no ROS needed); :class:`RclpyTransport`
lazy-imports ``rclpy`` for a live graph.

Coordinate handshake (spec §14.5): the mainline ground plane is graphics-world
``(x, y_kin) = (world_x, world_z)`` with ``θ`` CCW from +X toward +Z. ROS is
REP-103 (X forward, Y left, Z up). We map the ground plane by planar identity
``(rx, ry) = (world_x, world_z)``, ``yaw = θ``, ``z = 0`` — rotation-preserving,
so ``ω > 0`` stays ``angular.z > 0``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..control.kinematic import UnicycleState
from ..planning.schema import NavigationGoal

# A Twist reduced to the two diff-drive DOFs: body-forward speed and yaw rate.
Twist2D = tuple[float, float]  # (linear_x, angular_z)
# A PoseStamped reduced to a plain JSON-serialisable dict (no rclpy types).
PoseStampedDict = dict[str, Any]


@dataclass(frozen=True)
class Nav2BridgeConfig:
    """Limits + framing for the bridge. In a real deployment these would move
    to ``configs/ros.yaml``; kept inline to keep Stage A self-contained."""

    frame_id: str = "map"
    v_max: float = 0.8
    omega_max: float = 1.5
    allow_reverse: bool = True


def yaw_to_quaternion(yaw: float) -> dict[str, float]:
    """Yaw (rotation about REP-103 +Z) → unit quaternion ``{x, y, z, w}``."""
    half = 0.5 * yaw
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def quaternion_to_yaw(q: dict[str, float]) -> float:
    """Inverse of :func:`yaw_to_quaternion` (assumes a pure-yaw quaternion)."""
    return 2.0 * math.atan2(q["z"], q["w"])


def navigation_goal_to_pose_stamped(
    goal: NavigationGoal,
    *,
    frame_id: str = "map",
    yaw: float = 0.0,
    stamp: float | None = None,
) -> PoseStampedDict:
    """Convert a graphics-world :class:`NavigationGoal` into a
    ``geometry_msgs/PoseStamped``-shaped dict in the ROS ground frame.

    Only ``goal_type="pose"`` carries a concrete ``target_position_world``;
    other goal types must be resolved to a position upstream first.
    """
    pos = goal.target_position_world
    if pos is None:
        raise ValueError(
            f"goal {goal.goal_id!r} has no target_position_world; resolve it before bridging"
        )
    world_x, _world_y, world_z = pos
    return {
        "header": {"frame_id": frame_id, "stamp": stamp if stamp is not None else time.time()},
        "pose": {
            # Planar identity: graphics (world_x, world_z) → ROS (x, y); ground z=0.
            "position": {"x": world_x, "y": world_z, "z": 0.0},
            "orientation": yaw_to_quaternion(yaw),
        },
    }


def integrate_twist(
    state: UnicycleState,
    twist: Twist2D,
    dt: float,
    cfg: Nav2BridgeConfig,
) -> UnicycleState:
    """Integrate one ``/cmd_vel`` Twist into a new unicycle pose.

    Mirrors :func:`control.kinematic.kinematic_step` but, unlike the cat model,
    permits reverse (``allow_reverse``) since a real base can back up. ``v`` is
    the body-frame forward speed (``linear.x``); ``ω`` the yaw rate
    (``angular.z``). Heading and yaw share the CCW convention, so signs pass
    through unchanged.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    v, omega = twist
    v_lo = -cfg.v_max if cfg.allow_reverse else 0.0
    v = max(v_lo, min(v, cfg.v_max))
    omega = max(-cfg.omega_max, min(omega, cfg.omega_max))
    new_theta = state.theta + omega * dt
    new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
    new_x = state.x + v * math.cos(state.theta) * dt
    new_y = state.y + v * math.sin(state.theta) * dt
    return UnicycleState(x=new_x, y=new_y, theta=new_theta, t=state.t + dt)


def unicycle_to_world(state: UnicycleState) -> tuple[float, float, float]:
    """Ground-plane unicycle pose → graphics-world position ``(x, y_up, z)``
    (kinematic ``y`` is world Z; ground sits at world Y = 0)."""
    return (state.x, 0.0, state.y)


class RosTransport(Protocol):
    """The minimal publish/subscribe surface the bridge needs. A real
    ``rclpy`` node and a test double both satisfy it."""

    def publish_goal(self, pose: PoseStampedDict) -> None: ...

    def set_cmd_vel_callback(self, cb: Callable[[Twist2D], None]) -> None: ...


@dataclass
class RecordingTransport:
    """In-memory :class:`RosTransport` for tests — records published goals and
    lets the test inject ``/cmd_vel`` messages via :meth:`feed_cmd_vel`."""

    published_goals: list[PoseStampedDict] = field(default_factory=list)
    _cb: Callable[[Twist2D], None] | None = None

    def publish_goal(self, pose: PoseStampedDict) -> None:
        self.published_goals.append(pose)

    def set_cmd_vel_callback(self, cb: Callable[[Twist2D], None]) -> None:
        self._cb = cb

    def feed_cmd_vel(self, twist: Twist2D) -> None:
        if self._cb is not None:
            self._cb(twist)


class Nav2Bridge:
    """Ties the navigation output to a :class:`RosTransport`.

    :meth:`send_goal` publishes a :class:`NavigationGoal`; the integrated pose
    from ``/cmd_vel`` is exposed via :attr:`state` and pushed to an optional
    ``on_pose`` callback (e.g. the pet runtime / websocket broadcaster).
    """

    def __init__(
        self,
        transport: RosTransport,
        *,
        config: Nav2BridgeConfig | None = None,
        on_pose: Callable[[tuple[float, float, float]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport
        self.cfg = config or Nav2BridgeConfig()
        self._on_pose = on_pose
        self._clock = clock
        self.state = UnicycleState()
        self._last_t: float | None = None
        transport.set_cmd_vel_callback(self._on_cmd_vel)

    def send_goal(self, goal: NavigationGoal, *, yaw: float = 0.0) -> PoseStampedDict:
        pose = navigation_goal_to_pose_stamped(goal, frame_id=self.cfg.frame_id, yaw=yaw)
        self.transport.publish_goal(pose)
        return pose

    def _on_cmd_vel(self, twist: Twist2D) -> None:
        now = self._clock()
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        self.state = integrate_twist(self.state, twist, dt, self.cfg)
        if self._on_pose is not None:
            self._on_pose(unicycle_to_world(self.state))

    def reset(self, state: UnicycleState | None = None) -> None:
        self.state = state or UnicycleState()
        self._last_t = None


class RclpyTransport:  # pragma: no cover - requires a live ROS 2 graph
    """Live :class:`RosTransport` backed by ``rclpy``. Imported lazily so the
    package has no hard ROS dependency; only constructed on real hardware."""

    def __init__(self, node_name: str = "pet_nav2_bridge") -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped, Twist
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._PoseStamped = PoseStamped
        self._node: Node = rclpy.create_node(node_name)
        self._goal_pub = self._node.create_publisher(PoseStamped, "/goal_pose", 10)
        self._cb: Callable[[Twist2D], None] | None = None
        self._node.create_subscription(Twist, "/cmd_vel", self._forward_twist, 10)

    def publish_goal(self, pose: PoseStampedDict) -> None:
        msg = self._PoseStamped()
        msg.header.frame_id = pose["header"]["frame_id"]
        p, o = pose["pose"]["position"], pose["pose"]["orientation"]
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p["x"], p["y"], p["z"]
        msg.pose.orientation.x = o["x"]
        msg.pose.orientation.y = o["y"]
        msg.pose.orientation.z = o["z"]
        msg.pose.orientation.w = o["w"]
        self._goal_pub.publish(msg)

    def set_cmd_vel_callback(self, cb: Callable[[Twist2D], None]) -> None:
        self._cb = cb

    def _forward_twist(self, msg: Any) -> None:
        if self._cb is not None:
            self._cb((msg.linear.x, msg.angular.z))
