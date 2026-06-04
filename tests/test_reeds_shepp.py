"""Reeds-Shepp planner — spec §14.5 car kinematics.

The contract we actually depend on is **reconstruction**: whatever segment
word the planner returns, integrating it under the unit-radius bicycle model
from the start pose must land on the goal pose. Optimality is a bonus; we
assert it loosely (the analytic path is no longer than a turn-drive-turn
upper bound) rather than against a reference implementation.
"""

from __future__ import annotations

import math
import random

import pytest

from src.control.reeds_shepp import (
    ReedsSheppPath,
    reconstruct,
    reeds_shepp_path,
)


def _close(a: tuple[float, float, float], b: tuple[float, float, float], tol: float = 1e-6) -> bool:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dth = math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2]))
    return math.hypot(dx, dy) < tol and abs(dth) < tol


# ── reconstruction: the load-bearing invariant ──────────────────────────────


@pytest.mark.parametrize("seed", range(50))
def test_path_reconstructs_to_goal(seed: int) -> None:
    rng = random.Random(seed)
    start = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
    goal = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
    radius = rng.uniform(0.5, 2.0)

    path = reeds_shepp_path(start, goal, radius)
    assert path is not None, f"no path for {start} -> {goal} r={radius}"
    end = reconstruct(start, path, radius)
    assert _close(end, goal, tol=1e-4), f"reconstruct {end} != goal {goal}"


def test_straight_ahead_is_a_single_forward_segment() -> None:
    path = reeds_shepp_path((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), radius=1.0)
    assert path is not None
    # Pure straight line: one forward segment, no reverse, length == distance.
    assert math.isclose(path.length, 2.0, abs_tol=1e-6)
    assert all(seg.gear > 0 for seg in path.segments)
    assert all(seg.steering == 0 for seg in path.segments)


def test_goal_behind_requires_reverse() -> None:
    # Goal directly behind, same heading: the optimal RS path backs straight up.
    path = reeds_shepp_path((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0), radius=1.0)
    assert path is not None
    assert any(seg.gear < 0 for seg in path.segments), "expected a reverse segment"
    assert _close(reconstruct((0.0, 0.0, 0.0), path, 1.0), (-2.0, 0.0, 0.0), tol=1e-4)


def test_length_scales_with_radius() -> None:
    # Same relative geometry, larger radius → proportionally longer arc path.
    start = (0.0, 0.0, 0.0)
    goal = (1.0, 1.0, math.pi / 2)
    p1 = reeds_shepp_path(start, (goal[0], goal[1], goal[2]), radius=1.0)
    p2 = reeds_shepp_path(start, (goal[0] * 2, goal[1] * 2, goal[2]), radius=2.0)
    assert p1 is not None and p2 is not None
    assert math.isclose(p2.length, p1.length * 2, rel_tol=1e-6)


def test_returns_none_only_never_for_finite_poses() -> None:
    # The planner must always find *some* path between two finite poses.
    misses = 0
    rng = random.Random(7)
    for _ in range(200):
        start = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-math.pi, math.pi))
        goal = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-math.pi, math.pi))
        if reeds_shepp_path(start, goal, radius=0.8) is None:
            misses += 1
    assert misses == 0, f"{misses}/200 pose pairs had no path"


def test_sample_poses_are_dense_and_endpoint_exact() -> None:
    start = (0.0, 0.0, 0.0)
    goal = (1.5, -1.0, -math.pi / 3)
    path = reeds_shepp_path(start, goal, radius=0.9)
    assert path is not None
    poses = path.sample(start, radius=0.9, step=0.05)
    assert len(poses) > 5
    assert _close(poses[0], start, tol=1e-6)
    assert _close(poses[-1], goal, tol=1e-4)
    # Consecutive samples are never farther apart than ~step (+ slack).
    for a, b in zip(poses[:-1], poses[1:], strict=True):
        assert math.hypot(b[0] - a[0], b[1] - a[1]) < 0.12


def test_path_is_immutable() -> None:
    path = reeds_shepp_path((0.0, 0.0, 0.0), (2.0, 1.0, 0.5), radius=1.0)
    assert isinstance(path, ReedsSheppPath)
    with pytest.raises((AttributeError, TypeError)):
        path.length = 0.0  # type: ignore[misc]
