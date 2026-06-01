"""Phase 7 — planner orchestrator tests.

Spec §10.4 acceptance: planner finds a path around obstacles, returns
structured failure when impossible, and clears ≥ 80% of a hand-built set of
desk planning cases.
"""

from __future__ import annotations

import pytest

from src.planning import GridConfig, Planner, PlannerConfig
from src.planning.schema import NavigationConstraint, NavigationGoal
from src.spatial import SemanticMap
from src.spatial.object_lifter import ObjectConfidence, ObjectState3D


def _obj(
    *,
    object_id: str,
    center: tuple[float, float, float],
    extent: tuple[float, float, float] = (0.15, 0.15, 0.15),
    label: str = "cup",
    tracking_status: str = "tracked",
) -> ObjectState3D:
    return ObjectState3D(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=(100, 100, 200, 200),
        mask_path=None,
        center_2d=(150, 150),
        coordinate_frame="world",
        center_3d_world=center,
        extent_3d=extent,
        median_depth=2.0,
        depth_uncertainty=0.1,
        confidence=ObjectConfidence(
            detector=0.8, mask_quality=0.7, depth_quality=0.7, tracking=1.0, overall=0.8
        ),
        last_seen_frame=0,
        tracking_status=tracking_status,  # type: ignore[arg-type]
    )


def _planner() -> Planner:
    cfg = PlannerConfig(
        grid=GridConfig(
            resolution=0.1, origin_x=-3.0, origin_z=-3.0, width=60, height=60, obstacle_padding=0.1
        ),
        nearest_free_radius_m=0.6,
    )
    return Planner(cfg)


def _goal(
    *,
    target_xyz: tuple[float, float, float],
    target_id: str | None = None,
    constraints: list[NavigationConstraint] | None = None,
) -> NavigationGoal:
    return NavigationGoal(
        goal_id="g0",
        goal_type="pose",
        target_position_world=target_xyz,
        target_object_id=target_id,
        constraints=constraints or [],
        source_command="test",
        explanation="test",
    )


def test_straight_path_when_no_obstacles() -> None:
    p = _planner()
    m = SemanticMap()
    r = p.plan(_goal(target_xyz=(1.0, 0.0, 0.0)), m, start_world=(-1.0, 0.0, 0.0))
    assert r.status == "success"
    assert len(r.path_world) >= 2
    # End within one cell of the goal.
    end_x, _, end_z = r.path_world[-1]
    assert abs(end_x - 1.0) <= 0.1
    assert abs(end_z - 0.0) <= 0.1


def test_planner_detours_around_obstacle() -> None:
    p = _planner()
    m = SemanticMap()
    m.update([_obj(object_id="wall", center=(0.0, 0.0, 0.0), extent=(0.3, 0.5, 1.6))], frame_id=0)
    r = p.plan(_goal(target_xyz=(1.0, 0.0, 0.0)), m, start_world=(-1.0, 0.0, 0.0))
    assert r.status == "success"
    # Some waypoint must leave the y=0 line of the wall (z != 0 by > one cell).
    assert any(abs(z) > 0.2 for _, _, z in r.path_world)


def test_no_path_when_goal_walled_off() -> None:
    p = _planner()
    m = SemanticMap()
    # Wall extent overruns the grid width on both sides so the cat truly
    # cannot detour around it.
    m.update([_obj(object_id="wall", center=(0.0, 0.0, 0.0), extent=(10.0, 0.3, 0.4))], frame_id=0)
    r = p.plan(_goal(target_xyz=(0.0, 0.0, 1.0)), m, start_world=(0.0, 0.0, -1.0))
    assert r.status in {"no_path", "start_blocked", "goal_unreachable"}
    assert r.path_world == []


def test_blocked_goal_cell_relocates_to_nearest_free() -> None:
    """When the resolver picks a goal pose right on top of a cup, the planner
    must nudge to the nearest free cell rather than fail."""
    p = _planner()
    m = SemanticMap()
    m.update([_obj(object_id="cup", center=(1.0, 0.0, 0.0))], frame_id=0)
    # Goal sits in the cup's inflated halo; planner relocates.
    r = p.plan(
        _goal(target_xyz=(1.0, 0.0, 0.0), target_id="other"),  # different id so we don't carve
        m,
        start_world=(-1.0, 0.0, 0.0),
    )
    assert r.status == "success"


def test_target_object_footprint_is_carved() -> None:
    """If the goal sits on top of *its own* target object, we still succeed."""
    p = _planner()
    m = SemanticMap()
    m.update([_obj(object_id="cup", center=(1.0, 0.0, 0.0))], frame_id=0)
    r = p.plan(
        _goal(target_xyz=(1.0, 0.0, 0.0), target_id="cup"),
        m,
        start_world=(-1.0, 0.0, 0.0),
    )
    assert r.status == "success"


def test_no_goal_goal_type_returns_no_goal_status() -> None:
    p = _planner()
    m = SemanticMap()
    g = NavigationGoal(
        goal_id="g",
        goal_type="region",
        target_position_world=None,
        source_command="explore",
        explanation="-",
    )
    r = p.plan(g, m, start_world=(0.0, 0.0, 0.0))
    assert r.status == "no_goal"
    assert r.path_world == []


def test_avoid_constraint_inflates_obstacle() -> None:
    """A path that would normally clip near a mouse should detour further."""
    p = _planner()
    m = SemanticMap()
    m.update(
        [_obj(object_id="mouse", center=(0.0, 0.0, 0.0), extent=(0.1, 0.1, 0.1))],
        frame_id=0,
    )
    # Without avoid, planner happily goes near mouse.
    plain = p.plan(_goal(target_xyz=(1.0, 0.0, 0.0)), m, start_world=(-1.0, 0.0, 0.0))
    assert plain.status == "success"
    # With avoid + big halo, path detours further from origin.
    avoid_goal = _goal(
        target_xyz=(1.0, 0.0, 0.0),
        constraints=[
            NavigationConstraint(type="avoid_object", object_id="mouse", min_distance=0.5)
        ],
    )
    avoidy = p.plan(avoid_goal, m, start_world=(-1.0, 0.0, 0.0))
    assert avoidy.status == "success"
    max_dz_plain = max(abs(z) for _, _, z in plain.path_world)
    max_dz_avoid = max(abs(z) for _, _, z in avoidy.path_world)
    assert max_dz_avoid >= max_dz_plain


# Spec §10.4 acceptance: ≥ 80% success on 20 hand-built planning cases.
_CASES = [
    # (label, start, goal, [obstacle_centers], expected_success)
    ("clear", (-1.5, 0, 0), (1.5, 0, 0), [], True),
    ("clear_diag", (-1.5, 0, -1.0), (1.5, 0, 1.0), [], True),
    ("small_obstacle", (-1.5, 0, 0), (1.5, 0, 0), [(0, 0)], True),
    ("two_obstacles", (-1.5, 0, 0), (1.5, 0, 0), [(-0.4, 0), (0.4, 0)], True),
    ("offset_obstacle", (-1.5, 0, 0), (1.5, 0, 0), [(0, 0.5)], True),
    ("dense_cluster", (-1.5, 0, -1), (1.5, 0, 1), [(0, 0), (0.3, 0.3), (-0.3, -0.3)], True),
    ("approach_from_front", (-1.5, 0, 1.0), (0, 0, 0), [(0, 0)], True),
    ("hide_behind", (-1.5, 0, 1.0), (0, 0, -0.5), [(0, 0)], True),
    ("desk_corner", (-1.5, 0, -1.5), (1.0, 0, 1.0), [(0, 0)], True),
    ("near_grid_edge", (-2.8, 0, -2.5), (2.8, 0, 2.5), [], True),
    ("multi_object_room", (-1.5, 0, 0), (1.0, 0, 0), [(-0.5, 0.5), (0.5, -0.5), (0, 1)], True),
    ("path_through_corridor", (-1.5, 0, 0), (1.5, 0, 0), [(0, 0.5), (0, -0.5)], True),
    ("close_obstacle", (-0.5, 0, 0), (0.5, 0, 0), [(0, 0.4)], True),
    ("diag_detour", (-1, 0, -1), (1, 0, 1), [(0, 0)], True),
    ("retreat", (1, 0, 0), (-1.5, 0, 0), [(0, 0)], True),
    ("approach_object_short", (0, 0, -1.5), (0, 0, 0.4), [(0, 0)], True),
    ("two_walls_passable", (-1.5, 0, 0), (1.5, 0, 0), [(0, 0.3), (0, -0.3)], True),
    ("wide_gap_clear", (-1, 0, -1), (1, 0, 1), [(0, 1.5)], True),
    # Wall spans the full grid z range with no gap → no detour possible.
    (
        "walled_off",
        (-1.5, 0, 0),
        (1.5, 0, 0),
        [
            (0, c)
            for c in [
                -2.8,
                -2.4,
                -2.0,
                -1.6,
                -1.2,
                -0.8,
                -0.4,
                0.0,
                0.4,
                0.8,
                1.2,
                1.6,
                2.0,
                2.4,
                2.8,
            ]
        ],
        False,
    ),
    ("near_goal_obstacle", (-1, 0, 0), (1.0, 0, 0), [(0.9, 0)], True),
]


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c[0])
def test_desk_planning_suite(case) -> None:
    label, start, goal_xyz, obs, expected = case
    p = _planner()
    m = SemanticMap()
    m.update(
        [
            _obj(
                object_id=f"o{i}",
                center=(ox, 0.0, oz),
                extent=(0.2, 0.2, 0.2),
            )
            for i, (ox, oz) in enumerate(obs)
        ],
        frame_id=0,
    )
    r = p.plan(_goal(target_xyz=goal_xyz), m, start_world=start)
    success = r.status == "success" and len(r.path_world) >= 1
    assert success == expected, f"{label}: status={r.status}"
