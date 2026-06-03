"""Phase 7 — A* + LOS smoothing tests."""

from __future__ import annotations

import numpy as np

from src.planning import (
    GridConfig,
    OccupancyGrid,
    astar,
    count_path_collisions,
    smooth_path,
)
from src.planning.astar import iter_line_cells


def _empty_grid(w: int = 20, h: int = 20) -> OccupancyGrid:
    cfg = GridConfig(resolution=0.1, origin_x=0.0, origin_z=0.0, width=w, height=h)
    return OccupancyGrid(cfg=cfg, data=np.zeros((h, w), dtype=np.uint8))


def _grid_with_wall(*, gap_at: int | None = None) -> OccupancyGrid:
    grid = _empty_grid()
    # Vertical wall at x=10 spanning the whole height; optional 1-cell gap.
    grid.data[:, 10] = 1
    if gap_at is not None:
        grid.data[gap_at, 10] = 0
    return grid


def test_straight_path_in_empty_grid() -> None:
    grid = _empty_grid()
    res = astar(grid, (0, 0), (10, 0))
    assert res.success
    assert res.path[0] == (0, 0)
    assert res.path[-1] == (10, 0)


def test_diagonal_uses_8_connectivity() -> None:
    grid = _empty_grid()
    res = astar(grid, (0, 0), (5, 5), connectivity=8)
    assert res.success
    # 8-conn cost: 5 * sqrt(2) ≈ 7.07.
    assert abs(res.cost - 5 * 2**0.5) < 1e-6


def test_4_connectivity_costs_more() -> None:
    grid = _empty_grid()
    res = astar(grid, (0, 0), (5, 5), connectivity=4)
    assert res.success
    assert res.cost == 10.0  # forced to L-shape


def test_path_goes_around_wall_through_gap() -> None:
    grid = _grid_with_wall(gap_at=10)
    res = astar(grid, (0, 10), (19, 10))
    assert res.success
    assert (10, 10) in res.path  # used the gap


def test_no_path_returns_structured_failure() -> None:
    grid = _grid_with_wall(gap_at=None)
    res = astar(grid, (0, 10), (19, 10))
    assert not res.success
    assert res.failure == "no_path"
    assert res.path == []


def test_start_blocked_failure() -> None:
    grid = _empty_grid()
    grid.data[5, 5] = 1
    res = astar(grid, (5, 5), (10, 10))
    assert res.failure == "start_blocked"


def test_goal_unreachable_failure() -> None:
    grid = _empty_grid()
    grid.data[10, 10] = 1
    res = astar(grid, (0, 0), (10, 10))
    assert res.failure == "goal_unreachable"


def test_corner_cutting_is_disallowed() -> None:
    grid = _empty_grid()
    # Block both orthogonal neighbours of a diagonal step — A* must detour.
    grid.data[0, 1] = 1
    grid.data[1, 0] = 1
    res = astar(grid, (0, 0), (1, 1))
    # No corner-cut allowed → no path through this 2x2 with both ortho cells blocked.
    assert not res.success


def test_smooth_path_prunes_collinear_nodes() -> None:
    grid = _empty_grid()
    raw = [(i, 0) for i in range(11)]
    smoothed = smooth_path(grid, raw)
    assert smoothed[0] == (0, 0)
    assert smoothed[-1] == (10, 0)
    # LOS should collapse the straight line down to its endpoints.
    assert len(smoothed) == 2


def test_smooth_path_respects_obstacle_blocking_los() -> None:
    grid = _empty_grid()
    grid.data[5, 5] = 1  # block the direct diagonal
    raw = [(0, 0), (5, 0), (10, 10)]
    smoothed = smooth_path(grid, raw)
    # The middle waypoint stays because LOS from start → end is broken.
    assert (5, 0) in smoothed


# ── A4: segment-wise collision counting ──────────────────────────────────────
def test_iter_line_cells_includes_endpoints_and_interior() -> None:
    cells = list(iter_line_cells((0, 0), (3, 0)))
    assert cells == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_count_path_collisions_catches_mid_segment_obstacle() -> None:
    # Two free endpoints with an obstacle dead-centre between them. Endpoint
    # sampling would report 0; the Bresenham walk must report 1.
    grid = _empty_grid()
    grid.data[0, 5] = 1  # blocked cell at (gx=5, gz=0), mid-way on the segment
    path = [(0, 0), (10, 0)]
    assert grid.is_free(*path[0]) and grid.is_free(*path[1])
    assert count_path_collisions(grid, path) == 1


def test_count_path_collisions_counts_shared_turn_cell_once() -> None:
    grid = _empty_grid()
    grid.data[5, 5] = 1  # the corner cell shared by both segments
    path = [(5, 0), (5, 5), (10, 5)]  # bends through the blocked corner
    assert count_path_collisions(grid, path) == 1


def test_count_path_collisions_zero_on_clear_path() -> None:
    grid = _empty_grid()
    assert count_path_collisions(grid, [(0, 0), (9, 9), (0, 9)]) == 0


def test_count_path_collisions_single_and_empty() -> None:
    grid = _empty_grid()
    grid.data[3, 3] = 1
    assert count_path_collisions(grid, []) == 0
    assert count_path_collisions(grid, [(3, 3)]) == 1
    assert count_path_collisions(grid, [(0, 0)]) == 0
