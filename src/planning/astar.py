"""Grid A* with structured failure (spec §10.1).

Heuristic: Euclidean (admissible for 8-connectivity grids with diagonal cost
``√2``). Goal-cell unblocking is the caller's job — :func:`astar` reports
``start_blocked`` / ``goal_unreachable`` so the orchestrator can decide
whether to relocate the goal via :meth:`OccupancyGrid.nearest_free` and retry.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Literal

from .occupancy_grid import OccupancyGrid

Cell = tuple[int, int]
FailureReason = Literal["no_path", "goal_unreachable", "start_blocked"]


@dataclass
class AStarResult:
    """Discriminated A* outcome. ``path`` is in grid cells, start → goal inclusive."""

    success: bool
    path: list[Cell]
    cost: float
    expanded: int
    failure: FailureReason | None = None


_DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_DIAGS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_SQRT2 = math.sqrt(2.0)


def _euclid(a: Cell, b: Cell) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def astar(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    *,
    connectivity: int = 8,
) -> AStarResult:
    """Plan a grid path from ``start`` to ``goal`` over ``grid.data``.

    Returns :class:`AStarResult`; on failure ``path`` is empty and ``failure``
    is set. Diagonal moves are forbidden when either orthogonal neighbour is
    blocked — this avoids cutting through obstacle corners.
    """
    if not grid.in_bounds(*start) or grid.is_blocked(*start):
        return AStarResult(False, [], 0.0, 0, failure="start_blocked")
    if not grid.in_bounds(*goal) or grid.is_blocked(*goal):
        return AStarResult(False, [], 0.0, 0, failure="goal_unreachable")
    if start == goal:
        return AStarResult(True, [start], 0.0, 0)

    dirs = list(_DIRS_4)
    if connectivity == 8:
        dirs = dirs + _DIAGS

    # f, counter, cell  — counter breaks ties deterministically.
    counter = 0
    open_heap: list[tuple[float, int, Cell]] = [(0.0, counter, start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            path = _reconstruct(came_from, current)
            return AStarResult(True, path, g_score[current], expanded)
        expanded += 1
        cx, cz = current
        for dx, dz in dirs:
            nx, nz = cx + dx, cz + dz
            if not grid.is_free(nx, nz):
                continue
            # Forbid corner-cutting when moving diagonally.
            if (
                dx != 0
                and dz != 0
                and (grid.is_blocked(cx + dx, cz) or grid.is_blocked(cx, cz + dz))
            ):
                continue
            step = _SQRT2 if (dx != 0 and dz != 0) else 1.0
            tentative = g_score[current] + step
            neighbor = (nx, nz)
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            f = tentative + _euclid(neighbor, goal)
            counter += 1
            heapq.heappush(open_heap, (f, counter, neighbor))

    return AStarResult(False, [], 0.0, expanded, failure="no_path")


def _reconstruct(came_from: dict[Cell, Cell], end: Cell) -> list[Cell]:
    path = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


# ── path smoothing helpers ─────────────────────────────────────────────────
def line_of_sight(grid: OccupancyGrid, a: Cell, b: Cell) -> bool:
    """Bresenham line check — return True if every cell from ``a`` to ``b``
    is free. Used by LOS pruning in :func:`smooth_path`."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if not grid.is_free(x, y):
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def smooth_path(grid: OccupancyGrid, path: list[Cell]) -> list[Cell]:
    """Greedy LOS pruning: keep node ``i`` if cell ``i-1`` cannot see ``i+1``.

    Equivalent to the classic "string-pulling" simplification used after grid
    A*. Endpoints are always preserved.
    """
    if len(path) <= 2:
        return list(path)
    out: list[Cell] = [path[0]]
    anchor = 0
    i = 1
    while i < len(path) - 1:
        if line_of_sight(grid, path[anchor], path[i + 1]):
            i += 1
            continue
        out.append(path[i])
        anchor = i
        i += 1
    out.append(path[-1])
    return out
