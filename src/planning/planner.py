"""Planner orchestrator (spec §10.2).

Takes a :class:`NavigationGoal` + current :class:`SemanticMap` and returns a
3D path in graphics-world coordinates. Internals:

    1. Rasterise the SemanticMap into an :class:`OccupancyGrid` with
       constraint halos applied (avoid_object inflates that specific object).
    2. Project ``start`` (cat current pose, snap to grid) and ``goal``
       (target_position_world XZ) to grid cells.
    3. If the goal cell is blocked, replace it with the nearest free cell
       within ``nearest_free_radius``.
    4. Run :func:`astar`.
    5. Smooth the resulting path with LOS pruning.
    6. Convert grid cells back to world XYZ (Y from the goal pose so a hide
       behind-a-tall-cup keeps the hint).

The planner is pure — no global state, no I/O — so tests can spin one up
with a SemanticMap and a hand-crafted goal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from ..spatial.semantic_map import SemanticMap
from .astar import AStarResult, astar, smooth_path
from .occupancy_grid import GridConfig, OccupancyGrid, build_occupancy_grid
from .schema import NavigationGoal

log = logging.getLogger("pet_agent.planner")

PlannerStatus = Literal["success", "no_path", "goal_unreachable", "start_blocked", "no_goal"]


@dataclass
class PlannerResult:
    """What the planner hands back to the controller / server."""

    status: PlannerStatus
    path_world: list[tuple[float, float, float]]
    grid: OccupancyGrid | None = None
    astar: AStarResult | None = None
    explanation: str = ""


@dataclass
class PlannerConfig:
    """Mirrors ``configs/navigation.yaml::planner`` plus a grid handle."""

    grid: GridConfig
    connectivity: int = 8
    nearest_free_radius_m: float = 1.0
    smoothing: Literal["line_of_sight", "none"] = "line_of_sight"
    smoothing_subdivisions: int = 0
    avoid_default_min_distance: float = 0.25


class Planner:
    """Stateless planner; one instance reused across requests."""

    def __init__(self, cfg: PlannerConfig | None = None) -> None:
        self.cfg = cfg or PlannerConfig(grid=GridConfig())

    def plan(
        self,
        goal: NavigationGoal,
        semantic_map: SemanticMap,
        *,
        start_world: tuple[float, float, float],
    ) -> PlannerResult:
        # 1. No-target goal types (stop / explore / report) can't be planned
        # against a specific cell — return an empty success so the server
        # falls through to its own handling (sit, look_at the camera, etc.).
        if goal.target_position_world is None:
            return PlannerResult(
                status="no_goal",
                path_world=[],
                explanation="Goal has no world target — caller handles directly.",
            )

        # Don't let the target object's own footprint block the goal cell —
        # otherwise "go to the cup" plans to the cup's halo edge, not into
        # the standoff position the resolver picked.
        excluded: set[str] = {goal.target_object_id} if goal.target_object_id else set()
        grid = build_occupancy_grid(
            semantic_map,
            cfg=self.cfg.grid,
            constraints=goal.constraints,
            exclude_object_ids=excluded,
            avoid_default_min_distance=self.cfg.avoid_default_min_distance,
        )

        sx, _, sz = start_world
        gx, _, gz = goal.target_position_world
        start_cell = grid.world_to_cell(sx, sz)
        goal_cell = grid.world_to_cell(gx, gz)

        # 2. Try to recover blocked start by nudging out of the obstacle.
        if grid.is_blocked(*start_cell):
            relocated = grid.nearest_free(
                *start_cell,
                max_radius_cells=int(self.cfg.nearest_free_radius_m / grid.cfg.resolution),
            )
            if relocated is None:
                return PlannerResult(
                    status="start_blocked",
                    path_world=[],
                    grid=grid,
                    explanation="Cat is inside an obstacle and no free cell nearby.",
                )
            start_cell = relocated

        # 3. Relocate blocked goal.
        if grid.is_blocked(*goal_cell):
            relocated = grid.nearest_free(
                *goal_cell,
                max_radius_cells=int(self.cfg.nearest_free_radius_m / grid.cfg.resolution),
            )
            if relocated is None:
                return PlannerResult(
                    status="goal_unreachable",
                    path_world=[],
                    grid=grid,
                    explanation="Goal cell is in an obstacle and surroundings are blocked.",
                )
            goal_cell = relocated

        # 4. Run A*.
        result = astar(grid, start_cell, goal_cell, connectivity=self.cfg.connectivity)
        if not result.success:
            return PlannerResult(
                status=result.failure or "no_path",
                path_world=[],
                grid=grid,
                astar=result,
                explanation=f"A* failed: {result.failure}.",
            )

        # 5. Smooth + lift to world.
        cells = (
            smooth_path(grid, result.path)
            if self.cfg.smoothing == "line_of_sight"
            else list(result.path)
        )
        path_world = self._cells_to_world(cells, grid, y_hint=goal.target_position_world[1])

        log.info(
            "planner: %d cells → %d waypoints; expanded=%d",
            len(result.path),
            len(path_world),
            result.expanded,
        )
        return PlannerResult(
            status="success",
            path_world=path_world,
            grid=grid,
            astar=result,
            explanation=(
                f"Planned {len(path_world)} waypoints (expanded {result.expanded} cells)."
            ),
        )

    # ── helpers ─────────────────────────────────────────────────────────────
    def _cells_to_world(
        self, cells: list[tuple[int, int]], grid: OccupancyGrid, *, y_hint: float
    ) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for gx, gz in cells:
            x, z = grid.cell_to_world(gx, gz)
            out.append((x, 0.0, z))
        # Drop the very first waypoint if it's effectively the start — keeps
        # the move_follow_path tween from teleporting back a few cm.
        if len(out) >= 2:
            dx = out[1][0] - out[0][0]
            dz = out[1][2] - out[0][2]
            if (dx * dx + dz * dz) < (grid.cfg.resolution * 0.5) ** 2:
                out = out[1:]
        _ = y_hint  # height is a renderer concern; the floor plane wins here
        return out
