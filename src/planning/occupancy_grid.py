"""2D occupancy grid over the world XZ plane (spec §10.1).

Why XZ?  The cat moves on a flat surface (``y = 0``); planning collapses to
the floor plane. World convention is graphics: X right, Y up, Z toward the
viewer (camera looks down −Z). So a grid cell at ``(gx, gz)`` maps to world
``(origin_x + gx·res, 0, origin_z + gz·res)``.

A cell is *blocked* when an obstacle (any tracked ObjectState3D) overlaps it,
inflated by ``obstacle_padding`` plus any per-constraint halo. Inflation is a
dilation on the rasterised obstacles — cheap, deterministic, no scipy
required.

This module is build-once-per-frame. Phase 7's planner pairs it with the A*
search; Phase 9 (exploration) will read the same grid to score viewpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..spatial.object_lifter import ObjectState3D
from ..spatial.semantic_map import SemanticMap
from .schema import NavigationConstraint

log = logging.getLogger("pet_agent.occupancy_grid")


@dataclass
class GridConfig:
    """Tunables, mirroring ``configs/navigation.yaml::grid``."""

    resolution: float = 0.05
    origin_x: float = -3.0
    origin_z: float = -4.0
    width: int = 120  # cells along +X
    height: int = 120  # cells along +Z
    obstacle_padding: float = 0.15


@dataclass
class OccupancyGrid:
    """A binary occupancy snapshot of the world XZ plane.

    The data array is ``(height, width)`` row-major, so ``data[gz, gx] == 1``
    means the cell ``(gx, gz)`` is blocked. Match the matplotlib convention so
    debug dumps look right out of the box.
    """

    cfg: GridConfig
    data: np.ndarray  # uint8, shape (height, width)
    obstacle_ids: list[str] = field(default_factory=list)

    # ── world ↔ grid math ─────────────────────────────────────────────────
    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        gx = int(round((x - self.cfg.origin_x) / self.cfg.resolution))
        gz = int(round((z - self.cfg.origin_z) / self.cfg.resolution))
        return gx, gz

    def cell_to_world(self, gx: int, gz: int) -> tuple[float, float]:
        x = self.cfg.origin_x + gx * self.cfg.resolution
        z = self.cfg.origin_z + gz * self.cfg.resolution
        return x, z

    def in_bounds(self, gx: int, gz: int) -> bool:
        return 0 <= gx < self.cfg.width and 0 <= gz < self.cfg.height

    def is_free(self, gx: int, gz: int) -> bool:
        return self.in_bounds(gx, gz) and self.data[gz, gx] == 0

    def is_blocked(self, gx: int, gz: int) -> bool:
        return not self.is_free(gx, gz)

    def nearest_free(self, gx: int, gz: int, *, max_radius_cells: int) -> tuple[int, int] | None:
        """BFS-style ring search for the nearest free cell.

        Returns ``None`` if the whole disc is blocked or out of bounds. Used
        when the goal cell itself is inside an obstacle (spec §10.2).
        """
        if self.is_free(gx, gz):
            return (gx, gz)
        for r in range(1, max_radius_cells + 1):
            best: tuple[int, int] | None = None
            best_d2 = None
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dz)) != r:
                        continue  # only the ring at radius r
                    cx, cz = gx + dx, gz + dz
                    if not self.is_free(cx, cz):
                        continue
                    d2 = dx * dx + dz * dz
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = (cx, cz)
            if best is not None:
                return best
        return None

    def to_dict(self) -> dict:
        """Stable JSON shape for debug overlays (Phase 10 eval can diff these)."""
        return {
            "resolution": self.cfg.resolution,
            "origin": [self.cfg.origin_x, self.cfg.origin_z],
            "width": self.cfg.width,
            "height": self.cfg.height,
            "obstacle_ids": list(self.obstacle_ids),
            # Flattened row-major; eval/debug code can reshape with (h, w).
            "data": self.data.flatten().tolist(),
        }


# ── building ──────────────────────────────────────────────────────────────
def build_occupancy_grid(
    semantic_map: SemanticMap,
    *,
    cfg: GridConfig | None = None,
    constraints: list[NavigationConstraint] | None = None,
    exclude_object_ids: set[str] | None = None,
    avoid_default_min_distance: float = 0.25,
) -> OccupancyGrid:
    """Rasterise the SemanticMap into a binary grid + apply constraint halos.

    ``constraints`` lets the resolver bias the grid per-command (avoid the
    mouse, keep distance from the cup). ``avoid_default_min_distance`` is the
    fallback halo when an ``avoid_object`` constraint omits ``min_distance``.
    ``exclude_object_ids`` skips the listed objects entirely — used by the
    planner to leave the *target* object's footprint clear so the goal pose
    can land right next to it without being trapped in its own halo.
    """
    cfg = cfg or GridConfig()
    grid = np.zeros((cfg.height, cfg.width), dtype=np.uint8)
    obstacle_ids: list[str] = []
    skip: set[str] = exclude_object_ids or set()

    objects = semantic_map.values()
    # Per-object inflation (metres). avoid_object adds on top of the default.
    extra_inflation: dict[str, float] = {}
    for c in constraints or []:
        if c.type == "avoid_object" and c.object_id:
            halo = c.min_distance if c.min_distance is not None else avoid_default_min_distance
            extra_inflation[c.object_id] = max(extra_inflation.get(c.object_id, 0.0), halo)

    for obj in objects:
        if obj.object_id in skip:
            continue
        # Only treat *visible* objects as obstacles; stale/lost would freeze
        # the planner around a phantom long after the user moved the cup.
        if obj.tracking_status in {"stale", "lost"}:
            continue
        inflation = cfg.obstacle_padding + extra_inflation.get(obj.object_id, 0.0)
        if _rasterise_obstacle(grid, obj, cfg, inflation=inflation):
            obstacle_ids.append(obj.object_id)

    log.debug(
        "occupancy grid: %dx%d, %d obstacles, %d blocked cells",
        cfg.width,
        cfg.height,
        len(obstacle_ids),
        int(grid.sum()),
    )
    return OccupancyGrid(cfg=cfg, data=grid, obstacle_ids=obstacle_ids)


def _rasterise_obstacle(
    grid: np.ndarray, obj: ObjectState3D, cfg: GridConfig, *, inflation: float
) -> bool:
    """Mark all cells overlapping ``obj``'s XZ footprint (inflated). Returns
    True if any cell was touched."""
    cx, _, cz = obj.center_3d_world
    ex, _, ez = obj.extent_3d
    half_x = max(0.05, ex * 0.5) + inflation
    half_z = max(0.05, ez * 0.5) + inflation
    res = cfg.resolution

    gx_min = max(0, int(np.floor((cx - half_x - cfg.origin_x) / res)))
    gx_max = min(cfg.width - 1, int(np.ceil((cx + half_x - cfg.origin_x) / res)))
    gz_min = max(0, int(np.floor((cz - half_z - cfg.origin_z) / res)))
    gz_max = min(cfg.height - 1, int(np.ceil((cz + half_z - cfg.origin_z) / res)))

    if gx_min > gx_max or gz_min > gz_max:
        return False  # entirely outside the grid
    grid[gz_min : gz_max + 1, gx_min : gx_max + 1] = 1
    return True
