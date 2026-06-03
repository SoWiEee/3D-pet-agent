"""Metric occupancy mapping — Stage B of the mobile-manipulator track (spec §14.5).

The mainline ``OccupancyGrid`` (``planning/occupancy_grid.py``) is *semantic*:
it rasterises tracked objects from the SemanticMap. A real base also needs a
**metric** layer fused from range sensors (LiDAR / depth), independent of
object recognition, so Nav2's costmap and the planner can avoid walls and
clutter the detector never labelled.

This module implements the classic **log-odds occupancy grid** updated by
ray-casting each beam of a :class:`RangeScan`: cells the beam passes through
are evidence of *free* space, the cell it terminates on is evidence of an
*obstacle*. Ray traversal reuses ``planning.astar.iter_line_cells`` (Bresenham),
so the metric and planning layers share one line-walk. The result exports to a
binary :class:`OccupancyGrid`, slotting straight beneath the semantic layer.

Pip-only and fully testable: :func:`simulate_scan` casts synthetic beams
against any binary grid, so the mapper round-trips without a real LiDAR. A real
``sensor_msgs/LaserScan`` (Stage A's ROS bridge) drops in by filling a
:class:`RangeScan` from the message.

World ↔ grid convention matches ``OccupancyGrid``: cell ``(gx, gz)`` ↔ world
``(origin_x + gx·res, 0, origin_z + gz·res)`` on the XZ ground plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..planning.astar import iter_line_cells
from ..planning.occupancy_grid import GridConfig, OccupancyGrid

# Trinary classification returned by :meth:`MetricOccupancyMap.classify`.
UNKNOWN = -1
FREE = 0
OCCUPIED = 1


@dataclass(frozen=True)
class MetricMapConfig:
    """Grid extent (mirrors ``GridConfig``) plus the log-odds sensor model."""

    resolution: float = 0.05
    origin_x: float = -3.0
    origin_z: float = -4.0
    width: int = 120  # cells along +X
    height: int = 120  # cells along +Z
    # Inverse sensor model: log-odds added per hit / passed-through cell.
    l_occ: float = 0.85
    l_free: float = -0.40
    l_min: float = -4.0  # clamp so the map stays correctable when the world moves
    l_max: float = 4.0
    occ_threshold: float = 0.65  # prob ≥ this ⇒ OCCUPIED
    free_threshold: float = 0.35  # prob ≤ this ⇒ FREE; in between ⇒ UNKNOWN

    def to_grid_config(self) -> GridConfig:
        return GridConfig(
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_z=self.origin_z,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class RangeScan:
    """A planar range scan from a sensor at world ground pose ``(x, z, theta)``.

    ``angles`` are beam bearings **relative to the sensor heading** (rad);
    ``ranges`` the measured distance per beam (m). A non-finite range or one
    ``≥ max_range`` is a no-return beam (free space all the way out, no hit).
    """

    pose: tuple[float, float, float]
    angles: np.ndarray
    ranges: np.ndarray
    max_range: float = 5.0


class MetricOccupancyMap:
    """Log-odds occupancy accumulator over the XZ ground plane.

    Like the existing ``CoverageGrid``, this is a mutable accumulator: each
    :meth:`integrate` fuses a scan into the running ``log_odds`` field.
    """

    def __init__(self, cfg: MetricMapConfig | None = None) -> None:
        self.cfg = cfg or MetricMapConfig()
        self.log_odds = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)

    # ── world ↔ grid math (matches OccupancyGrid) ─────────────────────────
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

    # ── update ────────────────────────────────────────────────────────────
    def integrate(self, scan: RangeScan) -> None:
        """Fuse one scan: free along each beam, occupied at the hit cell."""
        sx, sz, theta = scan.pose
        origin = self.world_to_cell(sx, sz)
        for angle, rng in zip(scan.angles, scan.ranges, strict=True):
            bearing = theta + float(angle)
            hit = math.isfinite(rng) and rng < scan.max_range
            reach = float(rng) if hit else scan.max_range
            ex = sx + reach * math.cos(bearing)
            ez = sz + reach * math.sin(bearing)
            end = self.world_to_cell(ex, ez)
            self._cast(origin, end, mark_hit=hit)

    def _cast(self, origin: tuple[int, int], end: tuple[int, int], *, mark_hit: bool) -> None:
        cells = list(iter_line_cells(origin, end))
        for cell in cells[:-1]:  # everything up to the endpoint is free evidence
            self._bump(cell, self.cfg.l_free)
        last = cells[-1]
        # The endpoint is an obstacle only on a real return; a max-range beam
        # ends in open space, so its last cell is free too.
        self._bump(last, self.cfg.l_occ if mark_hit else self.cfg.l_free)

    def _bump(self, cell: tuple[int, int], delta: float) -> None:
        gx, gz = cell
        if not self.in_bounds(gx, gz):
            return
        self.log_odds[gz, gx] = float(
            np.clip(self.log_odds[gz, gx] + delta, self.cfg.l_min, self.cfg.l_max)
        )

    # ── query ─────────────────────────────────────────────────────────────
    def prob(self) -> np.ndarray:
        """Per-cell occupancy probability in ``[0, 1]`` (sigmoid of log-odds)."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def classify(self) -> np.ndarray:
        """Trinary map of ``FREE`` / ``OCCUPIED`` / ``UNKNOWN`` (int8)."""
        p = self.prob()
        out = np.full(p.shape, UNKNOWN, dtype=np.int8)
        out[p >= self.cfg.occ_threshold] = OCCUPIED
        out[p <= self.cfg.free_threshold] = FREE
        return out

    def to_occupancy_grid(self, *, unknown_is_blocked: bool = False) -> OccupancyGrid:
        """Export a binary :class:`OccupancyGrid`: ``1`` where OCCUPIED.

        ``unknown_is_blocked`` controls how never-observed cells read — default
        optimistic (free), matching Nav2's costmap default; set True for a
        conservative planner that refuses to cross unmapped space.
        """
        klass = self.classify()
        data = (klass == OCCUPIED).astype(np.uint8)
        if unknown_is_blocked:
            data[klass == UNKNOWN] = 1
        return OccupancyGrid(cfg=self.cfg.to_grid_config(), data=data)


# ── synthetic sensor (testing / offline replay) ─────────────────────────────
def simulate_scan(
    world: OccupancyGrid,
    pose: tuple[float, float, float],
    *,
    n_beams: int = 90,
    fov: float = 2 * math.pi,
    max_range: float = 5.0,
) -> RangeScan:
    """Cast ``n_beams`` against a binary ``world`` grid, returning measured
    ranges. A beam that leaves the grid or reaches ``max_range`` with no
    blocked cell is a no-return (range ``inf``)."""
    sx, sz, theta = pose
    half = fov / 2.0
    # Open arc (2π): drop the duplicate endpoint so 0 and 2π aren't both cast.
    endpoint = not math.isclose(fov, 2 * math.pi)
    angles = np.linspace(-half, half, n_beams, endpoint=endpoint)
    ranges = np.full(n_beams, np.inf, dtype=np.float64)
    origin = world.world_to_cell(sx, sz)
    for i, a in enumerate(angles):
        bearing = theta + float(a)
        far = (sx + max_range * math.cos(bearing), sz + max_range * math.sin(bearing))
        far_cell = world.world_to_cell(*far)
        for cell in iter_line_cells(origin, far_cell):
            if cell == origin:
                continue
            if not world.in_bounds(*cell):
                break  # left the map: no return
            if world.is_blocked(*cell):
                wx, wz = world.cell_to_world(*cell)
                ranges[i] = math.hypot(wx - sx, wz - sz)
                break
    return RangeScan(pose=pose, angles=angles, ranges=ranges, max_range=max_range)
