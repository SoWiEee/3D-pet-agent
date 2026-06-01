"""Observed/unobserved coverage grid (spec §12.1).

A bool grid aligned with the navigation grid (same world frame, same XZ
resolution). A cell becomes ``observed`` once it falls inside a viewpoint
cone (camera at (cx, cz), heading θ, half-fov α, max range r). The cone
test is a circular sector; close enough for desktop scenes.

The grid is intentionally separate from :class:`OccupancyGrid` so that
"unknown" and "obstacle" are decoupled — exploration wants to reach
unobserved free space, planning wants to avoid obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoverageGridConfig:
    """Mirrors the navigation grid extents so an XZ cell maps 1:1.

    Defaults match ``configs/navigation.yaml`` so the live server can spin
    one up without per-call wiring; tests construct their own.
    """

    resolution: float = 0.05
    origin_x: float = -3.0
    origin_z: float = -4.0
    width: int = 120
    height: int = 120


class CoverageGrid:
    """Mutable observation footprint.

    The grid stores a single uint16 counter per cell: number of frames in
    which the cell has been observed. Zero = never observed (unknown). A
    cell that has been seen at least ``min_observations`` times is treated
    as "known" by the exploration heuristic.
    """

    def __init__(self, cfg: CoverageGridConfig | None = None) -> None:
        self.cfg = cfg or CoverageGridConfig()
        self.grid: np.ndarray = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint16)

    # ── coordinate helpers (mirror OccupancyGrid for testability) ──────────
    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        gx = int((x - self.cfg.origin_x) / self.cfg.resolution)
        gz = int((z - self.cfg.origin_z) / self.cfg.resolution)
        return gx, gz

    def cell_to_world(self, gx: int, gz: int) -> tuple[float, float]:
        x = self.cfg.origin_x + (gx + 0.5) * self.cfg.resolution
        z = self.cfg.origin_z + (gz + 0.5) * self.cfg.resolution
        return x, z

    def in_bounds(self, gx: int, gz: int) -> bool:
        return 0 <= gx < self.cfg.width and 0 <= gz < self.cfg.height

    # ── reset / queries ────────────────────────────────────────────────────
    def reset(self) -> None:
        self.grid.fill(0)

    def is_observed(self, gx: int, gz: int, *, min_observations: int = 1) -> bool:
        if not self.in_bounds(gx, gz):
            return False
        return bool(self.grid[gz, gx] >= min_observations)

    def unobserved_ratio(self, *, min_observations: int = 1) -> float:
        total = self.grid.size
        if total == 0:
            return 0.0
        observed = int((self.grid >= min_observations).sum())
        return 1.0 - observed / total

    # ── mutation ───────────────────────────────────────────────────────────
    def observe_cone(
        self,
        camera_xz: tuple[float, float],
        heading: float,
        fov_rad: float,
        range_m: float,
    ) -> int:
        """Mark all cells inside the circular sector defined by the camera
        pose as observed (increment by 1). Returns the number of newly
        observed cells.

        The implementation is vectorised over a bounding-box of cells so the
        cost is O(r²/res²), not O(width·height) — important when the camera
        only sees a small slice of the room.
        """
        cx, cz = camera_xz
        if range_m <= 0.0 or fov_rad <= 0.0:
            return 0
        half_fov = fov_rad * 0.5
        gx_lo, gz_lo = self.world_to_cell(cx - range_m, cz - range_m)
        gx_hi, gz_hi = self.world_to_cell(cx + range_m, cz + range_m)
        gx_lo = max(0, gx_lo)
        gz_lo = max(0, gz_lo)
        gx_hi = min(self.cfg.width - 1, gx_hi)
        gz_hi = min(self.cfg.height - 1, gz_hi)
        if gx_lo > gx_hi or gz_lo > gz_hi:
            return 0

        gx_idx = np.arange(gx_lo, gx_hi + 1)
        gz_idx = np.arange(gz_lo, gz_hi + 1)
        xs = self.cfg.origin_x + (gx_idx + 0.5) * self.cfg.resolution
        zs = self.cfg.origin_z + (gz_idx + 0.5) * self.cfg.resolution
        dx = xs[None, :] - cx
        dz = zs[:, None] - cz
        dist = np.hypot(dx, dz)
        in_range = dist <= range_m
        bearing = np.arctan2(dz, dx)
        delta = np.arctan2(np.sin(bearing - heading), np.cos(bearing - heading))
        in_cone = np.abs(delta) <= half_fov
        mask = in_range & in_cone
        sub = self.grid[gz_lo : gz_hi + 1, gx_lo : gx_hi + 1]
        before = int((sub > 0).sum())
        sub[mask] = sub[mask] + 1
        after = int((sub > 0).sum())
        return after - before

    # ── unknown-region analysis ─────────────────────────────────────────────
    def unknown_clusters(
        self,
        *,
        min_cluster_cells: int = 8,
        min_observations: int = 1,
    ) -> list[dict[str, float | int]]:
        """Connected components of unobserved cells.

        Each cluster is summarised as ``{centroid_x, centroid_z, cell_count,
        bbox_min_x, bbox_max_x, bbox_min_z, bbox_max_z}``. Components smaller
        than ``min_cluster_cells`` are dropped — they are usually rasterisation
        noise at the corners of an observation cone.
        """
        unknown = self.grid < min_observations
        if not unknown.any():
            return []
        labels = np.zeros_like(unknown, dtype=np.int32)
        next_label = 1
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        h, w = unknown.shape
        for r in range(h):
            for c in range(w):
                if not unknown[r, c]:
                    continue
                left = labels[r, c - 1] if c > 0 else 0
                up = labels[r - 1, c] if r > 0 else 0
                if left == 0 and up == 0:
                    labels[r, c] = next_label
                    parent[next_label] = next_label
                    next_label += 1
                elif left != 0 and up == 0:
                    labels[r, c] = left
                elif up != 0 and left == 0:
                    labels[r, c] = up
                else:
                    labels[r, c] = min(left, up)
                    if left != up:
                        union(left, up)

        for r in range(h):
            for c in range(w):
                if labels[r, c] != 0:
                    labels[r, c] = find(int(labels[r, c]))

        clusters: dict[int, list[tuple[int, int]]] = {}
        for r in range(h):
            for c in range(w):
                lbl = int(labels[r, c])
                if lbl == 0:
                    continue
                clusters.setdefault(lbl, []).append((c, r))

        out: list[dict[str, float | int]] = []
        for cells in clusters.values():
            if len(cells) < min_cluster_cells:
                continue
            xs = [self.cfg.origin_x + (cx + 0.5) * self.cfg.resolution for cx, _ in cells]
            zs = [self.cfg.origin_z + (cz + 0.5) * self.cfg.resolution for _, cz in cells]
            out.append(
                {
                    "centroid_x": sum(xs) / len(xs),
                    "centroid_z": sum(zs) / len(zs),
                    "cell_count": len(cells),
                    "bbox_min_x": min(xs),
                    "bbox_max_x": max(xs),
                    "bbox_min_z": min(zs),
                    "bbox_max_z": max(zs),
                }
            )
        out.sort(key=lambda c: -int(c["cell_count"]))
        return out

    def nearest_unknown(
        self,
        from_xz: tuple[float, float],
        *,
        min_observations: int = 1,
        max_radius_cells: int = 200,
    ) -> tuple[float, float] | None:
        """Centre coordinates of the nearest unobserved cell, or ``None``."""
        sx, sz = self.world_to_cell(*from_xz)
        if not self.in_bounds(sx, sz):
            return None
        for radius in range(0, max_radius_cells + 1):
            for gz in range(sz - radius, sz + radius + 1):
                for gx in range(sx - radius, sx + radius + 1):
                    if not self.in_bounds(gx, gz):
                        continue
                    if max(abs(gx - sx), abs(gz - sz)) != radius:
                        continue
                    if self.grid[gz, gx] < min_observations:
                        return self.cell_to_world(gx, gz)
        return None

    # ── serialisation ──────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, object]:
        return {
            "resolution": self.cfg.resolution,
            "origin_x": self.cfg.origin_x,
            "origin_z": self.cfg.origin_z,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "unobserved_ratio": self.unobserved_ratio(),
            "cells": self.grid.tolist(),
        }
