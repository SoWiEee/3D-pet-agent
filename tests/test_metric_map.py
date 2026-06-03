"""Tests for the metric occupancy layer — spec §14.5 Stage B.

All offline: synthetic scans cast against a binary grid, fused via log-odds,
must recover the obstacle (occupied) and the swept space (free), and export a
binary OccupancyGrid that the planner can consume.
"""

from __future__ import annotations

import math

import numpy as np

from src.planning.occupancy_grid import OccupancyGrid
from src.research.metric_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    MetricMapConfig,
    MetricOccupancyMap,
    RangeScan,
    simulate_scan,
)


def _cfg() -> MetricMapConfig:
    # Small grid centred on the origin for fast, readable tests.
    return MetricMapConfig(resolution=0.1, origin_x=-2.0, origin_z=-2.0, width=40, height=40)


def _world_with_block(cfg: MetricMapConfig) -> OccupancyGrid:
    """Binary world: a single obstacle block to the +X side of the origin."""
    gcfg = cfg.to_grid_config()
    data = np.zeros((gcfg.height, gcfg.width), dtype=np.uint8)
    grid = OccupancyGrid(cfg=gcfg, data=data)
    gx, gz = grid.world_to_cell(1.0, 0.0)
    data[gz - 1 : gz + 2, gx - 1 : gx + 2] = 1  # 3×3 block around world (1.0, 0.0)
    return grid


# ── geometry / config ──────────────────────────────────────────────────────


def test_world_cell_round_trip():
    m = MetricOccupancyMap(_cfg())
    for wx, wz in [(0.0, 0.0), (1.0, -0.5), (-1.3, 1.2)]:
        gx, gz = m.world_to_cell(wx, wz)
        bx, bz = m.cell_to_world(gx, gz)
        assert abs(bx - wx) <= m.cfg.resolution
        assert abs(bz - wz) <= m.cfg.resolution


def test_to_grid_config_matches_extent():
    cfg = _cfg()
    gc = cfg.to_grid_config()
    assert (gc.width, gc.height, gc.resolution) == (cfg.width, cfg.height, cfg.resolution)
    assert (gc.origin_x, gc.origin_z) == (cfg.origin_x, cfg.origin_z)


# ── single-beam sensor model ────────────────────────────────────────────────


def test_hit_raises_endpoint_above_half_free_lowers_path():
    m = MetricOccupancyMap(_cfg())
    # One beam straight along +X (heading 0) that returns at 1.0 m.
    scan = RangeScan(pose=(0.0, 0.0, 0.0), angles=np.array([0.0]), ranges=np.array([1.0]))
    m.integrate(scan)
    p = m.prob()

    hit_gx, hit_gz = m.world_to_cell(1.0, 0.0)
    assert p[hit_gz, hit_gx] > 0.5  # endpoint reads occupied

    mid_gx, mid_gz = m.world_to_cell(0.5, 0.0)
    assert p[mid_gz, mid_gx] < 0.5  # swept space reads free


def test_no_return_beam_marks_free_no_occupied():
    m = MetricOccupancyMap(_cfg())
    scan = RangeScan(
        pose=(0.0, 0.0, 0.0),
        angles=np.array([0.0]),
        ranges=np.array([np.inf]),
        max_range=1.5,
    )
    # Two sweeps so the swept cells cross the FREE threshold (one l_free pass
    # alone leaves prob ≈ 0.40, still UNKNOWN — by design, evidence accrues).
    m.integrate(scan)
    m.integrate(scan)
    klass = m.classify()
    # Nothing should be occupied along an empty beam.
    assert not np.any(klass == OCCUPIED)
    near_gx, near_gz = m.world_to_cell(0.5, 0.0)
    assert klass[near_gz, near_gx] == FREE


def test_log_odds_clamped():
    cfg = MetricMapConfig(resolution=0.1, origin_x=-2.0, origin_z=-2.0, width=40, height=40)
    m = MetricOccupancyMap(cfg)
    scan = RangeScan(pose=(0.0, 0.0, 0.0), angles=np.array([0.0]), ranges=np.array([1.0]))
    for _ in range(100):
        m.integrate(scan)
    assert m.log_odds.max() <= cfg.l_max + 1e-6
    assert m.log_odds.min() >= cfg.l_min - 1e-6


# ── multi-scan fusion recovers a known obstacle ─────────────────────────────


def test_fusion_recovers_obstacle_and_free_space():
    cfg = _cfg()
    world = _world_with_block(cfg)
    m = MetricOccupancyMap(cfg)

    # Scan the block from several poses around it.
    for pose in [(-1.0, 0.0, 0.0), (0.0, -1.0, math.pi / 2), (0.0, 1.0, -math.pi / 2)]:
        m.integrate(simulate_scan(world, pose, n_beams=180, max_range=4.0))

    klass = m.classify()
    # Rays stop at the surface, so the block's *front face* (x≈0.9, facing the
    # -X sensor) reads occupied; its solid centre is never observed.
    bgx, bgz = m.world_to_cell(0.9, 0.0)
    assert klass[bgz, bgx] == OCCUPIED
    # A swept open cell between the sensors and the block must read free.
    fgx, fgz = m.world_to_cell(-0.5, 0.0)
    assert klass[fgz, fgx] == FREE


def test_export_binary_grid_blocks_obstacle():
    cfg = _cfg()
    world = _world_with_block(cfg)
    m = MetricOccupancyMap(cfg)
    for pose in [(-1.0, 0.0, 0.0), (0.0, -1.0, math.pi / 2)]:
        m.integrate(simulate_scan(world, pose, n_beams=180, max_range=4.0))

    grid = m.to_occupancy_grid()
    # Front face (observed surface) exports as a blocked cell.
    bgx, bgz = grid.world_to_cell(0.9, 0.0)
    assert grid.is_blocked(bgx, bgz)
    # A swept-free cell exports as free under the optimistic default.
    fgx, fgz = grid.world_to_cell(-0.5, 0.0)
    assert grid.is_free(fgx, fgz)


def test_unknown_is_blocked_flag():
    cfg = _cfg()
    m = MetricOccupancyMap(cfg)  # nothing integrated → everything unknown
    optimistic = m.to_occupancy_grid(unknown_is_blocked=False)
    conservative = m.to_occupancy_grid(unknown_is_blocked=True)
    assert int(optimistic.data.sum()) == 0
    assert int(conservative.data.sum()) == cfg.width * cfg.height


# ── synthetic sensor ────────────────────────────────────────────────────────


def test_simulate_scan_measures_block_distance():
    cfg = _cfg()
    world = _world_with_block(cfg)
    # From the origin looking along +X, the 3×3 block starts ~0.8 m away.
    scan = simulate_scan(world, (0.0, 0.0, 0.0), n_beams=1, fov=0.0, max_range=4.0)
    assert math.isfinite(scan.ranges[0])
    assert 0.7 <= scan.ranges[0] <= 1.0


def test_simulate_scan_open_space_returns_inf():
    cfg = _cfg()
    gc = cfg.to_grid_config()
    world = OccupancyGrid(cfg=gc, data=np.zeros((gc.height, gc.width), dtype=np.uint8))
    scan = simulate_scan(world, (0.0, 0.0, 0.0), n_beams=8, max_range=1.0)
    assert np.all(~np.isfinite(scan.ranges))


def test_classify_unknown_for_unscanned_cells():
    m = MetricOccupancyMap(_cfg())
    assert np.all(m.classify() == UNKNOWN)
