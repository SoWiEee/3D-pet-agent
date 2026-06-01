"""Phase 9 — coverage grid unit tests."""

from __future__ import annotations

import math

import pytest

from src.exploration.coverage_grid import CoverageGrid, CoverageGridConfig


def _small_grid() -> CoverageGrid:
    cfg = CoverageGridConfig(resolution=0.1, origin_x=-1.0, origin_z=-1.0, width=20, height=20)
    return CoverageGrid(cfg)


def test_fresh_grid_is_fully_unknown() -> None:
    g = _small_grid()
    assert g.unobserved_ratio() == pytest.approx(1.0)
    assert not g.is_observed(5, 5)


def test_world_cell_roundtrip_near_origin() -> None:
    g = _small_grid()
    gx, gz = g.world_to_cell(0.0, 0.0)
    x, z = g.cell_to_world(gx, gz)
    assert abs(x - 0.0) <= 0.06
    assert abs(z - 0.0) <= 0.06


def test_observe_cone_marks_cells_in_sector() -> None:
    g = _small_grid()
    new_cells = g.observe_cone(camera_xz=(0.0, 0.0), heading=0.0, fov_rad=math.pi / 2, range_m=0.5)
    assert new_cells > 0
    gx, gz = g.world_to_cell(0.3, 0.0)
    assert g.is_observed(gx, gz)
    gxb, gzb = g.world_to_cell(-0.3, 0.0)
    assert not g.is_observed(gxb, gzb)


def test_observe_cone_outside_fov_skipped() -> None:
    g = _small_grid()
    g.observe_cone(camera_xz=(0.0, 0.0), heading=0.0, fov_rad=math.radians(10.0), range_m=0.6)
    gx, gz = g.world_to_cell(0.5 * math.cos(math.radians(30)), 0.5 * math.sin(math.radians(30)))
    assert not g.is_observed(gx, gz)


def test_observe_cone_with_zero_range_is_noop() -> None:
    g = _small_grid()
    n = g.observe_cone(camera_xz=(0.0, 0.0), heading=0.0, fov_rad=math.pi / 2, range_m=0.0)
    assert n == 0
    assert g.unobserved_ratio() == pytest.approx(1.0)


def test_unknown_clusters_finds_remaining_region() -> None:
    g = _small_grid()
    # Narrow cone facing +x; +y / -y / -x sectors stay unknown.
    g.observe_cone(camera_xz=(0.0, 0.0), heading=0.0, fov_rad=math.radians(40.0), range_m=1.5)
    clusters = g.unknown_clusters(min_cluster_cells=4)
    assert len(clusters) >= 1
    assert all("centroid_x" in c for c in clusters)


def test_unknown_clusters_filters_tiny_components() -> None:
    g = _small_grid()
    for h in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        g.observe_cone(camera_xz=(0.0, 0.0), heading=h, fov_rad=math.pi / 2 + 0.05, range_m=3.0)
    clusters = g.unknown_clusters(min_cluster_cells=8)
    assert all(c["cell_count"] >= 8 for c in clusters)


def test_nearest_unknown_returns_none_when_fully_observed() -> None:
    g = _small_grid()
    g.grid[:, :] = 5
    assert g.nearest_unknown((0.0, 0.0)) is None


def test_nearest_unknown_finds_closest_cell() -> None:
    g = _small_grid()
    g.grid[:, :] = 5
    gx0, gz0 = g.world_to_cell(0.5, 0.0)
    g.grid[gz0, gx0] = 0
    nearest = g.nearest_unknown((0.0, 0.0))
    assert nearest is not None
    assert nearest[0] == pytest.approx(0.55, abs=0.06)


def test_reset_clears_all_observations() -> None:
    g = _small_grid()
    g.observe_cone((0.0, 0.0), 0.0, math.pi / 2, 0.5)
    assert g.unobserved_ratio() < 1.0
    g.reset()
    assert g.unobserved_ratio() == pytest.approx(1.0)


def test_to_dict_shape() -> None:
    g = _small_grid()
    g.observe_cone((0.0, 0.0), 0.0, math.pi / 2, 0.5)
    d = g.to_dict()
    assert d["width"] == 20
    assert d["height"] == 20
    assert len(d["cells"]) == 20
    assert len(d["cells"][0]) == 20
