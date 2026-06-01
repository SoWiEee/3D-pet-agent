"""Phase 7 — occupancy grid tests."""

from __future__ import annotations

from src.planning import GridConfig, build_occupancy_grid
from src.planning.schema import NavigationConstraint
from src.spatial import SemanticMap
from src.spatial.object_lifter import ObjectConfidence, ObjectState3D


def _obj(
    *,
    object_id: str,
    center: tuple[float, float, float],
    extent: tuple[float, float, float] = (0.2, 0.2, 0.2),
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


def _small_map(*objects: ObjectState3D) -> SemanticMap:
    m = SemanticMap()
    m.update(list(objects), frame_id=0)
    return m


def _tiny_cfg(*, padding: float = 0.0) -> GridConfig:
    return GridConfig(
        resolution=0.1,
        origin_x=-1.0,
        origin_z=-1.0,
        width=20,
        height=20,
        obstacle_padding=padding,
    )


def test_empty_map_yields_clear_grid() -> None:
    g = build_occupancy_grid(SemanticMap(), cfg=_tiny_cfg())
    assert int(g.data.sum()) == 0


def test_object_marks_cells_blocked() -> None:
    m = _small_map(_obj(object_id="cup", center=(0.0, 0.0, 0.0), extent=(0.2, 0.1, 0.2)))
    g = build_occupancy_grid(m, cfg=_tiny_cfg(padding=0.0))
    cx, cz = g.world_to_cell(0.0, 0.0)
    assert g.is_blocked(cx, cz)
    # A cell well outside the footprint stays free.
    assert g.is_free(*g.world_to_cell(0.8, 0.8))


def test_padding_inflates_footprint() -> None:
    m = _small_map(_obj(object_id="cup", center=(0.0, 0.0, 0.0), extent=(0.1, 0.1, 0.1)))
    nopad = build_occupancy_grid(m, cfg=_tiny_cfg(padding=0.0))
    padded = build_occupancy_grid(m, cfg=_tiny_cfg(padding=0.25))
    assert int(padded.data.sum()) > int(nopad.data.sum())


def test_stale_object_does_not_block() -> None:
    """SemanticMap.update stamps fresh inserts as ``tracked`` regardless of
    the input ``tracking_status``; age the entry by advancing frames past
    persistence + stale thresholds, then check the grid ignores it."""
    m = SemanticMap(persistence_frames=1, stale_frames=2, lost_frames=100)
    m.update([_obj(object_id="ghost", center=(0.0, 0.0, 0.0))], frame_id=0)
    # Advance frames without observing → tracked → occluded → stale.
    for fi in range(1, 5):
        m.update([], frame_id=fi)
    assert m.objects["ghost"].tracking_status in {"stale", "lost"}
    g = build_occupancy_grid(m, cfg=_tiny_cfg(padding=0.0))
    assert int(g.data.sum()) == 0


def test_avoid_constraint_inflates_specific_object() -> None:
    m = _small_map(
        _obj(object_id="cup", center=(-0.5, 0.0, 0.0), extent=(0.1, 0.1, 0.1)),
        _obj(object_id="mouse", center=(0.5, 0.0, 0.0), extent=(0.1, 0.1, 0.1)),
    )
    plain = build_occupancy_grid(m, cfg=_tiny_cfg(padding=0.05))
    avoidy = build_occupancy_grid(
        m,
        cfg=_tiny_cfg(padding=0.05),
        constraints=[
            NavigationConstraint(type="avoid_object", object_id="mouse", min_distance=0.4)
        ],
    )
    assert int(avoidy.data.sum()) > int(plain.data.sum())


def test_world_to_cell_roundtrip() -> None:
    g = build_occupancy_grid(SemanticMap(), cfg=_tiny_cfg())
    for wx, wz in [(-0.5, -0.5), (0.3, -0.2), (0.95, 0.95)]:
        cx, cz = g.world_to_cell(wx, wz)
        x, z = g.cell_to_world(cx, cz)
        assert abs(x - wx) <= g.cfg.resolution
        assert abs(z - wz) <= g.cfg.resolution


def test_nearest_free_inside_block_finds_edge() -> None:
    m = _small_map(_obj(object_id="cup", center=(0.0, 0.0, 0.0), extent=(0.2, 0.1, 0.2)))
    g = build_occupancy_grid(m, cfg=_tiny_cfg(padding=0.0))
    cx, cz = g.world_to_cell(0.0, 0.0)
    assert g.is_blocked(cx, cz)
    free = g.nearest_free(cx, cz, max_radius_cells=8)
    assert free is not None
    assert g.is_free(*free)


def test_to_dict_shape_for_debug_overlay() -> None:
    m = _small_map(_obj(object_id="cup", center=(0.0, 0.0, 0.0)))
    g = build_occupancy_grid(m, cfg=_tiny_cfg(padding=0.05))
    d = g.to_dict()
    assert d["width"] == 20 and d["height"] == 20
    assert "cup" in d["obstacle_ids"]
    assert len(d["data"]) == 20 * 20
