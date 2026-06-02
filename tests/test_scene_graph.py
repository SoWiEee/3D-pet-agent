"""Phase 5 — SceneGraphBuilder over SemanticMap tests."""

from __future__ import annotations

import json

from src.spatial import SceneGraphBuilder, SemanticMap
from src.spatial.object_lifter import ObjectState3D
from tests.factories import make_object


def _obj(
    *,
    object_id: str,
    label: str,
    center: tuple[float, float, float],
    extent: tuple[float, float, float] = (0.1, 0.1, 0.1),
    bbox: tuple[float, float, float, float] = (100, 100, 200, 200),
    median_depth: float = 2.0,
) -> ObjectState3D:
    return make_object(
        object_id=object_id,
        class_label=label,
        center_3d_world=center,
        extent_3d=extent,
        bbox_xyxy=bbox,
        median_depth=median_depth,
    )


def _populate(map_: SemanticMap, objects: list[ObjectState3D]) -> None:
    map_.update(objects, frame_id=0)


def test_empty_map_returns_empty_graph() -> None:
    g = SceneGraphBuilder().build(SemanticMap())
    assert g.objects == []
    assert g.relations == []


def test_single_object_no_edges() -> None:
    m = SemanticMap()
    _populate(m, [_obj(object_id="t1", label="cup", center=(0, 0, -2))])
    g = SceneGraphBuilder().build(m)
    assert g.objects == ["t1"]
    assert g.relations == []


def test_simple_desk_emits_directional_edges() -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="cup_001", label="cup", center=(-0.5, 0, -2)),
            _obj(object_id="kbd_001", label="keyboard", center=(0.5, 0, -2)),
        ],
    )
    g = SceneGraphBuilder(min_relation_score=0.5).build(m)
    pairs = {(r.subject, r.relation, r.object) for r in g.relations}
    assert ("cup_001", "left_of", "kbd_001") in pairs
    assert ("kbd_001", "right_of", "cup_001") in pairs
    assert all(r.score >= 0.5 for r in g.relations)


def test_edges_sorted_by_score_descending() -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="a", label="cup", center=(-1.0, 0, -2)),
            _obj(object_id="b", label="cup", center=(1.0, 0, -2)),
        ],
    )
    g = SceneGraphBuilder().build(m)
    scores = [r.score for r in g.relations]
    assert scores == sorted(scores, reverse=True)


def test_between_emitted_for_collinear_triple() -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="L", label="cup", center=(-1.0, 0, -2)),
            _obj(object_id="M", label="mouse", center=(0.0, 0, -2)),
            _obj(object_id="R", label="cup", center=(1.0, 0, -2)),
        ],
    )
    g = SceneGraphBuilder(min_relation_score=0.4).build(m)
    rels = [(r.subject, r.relation, r.object, r.object_2) for r in g.relations]
    assert any(r[0] == "M" and r[1] == "between" and {r[2], r[3]} == {"L", "R"} for r in rels)


def test_scene_graph_json_round_trip_matches_spec_shape() -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="cup_001", label="cup", center=(-0.5, 0, -2)),
            _obj(object_id="kbd_001", label="keyboard", center=(0.5, 0, -2)),
        ],
    )
    g = SceneGraphBuilder().build(m)
    js = json.loads(json.dumps(g.to_dict()))
    assert set(js) >= {"timestamp", "frame_id", "coordinate_frame", "objects", "relations"}
    assert all(
        {"subject", "relation", "object", "score", "evidence"} <= set(r) for r in js["relations"]
    )


def test_min_score_filters_low_confidence_edges() -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="a", label="cup", center=(0.0, 0, -2)),
            # Tiny x delta — below threshold, low score.
            _obj(object_id="b", label="cup", center=(0.01, 0, -2)),
        ],
    )
    high_bar = SceneGraphBuilder(min_relation_score=0.9).build(m)
    # near/far would still trigger; verify the axial low-confidence relations are dropped.
    rels = {r.relation for r in high_bar.relations}
    assert "left_of" not in rels
    assert "right_of" not in rels


def test_edges_for_filters_by_object_id() -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="a", label="cup", center=(-1.0, 0, -2)),
            _obj(object_id="b", label="cup", center=(0.0, 0, -2)),
            _obj(object_id="c", label="cup", center=(1.0, 0, -2)),
        ],
    )
    g = SceneGraphBuilder().build(m)
    only_b = g.edges_for("b")
    for r in only_b:
        assert "b" in (r.subject, r.object, r.object_2)
