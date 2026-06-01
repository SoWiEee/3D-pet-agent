"""Phase 9 — exploration planner unit tests."""

from __future__ import annotations

import math

from src.exploration.coverage_grid import CoverageGrid, CoverageGridConfig
from src.exploration.exploration_planner import (
    ExplorationPlanner,
    ExplorationPlannerConfig,
)
from src.spatial.object_lifter import ObjectConfidence, ObjectState3D
from src.spatial.semantic_map import SemanticMap


def _grid() -> CoverageGrid:
    return CoverageGrid(
        CoverageGridConfig(resolution=0.1, origin_x=-1.0, origin_z=-1.0, width=20, height=20)
    )


def _obj(
    *,
    object_id: str,
    label: str,
    pos: tuple[float, float, float],
    status: str = "tracked",
    confidence: float = 0.9,
) -> ObjectState3D:
    return ObjectState3D(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        center_2d=(5.0, 5.0),
        center_3d_world=pos,
        extent_3d=(0.1, 0.1, 0.1),
        median_depth=1.0,
        depth_uncertainty=0.05,
        confidence=ObjectConfidence(overall=confidence),
        tracking_status=status,
        last_seen_frame=1,
        source_backend="mainline_grounding_sam",
    )


def _semantic_map_with(*objects: ObjectState3D, frame_id: int = 1) -> SemanticMap:
    m = SemanticMap()
    m.update(list(objects), frame_id=frame_id)
    return m


def test_empty_world_yields_no_goal() -> None:
    planner = ExplorationPlanner()
    g = _grid()
    g.grid[:, :] = 5
    assert planner.next_goal(SemanticMap(), g, (0.0, 0.0)) is None


def test_inspect_unknown_returns_a_goal_on_half_known_map() -> None:
    planner = ExplorationPlanner(ExplorationPlannerConfig(min_cluster_cells=4))
    g = _grid()
    g.observe_cone((0.0, 0.0), 0.0, math.pi * 0.9, 3.0)
    goal = planner.next_goal(SemanticMap(), g, (0.0, 0.0))
    assert goal is not None
    assert goal.kind in ("inspect_unknown", "search_object", "look_behind")


def test_verify_stale_offered_for_stale_object() -> None:
    planner = ExplorationPlanner(ExplorationPlannerConfig(min_cluster_cells=10_000))
    g = _grid()
    g.grid[:, :] = 5
    stale_obj = _obj(
        object_id="track_001",
        label="cup",
        pos=(0.3, 0.0, 0.4),
        status="stale",
        confidence=0.2,
    )
    smap = _semantic_map_with(stale_obj)
    # SemanticMap.update overrides input tracking_status to "tracked" because
    # the observation arrived this frame; mutate the internal store to mark
    # the object stale for the heuristic.
    smap._objects["track_001"] = smap._objects["track_001"].model_copy(
        update={
            "tracking_status": "stale",
            "confidence": stale_obj.confidence,
        }
    )
    goal = planner.next_goal(smap, g, (0.0, 0.0))
    assert goal is not None
    assert goal.kind == "verify_stale"
    assert goal.related_object_id == "track_001"


def test_search_object_skipped_when_class_already_in_map() -> None:
    planner = ExplorationPlanner(ExplorationPlannerConfig(min_cluster_cells=4))
    g = _grid()
    g.grid[:, :] = 5
    obj = _obj(object_id="track_001", label="cup", pos=(0.3, 0.0, 0.4))
    smap = _semantic_map_with(obj)
    cands = planner.candidates(smap, g, (0.0, 0.0), target_class="cup")
    assert not any(c.kind == "search_object" for c in cands)


def test_search_object_yields_candidate_when_class_missing() -> None:
    planner = ExplorationPlanner(ExplorationPlannerConfig(min_cluster_cells=4))
    g = _grid()
    g.observe_cone((0.0, 0.0), 0.0, math.pi * 0.5, 0.6)
    cands = planner.candidates(SemanticMap(), g, (0.0, 0.0), target_class="cup")
    search = [c for c in cands if c.kind == "search_object"]
    assert search, "expected at least one search_object candidate"
    assert all(c.object_search_relevance == 1.0 for c in search)


def test_navigation_goal_conversion_preserves_position() -> None:
    planner = ExplorationPlanner(ExplorationPlannerConfig(min_cluster_cells=4))
    g = _grid()
    g.observe_cone((0.0, 0.0), 0.0, math.pi * 0.5, 0.6)
    goal = planner.next_goal(SemanticMap(), g, (0.0, 0.0))
    assert goal is not None
    nav = goal.to_navigation_goal()
    assert nav.target_position_world == goal.target_position_world
    assert nav.goal_type == "viewpoint"
    assert nav.source_command == "explore"


def test_scoring_weights_match_spec() -> None:
    planner = ExplorationPlanner()
    s = planner._combine(1.0, 1.0, 1.0, 1.0)
    assert abs(s - (0.40 + 0.25 + 0.20 - 0.15)) < 1e-9


def test_high_travel_cost_demotes_candidate() -> None:
    cfg = ExplorationPlannerConfig(min_cluster_cells=4)
    planner = ExplorationPlanner(cfg)
    g = _grid()
    g.observe_cone((-0.5, 0.0), 0.0, math.pi * 0.5, 0.4)
    near = planner.candidates(SemanticMap(), g, cat_xz=(0.0, 0.0))
    far = planner.candidates(SemanticMap(), g, cat_xz=(-10.0, -10.0))
    if near and far:
        assert far[0].score <= near[0].score
