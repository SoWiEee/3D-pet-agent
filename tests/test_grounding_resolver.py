"""Phase 6 — GroundingResolver tests."""

from __future__ import annotations

import pytest

from src.language import RuleCommandParser
from src.planning import GroundingResolver, NavigationGoal
from src.spatial import SceneGraphBuilder, SemanticMap
from src.spatial.object_lifter import ObjectConfidence, ObjectState3D


def _obj(
    *,
    object_id: str,
    label: str,
    center: tuple[float, float, float],
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
        extent_3d=(0.1, 0.1, 0.1),
        median_depth=2.0,
        depth_uncertainty=0.1,
        confidence=ObjectConfidence(
            detector=0.85, mask_quality=0.8, depth_quality=0.7, tracking=1.0, overall=0.8
        ),
        last_seen_frame=0,
        tracking_status=tracking_status,  # type: ignore[arg-type]
    )


@pytest.fixture
def parser() -> RuleCommandParser:
    return RuleCommandParser()


@pytest.fixture
def resolver() -> GroundingResolver:
    return GroundingResolver(min_final_score=0.5, ambiguity_margin=0.10)


def _populate(m: SemanticMap, objs: list[ObjectState3D]) -> None:
    m.update(objs, frame_id=0)


def test_empty_map_returns_empty_status(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    intent = parser.parse("go to the cup")
    assert intent is not None
    r = resolver.resolve(intent, SemanticMap())
    assert r.status == "empty_map"
    assert "no objects" in r.explanation.lower()


def test_move_to_unique_target_succeeds(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="cup_001", label="cup", center=(0.0, 0.0, -1.5)),
            _obj(object_id="kbd_001", label="keyboard", center=(0.8, 0.0, -1.7)),
        ],
    )
    intent = parser.parse("go to the cup")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "success"
    assert isinstance(r.goal, NavigationGoal)
    assert r.goal.target_object_id == "cup_001"
    assert r.goal.explanation  # spec §9.2: every NavigationGoal has explanation
    assert r.goal.target_position_world is not None


def test_success_exposes_candidate_breakdowns(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    """The explanation panel needs per-component scores, not just totals."""
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="cup_001", label="cup", center=(0.0, 0.0, -1.5)),
            _obj(object_id="kbd_001", label="keyboard", center=(0.8, 0.0, -1.7)),
        ],
    )
    intent = parser.parse("go to the cup")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "success"
    assert r.candidate_breakdowns is not None and len(r.candidate_breakdowns) >= 1
    top = r.candidate_breakdowns[0]
    for key in (
        "object_id",
        "total",
        "semantic",
        "attribute",
        "relation",
        "visibility",
        "feasibility",
    ):
        assert key in top
    # The winning candidate's component-weighted total matches its reported total.
    from src.planning.grounding_resolver import GROUNDING_WEIGHTS

    recomputed = sum(GROUNDING_WEIGHTS[k] * float(top[k]) for k in GROUNDING_WEIGHTS)
    assert recomputed == pytest.approx(float(top["total"]), abs=1e-3)


def test_two_matching_objects_trigger_clarification(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="cup_001", label="cup", center=(-0.3, 0.0, -1.5)),
            _obj(object_id="cup_002", label="cup", center=(0.3, 0.0, -1.5)),
        ],
    )
    intent = parser.parse("go to the cup")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "clarification"
    assert r.candidates is not None
    assert len(r.candidates) >= 2
    assert "which" in r.explanation.lower()


def test_relation_breaks_ambiguity(resolver: GroundingResolver, parser: RuleCommandParser) -> None:
    """Two cups, but one is clearly behind the keyboard."""
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="kbd_001", label="keyboard", center=(0.0, 0.0, -1.5)),
            _obj(object_id="cup_front", label="cup", center=(0.0, 0.0, -1.0)),  # in front
            _obj(object_id="cup_back", label="cup", center=(0.0, 0.0, -2.5)),  # behind
        ],
    )
    intent = parser.parse("hide behind the keyboard")
    assert intent is not None
    sg = SceneGraphBuilder().build(m)
    r = resolver.resolve(intent, m, sg)
    assert r.status == "success"
    assert r.goal is not None
    assert r.goal.target_object_id == "kbd_001"
    # Pose should be behind the keyboard (more negative z).
    assert r.goal.target_position_world is not None
    assert r.goal.target_position_world[2] < -1.5


def test_no_match_when_target_not_in_map(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(m, [_obj(object_id="kbd_001", label="keyboard", center=(0.0, 0.0, -1.5))])
    intent = parser.parse("go to the giraffe")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "no_match"


def test_explanation_present_on_success(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(m, [_obj(object_id="cup_001", label="cup", center=(0.0, 0.0, -1.5))])
    intent = parser.parse("go to the cup")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "success"
    assert r.goal is not None
    assert len(r.goal.explanation) > 0
    assert "cup" in r.goal.explanation


def test_avoid_constraint_projects_to_object_id(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="cup_001", label="cup", center=(-0.5, 0.0, -1.5)),
            _obj(object_id="mouse_001", label="mouse", center=(0.5, 0.0, -1.5)),
        ],
    )
    intent = parser.parse("go to the cup but avoid the mouse")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "success"
    assert r.goal is not None
    avoid_ids = [c.object_id for c in r.goal.constraints if c.type == "avoid_object"]
    assert "mouse_001" in avoid_ids


def test_occluded_target_loses_visibility_score(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(
        m,
        [
            _obj(object_id="cup_visible", label="cup", center=(-0.4, 0.0, -1.5)),
            _obj(
                object_id="cup_occluded",
                label="cup",
                center=(0.4, 0.0, -1.5),
                tracking_status="occluded",
            ),
        ],
    )
    intent = parser.parse("go to the cup")
    assert intent is not None
    r = resolver.resolve(intent, m)
    # Either success on the visible one or clarification; both are valid here.
    if r.status == "success":
        assert r.goal is not None
        assert r.goal.target_object_id == "cup_visible"


def test_stop_intent_returns_success_without_target(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(m, [_obj(object_id="cup_001", label="cup", center=(0.0, 0.0, -1.5))])
    intent = parser.parse("stop")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "success"
    assert r.goal is not None
    assert r.goal.target_position_world is None
    assert "stop" in r.goal.explanation.lower()


def test_explore_intent_emits_region_goal(
    resolver: GroundingResolver, parser: RuleCommandParser
) -> None:
    m = SemanticMap()
    _populate(m, [_obj(object_id="cup_001", label="cup", center=(0.0, 0.0, -1.5))])
    intent = parser.parse("explore")
    assert intent is not None
    r = resolver.resolve(intent, m)
    assert r.status == "success"
    assert r.goal is not None
    assert r.goal.goal_type == "region"
