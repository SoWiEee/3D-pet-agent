"""§14.6.4 — GroundingResolver.build_goal_for_object (explicit object pick)."""

from __future__ import annotations

from src.language import RuleCommandParser
from src.planning import GroundingResolver
from src.spatial.semantic_map import SemanticMap
from tests.factories import make_object


def _map() -> SemanticMap:
    m = SemanticMap()
    m.update(
        [
            make_object(object_id="box_a", class_label="box", center_3d_world=(-0.5, 0.0, -1.4)),
            make_object(object_id="box_b", class_label="box", center_3d_world=(0.7, 0.0, -1.6)),
        ],
        frame_id=0,
    )
    return m


def test_build_goal_for_known_object_succeeds() -> None:
    resolver = GroundingResolver()
    intent = RuleCommandParser().parse("go to the box")
    assert intent is not None
    r = resolver.build_goal_for_object(intent, _map(), "box_b", explanation="LLM chose box_b")
    assert r.status == "success"
    assert r.goal is not None
    assert r.goal.target_object_id == "box_b"
    assert r.goal.explanation == "LLM chose box_b"
    assert r.explanation == "LLM chose box_b"


def test_build_goal_without_override_uses_composed_explanation() -> None:
    resolver = GroundingResolver()
    intent = RuleCommandParser().parse("go to the box")
    assert intent is not None
    r = resolver.build_goal_for_object(intent, _map(), "box_a")
    assert r.status == "success"
    assert r.goal is not None
    assert "box_a" in r.goal.explanation


def test_build_goal_for_unknown_object_is_no_match() -> None:
    resolver = GroundingResolver()
    intent = RuleCommandParser().parse("go to the box")
    assert intent is not None
    r = resolver.build_goal_for_object(intent, _map(), "ghost_999")
    assert r.status == "no_match"
    assert r.goal is None
