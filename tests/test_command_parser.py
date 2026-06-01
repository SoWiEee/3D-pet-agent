"""Phase 6 — rule-based command parser tests.

Spec §9.2 acceptance: ≥ 20 predefined commands parse to a valid intent.
"""

from __future__ import annotations

import pytest

from src.language import CommandIntent, RuleCommandParser, parse_command
from src.language.schema import TargetSpec


@pytest.fixture
def parser() -> RuleCommandParser:
    return RuleCommandParser()


# Spec acceptance: at least 20 canonical commands → valid intent.
_CORPUS: list[tuple[str, str]] = [
    ("go to the cup", "move_to"),
    ("move to the keyboard", "move_to"),
    ("walk to the mouse", "move_to"),
    ("approach the red cup", "move_to"),
    ("come here", "move_to"),
    ("head to the bowl", "move_to"),
    ("hide behind the cup", "hide"),
    ("hide behind the red keyboard", "hide"),
    ("look at the mouse", "look_at"),
    ("watch the keyboard", "look_at"),
    ("stare at the cup", "look_at"),
    ("follow me", "follow"),
    ("follow the user", "follow"),
    ("inspect the cup", "inspect"),
    ("examine the bowl", "inspect"),
    ("find the red cup", "search"),
    ("look for the mouse", "search"),
    ("search for the keyboard", "search"),
    ("explore", "explore"),
    ("look around", "explore"),
    ("stop", "stop"),
    ("halt", "stop"),
    ("what do you see", "report"),
    ("describe the scene", "report"),
]


def test_corpus_has_at_least_20_commands() -> None:
    assert len(_CORPUS) >= 20


@pytest.mark.parametrize("text,expected_intent", _CORPUS)
def test_canonical_commands_parse(
    parser: RuleCommandParser, text: str, expected_intent: str
) -> None:
    intent = parser.parse(text)
    assert intent is not None, f"failed to parse: {text!r}"
    assert intent.intent_type == expected_intent, (
        f"{text!r} → {intent.intent_type} (expected {expected_intent})"
    )
    assert intent.raw_text == text


def test_empty_returns_none(parser: RuleCommandParser) -> None:
    assert parser.parse("") is None
    assert parser.parse("   ") is None


def test_target_class_label_extracted(parser: RuleCommandParser) -> None:
    intent = parser.parse("go to the red cup")
    assert intent is not None
    assert intent.target is not None
    assert intent.target.class_label == "cup"
    assert "red" in intent.target.attributes


def test_attributes_stripped_from_class_label(parser: RuleCommandParser) -> None:
    intent = parser.parse("approach the small blue bowl")
    assert intent is not None
    assert intent.target is not None
    assert intent.target.class_label == "bowl"
    assert set(intent.target.attributes) == {"small", "blue"}


def test_hide_extracts_behind_relation(parser: RuleCommandParser) -> None:
    intent = parser.parse("hide behind the red cup")
    assert intent is not None
    assert intent.intent_type == "hide"
    assert intent.spatial_relation is not None
    assert intent.spatial_relation.type == "behind"
    anchor = intent.spatial_relation.anchor
    assert isinstance(anchor, TargetSpec)
    assert anchor.class_label == "cup"
    assert "red" in anchor.attributes


def test_avoid_constraint_extracted(parser: RuleCommandParser) -> None:
    intent = parser.parse("hide behind the red cup but avoid the mouse")
    assert intent is not None
    assert len(intent.constraints) == 1
    constraint = intent.constraints[0]
    assert constraint.type == "avoid"
    assert constraint.object is not None
    assert constraint.object.class_label == "mouse"
    assert constraint.min_distance is not None


def test_left_right_relation_words(parser: RuleCommandParser) -> None:
    intent = parser.parse("go to the left of the keyboard")
    assert intent is not None
    assert intent.spatial_relation is not None
    assert intent.spatial_relation.type == "left_of"


def test_follow_user_normalizes_to_user_target(parser: RuleCommandParser) -> None:
    intent = parser.parse("follow me")
    assert intent is not None
    assert intent.intent_type == "follow"
    assert intent.target is not None
    assert intent.target.class_label == "user"


def test_bare_target_implies_move_to(parser: RuleCommandParser) -> None:
    intent = parser.parse("cup")
    assert intent is not None
    assert intent.intent_type == "move_to"
    assert intent.target is not None
    assert intent.target.class_label == "cup"
    assert intent.confidence < 1.0  # bare-target is a fuzzy guess


def test_garbage_returns_none(parser: RuleCommandParser) -> None:
    assert parser.parse("ababagalamaga jjkk ll mm nn oo pp qq") is None


def test_module_level_parse_command_works_without_llm() -> None:
    intent = parse_command("go to the cup")
    assert isinstance(intent, CommandIntent)
    assert intent.intent_type == "move_to"
