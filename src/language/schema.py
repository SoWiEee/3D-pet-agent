"""Command intent contracts (spec §3.5).

The parser emits ``CommandIntent``. The grounding resolver consumes it +
``SemanticMap`` + ``SceneGraph`` to produce a ``NavigationGoal``. The LLM (if
enabled) outputs JSON that validates against these models; on schema failure
we fall back to the rule parser.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntentType = Literal[
    "move_to",
    "hide",
    "look_at",
    "follow",
    "avoid",
    "search",
    "inspect",
    "explore",
    "report",
    "stop",
]

RelationType = Literal[
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "above",
    "below",
    "near",
    "far_from",
    "on_surface",
    "between",
]


class TargetSpec(BaseModel):
    """Coarse description of the object the command refers to."""

    class_label: str | None = None
    attributes: list[str] = Field(default_factory=list)
    object_id: str | None = None  # explicit pin from a panel click, optional


class RelationSpec(BaseModel):
    """Spatial qualifier on the target — "behind the red cup" etc."""

    type: RelationType
    anchor: TargetSpec | Literal["target", "self"] = "target"


class ConstraintSpec(BaseModel):
    """One constraint from the command — kept generic until Phase 7 consumes it."""

    type: Literal["avoid", "keep_distance", "stay_on_surface", "approach_from"]
    object: TargetSpec | None = None
    min_distance: float | None = None
    region_id: str | None = None
    direction: str | None = None


class CommandIntent(BaseModel):
    """Spec §3.5 — the structured form of one user utterance."""

    raw_text: str
    intent_type: IntentType
    target: TargetSpec | None = None
    spatial_relation: RelationSpec | None = None
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    fallback: Literal["ask_clarification", "stop", "noop"] = "ask_clarification"
    confidence: float = 1.0
