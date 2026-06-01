"""Navigation contracts (spec §3.6).

The grounding resolver's output and the planner's input.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

GoalType = Literal["pose", "region", "follow", "viewpoint"]


class NavigationConstraint(BaseModel):
    """Spec §3.6 — generic constraint forwarded into the planner."""

    type: Literal["avoid_object", "stay_on_surface", "keep_distance", "approach_from"]
    object_id: str | None = None
    region_id: str | None = None
    min_distance: float | None = None
    direction: str | None = None


class NavigationGoal(BaseModel):
    """Spec §3.6 — what the planner ultimately consumes."""

    goal_id: str
    goal_type: GoalType = "pose"
    target_position_world: tuple[float, float, float] | None = None
    target_object_id: str | None = None
    target_orientation_hint: str | None = None
    constraints: list[NavigationConstraint] = Field(default_factory=list)
    source_command: str
    explanation: str
    score: float = 0.0
    timestamp: float = Field(default_factory=time.time)
