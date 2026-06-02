"""Evaluation contracts (spec §3.8 + §13).

A :class:`DatasetEntry` describes one trial — the scene state (SemanticMap
snapshot), the user utterance, what the expected ground-truth target /
relation / outcome is, and optional notes. A :class:`EvaluationRecord` is
the per-trial result produced by the runner.

Datasets are JSONL — one entry per line — so they stream and diff cleanly.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

ExpectedOutcome = Literal[
    "navigate", "hide", "look_at", "stop", "explore", "report", "clarification", "no_match"
]


class DatasetSceneObject(BaseModel):
    """One pre-placed object in a scene fixture."""

    object_id: str
    class_label: str
    attributes: list[str] = Field(default_factory=list)
    center_3d_world: tuple[float, float, float]
    extent_3d: tuple[float, float, float] = (0.1, 0.1, 0.1)
    confidence: float = 0.85
    tracking_status: Literal["tracked", "occluded", "stale", "lost"] = "tracked"


class DatasetScene(BaseModel):
    """Pre-built SemanticMap snapshot the trial runs against.

    The runner pours these into a fresh SemanticMap before each trial so
    state is hermetic — one bad trial cannot poison the next one.
    """

    scene_id: str
    description: str = ""
    objects: list[DatasetSceneObject]


class DatasetEntry(BaseModel):
    """One evaluation trial. Stored one per line in the dataset JSONL."""

    trial_id: str
    scene: DatasetScene
    command: str
    expected_outcome: ExpectedOutcome
    expected_target: str | None = None
    expected_relation: str | None = None
    notes: str = ""


class ControllerMetrics(BaseModel):
    max_cross_track_error_m: float = 0.0
    max_heading_error_rad: float = 0.0
    mean_speed_mps: float = 0.0
    steps: int = 0


class EvaluationRecord(BaseModel):
    """Spec §3.8 EvaluationRecord — per-trial output."""

    trial_id: str
    scene_id: str
    command: str
    expected_outcome: ExpectedOutcome
    expected_target: str | None = None
    predicted_outcome: str = ""
    predicted_target: str | None = None
    grounding_success: bool = False
    path_success: bool = False
    collision_count: int = 0
    task_success: bool = False
    latency_ms: float = 0.0
    controller_metrics: ControllerMetrics = Field(default_factory=ControllerMetrics)
    notes: str = ""
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
