"""Perception data contracts. Spec §5.4 + a subset of §3.4 used in Phase 2."""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Detection2D(BaseModel):
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float


class ObjectCandidate2D(BaseModel):
    """A detection + mask + simple statistics — the unit consumed by tracking / lifting later."""

    id: str
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    mask_path: str | None = None
    detector_confidence: float
    mask_quality: float = 0.0
    # Normalized image-plane center (x, y) ∈ [0, 1]; used by the placeholder
    # perception→pet behavior wire-up until Phase 3 lifts these to 3D.
    center_normalized: tuple[float, float] = (0.5, 0.5)

    extras: dict[str, Any] = Field(default_factory=dict)


class PerceptionResult(BaseModel):
    frame_id: int
    image_size: tuple[int, int]  # (height, width)
    objects_2d: list[ObjectCandidate2D]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def mask_quality_proxy(mask: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    """Cheap proxy: ratio of mask area to bbox area, clamped to [0, 1].

    A SAM mask that fills most of its prompt bbox usually indicates a stable
    object — a tiny mask in a large bbox often signals a failure.
    """
    x1, y1, x2, y2 = bbox
    bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
    mask_area = float(mask.sum())
    return float(min(1.0, mask_area / bbox_area))
