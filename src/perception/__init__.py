"""Perception modules: detection, segmentation, optional depth + pipeline orchestrator."""

from .pipeline import PerceptionPipeline, PerceptionResult
from .schema import Detection2D, ObjectCandidate2D

__all__ = ["PerceptionPipeline", "PerceptionResult", "Detection2D", "ObjectCandidate2D"]
