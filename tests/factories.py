"""Shared test builders.

Every ``test_*`` module used to hand-roll its own ``ObjectState3D`` literal —
the same ~20-line construction with slightly different defaults. They now all
delegate to :func:`make_object`, so the boilerplate lives in one place while
each module keeps its own thin wrapper (and its existing call sites) for the
handful of fields that module actually cares about.

Importable helpers live here rather than in ``conftest.py`` because pytest
fixtures aren't meant to be imported; ``conftest.py`` holds the fixtures.
"""

from __future__ import annotations

from src.spatial.object_lifter import ObjectConfidence, ObjectState3D


def make_object(
    *,
    object_id: str = "obj_001",
    class_label: str = "cup",
    center_3d_world: tuple[float, float, float] = (0.0, 0.0, -2.0),
    extent_3d: tuple[float, float, float] = (0.1, 0.1, 0.1),
    bbox_xyxy: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 200.0),
    center_2d: tuple[float, float] | None = None,
    median_depth: float = 2.0,
    depth_uncertainty: float = 0.1,
    tracking_status: str = "tracked",
    last_seen_frame: int = 0,
    attributes: list[str] | None = None,
    source_backend: str = "mainline_grounding_sam",
    detector: float = 0.85,
    mask_quality: float = 0.8,
    depth_quality: float = 0.7,
    tracking: float = 1.0,
    overall: float = 0.8,
) -> ObjectState3D:
    """Build an ``ObjectState3D`` with sensible defaults; override per test.

    ``center_2d`` defaults to the centre of ``bbox_xyxy`` when omitted.
    """
    if center_2d is None:
        center_2d = ((bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2)
    return ObjectState3D(
        object_id=object_id,
        class_label=class_label,
        attributes=attributes or [],
        bbox_xyxy=bbox_xyxy,
        mask_path=None,
        center_2d=center_2d,
        coordinate_frame="world",
        center_3d_world=center_3d_world,
        extent_3d=extent_3d,
        median_depth=median_depth,
        depth_uncertainty=depth_uncertainty,
        source_backend=source_backend,  # type: ignore[arg-type]
        confidence=ObjectConfidence(
            detector=detector,
            mask_quality=mask_quality,
            depth_quality=depth_quality,
            tracking=tracking,
            overall=overall,
        ),
        last_seen_frame=last_seen_frame,
        tracking_status=tracking_status,  # type: ignore[arg-type]
    )
