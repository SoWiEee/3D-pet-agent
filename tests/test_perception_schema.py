"""Perception output schema. Spec §5.4."""
import numpy as np

from src.perception.schema import (
    Detection2D,
    ObjectCandidate2D,
    PerceptionResult,
    mask_quality_proxy,
)


def test_detection_validates():
    d = Detection2D(label="cup", bbox_xyxy=(10, 20, 100, 200), detector_confidence=0.8)
    assert d.label == "cup"


def test_object_candidate_default_extras():
    o = ObjectCandidate2D(
        id="obj_000000_001",
        label="cup",
        bbox_xyxy=(10, 20, 100, 200),
        detector_confidence=0.8,
    )
    assert o.extras == {}
    assert o.center_normalized == (0.5, 0.5)


def test_perception_result_serializes():
    r = PerceptionResult(frame_id=42, image_size=(480, 640), objects_2d=[])
    blob = r.to_dict()
    assert blob["frame_id"] == 42
    assert blob["image_size"] == (480, 640)
    assert blob["objects_2d"] == []


def test_mask_quality_proxy_basic():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:90] = True  # 60 × 60 = 3600 area, bbox 60×60 = 3600 → ratio 1.0
    q = mask_quality_proxy(mask, (30, 20, 90, 80))
    assert 0.99 <= q <= 1.0


def test_mask_quality_proxy_partial():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:50, 40:50] = True  # 100 area, bbox area 3600 → ratio ≈ 0.028
    q = mask_quality_proxy(mask, (30, 20, 90, 80))
    assert 0.0 < q < 0.05
