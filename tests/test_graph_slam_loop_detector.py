"""§14.6.2 — ORB appearance loop detector."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
from src.research.graph_slam import OrbBowLoopDetector  # noqa: E402


def _structured(seed: int) -> np.ndarray:
    """A textured, ORB-friendly image (random noise is feature-poor; draw
    deterministic rectangles so ORB finds stable, repeatable keypoints)."""
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((240, 320), 30, dtype=np.uint8)
    for _ in range(40):
        x1, y1 = int(rng.integers(0, 280)), int(rng.integers(0, 200))
        x2, y2 = x1 + int(rng.integers(8, 40)), y1 + int(rng.integers(8, 40))
        shade = int(rng.integers(80, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), shade, -1)
    return img


def test_revisited_frame_is_detected() -> None:
    # min_matches=50: same-image yields ~343 ratio-test matches; distinct images
    # yield at most ~34, so this threshold is an honest 10x margin.
    det = OrbBowLoopDetector(min_gap=2, min_matches=50)
    a = _structured(1)
    seq = [a, _structured(2), _structured(3), a]  # kf 3 revisits kf 0's image
    results = [det.add_keyframe(kf, img) for kf, img in enumerate(seq)]
    assert results[0] is None and results[1] is None and results[2] is None
    assert results[3] == 0  # the revisited image loops back to keyframe 0


def test_distinct_frames_no_loop() -> None:
    det = OrbBowLoopDetector(min_gap=1, min_matches=40)
    for kf in range(5):
        assert det.add_keyframe(kf, _structured(100 + kf)) is None
