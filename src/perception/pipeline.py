"""Perception orchestrator. Detects → segments → builds ObjectCandidate2D list.

Per spec §5.3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .detector import GroundingDinoDetector
from .schema import ObjectCandidate2D, PerceptionResult, mask_quality_proxy
from .segmenter import SamSegmenter

if TYPE_CHECKING:
    from ..config import AppConfig

log = logging.getLogger("pet_agent.perception")


_PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
        [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    ],
    dtype=np.uint8,
)


def _color(i: int) -> tuple[int, int, int]:
    r, g, b = _PALETTE[i % len(_PALETTE)]
    return int(b), int(g), int(r)  # cv2 wants BGR


class PerceptionPipeline:
    def __init__(self, cfg: AppConfig, *, run_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.detector = GroundingDinoDetector(cfg.models.detector)
        self.segmenter = SamSegmenter(cfg.models.segmenter)
        self.run_dir = run_dir or Path("runs")

    def run_frame(
        self,
        frame_bgr: np.ndarray,
        prompts: list[str],
        *,
        frame_id: int = 0,
        save_masks: bool = True,
    ) -> PerceptionResult:
        h, w = frame_bgr.shape[:2]
        detections = self.detector.predict(frame_bgr, prompts)
        log.info("frame %d: %d detections", frame_id, len(detections))

        boxes = [d.bbox_xyxy for d in detections]
        masks = self.segmenter.predict(frame_bgr, boxes) if boxes else []

        out_dir = self.run_dir / f"frame_{frame_id:06d}"
        if save_masks and masks:
            out_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[ObjectCandidate2D] = []
        for i, (det, mask) in enumerate(zip(detections, masks, strict=True)):
            mask_path: str | None = None
            if save_masks:
                p = out_dir / f"obj_{i:03d}_{det.label.replace(' ', '_')}.png"
                cv2.imwrite(str(p), (mask.astype(np.uint8) * 255))
                mask_path = str(p)
            x1, y1, x2, y2 = det.bbox_xyxy
            cx = (x1 + x2) / 2.0 / max(1, w)
            cy = (y1 + y2) / 2.0 / max(1, h)
            candidates.append(
                ObjectCandidate2D(
                    id=f"obj_{frame_id:06d}_{i:03d}",
                    label=det.label,
                    bbox_xyxy=det.bbox_xyxy,
                    mask_path=mask_path,
                    detector_confidence=det.detector_confidence,
                    mask_quality=mask_quality_proxy(mask, det.bbox_xyxy),
                    center_normalized=(cx, cy),
                )
            )
        return PerceptionResult(frame_id=frame_id, image_size=(h, w), objects_2d=candidates)

    # ── debug visualisation ─────────────────────────────────────────────────
    def visualize(self, frame_bgr: np.ndarray, result: PerceptionResult) -> np.ndarray:
        canvas = frame_bgr.copy()
        for i, obj in enumerate(result.objects_2d):
            color = _color(i)
            x1, y1, x2, y2 = (int(v) for v in obj.bbox_xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.label} {obj.detector_confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas, label, (x1 + 2, max(th, y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            if obj.mask_path:
                mask = cv2.imread(obj.mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    overlay = np.zeros_like(canvas)
                    overlay[mask > 0] = color
                    canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.35, 0)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
