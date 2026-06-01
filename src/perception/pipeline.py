"""Perception orchestrator. Detects → segments → builds ObjectCandidate2D list.

Per spec §5.3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .depth import DepthAnythingV2
from .detector import GroundingDinoDetector
from .schema import ObjectCandidate2D, PerceptionResult, mask_quality_proxy
from .segmenter import SamSegmenter

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..spatial import (
        CameraIntrinsics,
        ObjectLifter,
        ObjectState3D,
        PoseSource,
        SemanticMap,
    )
    from ..tracking import Tracker

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
        self.depth: DepthAnythingV2 | None = None  # lazy
        self.lifter: ObjectLifter | None = None  # lazy
        self.run_dir = run_dir or Path("runs")

    def _ensure_3d(self) -> None:
        if self.depth is None:
            self.depth = DepthAnythingV2(self.cfg.models.depth)
        if self.lifter is None:
            from ..spatial import ObjectLifter as _Lifter
            self.lifter = _Lifter()

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

    # ── Phase 3: 2D candidates + depth + lifting ────────────────────────────
    def run_frame_3d(
        self,
        frame_bgr: np.ndarray,
        prompts: list[str],
        *,
        frame_id: int = 0,
        intrinsics: CameraIntrinsics | None = None,
        pose_source: PoseSource | None = None,
        save_masks: bool = True,
    ) -> tuple[PerceptionResult, np.ndarray, list[ObjectState3D]]:
        """Run detection + segmentation + depth + 3D lifting on one frame.

        Returns ``(2d_result, depth_map, lifted_objects)``. Depth is float32
        ``(H, W)`` in the model's native (relative) units; lifted objects use
        the (graphics-)world frame when a pose is provided, otherwise the
        camera frame — see ``coordinate_frame``.
        """
        from ..spatial import CameraIntrinsics as _CameraIntrinsics
        from ..spatial import FixedPoseSource

        self._ensure_3d()
        assert self.depth is not None and self.lifter is not None

        result = self.run_frame(frame_bgr, prompts, frame_id=frame_id, save_masks=save_masks)
        depth = self.depth.predict(frame_bgr)
        log.info(
            "frame %d: depth shape=%s range=[%.3f, %.3f]",
            frame_id, depth.shape, float(depth.min()), float(depth.max()),
        )

        intr = intrinsics or _CameraIntrinsics.from_fov(
            image_size=result.image_size, horizontal_fov_deg=60.0
        )
        pose = (pose_source or FixedPoseSource()).get(frame_id)

        lifted = self.lifter.lift_many(
            result.objects_2d, depth, intr, pose, frame_id=frame_id
        )
        log.info("frame %d: lifted %d / %d objects", frame_id, len(lifted), len(result.objects_2d))
        return result, depth, lifted

    # ── Phase 4: 3D pipeline + tracker + SemanticMap fusion ─────────────────
    def run_frame_tracked(
        self,
        frame_bgr: np.ndarray,
        prompts: list[str],
        *,
        tracker: Tracker,
        semantic_map: SemanticMap,
        frame_id: int = 0,
        intrinsics: CameraIntrinsics | None = None,
        pose_source: PoseSource | None = None,
        save_masks: bool = True,
    ) -> tuple[PerceptionResult, np.ndarray, list[ObjectState3D]]:
        """Phase 4 hot path: 2D → 3D → track → fuse into SemanticMap.

        Returns the same triple as :meth:`run_frame_3d` but the lifted objects
        carry stable ``track_NNN`` ids. The ``semantic_map`` parameter is
        mutated in place (one persistent instance over the whole session).
        """
        result, depth, lifted = self.run_frame_3d(
            frame_bgr,
            prompts,
            frame_id=frame_id,
            intrinsics=intrinsics,
            pose_source=pose_source,
            save_masks=save_masks,
        )
        tracked = tracker.update(lifted, frame_id)
        semantic_map.update(tracked, frame_id)
        return result, depth, tracked

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

    @staticmethod
    def colorize_depth(depth: np.ndarray) -> np.ndarray:
        """Visualize a depth map as an inferno heatmap, RGB ndarray."""
        d = depth.astype(np.float32)
        d_min, d_max = float(np.nanmin(d)), float(np.nanmax(d))
        if d_max - d_min < 1e-6:
            d = np.zeros_like(d, dtype=np.uint8)
        else:
            d = ((d - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
