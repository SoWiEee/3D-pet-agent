"""Monocular depth via Depth Anything V2.

Optional in Phase 2 — included so Phase 3 (3D lifting) can wire in without restructuring.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import DepthConfig

log = logging.getLogger("pet_agent.depth")


def _pick_device(requested: str) -> str:
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


class DepthAnythingV2:
    def __init__(self, cfg: DepthConfig) -> None:
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        log.info("loading Depth Anything V2 weights: %s (device=%s)", self.cfg.hf_model_id, self.device)
        self._processor = AutoImageProcessor.from_pretrained(self.cfg.hf_model_id)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(self.cfg.hf_model_id)
            .to(self.device)
            .eval()
        )

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a depth map (H, W) float32 in relative units."""
        self._ensure_loaded()
        import torch
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        predicted = outputs.predicted_depth  # (B, h', w') or (B, 1, h', w')
        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)
        depth = torch.nn.functional.interpolate(
            predicted, size=(h, w), mode="bicubic", align_corners=False
        )[0, 0].cpu().numpy().astype(np.float32)
        return depth
