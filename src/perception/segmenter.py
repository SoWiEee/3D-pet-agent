"""Promptable segmentation backed by SAM (facebook/sam-vit-base) via transformers.

Takes BGR frames + detector boxes; returns binary masks aligned to the frame.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import SegmenterConfig

log = logging.getLogger("pet_agent.segmenter")


def _pick_device(requested: str) -> str:
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


class SamSegmenter:
    def __init__(self, cfg: SegmenterConfig) -> None:
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import SamModel, SamProcessor

        log.info("loading SAM weights: %s (device=%s)", self.cfg.hf_model_id, self.device)
        self._processor = SamProcessor.from_pretrained(self.cfg.hf_model_id)
        self._model = SamModel.from_pretrained(self.cfg.hf_model_id).to(self.device).eval()

    def predict(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> list[np.ndarray]:
        """Return one binary mask (H, W) bool per box. Empty input → empty output."""
        if not boxes_xyxy:
            return []
        self._ensure_loaded()
        import torch
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)
        # SamProcessor expects input_boxes shape (batch=1, n_boxes, 4) in image coords.
        input_boxes = [[list(b) for b in boxes_xyxy]]
        inputs = self._processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        # masks: tensor (n_boxes, num_mask_per_box, H, W). Take best per box (max IoU).
        scores = outputs.iou_scores.cpu().numpy()[0]  # (n_boxes, num_mask_per_box)
        result: list[np.ndarray] = []
        for i in range(masks.shape[0]):
            best = int(scores[i].argmax())
            mask = masks[i, best].numpy().astype(bool)
            result.append(mask)
        return result
