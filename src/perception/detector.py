"""Open-vocabulary detector backed by GroundingDINO via HuggingFace transformers.

Loading is deferred — first call to .predict() pulls weights. This keeps `import` fast
and `--mode sandbox` runnable without any model download.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .schema import Detection2D

if TYPE_CHECKING:
    from ..config import DetectorConfig

log = logging.getLogger("pet_agent.detector")


def _pick_device(requested: str) -> str:
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


class GroundingDinoDetector:
    """Text-conditioned open-vocabulary detector."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        log.info("loading GroundingDINO weights: %s (device=%s)", self.cfg.hf_model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.cfg.hf_model_id)
        # GroundingDINO's deformable attention path mixes fp32 tensors internally;
        # running the whole model in fp16 trips ``grid_sample`` with a dtype mismatch.
        # fp32 is fast enough on a 4070 for snapshot mode (~0.5–1 s/frame).
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.hf_model_id)
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _build_text_prompt(prompts: list[str]) -> str:
        # GroundingDINO expects lowercased phrases joined by ". " with a trailing period.
        cleaned = [p.strip().lower().rstrip(".") for p in prompts if p.strip()]
        return ". ".join(cleaned) + "."

    def predict(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection2D]:
        """Run detection. ``frame_bgr`` is (H, W, 3) BGR as returned by OpenCV."""
        self._ensure_loaded()
        import torch
        from PIL import Image

        text = self._build_text_prompt(prompts)
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # transformers ≥4.51 expects ``threshold`` + ``text_threshold``; older versions used
        # ``box_threshold``. Probe to stay compatible.
        post = self._processor.post_process_grounded_object_detection
        target_sizes = [image.size[::-1]]
        try:
            results = post(
                outputs,
                inputs.input_ids,
                threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = post(
                outputs,
                inputs.input_ids,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=target_sizes,
            )

        r = results[0]
        boxes = r["boxes"].detach().cpu().tolist()
        scores = r["scores"].detach().cpu().tolist()
        labels = r.get("labels") or r.get("text_labels") or []

        out: list[Detection2D] = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            x1, y1, x2, y2 = (float(v) for v in box)
            label_text = label if isinstance(label, str) else str(label)
            label_text = label_text.strip() or "object"
            out.append(
                Detection2D(
                    label=label_text,
                    bbox_xyxy=(x1, y1, x2, y2),
                    detector_confidence=float(score),
                )
            )
        return out
