"""Webcam reader. Thin wrapper around cv2.VideoCapture."""
from __future__ import annotations

import numpy as np


class Webcam:
    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        import cv2
        self.cap = cv2.VideoCapture(index)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open webcam index {index}")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> Webcam:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
