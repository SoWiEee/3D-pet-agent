"""Video file reader. Yields BGR frames in order; supports basic seeking."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
