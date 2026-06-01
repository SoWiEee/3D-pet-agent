"""Single-image input adapter for snapshot mode."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path | str) -> np.ndarray:
    """Read an image as BGR ndarray (H, W, 3)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2 could not decode {p}")
    return img
