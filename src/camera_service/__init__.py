"""Frame capture (webcam, video, single image). Returns BGR ndarrays consistent with OpenCV."""

from .image_reader import read_image
from .video_reader import VideoReader
from .webcam import Webcam

__all__ = ["read_image", "VideoReader", "Webcam"]
