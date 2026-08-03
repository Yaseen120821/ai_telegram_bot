"""
app/vision/image_processing/image_utils.py - Utility Routines for Image Processing
===================================================================================
Provides path normalization, aspect ratio math, and bidirectional conversions
between Pillow (PIL) and OpenCV (NumPy) representations.
"""

import os
import cv2
import numpy as np
from PIL import Image as PILImage


def normalize_path(path_str: str) -> str:
    """Normalizes path strings to absolute clean OS format."""
    return os.path.abspath(os.path.normpath(path_str))


def calculate_aspect_ratio(width: int, height: int) -> float:
    """Calculates aspect ratio rounded to 2 decimal places."""
    return round(width / height, 2) if height > 0 else 1.0


def pil_to_opencv(pil_img: PILImage.Image) -> np.ndarray:
    """Converts PIL Image to OpenCV BGR/RGB NumPy array."""
    rgb_arr = np.array(pil_img.convert("RGB"))
    # Convert RGB to BGR for OpenCV
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return bgr_arr


def opencv_to_pil(cv_img: np.ndarray) -> PILImage.Image:
    """Converts OpenCV BGR NumPy array to PIL Image."""
    if len(cv_img.shape) == 2:
        return PILImage.fromarray(cv_img)
    rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb_arr)
