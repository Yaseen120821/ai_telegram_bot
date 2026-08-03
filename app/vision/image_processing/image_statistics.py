"""
app/vision/image_processing/image_statistics.py - Quantitative Image Statistics Engine
=======================================================================================
Calculates statistical metrics (brightness, contrast, Laplacian sharpness, color histograms,
and primary color distribution) using NumPy and OpenCV array algorithms.
"""

import cv2
import numpy as np
from PIL import Image as PILImage
from typing import Dict, List, Tuple
from app.vision.image_processing.image_models import ImageStatistics
from app.vision.image_processing.image_utils import pil_to_opencv


class ImageStatisticsCalculator:
    """Calculates quantitative visual metrics via NumPy and OpenCV array operations."""

    @staticmethod
    def calculate_statistics(pil_img: PILImage.Image) -> ImageStatistics:
        """Computes ImageStatistics dataclass from a PIL Image."""
        # Convert PIL to BGR OpenCV array
        bgr = pil_to_opencv(pil_img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Brightness: Mean intensity of grayscale image
        brightness = float(np.mean(gray))

        # 2. Contrast: Standard deviation of pixel intensities
        contrast = float(np.std(gray))

        # 3. Sharpness: Variance of Laplacian edge response
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 4. Grayscale Check
        is_grayscale = bool(np.array_equal(bgr[:, :, 0], bgr[:, :, 1]) and np.array_equal(bgr[:, :, 1], bgr[:, :, 2]))

        # 5. Color Histograms (16 bins per channel for efficiency)
        histogram: Dict[str, List[float]] = {}
        colors = ("b", "g", "r")
        for i, col in enumerate(colors):
            hist = cv2.calcHist([bgr], [i], None, [16], [0, 256])
            # Normalize histogram values to sum to 1.0
            total = hist.sum()
            norm_hist = (hist / total).flatten().tolist() if total > 0 else [0.0] * 16
            histogram[col] = [round(v, 4) for v in norm_hist]

        # 6. Primary Dominant Color (RGB)
        mean_b = int(np.mean(bgr[:, :, 0]))
        mean_g = int(np.mean(bgr[:, :, 1]))
        mean_r = int(np.mean(bgr[:, :, 2]))
        primary_color_rgb = (mean_r, mean_g, mean_b)

        return ImageStatistics(
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            sharpness=round(laplacian_var, 2),
            is_grayscale=is_grayscale,
            color_histogram=histogram,
            primary_color_rgb=primary_color_rgb
        )
