"""
app/vision/image_processing/image_optimizer.py - Image Compression & Optimization Engine
==========================================================================================
Optimizes image file sizes, byte streams, and RAM allocations via adjustable JPEG/WebP compression.
"""

import io
import logging
from PIL import Image as PILImage
from typing import Tuple

logger = logging.getLogger("sana_ai.vision.image_optimizer")


class ImageOptimizer:
    """Optimizes image memory footprint and encodes compressed JPEG/WebP byte payloads."""

    @staticmethod
    def compress_to_jpeg_bytes(pil_img: PILImage.Image, quality: int = 85) -> bytes:
        """Compresses PIL image to JPEG bytes buffer."""
        buf = io.BytesIO()
        rgb = pil_img.convert("RGB")
        rgb.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    @staticmethod
    def compress_to_webp_bytes(pil_img: PILImage.Image, quality: int = 85) -> bytes:
        """Compresses PIL image to WebP bytes buffer."""
        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP", quality=quality, method=4)
        return buf.getvalue()

    @staticmethod
    def optimize_memory_footprint(pil_img: PILImage.Image, max_dimension: int = 2048) -> PILImage.Image:
        """Downsamples large images to reduce RAM usage during Vision Model inference."""
        w, h = pil_img.width, pil_img.height
        if w > max_dimension or h > max_dimension:
            img_copy = pil_img.copy()
            img_copy.thumbnail((max_dimension, max_dimension), PILImage.Resampling.LANCZOS)
            logger.info(f"Optimized memory footprint: downsampled from {w}x{h} to {img_copy.width}x{img_copy.height}.")
            return img_copy
        return pil_img
