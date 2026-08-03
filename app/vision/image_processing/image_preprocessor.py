"""
app/vision/image_processing/image_preprocessor.py - Image Resizing & Normalization Engine
==========================================================================================
Executes aspect-ratio preserving resizing, letterbox padding, center cropping, color mode conversions,
and pixel array normalization for neural vision model ingestion.
"""

import numpy as np
from PIL import Image as PILImage, ImageOps
from typing import Tuple
from app.vision.image_processing.image_types import ResizeMode, NormalizationMode


class ImagePreprocessor:
    """Handles image aspect-ratio resizing, letterboxing, color mode conversion, and pixel tensor normalization."""

    @staticmethod
    def convert_to_rgb(pil_img: PILImage.Image) -> PILImage.Image:
        """
        Converts any PIL image mode (RGBA, LA, CMYK, P, L) to standard 3-channel RGB.
        Blends transparent alpha backgrounds onto solid white.
        """
        if pil_img.mode == "RGB":
            return pil_img

        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            # Convert to RGBA first
            rgba = pil_img.convert("RGBA")
            # Create white background canvas
            bg = PILImage.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            return bg

        return pil_img.convert("RGB")

    @staticmethod
    def resize_image(
        pil_img: PILImage.Image,
        target_size: Tuple[int, int] = (800, 800),
        mode: ResizeMode = ResizeMode.LETTERBOX
    ) -> PILImage.Image:
        """
        Resizes PIL image according to the specified ResizeMode strategy.
        """
        target_w, target_h = target_size
        img = ImagePreprocessor.convert_to_rgb(pil_img)

        if mode == ResizeMode.STRETCH:
            return img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)

        elif mode == ResizeMode.MAINTAIN_ASPECT_RATIO:
            scale = min(target_w / img.width, target_h / img.height)
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            return img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

        elif mode == ResizeMode.CENTER_CROP:
            return ImageOps.fit(img, (target_w, target_h), method=PILImage.Resampling.LANCZOS, centering=(0.5, 0.5))

        elif mode == ResizeMode.LETTERBOX:
            scale = min(target_w / img.width, target_h / img.height)
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            resized = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            padded = PILImage.new("RGB", (target_w, target_h), (0, 0, 0))
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            padded.paste(resized, (pad_x, pad_y))
            return padded

        return img

    @staticmethod
    def normalize_pixel_tensor(
        pil_img: PILImage.Image,
        mode: NormalizationMode = NormalizationMode.RESCALE_0_1
    ) -> np.ndarray:
        """
        Converts PIL Image to float32 NumPy array and applies normalization.
        """
        rgb_img = ImagePreprocessor.convert_to_rgb(pil_img)
        arr = np.array(rgb_img, dtype=np.float32)

        if mode == NormalizationMode.NONE:
            return arr

        if mode == NormalizationMode.RESCALE_0_1:
            return arr / 255.0

        if mode == NormalizationMode.MEAN_STD_IMAGENET:
            rescaled = arr / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rescaled - mean) / std
            return normalized

        return arr
