"""
app/vision/image_processing/image_loader.py - Safe Image Loading Subsystem
==========================================================================
Responsible for loading image files safely into memory using Pillow (PIL),
verifying file access and header integrity, while strictly preserving original dimensions and color modes.
"""

import os
import logging
from typing import Tuple, Optional
from PIL import Image as PILImage, UnidentifiedImageError
from app.vision.image_processing.image_utils import normalize_path

logger = logging.getLogger("sana_ai.vision.image_loader")


class ImageLoader:
    """
    Safely opens image files using Pillow (PIL).
    
    Architectural Strict Boundaries:
    - Never resizes or crops the image.
    - Never normalizes pixel values.
    - Never performs OCR or model inference.
    - Only reads and loads raw image data into memory.
    """

    @staticmethod
    def load_image(file_path: str) -> Tuple[Optional[PILImage.Image], Optional[str]]:
        """
        Loads an image file into a PIL Image object safely.
        Returns Tuple[PIL.Image, error_message].
        """
        norm_p = normalize_path(file_path)

        if not os.path.exists(norm_p):
            err = f"Image file not found at path: '{file_path}'"
            logger.error(err)
            return None, err

        if not os.path.isfile(norm_p):
            err = f"Target path is not a regular file: '{file_path}'"
            logger.error(err)
            return None, err

        try:
            # Open PIL image
            img = PILImage.open(norm_p)
            # Load pixel data into memory so file descriptor can be safely closed
            img.load()
            logger.info(f"Successfully loaded image '{os.path.basename(norm_p)}' ({img.width}x{img.height}, {img.mode}, {img.format}).")
            return img, None

        except (UnidentifiedImageError, OSError, ValueError, Exception) as e:
            err = f"Failed to load image '{file_path}': {str(e)}"
            logger.error(err)
            return None, err
