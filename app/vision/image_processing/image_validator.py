"""
app/vision/image_processing/image_validator.py - Comprehensive Image Validation Engine
=======================================================================================
Performs multi-point diagnostic validation on image files, including file extension checks,
MIME/header integrity verification, dimension limits, corruption tests, and animated frame blocking.
"""

import os
import logging
from typing import Optional
from PIL import Image as PILImage, UnidentifiedImageError

from app.vision.image_processing.image_types import ValidationStatus, ImageFormat
from app.vision.image_processing.image_config import get_image_config, ImageProcessingConfig
from app.vision.image_processing.image_models import ValidationResult
from app.vision.image_processing.image_utils import normalize_path

logger = logging.getLogger("sana_ai.vision.image_validator")


class ImageValidator:
    """Diagnostic validator verifying image file integrity, extensions, size ceilings, and corruption."""

    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        self.config = config or get_image_config()

    def validate(self, file_path: str, pil_img: Optional[PILImage.Image] = None) -> ValidationResult:
        """
        Executes complete validation checks on file path and loaded PIL Image.
        Returns ValidationResult dataclass.
        """
        norm_p = normalize_path(file_path)

        # 1. File existence check
        if not os.path.exists(norm_p):
            return ValidationResult(
                status=ValidationStatus.FILE_NOT_FOUND,
                is_valid=False,
                error_message=f"File not found: '{file_path}'"
            )

        # 2. File size ceiling check
        file_size = os.path.getsize(norm_p)
        if file_size > self.config.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            limit_mb = self.config.max_file_size_bytes / (1024 * 1024)
            return ValidationResult(
                status=ValidationStatus.FILE_TOO_LARGE,
                is_valid=False,
                error_message=f"Image file size ({size_mb:.2f}MB) exceeds maximum limit ({limit_mb:.1f}MB)."
            )

        # 3. File extension check
        ext = os.path.splitext(norm_p)[1].lstrip(".").lower()
        if ext not in self.config.supported_formats:
            return ValidationResult(
                status=ValidationStatus.INVALID_FORMAT,
                is_valid=False,
                error_message=f"Unsupported format extension '.{ext}'. Supported: {sorted(self.config.supported_formats)}"
            )

        # 4. Open PIL image if not provided
        img_to_verify = pil_img
        opened_internally = False
        if img_to_verify is None:
            try:
                img_to_verify = PILImage.open(norm_p)
                opened_internally = True
            except Exception as e:
                return ValidationResult(
                    status=ValidationStatus.CORRUPTED,
                    is_valid=False,
                    error_message=f"Corrupted image header or unreadable file: {str(e)}"
                )

        # 5. Dimension checks
        w, h = img_to_verify.width, img_to_verify.height

        if w > self.config.max_resolution_width or h > self.config.max_resolution_height:
            if opened_internally:
                img_to_verify.close()
            return ValidationResult(
                status=ValidationStatus.RESOLUTION_TOO_HIGH,
                is_valid=False,
                error_message=f"Image resolution ({w}x{h}) exceeds ceiling ({self.config.max_resolution_width}x{self.config.max_resolution_height})."
            )

        if w < self.config.min_resolution_width or h < self.config.min_resolution_height:
            if opened_internally:
                img_to_verify.close()
            return ValidationResult(
                status=ValidationStatus.RESOLUTION_TOO_LOW,
                is_valid=False,
                error_message=f"Image resolution ({w}x{h}) below minimum requirements ({self.config.min_resolution_width}x{self.config.min_resolution_height})."
            )

        # 6. Animated GIF check
        if getattr(img_to_verify, "is_animated", False) and not self.config.allow_animated_gif:
            if opened_internally:
                img_to_verify.close()
            return ValidationResult(
                status=ValidationStatus.ANIMATED_GIF_BLOCKED,
                is_valid=False,
                error_message="Animated GIF images are not supported for visual model analysis."
            )

        if opened_internally:
            img_to_verify.close()

        return ValidationResult(
            status=ValidationStatus.PASSED,
            is_valid=True,
            error_message="Validation passed cleanly."
        )
