"""
app/vision/image_processing/image_config.py - Image Processing Pipeline Configuration
=====================================================================================
Centralized configuration parameters controlling image limits, resize targets,
quality parameters, supported formats, and cache capacities.
"""

import os
from typing import Set, Tuple
from dataclasses import dataclass, field
from app.vision.image_processing.image_types import ResizeMode, NormalizationMode


@dataclass
class ImageProcessingConfig:
    """Central configuration for image loading, validation, resizing, and caching."""

    max_file_size_bytes: int = 15 * 1024 * 1024       # 15 MB limit
    max_resolution_width: int = 4096
    max_resolution_height: int = 4096
    min_resolution_width: int = 32
    min_resolution_height: int = 32
    target_width: int = 800
    target_height: int = 800
    resize_mode: ResizeMode = ResizeMode.LETTERBOX
    normalization_mode: NormalizationMode = NormalizationMode.RESCALE_0_1
    jpeg_quality: int = 85
    cache_max_items: int = 1000
    allow_animated_gif: bool = False
    supported_formats: Set[str] = field(default_factory=lambda: {
        "png", "jpeg", "jpg", "webp", "bmp", "tiff", "gif", "pdf"
    })

    @classmethod
    def from_env(cls) -> "ImageProcessingConfig":
        return cls(
            max_file_size_bytes=int(os.getenv("SANA_IMG_MAX_SIZE", str(15 * 1024 * 1024))),
            max_resolution_width=int(os.getenv("SANA_IMG_MAX_WIDTH", "4096")),
            max_resolution_height=int(os.getenv("SANA_IMG_MAX_HEIGHT", "4096")),
            min_resolution_width=int(os.getenv("SANA_IMG_MIN_WIDTH", "32")),
            min_resolution_height=int(os.getenv("SANA_IMG_MIN_HEIGHT", "32")),
            target_width=int(os.getenv("SANA_IMG_TARGET_WIDTH", "800")),
            target_height=int(os.getenv("SANA_IMG_TARGET_HEIGHT", "800")),
            jpeg_quality=int(os.getenv("SANA_IMG_JPEG_QUALITY", "85")),
            cache_max_items=int(os.getenv("SANA_IMG_CACHE_SIZE", "1000")),
            allow_animated_gif=os.getenv("SANA_IMG_ALLOW_ANIMATED", "false").lower() == "true",
        )


_image_config_instance: ImageProcessingConfig = ImageProcessingConfig.from_env()


def get_image_config() -> ImageProcessingConfig:
    """Returns global ImageProcessingConfig singleton instance."""
    return _image_config_instance


def set_image_config(config: ImageProcessingConfig) -> None:
    """Overrides global ImageProcessingConfig singleton instance."""
    global _image_config_instance
    _image_config_instance = config
