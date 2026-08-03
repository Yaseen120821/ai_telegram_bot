"""
app/vision/vision_config.py - Centralized Configuration for Vision AI Subsystem
==============================================================================
Manages image upload size limits, resolution ceilings, supported format sets,
vision model selection, device allocation, and caching settings.
"""

import os
from typing import Set
from dataclasses import dataclass, field


@dataclass
class VisionConfig:
    """Central configuration dataclass for SANA AI Vision Subsystem."""

    max_file_size_bytes: int = 15 * 1024 * 1024       # 15 MB limit
    max_resolution_width: int = 4096
    max_resolution_height: int = 4096
    vision_model_name: str = "florence-2-base"
    ocr_model_name: str = "easyocr"
    device: str = "cpu"
    cache_enabled: bool = True
    log_payloads: bool = True
    supported_formats: Set[str] = field(default_factory=lambda: {
        "png", "jpeg", "jpg", "webp", "bmp", "tiff", "gif", "pdf"
    })

    @classmethod
    def from_env(cls) -> "VisionConfig":
        return cls(
            max_file_size_bytes=int(os.getenv("SANA_VISION_MAX_FILE_SIZE", str(15 * 1024 * 1024))),
            max_resolution_width=int(os.getenv("SANA_VISION_MAX_WIDTH", "4096")),
            max_resolution_height=int(os.getenv("SANA_VISION_MAX_HEIGHT", "4096")),
            vision_model_name=os.getenv("SANA_VISION_MODEL", "florence-2-base"),
            ocr_model_name=os.getenv("SANA_OCR_MODEL", "easyocr"),
            device=os.getenv("SANA_VISION_DEVICE", "cpu"),
            cache_enabled=os.getenv("SANA_VISION_CACHE", "true").lower() == "true",
            log_payloads=os.getenv("SANA_VISION_LOG_PAYLOADS", "true").lower() == "true",
        )


_vision_config_instance: VisionConfig = VisionConfig.from_env()


def get_vision_config() -> VisionConfig:
    """Returns global VisionConfig singleton instance."""
    return _vision_config_instance


def set_vision_config(config: VisionConfig) -> None:
    """Overrides global VisionConfig singleton instance."""
    global _vision_config_instance
    _vision_config_instance = config
