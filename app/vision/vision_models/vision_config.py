"""
app/vision/vision_models/vision_config.py - Vision Model Subsystem Configuration
==================================================================================
Centralized configuration parameters controlling Florence-2 model parameters,
device allocation, generation beams, EasyOCR settings, and caching capacity.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List
from app.vision.vision_models.vision_types import DeviceType


@dataclass
class VisionModelConfig:
    """Central configuration for Florence-2 model loading, device allocation, and OCR."""

    model_name: str = "microsoft/florence-2-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "float16" if torch.cuda.is_available() else "float32"
    max_new_tokens: int = 1024
    num_beams: int = 3
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    use_ocr_gpu: bool = torch.cuda.is_available()
    cache_capacity: int = 500
    timeout_seconds: float = 30.0
    fallback_to_mock: bool = True

    @classmethod
    def from_env(cls) -> "VisionModelConfig":
        default_dev = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            model_name=os.getenv("SANA_FLORENCE_MODEL", "microsoft/florence-2-base"),
            device=os.getenv("SANA_VISION_DEVICE", default_dev),
            torch_dtype=os.getenv("SANA_VISION_DTYPE", "float16" if torch.cuda.is_available() else "float32"),
            max_new_tokens=int(os.getenv("SANA_VISION_MAX_TOKENS", "1024")),
            num_beams=int(os.getenv("SANA_VISION_NUM_BEAMS", "3")),
            use_ocr_gpu=os.getenv("SANA_OCR_GPU", "true").lower() == "true" and torch.cuda.is_available(),
            cache_capacity=int(os.getenv("SANA_VISION_CACHE_SIZE", "500")),
            fallback_to_mock=os.getenv("SANA_VISION_MOCK_FALLBACK", "true").lower() == "true",
        )


_vision_model_config_instance: VisionModelConfig = VisionModelConfig.from_env()


def get_vision_model_config() -> VisionModelConfig:
    """Returns global VisionModelConfig singleton instance."""
    return _vision_model_config_instance


def set_vision_model_config(config: VisionModelConfig) -> None:
    """Overrides global VisionModelConfig singleton instance."""
    global _vision_model_config_instance
    _vision_model_config_instance = config
