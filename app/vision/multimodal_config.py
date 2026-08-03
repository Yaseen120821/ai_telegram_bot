"""
app/vision/multimodal_config.py - Multimodal Pipeline Configuration
====================================================================
Central configuration settings governing image upload limits, token budgets,
OCR text truncation thresholds, and multimodal caching policies.
"""

import os
from dataclasses import dataclass


@dataclass
class MultimodalConfig:
    """Configuration parameters for Multimodal Vision & Reasoning Pipeline."""

    max_images_per_query: int = 3
    max_ocr_chars: int = 2000
    max_caption_chars: int = 500
    max_vision_prompt_tokens: int = 1500
    total_prompt_token_budget: int = 4096
    enable_image_caching: bool = True
    log_multimodal_traces: bool = True

    @classmethod
    def from_env(cls) -> "MultimodalConfig":
        return cls(
            max_images_per_query=int(os.getenv("SANA_MULTIMODAL_MAX_IMAGES", "3")),
            max_ocr_chars=int(os.getenv("SANA_MULTIMODAL_MAX_OCR_CHARS", "2000")),
            max_caption_chars=int(os.getenv("SANA_MULTIMODAL_MAX_CAPTION_CHARS", "500")),
            max_vision_prompt_tokens=int(os.getenv("SANA_MULTIMODAL_VISION_TOKENS", "1500")),
            total_prompt_token_budget=int(os.getenv("SANA_MULTIMODAL_TOTAL_TOKENS", "4096")),
            enable_image_caching=os.getenv("SANA_MULTIMODAL_ENABLE_CACHE", "true").lower() == "true",
            log_multimodal_traces=os.getenv("SANA_MULTIMODAL_LOG_TRACES", "true").lower() == "true",
        )


_multimodal_config_instance: MultimodalConfig = MultimodalConfig.from_env()


def get_multimodal_config() -> MultimodalConfig:
    """Returns global MultimodalConfig singleton instance."""
    return _multimodal_config_instance


def set_multimodal_config(config: MultimodalConfig) -> None:
    """Overrides global MultimodalConfig singleton instance."""
    global _multimodal_config_instance
    _multimodal_config_instance = config
