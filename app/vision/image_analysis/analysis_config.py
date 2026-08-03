"""
app/vision/image_analysis/analysis_config.py - Image Analysis Subsystem Configuration
====================================================================================
Centralized configuration parameters controlling classification confidence thresholds,
feature analyzer toggles, component bounds, and logging trace levels.
"""

import os
from dataclasses import dataclass


@dataclass
class ImageAnalysisConfig:
    """Central configuration dataclass for Specialized Image Analysis Framework."""

    min_category_confidence: float = 0.60
    ocr_keyword_matching: bool = True
    enable_ui_analysis: bool = True
    enable_code_analysis: bool = True
    enable_chart_analysis: bool = True
    max_detected_components: int = 20
    log_analysis_traces: bool = True

    @classmethod
    def from_env(cls) -> "ImageAnalysisConfig":
        return cls(
            min_category_confidence=float(os.getenv("SANA_ANALYSIS_MIN_CONF", "0.60")),
            ocr_keyword_matching=os.getenv("SANA_ANALYSIS_OCR_KEYWORDS", "true").lower() == "true",
            enable_ui_analysis=os.getenv("SANA_ANALYSIS_UI", "true").lower() == "true",
            enable_code_analysis=os.getenv("SANA_ANALYSIS_CODE", "true").lower() == "true",
            enable_chart_analysis=os.getenv("SANA_ANALYSIS_CHART", "true").lower() == "true",
            max_detected_components=int(os.getenv("SANA_ANALYSIS_MAX_COMPONENTS", "20")),
            log_analysis_traces=os.getenv("SANA_ANALYSIS_LOG_TRACES", "true").lower() == "true",
        )


_analysis_config_instance: ImageAnalysisConfig = ImageAnalysisConfig.from_env()


def get_analysis_config() -> ImageAnalysisConfig:
    """Returns global ImageAnalysisConfig singleton instance."""
    return _analysis_config_instance


def set_analysis_config(config: ImageAnalysisConfig) -> None:
    """Overrides global ImageAnalysisConfig singleton instance."""
    global _analysis_config_instance
    _analysis_config_instance = config
