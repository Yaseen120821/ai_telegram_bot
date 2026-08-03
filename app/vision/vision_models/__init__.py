"""
app/vision/vision_models package initializer
=============================================
Exposes public API for Florence-2 Base vision models, EasyOCR engine,
VisionModelManager, VisionInferenceEngine, VisionCache, and master VisionModelPipeline.
"""

from app.vision.vision_models.vision_types import (
    VisionTask,
    VisionStatus,
    DetectionType,
    CaptionMode,
    OCRMode,
    OCRStatus,
    DeviceType
)
from app.vision.vision_models.vision_config import (
    VisionModelConfig,
    get_vision_model_config,
    set_vision_model_config
)
from app.vision.vision_models.vision_schemas import (
    VisionRequest,
    VisionResponse,
    CaptionResult,
    ObjectDetectionResult,
    OCRTextResult,
    ImageContext,
    VisionStatistics
)
from app.vision.vision_models.vision_utils import (
    clean_ocr_text,
    parse_bounding_boxes,
    format_vision_context_block
)
from app.vision.vision_models.florence2_model import Florence2Model
from app.vision.vision_models.vision_cache import VisionCache
from app.vision.vision_models.vision_statistics import VisionStatisticsTracker
from app.vision.vision_models.vision_model_manager import VisionModelManager
from app.vision.vision_models.vision_inference import VisionInferenceEngine
from app.vision.vision_models.vision_factory import VisionFactory
from app.vision.vision_models.vision_pipeline import VisionModelPipeline

__all__ = [
    # Enums
    "VisionTask",
    "VisionStatus",
    "DetectionType",
    "CaptionMode",
    "OCRMode",
    "OCRStatus",
    "DeviceType",
    # Config
    "VisionModelConfig",
    "get_vision_model_config",
    "set_vision_model_config",
    # Schemas
    "VisionRequest",
    "VisionResponse",
    "CaptionResult",
    "ObjectDetectionResult",
    "OCRTextResult",
    "ImageContext",
    "VisionStatistics",
    # Utils
    "clean_ocr_text",
    "parse_bounding_boxes",
    "format_vision_context_block",
    # Core Components
    "Florence2Model",
    "VisionCache",
    "VisionStatisticsTracker",
    "VisionModelManager",
    "VisionInferenceEngine",
    "VisionFactory",
    "VisionModelPipeline"
]
