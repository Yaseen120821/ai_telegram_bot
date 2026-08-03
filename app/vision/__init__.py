"""
app/vision package initializer
==============================
Exposes public API for SANA AI Vision AI Subsystem, including Image Processing,
Florence-2 Vision Models, EasyOCR, Specialized Image Analysis, and Multimodal Pipeline.
"""

from app.vision.vision_types import (
    VisionStatus,
    ImageType,
    OCRStatus,
    VisionTask,
    ImageFormat,
    AnalysisMode
)
from app.vision.image_processing.image_types import (
    ImageColorMode,
    ValidationStatus,
    ResizeMode,
    NormalizationMode,
    HashAlgorithm
)
from app.vision.vision_config import (
    VisionConfig,
    get_vision_config,
    set_vision_config
)
from app.vision.vision_schemas import (
    VisionRequest,
    VisionResult,
    ImageMetadata,
    ImageContext,
    CaptionResult,
    OCRResult,
    ImageAnalysis
)
from app.vision.vision_utils import (
    validate_image_file,
    extract_image_metadata,
    normalize_image_path,
    generate_image_id,
    calculate_file_hash
)
from app.vision.vision_manager import VisionManager
from app.vision.multimodal_types import (
    ContextSource,
    ImageRole,
    VisionConfidence,
    PromptSection,
    ReasoningMode,
    ResponseMode
)
from app.vision.multimodal_config import (
    MultimodalConfig,
    get_multimodal_config,
    set_multimodal_config
)
from app.vision.multimodal_models import (
    VisionContext,
    OCRContext,
    AnalysisContextData,
    ImageContextData,
    MultimodalContext,
    IntegratedPrompt,
    ImageConversationReference
)
from app.vision.multimodal_utils import (
    merge_image_contexts,
    estimate_tokens
)

__all__ = [
    # Core Enums
    "VisionStatus",
    "ImageType",
    "OCRStatus",
    "VisionTask",
    "ImageFormat",
    "AnalysisMode",
    "ImageColorMode",
    "ValidationStatus",
    "ResizeMode",
    "NormalizationMode",
    "HashAlgorithm",
    # Config
    "VisionConfig",
    "get_vision_config",
    "set_vision_config",
    # Schemas
    "VisionRequest",
    "VisionResult",
    "ImageMetadata",
    "ImageContext",
    "CaptionResult",
    "OCRResult",
    "ImageAnalysis",
    # Utils
    "validate_image_file",
    "extract_image_metadata",
    "normalize_image_path",
    "generate_image_id",
    "calculate_file_hash",
    # Facade Manager
    "VisionManager",
    # Multimodal Enums
    "ContextSource",
    "ImageRole",
    "VisionConfidence",
    "PromptSection",
    "ReasoningMode",
    "ResponseMode",
    # Multimodal Config
    "MultimodalConfig",
    "get_multimodal_config",
    "set_multimodal_config",
    # Multimodal Schemas
    "VisionContext",
    "OCRContext",
    "AnalysisContextData",
    "ImageContextData",
    "MultimodalContext",
    "IntegratedPrompt",
    "ImageConversationReference",
    # Multimodal Utils
    "merge_image_contexts",
    "estimate_tokens"
]
