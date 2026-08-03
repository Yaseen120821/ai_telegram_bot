"""
app/vision/vision_models/vision_types.py - Vision Model & Task Enums
====================================================================
Defines standard enumerations for Florence-2 vision tasks, model states,
detection spatial representation types, caption modes, OCR modes, and device targets.
"""

from enum import Enum


class VisionTask(str, Enum):
    """Supported Florence-2 vision task prompt tokens."""
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    OBJECT_DETECTION = "<OD>"
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    OCR = "<OCR>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"
    VQA = "<VQA>"


class VisionStatus(str, Enum):
    """Status state machine for vision model inference calls."""
    PENDING = "pending"
    MODEL_LOADING = "model_loading"
    INFERRING = "inferring"
    COMPLETED = "completed"
    FAILED = "failed"
    MODEL_UNAVAILABLE = "model_unavailable"


class DetectionType(str, Enum):
    """Spatial representation for detected object regions."""
    BOUNDING_BOX = "bounding_box"
    POLYGON = "polygon"
    POINT = "point"


class CaptionMode(str, Enum):
    """Verbosity mode for visual scene description."""
    BRIEF = "brief"
    DETAILED = "detailed"
    DENSE = "dense"


class OCRMode(str, Enum):
    """Operational mode for text recognition engine."""
    FAST = "fast"
    ACCURATE = "accurate"
    LAYOUT_AWARE = "layout_aware"


class OCRStatus(str, Enum):
    """Status state for text recognition engine."""
    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    NO_TEXT_FOUND = "no_text_found"
    FAILED = "failed"


class DeviceType(str, Enum):
    """Execution hardware device target."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
