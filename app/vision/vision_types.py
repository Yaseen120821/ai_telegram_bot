"""
app/vision/vision_types.py - Enumeration Types for Vision AI Subsystem
=====================================================================
Defines standard enums for image formats, visual content types, vision tasks,
OCR processing statuses, vision model pipeline states, and analysis modes.
"""

from enum import Enum


class ImageFormat(str, Enum):
    """Supported input image file extensions and MIME formats."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    GIF = "gif"
    PDF_IMAGE = "pdf"
    UNKNOWN = "unknown"


class ImageType(str, Enum):
    """Categorized visual content classification."""
    PHOTO = "photo"
    SCREENSHOT = "screenshot"
    DIAGRAM = "diagram"
    CHART = "chart"
    GRAPH = "graph"
    CODE_SCREENSHOT = "code_screenshot"
    DOCUMENT = "document"
    TABLE = "table"
    HANDWRITING = "handwriting"
    UI_SCREENSHOT = "ui_screenshot"
    UNKNOWN = "unknown"


class VisionTask(str, Enum):
    """Supported computer vision and multimodal vision tasks."""
    CAPTIONING = "captioning"
    DETAILED_DESCRIPTION = "detailed_description"
    OCR = "ocr"
    OBJECT_DETECTION = "object_detection"
    REGION_DESCRIPTION = "region_description"
    VISUAL_QUESTION_ANSWERING = "vqa"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    UI_UNDERSTANDING = "ui_understanding"
    CODE_EXTRACTION = "code_extraction"


class OCRStatus(str, Enum):
    """Processing state for Optical Character Recognition engine."""
    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    NO_TEXT_FOUND = "no_text_found"
    FAILED = "failed"


class VisionStatus(str, Enum):
    """Lifecycle state machine for Vision Manager requests."""
    PENDING = "pending"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNSUPPORTED_FORMAT = "unsupported_format"


class AnalysisMode(str, Enum):
    """Operational depth mode for Vision AI processing."""
    FAST = "fast"               # Quick captioning & basic text extraction
    BALANCED = "balanced"       # Detailed description & standard OCR
    DEEP = "deep"               # Full region detection, document layout & deep OCR
