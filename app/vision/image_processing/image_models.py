"""
app/vision/image_processing/image_models.py - Data Models for Image Processing
==============================================================================
Defines strongly-typed dataclasses for image metadata, validation results,
statistical metrics, perceptual hashes, processed image tensors, and cache entries.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from PIL import Image as PILImage

from app.vision.image_processing.image_types import (
    ImageFormat,
    ImageColorMode,
    ValidationStatus,
    ResizeMode,
    NormalizationMode
)


@dataclass
class ImageHash:
    """Checksum and perceptual hashes computed for an image."""
    sha256: str = ""
    md5: str = ""
    average_hash: str = ""
    difference_hash: str = ""
    perceptual_hash: str = ""


@dataclass
class ImageMetadata:
    """Detailed technical and structural metadata extracted from an image file."""
    image_id: str = field(default_factory=lambda: f"img_{uuid.uuid4().hex[:12]}")
    file_path: str = ""
    file_name: str = ""
    file_size_bytes: int = 0
    format: ImageFormat = ImageFormat.UNKNOWN
    color_mode: ImageColorMode = ImageColorMode.UNKNOWN
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    dpi: Tuple[int, int] = (72, 72)
    creation_timestamp: float = 0.0
    modified_timestamp: float = 0.0
    hashes: ImageHash = field(default_factory=ImageHash)


@dataclass
class ImageStatistics:
    """Quantitative image metrics computed via NumPy and OpenCV."""
    brightness: float = 0.0            # Mean pixel intensity (0..255)
    contrast: float = 0.0              # Standard deviation of pixel intensities
    sharpness: float = 0.0             # Variance of Laplacian edge filter
    is_grayscale: bool = False
    color_histogram: Dict[str, List[float]] = field(default_factory=dict)
    primary_color_rgb: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class ValidationResult:
    """Diagnostic outcome of image validation checks."""
    status: ValidationStatus = ValidationStatus.PASSED
    is_valid: bool = True
    error_message: str = "Validation successful."
    checked_at: float = field(default_factory=time.time)


@dataclass
class ProcessedImage:
    """Complete preprocessed output ready for Vision Model consumption."""
    request_id: str = ""
    original_metadata: ImageMetadata = field(default_factory=ImageMetadata)
    processed_width: int = 0
    processed_height: int = 0
    processed_color_mode: ImageColorMode = ImageColorMode.RGB
    numpy_tensor: Optional[np.ndarray] = None
    pillow_image: Optional[PILImage.Image] = None
    statistics: ImageStatistics = field(default_factory=ImageStatistics)
    hashes: ImageHash = field(default_factory=ImageHash)
    processing_time_ms: float = 0.0


@dataclass
class ImageCacheEntry:
    """In-memory cache record storing preprocessed image results."""
    key: str
    processed_image: ProcessedImage
    metadata: ImageMetadata
    created_at: float = field(default_factory=time.time)
    access_count: int = 1


@dataclass
class ImageRequest:
    """Request payload passed to ImageProcessingPipeline."""
    image_path: str
    user_id: str = "default_user"
    conversation_id: str = "default_session"
    request_id: str = field(default_factory=lambda: f"ireq_{uuid.uuid4().hex[:12]}")
    target_size: Optional[Tuple[int, int]] = None
    resize_mode: ResizeMode = ResizeMode.LETTERBOX
    normalization_mode: NormalizationMode = NormalizationMode.RESCALE_0_1
    timestamp: float = field(default_factory=time.time)
