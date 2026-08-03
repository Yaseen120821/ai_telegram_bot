"""
app/vision/image_processing/image_types.py - Type Enumerations for Image Processing
===================================================================================
Defines standard enums for image formats, color modes, validation statuses,
resizing algorithms, normalization modes, and hashing algorithms.
"""

from enum import Enum


class ImageFormat(str, Enum):
    """Supported image file extensions and formats."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    GIF = "gif"
    PDF_IMAGE = "pdf"
    UNKNOWN = "unknown"


class ImageColorMode(str, Enum):
    """Image color channels and pixel representations."""
    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "L"
    CMYK = "CMYK"
    BGR = "BGR"
    LAB = "LAB"
    HSV = "HSV"
    UNKNOWN = "UNKNOWN"


class ValidationStatus(str, Enum):
    """Outcome status for image validation checks."""
    PASSED = "passed"
    FAILED = "failed"
    INVALID_FORMAT = "invalid_format"
    FILE_NOT_FOUND = "file_not_found"
    FILE_TOO_LARGE = "file_too_large"
    RESOLUTION_TOO_HIGH = "resolution_too_high"
    RESOLUTION_TOO_LOW = "resolution_too_low"
    CORRUPTED = "corrupted"
    ANIMATED_GIF_BLOCKED = "animated_gif_blocked"


class ResizeMode(str, Enum):
    """Strategy for image dimension resizing."""
    MAINTAIN_ASPECT_RATIO = "maintain_aspect_ratio"
    LETTERBOX = "letterbox"               # Pad borders to exact target box
    CENTER_CROP = "center_crop"           # Crop edges to fit target box
    STRETCH = "stretch"                   # Direct non-proportional scaling


class NormalizationMode(str, Enum):
    """Pixel tensor normalization strategy for vision neural networks."""
    RESCALE_0_1 = "rescale_0_1"           # Scale pixel values uint8 [0..255] -> float32 [0.0..1.0]
    MEAN_STD_IMAGENET = "mean_std_imagenet" # Normalize with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    NONE = "none"


class HashAlgorithm(str, Enum):
    """Algorithm choices for image hash computation."""
    SHA256 = "sha256"
    MD5 = "md5"
    AVERAGE_HASH = "average_hash"
    DIFFERENCE_HASH = "difference_hash"
    PERCEPTUAL_HASH = "perceptual_hash"
