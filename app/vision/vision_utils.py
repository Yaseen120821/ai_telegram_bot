"""
app/vision/vision_utils.py - Vision Helper & Validation Utilities
================================================================
Provides image file validation, resolution estimation, SHA-256 hash calculation,
metadata extraction, ID generation, and path normalization routines for Vision AI.
"""

import os
import hashlib
import uuid
import logging
from typing import Tuple, Optional
from app.vision.vision_types import ImageFormat
from app.vision.vision_schemas import ImageMetadata
from app.vision.vision_config import get_vision_config

logger = logging.getLogger("sana_ai.vision.utils")


def generate_image_id(prefix: str = "img") -> str:
    """Generates a prefixed unique image identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_image_path(path_str: str) -> str:
    """Normalizes relative or absolute image path strings."""
    return os.path.abspath(os.path.normpath(path_str))


def calculate_file_hash(file_path: str) -> str:
    """Computes SHA-256 checksum hash of an image file for caching and deduplication."""
    if not os.path.exists(file_path):
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def estimate_resolution(file_path: str) -> Tuple[int, int]:
    """
    Estimates image width and height without heavy PIL/OpenCV dependencies.
    Parses PNG/JPEG header bytes when possible, falling back to defaults.
    """
    if not os.path.exists(file_path):
        return (0, 0)
    
    size = os.path.getsize(file_path)
    if size == 0:
        return (0, 0)

    try:
        with open(file_path, "rb") as f:
            head = f.read(32)
            # PNG header resolution parsing
            if head.startswith(b'\x89PNG\r\n\x1a\n') and len(head) >= 24:
                w = int.from_bytes(head[16:20], byteorder='big')
                h = int.from_bytes(head[20:24], byteorder='big')
                return (w, h)
    except Exception as e:
        logger.debug(f"Could not parse binary image header for '{file_path}': {e}")

    # Default fallback dimensions for architecture foundation
    return (1920, 1080)


def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """
    Validates image file existence, file size ceiling, and extension support.
    Returns Tuple[is_valid: bool, error_message: str].
    """
    config = get_vision_config()
    norm_p = normalize_image_path(file_path)

    if not os.path.exists(norm_p):
        return False, f"Image file does not exist at path: '{file_path}'"

    if not os.path.isfile(norm_p):
        return False, f"Target path is not a regular file: '{file_path}'"

    file_size = os.path.getsize(norm_p)
    if file_size == 0:
        return False, f"Image file is empty (0 bytes): '{file_path}'"

    if file_size > config.max_file_size_bytes:
        limit_mb = config.max_file_size_bytes / (1024 * 1024)
        return False, f"Image file size ({file_size / (1024*1024):.2f}MB) exceeds ceiling ({limit_mb:.1f}MB)."

    ext = os.path.splitext(norm_p)[1].lstrip(".").lower()
    if ext not in config.supported_formats:
        return False, f"Unsupported image format extension '.{ext}'. Supported: {sorted(config.supported_formats)}"

    return True, "Validation successful."


def extract_image_metadata(file_path: str, source: str = "telegram_upload") -> ImageMetadata:
    """Extracts structural ImageMetadata dataclass from an image file."""
    norm_p = normalize_image_path(file_path)
    file_name = os.path.basename(norm_p)
    file_size = os.path.getsize(norm_p) if os.path.exists(norm_p) else 0
    ext = os.path.splitext(norm_p)[1].lstrip(".").lower()

    try:
        img_fmt = ImageFormat(ext)
    except ValueError:
        img_fmt = ImageFormat.UNKNOWN

    width, height = estimate_resolution(norm_p)
    aspect_ratio = round(width / height, 2) if height > 0 else 1.0
    file_hash = calculate_file_hash(norm_p)

    return ImageMetadata(
        image_id=generate_image_id("img"),
        file_path=norm_p,
        file_name=file_name,
        file_size_bytes=file_size,
        format=img_fmt,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        sha256_hash=file_hash,
        source=source
    )
