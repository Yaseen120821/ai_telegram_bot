"""
app/vision/image_processing/metadata_extractor.py - Image Metadata Extractor
=============================================================================
Extracts technical and structural metadata from image files and loaded PIL images,
including file size, dimensions, color modes, DPI, aspect ratios, timestamps, and checksum hashes.
"""

import os
import logging
from PIL import Image as PILImage

from app.vision.image_processing.image_types import ImageFormat, ImageColorMode
from app.vision.image_processing.image_models import ImageMetadata
from app.vision.image_processing.image_utils import normalize_path, calculate_aspect_ratio
from app.vision.image_processing.image_hash import compute_all_hashes

logger = logging.getLogger("sana_ai.vision.metadata_extractor")


class MetadataExtractor:
    """Extracts structural metadata and checksum hashes from image assets."""

    @staticmethod
    def extract_metadata(file_path: str, pil_img: PILImage.Image) -> ImageMetadata:
        """Extracts complete ImageMetadata dataclass."""
        norm_p = normalize_path(file_path)
        file_name = os.path.basename(norm_p)
        file_size = os.path.getsize(norm_p) if os.path.exists(norm_p) else 0

        ext = os.path.splitext(norm_p)[1].lstrip(".").lower()
        try:
            img_fmt = ImageFormat(ext)
        except ValueError:
            img_fmt = ImageFormat.UNKNOWN

        try:
            color_mode = ImageColorMode(pil_img.mode)
        except ValueError:
            color_mode = ImageColorMode.UNKNOWN

        w, h = pil_img.width, pil_img.height
        aspect_ratio = calculate_aspect_ratio(w, h)

        dpi_val = (72, 72)
        if "dpi" in pil_img.info and isinstance(pil_img.info["dpi"], tuple):
            dpi_val = (int(pil_img.info["dpi"][0]), int(pil_img.info["dpi"][1]))

        c_time = os.path.getctime(norm_p) if os.path.exists(norm_p) else 0.0
        m_time = os.path.getmtime(norm_p) if os.path.exists(norm_p) else 0.0

        hashes = compute_all_hashes(norm_p, pil_img)

        return ImageMetadata(
            file_path=norm_p,
            file_name=file_name,
            file_size_bytes=file_size,
            format=img_fmt,
            color_mode=color_mode,
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            dpi=dpi_val,
            creation_timestamp=c_time,
            modified_timestamp=m_time,
            hashes=hashes
        )
