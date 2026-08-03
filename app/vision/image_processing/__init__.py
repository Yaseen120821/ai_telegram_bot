"""
app/vision/image_processing package initializer
================================================
Exposes public API and the unified ImageProcessingPipeline facade for image loading,
validation, metadata extraction, quantitative statistics, aspect-ratio resizing,
color mode conversion, pixel tensor normalization, perceptual hashing, and LRU caching.
"""

import time
import logging
from typing import Optional, Tuple

from app.vision.image_processing.image_types import (
    ImageFormat,
    ImageColorMode,
    ValidationStatus,
    ResizeMode,
    NormalizationMode,
    HashAlgorithm
)
from app.vision.image_processing.image_config import (
    ImageProcessingConfig,
    get_image_config,
    set_image_config
)
from app.vision.image_processing.image_models import (
    ImageRequest,
    ImageMetadata,
    ImageStatistics,
    ProcessedImage,
    ValidationResult,
    ImageCacheEntry,
    ImageHash
)
from app.vision.image_processing.image_utils import (
    normalize_path,
    calculate_aspect_ratio,
    pil_to_opencv,
    opencv_to_pil
)
from app.vision.image_processing.image_hash import (
    compute_file_sha256,
    compute_file_md5,
    compute_average_hash,
    compute_difference_hash,
    compute_perceptual_hash,
    compute_all_hashes
)
from app.vision.image_processing.image_loader import ImageLoader
from app.vision.image_processing.image_validator import ImageValidator
from app.vision.image_processing.metadata_extractor import MetadataExtractor
from app.vision.image_processing.image_statistics import ImageStatisticsCalculator
from app.vision.image_processing.image_preprocessor import ImagePreprocessor
from app.vision.image_processing.image_optimizer import ImageOptimizer
from app.vision.image_processing.image_cache import ImageCache

logger = logging.getLogger("sana_ai.vision.image_processing_pipeline")


class ImageProcessingPipeline:
    """
    Main Orchestrator Facade for the Image Processing Subsystem.
    
    Coordinates:
    Loader ──► Validator ──► Metadata ──► Statistics ──► Preprocessor ──► Hashes ──► Cache
    """

    _instance: Optional["ImageProcessingPipeline"] = None

    def __init__(
        self,
        config: Optional[ImageProcessingConfig] = None,
        validator: Optional[ImageValidator] = None,
        cache: Optional[ImageCache] = None
    ):
        self.config = config or get_image_config()
        self.validator = validator or ImageValidator(self.config)
        self.cache = cache or ImageCache.get_instance()

    @classmethod
    def get_instance(cls) -> "ImageProcessingPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_image(self, request: ImageRequest) -> Tuple[Optional[ProcessedImage], Optional[ValidationResult]]:
        """
        Executes end-to-end image loading, validation, preprocessing, statistics calculation,
        and pixel array normalization.
        """
        start_time = time.perf_counter()
        req_id = request.request_id
        img_path = request.image_path

        logger.info(f"ImageProcessingPipeline processing request [{req_id}] for path '{img_path}'.")

        # 1. Load raw image safely
        raw_pil, load_err = ImageLoader.load_image(img_path)
        if load_err or raw_pil is None:
            val_res = ValidationResult(
                status=ValidationStatus.FILE_NOT_FOUND if "not found" in (load_err or "").lower() else ValidationStatus.FAILED,
                is_valid=False,
                error_message=load_err or "Failed to load image."
            )
            return None, val_res

        # 2. Validate image integrity and constraints
        val_res = self.validator.validate(img_path, raw_pil)
        if not val_res.is_valid:
            logger.warning(f"Image validation failed for [{req_id}]: {val_res.error_message}")
            return None, val_res

        # 3. Extract structural metadata
        metadata = MetadataExtractor.extract_metadata(img_path, raw_pil)
        file_hash = metadata.hashes.sha256

        # 4. Check LRU cache hit
        if self.cache.has(file_hash):
            cached_img = self.cache.get(file_hash)
            if cached_img:
                logger.info(f"ImageProcessingPipeline cache hit for SHA-256 '{file_hash[:8]}'.")
                return cached_img, val_res

        # 5. Compute quantitative statistics
        stats = ImageStatisticsCalculator.calculate_statistics(raw_pil)

        # 6. Preprocess & Resize image
        target_size = request.target_size or (self.config.target_width, self.config.target_height)
        resize_mode = request.resize_mode or self.config.resize_mode

        processed_pil = ImagePreprocessor.resize_image(
            pil_img=raw_pil,
            target_size=target_size,
            mode=resize_mode
        )

        # 7. Convert color mode to RGB
        rgb_pil = ImagePreprocessor.convert_to_rgb(processed_pil)

        # 8. Normalize pixel array tensor
        norm_mode = request.normalization_mode or self.config.normalization_mode
        numpy_tensor = ImagePreprocessor.normalize_pixel_tensor(
            pil_img=rgb_pil,
            mode=norm_mode
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        processed_result = ProcessedImage(
            request_id=req_id,
            original_metadata=metadata,
            processed_width=rgb_pil.width,
            processed_height=rgb_pil.height,
            processed_color_mode=ImageColorMode.RGB,
            numpy_tensor=numpy_tensor,
            pillow_image=rgb_pil,
            statistics=stats,
            hashes=metadata.hashes,
            processing_time_ms=elapsed_ms
        )

        # Store in LRU cache
        if file_hash:
            self.cache.put(file_hash, processed_result, metadata)

        logger.info(f"ImageProcessingPipeline finished request [{req_id}] in {elapsed_ms:.2f}ms ({rgb_pil.width}x{rgb_pil.height}).")

        return processed_result, val_res


__all__ = [
    # Enums
    "ImageFormat",
    "ImageColorMode",
    "ValidationStatus",
    "ResizeMode",
    "NormalizationMode",
    "HashAlgorithm",
    # Config
    "ImageProcessingConfig",
    "get_image_config",
    "set_image_config",
    # Models
    "ImageRequest",
    "ImageMetadata",
    "ImageStatistics",
    "ProcessedImage",
    "ValidationResult",
    "ImageCacheEntry",
    "ImageHash",
    # Utils & Hash
    "normalize_path",
    "calculate_aspect_ratio",
    "pil_to_opencv",
    "opencv_to_pil",
    "compute_file_sha256",
    "compute_file_md5",
    "compute_average_hash",
    "compute_difference_hash",
    "compute_perceptual_hash",
    "compute_all_hashes",
    # Subsystem Classes
    "ImageLoader",
    "ImageValidator",
    "MetadataExtractor",
    "ImageStatisticsCalculator",
    "ImagePreprocessor",
    "ImageOptimizer",
    "ImageCache",
    "ImageProcessingPipeline"
]
