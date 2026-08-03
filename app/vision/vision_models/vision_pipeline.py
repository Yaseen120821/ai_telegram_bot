"""
app/vision/vision_models/vision_pipeline.py - Master Multimodal Vision Orchestrator
==================================================================================
Coordinates image pre-processing, Florence-2 visual scene understanding,
EasyOCR text extraction, caching, and prompt context formatting into structured VisionResponse objects.
"""

import time
import os
import logging
from typing import Optional

from app.vision.image_processing import ImageProcessingPipeline, ImageRequest, ResizeMode
from app.vision.vision_models.vision_types import VisionStatus, VisionTask
from app.vision.vision_models.vision_config import get_vision_model_config, VisionModelConfig
from app.vision.vision_models.vision_schemas import (
    VisionRequest,
    VisionResponse,
    ImageContext,
    CaptionResult,
    OCRTextResult
)
from app.vision.vision_models.vision_utils import format_vision_context_block
from app.vision.vision_models.vision_inference import VisionInferenceEngine
from app.vision.vision_models.vision_cache import VisionCache
from app.vision.vision_models.vision_statistics import VisionStatisticsTracker

logger = logging.getLogger("sana_ai.vision.models.pipeline")


class VisionModelPipeline:
    """Master facade orchestrating Image Processing, Florence-2, EasyOCR, and Context Assembly."""

    _instance: Optional["VisionModelPipeline"] = None

    def __init__(
        self,
        config: Optional[VisionModelConfig] = None,
        inference_engine: Optional[VisionInferenceEngine] = None,
        image_pipeline: Optional[ImageProcessingPipeline] = None,
        cache: Optional[VisionCache] = None,
        tracker: Optional[VisionStatisticsTracker] = None
    ):
        self.config = config or get_vision_model_config()
        self.inference_engine = inference_engine or VisionInferenceEngine(self.config)
        self.image_pipeline = image_pipeline or ImageProcessingPipeline.get_instance()
        self.cache = cache or VisionCache.get_instance()
        self.tracker = tracker or VisionStatisticsTracker.get_instance()

    @classmethod
    def get_instance(cls) -> "VisionModelPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_vision_request(self, request: VisionRequest) -> VisionResponse:
        """
        Main multimodal inference pipeline entry point.
        
        Workflow:
        1. Preprocess image via ImageProcessingPipeline (scaling, RGB conversion, SHA256 hash).
        2. Check VisionCache for hit.
        3. Execute Florence-2 captioning & object detection.
        4. Execute EasyOCR text extraction.
        5. Merge results into structured ImageContext.
        6. Construct formatted prompt context block.
        7. Record telemetry statistics & update cache.
        """
        start_time = time.perf_counter()
        req_id = request.request_id
        img_path = request.image_path

        logger.info(f"VisionModelPipeline started request [{req_id}] for file '{img_path}'.")

        # 1. Preprocess image asset
        img_req = ImageRequest(image_path=img_path, resize_mode=ResizeMode.LETTERBOX)
        processed_img, val_res = self.image_pipeline.process_image(img_req)

        if not val_res.is_valid or processed_img is None or processed_img.pillow_image is None:
            logger.warning(f"Image processing failed for [{req_id}]: {val_res.error_message if val_res else 'Null image'}")
            self.tracker.record_request(success=False, latency_ms=0.0)
            return VisionResponse(
                request_id=req_id,
                status=VisionStatus.FAILED,
                error_message=val_res.error_message if val_res else "Failed to load/process image file."
            )

        file_hash = processed_img.hashes.sha256
        cache_key = f"{file_hash}_{request.task.value}"

        # 2. Check Cache
        cached_res = self.cache.get(cache_key)
        if cached_res:
            logger.info(f"VisionModelPipeline cache hit for key '{cache_key[:12]}'.")
            self.tracker.record_request(success=True, latency_ms=cached_res.inference_time_ms, cache_hit=True)
            return cached_res

        # 3. Execute Florence-2 Visual Tasks
        pil_img = processed_img.pillow_image
        caption_result = self.inference_engine.execute_captioning(pil_img)
        objects_detected = self.inference_engine.execute_object_detection(pil_img)

        # 4. Execute EasyOCR Text Extraction
        ocr_result = self.inference_engine.execute_ocr(pil_img)

        # 5. Format Prompt Context Block
        formatted_prompt_block = format_vision_context_block(
            caption_short=caption_result.short_caption,
            caption_detailed=caption_result.detailed_caption,
            ocr_text=ocr_result.raw_text,
            objects=objects_detected,
            file_name=os.path.basename(img_path)
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 6. Assemble ImageContext
        image_context = ImageContext(
            request_id=req_id,
            file_path=img_path,
            caption=caption_result,
            objects=objects_detected,
            ocr=ocr_result,
            image_type="screenshot" if "screen" in img_path.lower() else "photo",
            formatted_prompt_block=formatted_prompt_block,
            inference_time_ms=elapsed_ms
        )

        response = VisionResponse(
            request_id=req_id,
            status=VisionStatus.COMPLETED,
            context=image_context,
            inference_time_ms=elapsed_ms
        )

        # 7. Record Telemetry & Cache
        self.tracker.record_request(success=True, latency_ms=elapsed_ms)
        self.cache.put(cache_key, response)

        logger.info(f"VisionModelPipeline finished request [{req_id}] in {elapsed_ms:.2f}ms.")
        return response
