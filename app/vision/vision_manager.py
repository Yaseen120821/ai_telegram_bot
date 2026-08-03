r"""
app/vision/vision_manager.py - Main Facade for Vision AI Subsystem
==================================================================
Coordinates image validation, structural metadata extraction, visual category classification,
Florence-2 neural model captioning, EasyOCR text extraction, specialized domain analysis,
and formatted context block generation for PromptBuilder injection.
"""

import time
import os
import logging
from typing import Optional, Dict, Any, List
from app.vision.vision_types import VisionStatus, ImageType, OCRStatus, VisionTask, ImageFormat
from app.vision.vision_config import get_vision_config, VisionConfig
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
    normalize_image_path
)

logger = logging.getLogger("sana_ai.vision.manager")


class VisionManager:
    """
    Central Orchestrator Facade for SANA AI Vision System.
    """

    _instance: Optional["VisionManager"] = None

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or get_vision_config()
        self.cache: Dict[str, ImageContext] = {}

    @classmethod
    def get_instance(cls) -> "VisionManager":
        """Returns global VisionManager singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_image(self, request: VisionRequest) -> VisionResult:
        """
        Main entry point for processing an image request.
        
        Workflow:
        1. Validate image file existence, size, and format.
        2. Extract structural ImageMetadata.
        3. Check Cache.
        4. Execute Florence-2 captioning & object detection + EasyOCR via VisionModelPipeline.
        5. Execute specialized domain analysis via AnalysisManager.
        6. Format prompt context block and return standardized VisionResult.
        """
        start_time = time.perf_counter()
        req_id = request.request_id
        img_path = request.image_path

        logger.info(f"👁️ VisionManager started processing request [{req_id}] for image '{img_path}'.")

        # 1. Validate Image File
        is_valid, err_msg = validate_image_file(img_path)
        if not is_valid:
            logger.warning(f"Vision validation failed for '{img_path}': {err_msg}")
            return VisionResult(
                request_id=req_id,
                status=VisionStatus.UNSUPPORTED_FORMAT if "Unsupported" in err_msg else VisionStatus.FAILED,
                error_message=err_msg
            )

        # 2. Extract Structural Metadata
        metadata = extract_image_metadata(img_path)

        # 3. Check Cache
        if self.config.cache_enabled and metadata.sha256_hash in self.cache:
            logger.info(f"⚡ Retrieved ImageContext from cache for hash '{metadata.sha256_hash[:8]}'.")
            cached_ctx = self.cache[metadata.sha256_hash]
            return VisionResult(
                request_id=req_id,
                status=VisionStatus.COMPLETED,
                image_context=cached_ctx
            )

        # 4. Execute Florence-2 & EasyOCR via VisionModelPipeline
        caption_res = CaptionResult(short_caption=f"Image {metadata.file_name} attached.", detailed_caption="", confidence=0.9)
        ocr_res = OCRResult(status=OCRStatus.NO_TEXT_FOUND, raw_text="", detected_languages=["en"], line_count=0, word_count=0)
        objects_detected: List[str] = []

        try:
            from app.vision.vision_models.vision_pipeline import VisionModelPipeline, VisionRequest as VReq
            v_pipe = VisionModelPipeline.get_instance()
            v_resp = v_pipe.process_vision_request(VReq(image_path=img_path))

            if v_resp and v_resp.context:
                if v_resp.context.caption:
                    caption_res = CaptionResult(
                        short_caption=v_resp.context.caption.short_caption or f"Image {metadata.file_name} attached.",
                        detailed_caption=v_resp.context.caption.detailed_caption or "",
                        confidence=v_resp.context.caption.confidence or 0.95
                    )
                if v_resp.context.ocr:
                    ocr_res = OCRResult(
                        status=OCRStatus.SUCCESS if v_resp.context.ocr.raw_text else OCRStatus.NO_TEXT_FOUND,
                        raw_text=v_resp.context.ocr.raw_text or "",
                        detected_languages=["en"],
                        line_count=v_resp.context.ocr.line_count or 0,
                        word_count=v_resp.context.ocr.word_count or 0
                    )
                if v_resp.context.objects:
                    objects_detected = [obj.label for obj in v_resp.context.objects if hasattr(obj, 'label')]
        except Exception as v_err:
            logger.warning(f"VisionModelPipeline execution notice for '{img_path}' ({v_err}). Falling back to OCREngine...")
            try:
                from app.vision.ocr.ocr_engine import OCREngine
                ocr_out = OCREngine.get_instance().extract_text(img_path)
                ocr_res = OCRResult(
                    status=ocr_out.status,
                    raw_text=ocr_out.raw_text,
                    detected_languages=ocr_out.detected_languages,
                    line_count=ocr_out.line_count,
                    word_count=ocr_out.word_count
                )
            except Exception as ocr_err:
                logger.error(f"Fallback OCREngine error for '{img_path}': {ocr_err}")

        # 5. Execute Domain Analysis via AnalysisManager
        image_type = self._classify_image_type(metadata, caption=caption_res, ocr=ocr_res)
        analysis_res = ImageAnalysis(detected_objects=objects_detected, scene_type=image_type.value, summary="")

        try:
            from app.vision.image_analysis import AnalysisManager, AnalysisRequest as AReq
            a_mgr = AnalysisManager.get_instance()
            a_req = AReq(
                image_path=img_path,
                caption=caption_res.detailed_caption or caption_res.short_caption,
                ocr_text=ocr_res.raw_text,
                objects=objects_detected
            )
            a_ctx = a_mgr.analyze_image(a_req)
            if a_ctx and a_ctx.result:
                summary_val = getattr(a_ctx.result, 'domain_summary', None) or getattr(a_ctx.result, 'summary', '')
                analysis_res = ImageAnalysis(
                    detected_objects=objects_detected,
                    scene_type=a_ctx.category.value if hasattr(a_ctx, 'category') else image_type.value,
                    summary=summary_val or f"Visual asset categorized as {image_type.value}."
                )
        except Exception as a_err:
            logger.warning(f"AnalysisManager notice for '{img_path}': {a_err}")

        # 6. Format Prompt Context Block
        formatted_prompt_block = self.format_image_context_block(
            metadata=metadata,
            image_type=image_type,
            caption=caption_res,
            ocr=ocr_res,
            analysis=analysis_res
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        image_ctx = ImageContext(
            request_id=req_id,
            metadata=metadata,
            image_type=image_type,
            caption_result=caption_res,
            ocr_result=ocr_res,
            analysis=analysis_res,
            formatted_prompt_block=formatted_prompt_block,
            processing_time_ms=elapsed_ms
        )

        # Cache result
        if self.config.cache_enabled and metadata.sha256_hash:
            self.cache[metadata.sha256_hash] = image_ctx

        logger.info(f"✅ VisionManager completed request [{req_id}] in {elapsed_ms:.2f}ms [Type: {image_type.value}].")

        return VisionResult(
            request_id=req_id,
            status=VisionStatus.COMPLETED,
            image_context=image_ctx
        )

    def process_multimodal_images(self, image_paths: List[str]) -> str:
        """
        Multimodal facade method called by IntelligentPipeline & Telegram handlers.
        Executes Vision AI processing for each image path and returns merged prompt context blocks.
        """
        if not image_paths:
            return ""

        formatted_blocks: List[str] = []

        for path in image_paths:
            try:
                from app.vision.vision_schemas import VisionRequest
                req = VisionRequest(image_path=path)
                res = self.process_image(req)

                if res.status == VisionStatus.COMPLETED and res.image_context:
                    formatted_blocks.append(res.image_context.formatted_prompt_block)
                else:
                    formatted_blocks.append(f"=== ATTACHED IMAGE: {os.path.basename(path)} ===\nNotice: Image attached.\n=== END ATTACHED IMAGE ===")
            except Exception as e:
                logger.error(f"Failed multimodal processing for '{path}': {e}")
                formatted_blocks.append(f"=== ATTACHED IMAGE: {os.path.basename(path)} ===\nNotice: Image attached ({e}).\n=== END ATTACHED IMAGE ===")

        return "\n\n".join(formatted_blocks)

    def format_image_context_block(
        self,
        metadata: ImageMetadata,
        image_type: ImageType,
        caption: CaptionResult,
        ocr: OCRResult,
        analysis: ImageAnalysis
    ) -> str:
        """Constructs standardized text block representation for PromptBuilder injection."""
        lines = [
            f"=== ATTACHED IMAGE CONTEXT: {metadata.file_name} ===",
            f"• File Name: {metadata.file_name} ({metadata.width}x{metadata.height}, {metadata.format.value.upper()})",
            f"• Image Category: {image_type.value.upper()}",
            f"• Visual Description: {caption.short_caption or 'Image attached by user.'}"
        ]

        if caption.detailed_caption and caption.detailed_caption != caption.short_caption:
            lines.append(f"• Detailed Description: {caption.detailed_caption}")

        if ocr.raw_text and ocr.raw_text.strip():
            lines.append(f"• Extracted Text (OCR):\n{ocr.raw_text.strip()}")

        if analysis.summary:
            lines.append(f"• Analysis Summary: {analysis.summary}")

        lines.append("=== END ATTACHED IMAGE CONTEXT ===")
        return "\n".join(lines)

    def _classify_image_type(
        self,
        metadata: ImageMetadata,
        caption: Optional[CaptionResult] = None,
        ocr: Optional[OCRResult] = None
    ) -> ImageType:
        """Classifies image category based on file name, visual caption, and extracted OCR text."""
        fname = metadata.file_name.lower()
        ocr_text = (ocr.raw_text if ocr and ocr.raw_text else "").lower()
        cap_text = (caption.short_caption + " " + (caption.detailed_caption or "") if caption else "").lower()

        combined = f"{fname} {cap_text} {ocr_text}"

        code_keywords = ["def ", "class ", "import ", "from ", "return ", "function", "var ", "const ", "traceback", "syntaxerror", "vscode", "terminal", "python", "implementation plan", "pipeline", "code"]
        if any(kw in combined for kw in code_keywords):
            return ImageType.CODE_SCREENSHOT

        if "screenshot" in combined or "screen" in combined or "window" in combined or "editor" in combined:
            return ImageType.SCREENSHOT

        if "doc" in combined or "invoice" in combined or "abstract" in combined or metadata.format == ImageFormat.PDF_IMAGE:
            return ImageType.DOCUMENT

        if "chart" in combined or "graph" in combined or "plot" in combined:
            return ImageType.CHART

        if "table" in combined or "grid" in combined:
            return ImageType.TABLE

        return ImageType.PHOTO

    def clear_cache(self) -> None:
        """Clears image context cache."""
        self.cache.clear()
