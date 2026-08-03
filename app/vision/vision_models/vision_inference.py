r"""
app/vision/vision_models/vision_inference.py - Vision Task Inference Engine
=============================================================================
Coordinates execution of Florence-2 visual tasks (Captioning, Detailed Scene Analysis, Object Detection)
and EasyOCR text extraction routines, producing unified structured outputs.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from PIL import Image as PILImage
import numpy as np

from app.vision.vision_models.vision_types import VisionTask, OCRStatus
from app.vision.vision_models.vision_config import get_vision_model_config, VisionModelConfig
from app.vision.vision_models.vision_schemas import (
    CaptionResult,
    ObjectDetectionResult,
    OCRTextResult
)
from app.vision.vision_models.vision_utils import (
    parse_bounding_boxes,
    clean_ocr_text
)
from app.vision.vision_models.vision_model_manager import VisionModelManager

logger = logging.getLogger("sana_ai.vision.models.inference")


class VisionInferenceEngine:
    """Executes Florence-2 neural model tasks and EasyOCR text extraction."""

    def __init__(self, config: Optional[VisionModelConfig] = None):
        self.config = config or get_vision_model_config()
        self.model_manager = VisionModelManager.get_instance()
        self._easyocr_reader = None

    def _get_ocr_reader(self) -> Optional[Any]:
        """Lazy initializer for EasyOCR Reader."""
        if self._easyocr_reader is None:
            logger.info(f"Initializing EasyOCR Reader (languages={self.config.ocr_languages}, gpu={self.config.use_ocr_gpu})...")
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(
                    lang_list=self.config.ocr_languages,
                    gpu=self.config.use_ocr_gpu,
                    verbose=False
                )
            except Exception as e:
                logger.warning(f"EasyOCR init notice ({e}). Trying CPU fallback...")
                try:
                    import easyocr
                    self._easyocr_reader = easyocr.Reader(
                        lang_list=self.config.ocr_languages,
                        gpu=False,
                        verbose=False
                    )
                except Exception as ex:
                    logger.warning(f"Could not load EasyOCR reader ({ex}). Continuing with empty OCR.")
                    self._easyocr_reader = None
        return self._easyocr_reader

    def execute_captioning(self, pil_img: PILImage.Image) -> CaptionResult:
        """Executes Florence-2 captioning tasks (<CAPTION> & <DETAILED_CAPTION>)."""
        florence = self.model_manager.get_florence_model()

        # 1. Short Caption
        short_res = florence.run_task(pil_img, task=VisionTask.CAPTION)
        short_text = short_res.get(VisionTask.CAPTION.value, "Image attached.")

        # 2. Detailed Caption
        detailed_res = florence.run_task(pil_img, task=VisionTask.DETAILED_CAPTION)
        detailed_text = detailed_res.get(VisionTask.DETAILED_CAPTION.value, short_text)

        return CaptionResult(
            short_caption=str(short_text),
            detailed_caption=str(detailed_text),
            confidence=0.92
        )

    def execute_object_detection(self, pil_img: PILImage.Image) -> List[ObjectDetectionResult]:
        """Executes Florence-2 object detection task (<OD>)."""
        florence = self.model_manager.get_florence_model()
        od_res = florence.run_task(pil_img, task=VisionTask.OBJECT_DETECTION)
        return parse_bounding_boxes(od_res)

    def execute_ocr(self, pil_img: PILImage.Image) -> OCRTextResult:
        """Executes EasyOCR text extraction on PIL Image."""
        try:
            reader = self._get_ocr_reader()
            if reader is None:
                return OCRTextResult(raw_text="", line_count=0, word_count=0, confidence=0.0)

            img_arr = np.array(pil_img.convert("RGB"))
            
            # EasyOCR detail=1 returns list of [bbox, text, prob]
            results = reader.readtext(img_arr, detail=1)

            extracted_lines: List[str] = []
            regions: List[Dict[str, Any]] = []

            for bbox, text, prob in results:
                if prob >= 0.2:  # Confidence threshold
                    extracted_lines.append(text)
                    regions.append({
                        "text": text,
                        "confidence": round(float(prob), 2),
                        "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox]
                    })

            raw_text = clean_ocr_text("\n".join(extracted_lines))
            word_count = len(raw_text.split())

            return OCRTextResult(
                raw_text=raw_text,
                line_count=len(extracted_lines),
                word_count=word_count,
                confidence=0.88 if extracted_lines else 0.0,
                regions=regions
            )

        except Exception as e:
            logger.error(f"EasyOCR text extraction failed: {e}")
            return OCRTextResult(
                raw_text="",
                line_count=0,
                word_count=0,
                confidence=0.0
            )
