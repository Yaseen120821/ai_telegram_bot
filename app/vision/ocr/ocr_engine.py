r"""
app/vision/ocr/ocr_engine.py - Thread-Safe EasyOCR Engine
===========================================================

1. PURPOSE:
-----------
Executes Optical Character Recognition (OCR) on image assets and scanned document pages using EasyOCR.
Extracts clean text, line counts, word counts, region bounding boxes, and numerical confidence scores.

2. WHY IT EXISTS:
-----------------
Visual assets (code screenshots, mobile UI, document scans, table screenshots, diagrams) contain textual data
that vision neural models (like Florence-2) cannot transcribe word-for-word. `OCREngine` bridges this gap.

3. RESPONSIBILITIES:
--------------------
- Lazy-initialize `easyocr.Reader` with `verbose=False` to prevent Windows cp1252 Unicode charmap errors.
- Support CUDA GPU with automatic CPU fallback.
- Support PIL Image, NumPy array, and local file path inputs.
- Filter low-confidence OCR text candidates (< 0.20 threshold).
- Clean and normalize extracted text blocks.
- Return structured `OCRResult` object.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/vision/vision_models/vision_inference.py`, `app/vision/vision_manager.py`, and `app/rag/document_processing/loaders/pdf_loader.py`.
"""

import logging
import threading
import numpy as np
from typing import Optional, List, Dict, Any, Union
from PIL import Image as PILImage

from app.vision.vision_types import OCRStatus
from app.vision.vision_schemas import OCRResult

logger = logging.getLogger("sana_ai.vision.ocr.engine")


class OCREngine:
    """
    Thread-safe Singleton wrapper for EasyOCR text extraction.
    """
    _instance: Optional["OCREngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = True) -> None:
        self.languages: List[str] = languages or ["en"]
        self.use_gpu: bool = use_gpu
        self._reader: Optional[Any] = None
        self._initialized: bool = False

    @classmethod
    def get_instance(cls) -> "OCREngine":
        """Returns shared Singleton instance of OCREngine."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_reader(self) -> Optional[Any]:
        """Lazy initializer for EasyOCR Reader with GPU -> CPU fallback and verbose=False."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    logger.info(f"Initializing EasyOCR Reader (languages={self.languages}, gpu={self.use_gpu})...")
                    try:
                        import easyocr
                        self._reader = easyocr.Reader(lang_list=self.languages, gpu=self.use_gpu, verbose=False)
                        logger.info("✅ EasyOCR Reader initialized successfully on GPU!")
                    except Exception as gpu_err:
                        logger.warning(f"EasyOCR GPU initialization notice ({gpu_err}). Falling back to CPU...")
                        try:
                            import easyocr
                            self._reader = easyocr.Reader(lang_list=self.languages, gpu=False, verbose=False)
                            logger.info("✅ EasyOCR Reader initialized on CPU fallback.")
                        except Exception as cpu_err:
                            logger.error(f"Failed to initialize EasyOCR Reader: {cpu_err}")
                            self._reader = None
                    self._initialized = True
        return self._reader

    def extract_text(
        self,
        image_input: Union[str, PILImage.Image, np.ndarray],
        confidence_threshold: float = 0.20
    ) -> OCRResult:
        """
        Extracts text from an image file path, PIL Image, or NumPy array.

        Args:
            image_input (Union[str, PILImage.Image, np.ndarray]): Target image input.
            confidence_threshold (float): Minimum candidate confidence score (default: 0.20).

        Returns:
            OCRResult: Structured result object containing extracted text, lines, words, and confidence.
        """
        reader = self._get_reader()
        if reader is None:
            return OCRResult(
                status=OCRStatus.FAILED,
                raw_text="",
                detected_languages=self.languages,
                line_count=0,
                word_count=0,
                confidence=0.0,
                error_message="EasyOCR reader could not be initialized."
            )

        try:
            # Normalize input to format accepted by EasyOCR
            if isinstance(image_input, PILImage.Image):
                img_data = np.array(image_input.convert("RGB"))
            elif isinstance(image_input, str):
                img_data = image_input
            else:
                img_data = image_input

            # Execute EasyOCR readtext
            results = reader.readtext(img_data, detail=1)

            extracted_lines: List[str] = []
            confidence_scores: List[float] = []

            for bbox, text, prob in results:
                prob_val = float(prob)
                if prob_val >= confidence_threshold:
                    clean_line = text.strip()
                    if clean_line:
                        extracted_lines.append(clean_line)
                        confidence_scores.append(prob_val)

            raw_text = "\n".join(extracted_lines)
            word_count = len(raw_text.split())
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            status = OCRStatus.SUCCESS if raw_text else OCRStatus.NO_TEXT_FOUND

            logger.info(
                f"🎯 EasyOCR Extraction Complete | Status: {status.value} | "
                f"Lines: {len(extracted_lines)} | Words: {word_count} | Avg Conf: {avg_confidence:.2f}"
            )

            return OCRResult(
                status=status,
                raw_text=raw_text,
                detected_languages=self.languages,
                line_count=len(extracted_lines),
                word_count=word_count,
                confidence=avg_confidence
            )

        except Exception as err:
            logger.error(f"Error during EasyOCR text extraction: {err}", exc_info=True)
            return OCRResult(
                status=OCRStatus.FAILED,
                raw_text="",
                detected_languages=self.languages,
                line_count=0,
                word_count=0,
                confidence=0.0,
                error_message=str(err)
            )
