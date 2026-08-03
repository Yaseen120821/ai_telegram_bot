"""
app/vision/ocr package initializer.
Exports OCREngine singleton for EasyOCR text extraction.
"""

from app.vision.ocr.ocr_engine import OCREngine

__all__ = ["OCREngine"]
