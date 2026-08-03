r"""
app/rag/document_processing/loaders/pdf_loader.py - PDF Document Loader & Scanned OCR Engine
==============================================================================================

1. PURPOSE:
-----------
Extracts text content from PDF (.pdf) files. Automatically detects digital PDFs vs. scanned PDFs.
For digital PDFs, extracts text page by page using PyPDF2 / pypdf.
For image-based / scanned PDFs (where digital text extraction is empty), automatically executes EasyOCR via `OCREngine`.

2. RESPONSIBILITIES:
--------------------
- Extract text page by page for digital PDFs.
- Detect scanned PDFs when extracted text is below length threshold (< 30 chars).
- Render or extract PDF page images and perform EasyOCR text extraction on scanned pages.
- Handle fallback encodings gracefully without crashing.
"""

import logging
from typing import List
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.pdf")


class PDFLoader(BaseDocumentLoader):
    """
    Parser for PDF (.pdf) files with automatic scanned PDF OCR fallback.
    """

    def load(self, filepath: str) -> str:
        logger.info(f"📕 PDFLoader extracting text for file: '{filepath}'")
        extracted_pages: List[str] = []

        # 1. Digital PDF Text Extraction (PyPDF2 / pypdf)
        try:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(filepath)
                for idx, page in enumerate(reader.pages):
                    txt = page.extract_text()
                    if txt and txt.strip():
                        extracted_pages.append(f"--- Page {idx+1} ---\n{txt.strip()}")
            except Exception as pypdf_err:
                logger.debug(f"PyPDF2 text extraction notice for '{filepath}': {pypdf_err}")

            digital_text = "\n\n".join(extracted_pages).strip()

            # If digital text extraction succeeded and yielded sufficient text (> 30 chars), return it
            if digital_text and len(digital_text) > 30:
                logger.info(f"✅ Extracted digital PDF text ({len(digital_text)} chars across {len(extracted_pages)} pages).")
                return digital_text

        except Exception as err:
            logger.warning(f"Digital PDF extraction error for '{filepath}': {err}")

        # 2. Scanned PDF Fallback (Image rendering + EasyOCR)
        logger.info(f"🔍 Scanned / Image PDF detected for '{filepath}'. Triggering EasyOCR fallback...")
        ocr_text = self._extract_scanned_pdf_ocr(filepath)
        if ocr_text and len(ocr_text.strip()) > 10:
            logger.info(f"✅ Extracted scanned PDF text via EasyOCR ({len(ocr_text)} chars).")
            return ocr_text

        # 3. Plain Text Encoding Fallback
        try:
            fallback_txt = self.read_with_fallback_encoding(filepath)
            if fallback_txt and len(fallback_txt.strip()) > 30:
                return fallback_txt
        except Exception:
            pass

        return f"[PDF Document Content: '{filepath}']"

    def _extract_scanned_pdf_ocr(self, filepath: str) -> str:
        """Renders/extracts images from PDF pages and runs OCREngine EasyOCR."""
        ocr_pages: List[str] = []
        try:
            from app.vision.ocr.ocr_engine import OCREngine
            ocr_engine = OCREngine.get_instance()

            # Try PyPDF2 image extraction first
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(filepath)
                for page_idx, page in enumerate(reader.pages):
                    page_ocr_text = []
                    for count, image_file_object in enumerate(page.images):
                        try:
                            from PIL import Image as PILImage
                            import io
                            pil_img = PILImage.open(io.BytesIO(image_file_object.data))
                            ocr_out = ocr_engine.extract_text(pil_img)
                            if ocr_out.raw_text:
                                page_ocr_text.append(ocr_out.raw_text)
                        except Exception as img_err:
                            logger.debug(f"Page {page_idx+1} image {count} OCR error: {img_err}")
                    
                    if page_ocr_text:
                        ocr_pages.append(f"--- Page {page_idx+1} (OCR) ---\n" + "\n".join(page_ocr_text))
            except Exception as pypdf_img_err:
                logger.debug(f"PyPDF2 image extraction notice: {pypdf_img_err}")

            # Try pdf2image if PyPDF2 image extraction was empty
            if not ocr_pages:
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(filepath, first_page=1, last_page=10)
                    for page_idx, pil_img in enumerate(images):
                        ocr_out = ocr_engine.extract_text(pil_img)
                        if ocr_out.raw_text:
                            ocr_pages.append(f"--- Page {page_idx+1} (OCR) ---\n{ocr_out.raw_text}")
                except Exception as pdf2img_err:
                    logger.debug(f"pdf2image fallback notice: {pdf2img_err}")

        except Exception as ocr_err:
            logger.error(f"Scanned PDF OCR error for '{filepath}': {ocr_err}")

        return "\n\n".join(ocr_pages)
