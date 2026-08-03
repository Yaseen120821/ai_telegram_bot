r"""
app/rag/pdf_analyzer.py - High-Level PDF Analyzer & Context Builder
====================================================================

1. PURPOSE:
-----------
Inspects PDF document assets, extracts page counts, structural metadata, digital page text, and scanned OCR text,
and constructs formatted context blocks for PromptBuilder injection.

2. RESPONSIBILITIES:
--------------------
- Inspect PDF file size, page count, and metadata.
- Extract digital page text or scanned EasyOCR text via PDFLoader.
- Format structured prompt context block (`=== ATTACHED PDF DOCUMENT: <filename> ===`).
"""

import os
import logging
from typing import Optional, Dict, Any
from app.rag.document_processing.loaders.pdf_loader import PDFLoader

logger = logging.getLogger("sana_ai.rag.pdf_analyzer")


class PDFAnalyzer:
    """High-Level PDF Document Analyzer and Context Formatter."""

    _instance: Optional["PDFAnalyzer"] = None

    def __init__(self):
        self.pdf_loader = PDFLoader()

    @classmethod
    def get_instance(cls) -> "PDFAnalyzer":
        """Returns global PDFAnalyzer singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_pdf(self, filepath: str) -> Dict[str, Any]:
        """
        Analyzes PDF document file and returns metadata and extracted text.

        Args:
            filepath (str): Local path to PDF document.

        Returns:
            Dict[str, Any]: Extracted metadata, page count, text content, and formatted context block.
        """
        if not os.path.exists(filepath):
            return {
                "file_name": os.path.basename(filepath),
                "page_count": 0,
                "extracted_text": "",
                "formatted_context": f"=== ATTACHED PDF DOCUMENT: {os.path.basename(filepath)} ===\nNotice: File not found.\n=== END ATTACHED PDF DOCUMENT ==="
            }

        filename = os.path.basename(filepath)
        file_size_kb = os.path.getsize(filepath) / 1024.0

        page_count = 1
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(filepath)
            page_count = len(reader.pages)
        except Exception:
            pass

        logger.info(f"📕 PDFAnalyzer processing '{filename}' ({page_count} pages, {file_size_kb:.1f} KB)...")
        extracted_text = self.pdf_loader.load(filepath)

        formatted_context = self.format_pdf_context_block(
            filename=filename,
            page_count=page_count,
            file_size_kb=file_size_kb,
            text=extracted_text
        )

        return {
            "file_name": filename,
            "file_size_kb": file_size_kb,
            "page_count": page_count,
            "extracted_text": extracted_text,
            "formatted_context": formatted_context
        }

    def format_pdf_context_block(
        self,
        filename: str,
        page_count: int,
        file_size_kb: float,
        text: str
    ) -> str:
        """Constructs standardized ChatML prompt context block for PDF documents."""
        lines = [
            f"=== ATTACHED PDF DOCUMENT: {filename} ===",
            f"• File Name: {filename} ({page_count} page(s), {file_size_kb:.1f} KB)",
            f"• Extracted Document Content:\n{text.strip() if text else 'No text could be extracted.'}",
            "=== END ATTACHED PDF DOCUMENT ==="
        ]
        return "\n".join(lines)
