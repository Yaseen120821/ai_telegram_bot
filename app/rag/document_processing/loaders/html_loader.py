"""
app/rag/document_processing/loaders/html_loader.py - HTML Document Loader
==========================================================================
"""

import re
import logging
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.html")


class HTMLLoader(BaseDocumentLoader):
    """
    Parser for HTML (.html, .htm) files, stripping HTML tags while preserving readable text content.
    """

    def load(self, filepath: str) -> str:
        logger.debug(f"🌐 Loading HTML Document: {filepath}")
        raw = self.read_with_fallback_encoding(filepath)
        try:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw, "html.parser")
                return soup.get_text(separator="\n", strip=True)
            except ImportError:
                # Regex fallback tag stripper
                clean = re.sub(r"<script.*?>.*?</script>", "", raw, flags=re.DOTALL)
                clean = re.sub(r"<style.*?>.*?</style>", "", clean, flags=re.DOTALL)
                clean = re.sub(r"<[^>]+>", " ", clean)
                return re.sub(r"\s+", " ", clean).strip()
        except Exception:
            return raw
