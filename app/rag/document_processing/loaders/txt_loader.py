"""
app/rag/document_processing/loaders/txt_loader.py - Plain Text Document Loader
==============================================================================
"""

import logging
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.txt")


class TXTLoader(BaseDocumentLoader):
    """
    Parser for plain text (.txt) files.
    """

    def load(self, filepath: str) -> str:
        logger.debug(f"📄 Loading TXT Document: {filepath}")
        return self.read_with_fallback_encoding(filepath)
