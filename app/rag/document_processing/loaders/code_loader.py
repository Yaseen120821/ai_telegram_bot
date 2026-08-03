"""
app/rag/document_processing/loaders/code_loader.py - Source Code Document Loader
=================================================================================
"""

import logging
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.code")


class CodeLoader(BaseDocumentLoader):
    """
    Parser for Source Code (.py, .js, .ts, .java, .cpp, .c, .h, .css) files.
    """

    def load(self, filepath: str) -> str:
        logger.debug(f"💻 Loading Source Code Document: {filepath}")
        return self.read_with_fallback_encoding(filepath)
