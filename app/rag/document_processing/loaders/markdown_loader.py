"""
app/rag/document_processing/loaders/markdown_loader.py - Markdown Document Loader
==================================================================================
"""

import logging
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.markdown")


class MarkdownLoader(BaseDocumentLoader):
    """
    Parser for Markdown (.md, .markdown) files, preserving section headers and code blocks.
    """

    def load(self, filepath: str) -> str:
        logger.debug(f"📝 Loading Markdown Document: {filepath}")
        return self.read_with_fallback_encoding(filepath)
