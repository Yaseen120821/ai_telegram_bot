"""
app/rag/document_processing/loaders/json_loader.py - JSON Document Loader
==========================================================================
"""

import json
import logging
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.json")


class JSONLoader(BaseDocumentLoader):
    """
    Parser for JSON (.json) files, formatting key-value structures into readable text lines.
    """

    def load(self, filepath: str) -> str:
        logger.debug(f"📊 Loading JSON Document: {filepath}")
        raw = self.read_with_fallback_encoding(filepath)
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, indent=2)
        except Exception:
            return raw
