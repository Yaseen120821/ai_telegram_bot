"""
app/rag/document_processing/loaders/docx_loader.py - DOCX Document Loader
==========================================================================
"""

import logging
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.docx")


class DOCXLoader(BaseDocumentLoader):
    """
    Parser for Word (.docx, .doc) files.
    """

    def load(self, filepath: str) -> str:
        logger.debug(f"📘 Loading DOCX Document: {filepath}")
        try:
            try:
                import docx
                doc = docx.Document(filepath)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                if paragraphs:
                    return "\n\n".join(paragraphs)
            except Exception as docx_err:
                logger.debug(f"python-docx extraction attempt failed for {filepath} ({docx_err}). Trying fallback.")

            # Fallback plain text read attempt
            return self.read_with_fallback_encoding(filepath)
        except Exception as err:
            logger.error(f"Failed to extract DOCX text for {filepath}: {err}")
            return f"[DOCX Document Content Placeholder for {filepath}]"
