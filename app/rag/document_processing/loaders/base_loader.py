"""
app/rag/document_processing/loaders/base_loader.py - Abstract Base Document Loader
====================================================================================

1. PURPOSE:
-----------
Defines the `BaseDocumentLoader` abstract base class establishing the standard interface for all document parsers.

2. WHY IT EXISTS (STRATEGY & FACTORY PATTERN):
----------------------------------------------
Each document format (PDF, DOCX, TXT, Markdown, HTML, JSON, CSV, Source Code) requires different extraction logic.
Defining an abstract base class enforces a uniform `load(filepath)` method contract across all format loaders.

3. RESPONSIBILITIES:
--------------------
- Enforce standard `load(filepath: str) -> str` interface.
- Provide common encoding detection fallback methods.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Inherited by `txt_loader.py`, `markdown_loader.py`, `code_loader.py`, `pdf_loader.py`, `docx_loader.py`, etc.
- Instantiated via `LoaderFactory` in `app/rag/document_processing/loaders/__init__.py`.

5. COMPLETE CODE:
-----------------
"""

from abc import ABC, abstractmethod
import logging
from app.rag.document_processing.document_utils import DocumentUtils

logger = logging.getLogger("sana_ai.rag.loaders.base")


class BaseDocumentLoader(ABC):
    """
    Abstract base class for all document format loaders.
    """

    @abstractmethod
    def load(self, filepath: str) -> str:
        """
        Extracts raw text content from the specified document file.

        Args:
            filepath (str): Absolute or relative path to target document.

        Returns:
            str: Extracted raw text string.

        Raises:
            Exception: If document loading or parsing fails.
        """
        pass

    def read_with_fallback_encoding(self, filepath: str) -> str:
        """
        Helper method reading plain text files using auto-detected or fallback encodings.

        Args:
            filepath (str): File path.

        Returns:
            str: Extracted text.
        """
        enc = DocumentUtils.detect_encoding(filepath)
        try:
            with open(filepath, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except Exception as err:
            logger.warning(f"Primary encoding '{enc}' failed for {filepath} ({err}). Retrying with utf-8.")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
