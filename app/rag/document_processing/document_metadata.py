"""
app/rag/document_processing/document_metadata.py - Document Metadata Extractor
================================================================================

1. PURPOSE:
-----------
Extracts file statistics, word/character counts, token estimates, creation timestamps, and SHA256 hashes for documents.

2. WHY IT EXISTS:
-----------------
Rich metadata is essential for source citations, vector filtering, and duplicate detection.
`MetadataExtractor` isolates metadata computation into a single, clean module.

3. RESPONSIBILITIES:
--------------------
- Compute word count, character count, and estimated tokens.
- Compute SHA256 file hash digest.
- Generate deterministic `file_id` and construct `DocumentMetadata` object.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `DocumentUtils` from `document_utils.py`.
- Used by `document_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import os
import logging
from pathlib import Path

from app.rag.document_processing.document_models import DocumentMetadata
from app.rag.document_processing.document_utils import DocumentUtils

logger = logging.getLogger("sana_ai.rag.document.metadata")


class MetadataExtractor:
    """
    Metadata extraction engine for document files.
    """

    def extract_metadata(self, filepath: str, text_content: str = "") -> DocumentMetadata:
        """
        Extracts comprehensive DocumentMetadata for a file.

        Args:
            filepath (str): Absolute or relative file path.
            text_content (str): Optional extracted text for word/token calculation.

        Returns:
            DocumentMetadata: DocumentMetadata dataclass instance.
        """
        path = Path(filepath)
        filename = path.name
        file_type = DocumentUtils.validate_file_extension(filename)

        try:
            file_size = path.stat().st_size if path.exists() else len(text_content.encode("utf-8"))
        except Exception:
            file_size = len(text_content.encode("utf-8"))

        file_hash = DocumentUtils.calculate_sha256(filepath) if path.exists() else ""
        doc_id = DocumentUtils.generate_doc_id(filename, file_hash)

        word_count, char_count = DocumentUtils.count_words_and_chars(text_content)
        estimated_tokens = DocumentUtils.estimate_tokens(text_content)

        metadata = DocumentMetadata(
            file_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_path=str(path.resolve()) if path.exists() else filepath,
            file_size_bytes=file_size,
            word_count=word_count,
            char_count=char_count,
            estimated_tokens=estimated_tokens,
            hash_digest=file_hash
        )

        logger.debug(
            f"📋 Extracted Metadata for '{filename}' | Doc ID: '{doc_id}' | "
            f"Words: {word_count} | Chars: {char_count} | Est. Tokens: {estimated_tokens}"
        )
        return metadata
