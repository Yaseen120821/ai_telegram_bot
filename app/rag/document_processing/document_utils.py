"""
app/rag/document_processing/document_utils.py - Document Processing Utility Engine
====================================================================================

1. PURPOSE:
-----------
Provides token estimation, SHA256 cryptographic hashing, deterministic ID generation, encoding detection,
and word/character count utilities for document processing.

2. WHY IT EXISTS:
-----------------
Decouples helper logic from loaders, chunkers, and facade managers. Guarantees deterministic document IDs,
reliable encoding detection, and accurate token estimates across the ingestion pipeline.

3. RESPONSIBILITIES:
--------------------
- Estimate LLM token counts (~4 characters per token) (`estimate_tokens`).
- Compute SHA256 cryptographic hash digests for files (`calculate_sha256`).
- Generate deterministic document IDs (`generate_doc_id`) and chunk IDs (`generate_chunk_id`).
- Detect text encodings with multi-encoding fallback (`detect_encoding`).
- Calculate word and character counts (`count_words_and_chars`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `document_validator.py`, `document_cleaner.py`, `document_chunker.py`, `document_metadata.py`, and `loaders/`.

5. COMPLETE CODE:
-----------------
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import Tuple, List

from app.rag.document_processing.document_config import DEFAULT_ENCODING, FALLBACK_ENCODINGS

logger = logging.getLogger("sana_ai.rag.document.utils")


class DocumentUtils:
    """
    Utility helper class for token estimation, SHA256 hashing, encoding detection, and ID generation.
    """

    @staticmethod
    def validate_file_extension(filename: str) -> str:
        """
        Validates whether a filename extension is supported and maps it to a DocumentType value.

        Args:
            filename (str): Base filename or full file path string.

        Returns:
            str: DocumentType string value or 'unknown'.
        """
        from app.rag.rag_config import SUPPORTED_FILE_EXTENSIONS
        from app.rag.document_processing.document_types import DocumentType

        if not filename:
            return DocumentType.UNKNOWN.value

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            return DocumentType.UNKNOWN.value

        if ext == ".pdf":
            return DocumentType.PDF.value
        elif ext in (".docx", ".doc"):
            return DocumentType.DOCX.value
        elif ext == ".txt":
            return DocumentType.TXT.value
        elif ext in (".md", ".markdown"):
            return DocumentType.MARKDOWN.value
        elif ext in (".html", ".htm"):
            return DocumentType.HTML.value
        elif ext == ".json":
            return DocumentType.JSON.value
        elif ext == ".csv":
            return DocumentType.CSV.value
        elif ext in (".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".css"):
            return DocumentType.CODE.value

        return DocumentType.UNKNOWN.value

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimates LLM token count for a text string using standard ~4 chars per token rule.

        Args:
            text (str): Input text statement.

        Returns:
            int: Estimated token count.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def calculate_sha256(filepath: str) -> str:
        """
        Calculates SHA256 cryptographic hash digest for a local file.

        Args:
            filepath (str): Absolute or relative file path.

        Returns:
            str: SHA256 hex digest string.
        """
        p = Path(filepath)
        if not p.exists() or not p.is_file():
            return ""

        try:
            hasher = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as err:
            logger.warning(f"Failed to calculate SHA256 hash for {filepath}: {err}")
            return ""

    @staticmethod
    def generate_doc_id(filename: str, hash_digest: str = "") -> str:
        """
        Generates a deterministic document ID string using filename and hash digest.

        Args:
            filename (str): Base filename string.
            hash_digest (str): SHA256 or content hash string.

        Returns:
            str: Document ID string (e.g. 'doc_sana_architecture_a1b2c3').
        """
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", Path(filename).stem).lower()
        if hash_digest:
            suffix = hash_digest[:8]
        else:
            suffix = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
        return f"doc_{clean_name}_{suffix}"

    @staticmethod
    def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
        """
        Generates a structured chunk ID.

        Args:
            doc_id (str): Parent document ID.
            chunk_index (int): 0-indexed position in document.

        Returns:
            str: Chunk ID string (e.g. 'chunk_doc_sana_architecture_a1b2c3_0001').
        """
        return f"chunk_{doc_id}_{chunk_index:04d}"

    @staticmethod
    def detect_encoding(filepath: str) -> str:
        """
        Detects text encoding for a file, testing fallback list if chardet is absent.

        Args:
            filepath (str): File path.

        Returns:
            str: Encoding name string.
        """
        p = Path(filepath)
        if not p.exists():
            return DEFAULT_ENCODING

        try:
            import chardet
            with open(p, "rb") as f:
                raw = f.read(10000)
                res = chardet.detect(raw)
                if res and res.get("encoding") and res.get("confidence", 0) > 0.6:
                    return res["encoding"].lower()
        except Exception:
            pass

        # Test reading file with fallback list
        for enc in FALLBACK_ENCODINGS:
            try:
                with open(p, "r", encoding=enc) as f:
                    f.read(2048)
                return enc
            except (UnicodeDecodeError, Exception):
                continue

        return DEFAULT_ENCODING

    @staticmethod
    def count_words_and_chars(text: str) -> Tuple[int, int]:
        """
        Calculates word count and character count for text.

        Args:
            text (str): Input text string.

        Returns:
            Tuple[int, int]: (word_count, char_count).
        """
        if not text:
            return 0, 0

        char_count = len(text)
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        return word_count, char_count
