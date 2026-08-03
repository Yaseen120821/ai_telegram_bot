"""
app/rag/rag_utils.py - RAG Utility & Helper Engine
===================================================

1. PURPOSE:
-----------
Provides text processing utilities, file extension validation, cryptographic file hashing, unique ID generation,
and prompt knowledge context formatting for the RAG Subsystem.

2. WHY IT EXISTS:
-----------------
Decouples document metadata calculation, string sanitization, and context formatting from the `RAGManager` facade.
Ensures query strings are clean, file types are validated against `SUPPORTED_FILE_EXTENSIONS`, and retrieved chunks are formatted consistently into ChatML prompt blocks.

3. RESPONSIBILITIES:
--------------------
- Clean and sanitize input search queries (`clean_query_text`).
- Validate and normalize document file extensions (`validate_file_extension`).
- Compute MD5/SHA256 file hashes (`calculate_file_hash`).
- Generate deterministic unique IDs (`generate_unique_id`).
- Format retrieved chunks into System Prompt context blocks (`format_knowledge_context`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `SUPPORTED_FILE_EXTENSIONS` and `INPUT_MAX_QUERY_CHARS` from `app/rag/rag_config.py`.
- Used by `app/rag/rag_manager.py` and `app/llm/prompt_builder.py`.

5. COMPLETE CODE:
-----------------
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from app.rag.rag_config import SUPPORTED_FILE_EXTENSIONS, INPUT_MAX_QUERY_CHARS
from app.rag.rag_types import DocumentType
from app.rag.rag_models import RetrievedChunk

logger = logging.getLogger("sana_ai.rag.utils")


class RAGUtils:
    """
    Utility helper class for text sanitization, file hashing, ID generation, and context formatting.
    """

    @staticmethod
    def clean_query_text(query: str) -> str:
        """
        Strips surrounding whitespace, normalizes Unicode, removes null bytes, and truncates query text.

        Args:
            query (str): Input query text statement.

        Returns:
            str: Cleaned and truncated query string.
        """
        if not query:
            return ""

        try:
            import unicodedata
            normalized = unicodedata.normalize("NFKC", query)
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized).strip()
            collapsed = re.sub(r"\s+", " ", cleaned)

            if len(collapsed) > INPUT_MAX_QUERY_CHARS:
                logger.debug(f"✂️ Truncated search query ({len(collapsed)} > {INPUT_MAX_QUERY_CHARS} chars).")
                collapsed = collapsed[:INPUT_MAX_QUERY_CHARS]

            return collapsed
        except Exception as err:
            logger.warning(f"Error cleaning query text ({err}). Returning basic trimmed string.")
            return query.strip()[:1000]

    @staticmethod
    def validate_file_extension(filename: str) -> str:
        """
        Validates whether a filename extension is supported and maps it to a DocumentType value.

        Args:
            filename (str): Base filename or full file path string.

        Returns:
            str: DocumentType string value or 'unknown'.
        """
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
    def calculate_file_hash(filepath: str) -> str:
        """
        Calculates MD5 hash digest of a local file.

        Args:
            filepath (str): Absolute file path.

        Returns:
            str: Hexadecimal hash digest string.
        """
        p = Path(filepath)
        if not p.exists() or not p.is_file():
            return ""

        try:
            hasher = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as err:
            logger.warning(f"Failed to calculate file hash for {filepath}: {err}")
            return ""

    @staticmethod
    def generate_unique_id(prefix: str, content: str) -> str:
        """
        Generates a deterministic unique ID string using MD5 hashing.

        Args:
            prefix (str): Prefix string (e.g. 'doc', 'chunk', 'src').
            content (str): Content text to hash.

        Returns:
            str: Formatted ID string (e.g. 'chunk_a1b2c3d4').
        """
        digest = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    @staticmethod
    def format_knowledge_context(chunks: List[RetrievedChunk]) -> str:
        """
        Formats a list of RetrievedChunk objects into a clean, structured ChatML System Prompt knowledge context block.

        Args:
            chunks (List[RetrievedChunk]): List of retrieved text chunks.

        Returns:
            str: Formatted prompt knowledge context string.
        """
        if not chunks:
            return ""

        lines = ["=== RELEVANT RETRIEVED KNOWLEDGE DOCUMENTS (RAG) ==="]
        for idx, chunk in enumerate(chunks, 1):
            doc_name = chunk.metadata.filename if chunk.metadata else f"Document_{chunk.doc_id}"
            score_pct = int(chunk.score * 100) if chunk.score <= 1.0 else int(chunk.score)
            lines.append(f"[{idx}] Source: {doc_name} (Relevance Match: {score_pct}%)")
            lines.append(f"{chunk.content.strip()}")
            lines.append("")

        lines.append("Instructions: Ground your answer primarily in the retrieved knowledge passages above. Cite source filenames where helpful.")
        return "\n".join(lines).strip()
