"""
app/rag/document_processing/document_validator.py - Document Ingestion Validator
====================================================================================

1. PURPOSE:
-----------
Validates document files prior to ingestion (verifying file existence, read permissions, file size limits, supported formats,
and SHA256 duplicate detection).

2. WHY IT EXISTS (DEFENSIVE PROGRAMMING & SECURITY):
---------------------------------------------------
Attempting to process corrupted, non-existent, 500MB, or unsupported files causes unhandled crashes, memory leaks,
and pipeline failures. `DocumentValidator` acts as a security gatekeeper, rejecting invalid inputs early.

3. RESPONSIBILITIES:
--------------------
- Verify file existence and readable permissions.
- Check extension against `SUPPORTED_FILE_EXTENSIONS`.
- Verify file size is within bounds (`0 < size <= MAX_FILE_SIZE_BYTES`).
- Compute SHA256 file hash and detect duplicates against a known hash registry.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `MAX_FILE_SIZE_BYTES` from `document_config.py`.
- Uses `DocumentUtils` from `document_utils.py`.
- Called by `document_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import os
import logging
from pathlib import Path
from typing import Set, Optional

from app.rag.document_processing.document_config import MAX_FILE_SIZE_BYTES
from app.rag.document_processing.document_types import DocumentType
from app.rag.document_processing.document_models import ValidationResult
from app.rag.document_processing.document_utils import DocumentUtils

logger = logging.getLogger("sana_ai.rag.document.validator")


class DocumentValidator:
    """
    Document validation engine enforcing safety limits, format verification, and duplicate detection.
    """

    def __init__(self) -> None:
        """Initializes known file hash registry for duplicate detection."""
        self._known_hashes: Set[str] = set()

    def validate_file(self, filepath: str) -> ValidationResult:
        """
        Performs thorough validation checks on a document file.

        Args:
            filepath (str): Absolute or relative file path string.

        Returns:
            ValidationResult: ValidationResult object detailing validation status and reasons.
        """
        if not filepath or not str(filepath).strip():
            return ValidationResult(is_valid=False, reason="Filepath is empty or invalid.")

        path = Path(filepath)

        # 1. File existence check
        if not path.exists():
            return ValidationResult(is_valid=False, reason=f"File does not exist: {filepath}")

        # 2. File type check (must be regular file, not folder)
        if not path.is_file():
            return ValidationResult(is_valid=False, reason=f"Path is a directory or special file, not a regular file: {filepath}")

        # 3. Readable permission check
        if not os.access(path, os.R_OK):
            return ValidationResult(is_valid=False, reason=f"File lacks read permission: {filepath}")

        # 4. Supported extension check
        file_type = DocumentUtils.validate_file_extension(path.name)
        if file_type == DocumentType.UNKNOWN.value:
            return ValidationResult(is_valid=False, reason=f"Unsupported document file extension: '{path.suffix}'")

        # 5. File size limits check
        try:
            file_size = path.stat().st_size
            if file_size == 0:
                return ValidationResult(is_valid=False, reason="File is empty (0 bytes).")
            if file_size > MAX_FILE_SIZE_BYTES:
                size_mb = file_size / (1024 * 1024)
                limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
                return ValidationResult(
                    is_valid=False,
                    reason=f"File size ({size_mb:.2f} MB) exceeds maximum limit ({limit_mb:.2f} MB)."
                )
        except Exception as err:
            return ValidationResult(is_valid=False, reason=f"Failed to read file status: {err}")

        # 6. SHA256 Duplicate Detection check
        file_hash = DocumentUtils.calculate_sha256(str(path))
        if file_hash and file_hash in self._known_hashes:
            logger.info(f"ℹ️ Duplicate document hash detected ({file_hash[:8]}...) for {path.name}.")
            return ValidationResult(is_valid=True, reason="Document is a duplicate", is_duplicate=True)

        return ValidationResult(is_valid=True, reason="Validation passed successfully.")

    def register_hash(self, hash_digest: str) -> None:
        """
        Registers a processed document SHA256 hash digest to prevent future duplicate processing.

        Args:
            hash_digest (str): SHA256 hex digest string.
        """
        if hash_digest:
            self._known_hashes.add(hash_digest)

    def clear_hash_registry(self) -> None:
        """Clears known duplicate hash registry."""
        self._known_hashes.clear()
