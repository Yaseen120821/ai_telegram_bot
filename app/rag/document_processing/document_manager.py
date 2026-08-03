"""
app/rag/document_processing/document_manager.py - Central Document Processing Manager Facade
==============================================================================================

1. PURPOSE:
-----------
Acts as the central Thread-safe Singleton orchestrator facade for the Document Processing Subsystem.
Coordinates document validation, format loader selection, text sanitization, metadata extraction, recursive chunking,
and chunk artifact storage.

2. WHY IT EXISTS (FACADE PATTERN):
----------------------------------
Provides a unified, clean public API for `RAGManager` and Telegram handlers.
Hides internal details of PDF parsing, Unicode normalization, SHA256 hashing, and chunk overlap math.

3. RESPONSIBILITIES:
--------------------
- Process document files and return structured `ProcessingResult` dataclasses (`process_document`).
- Process raw text strings directly (`process_text`).
- Retrieve processed chunks by `doc_id` (`get_document_chunks`).
- Manage processed chunk storage in `knowledge/processed/chunks/`.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Orchestrates `DocumentValidator`, `LoaderFactory`, `DocumentCleaner`, `MetadataExtractor`, and `DocumentChunker`.
- Interfaced by `RAGManager` in `app/rag/rag_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.rag.document_processing.document_config import PROCESSED_CHUNKS_DIR
from app.rag.document_processing.document_types import ProcessingStatus, DocumentType
from app.rag.document_processing.document_models import (
    Document,
    DocumentMetadata,
    Chunk,
    ChunkCollection,
    ProcessingResult,
    ValidationResult,
    CleaningResult
)
from app.rag.document_processing.document_validator import DocumentValidator
from app.rag.document_processing.document_cleaner import DocumentCleaner
from app.rag.document_processing.document_metadata import MetadataExtractor
from app.rag.document_processing.document_chunker import DocumentChunker
from app.rag.document_processing.loaders import LoaderFactory
from app.rag.document_processing.document_utils import DocumentUtils

logger = logging.getLogger("sana_ai.rag.document.manager")


class DocumentManager:
    """
    Thread-safe Singleton facade for the Document Processing & Ingestion Subsystem.
    """
    _instance: Optional["DocumentManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if DocumentManager._instance is not None:
            raise RuntimeError(
                "DocumentManager is a Singleton! Use `DocumentManager.get_instance()` instead."
            )

        self.validator: DocumentValidator = DocumentValidator()
        self.cleaner: DocumentCleaner = DocumentCleaner()
        self.metadata_extractor: MetadataExtractor = MetadataExtractor()
        self.chunker: DocumentChunker = DocumentChunker()

        self._processed_chunks_store: Dict[str, List[Chunk]] = {}
        self._ensure_storage_directories()

        logger.info("📁 DocumentManager Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "DocumentManager":
        """
        Thread-safe accessor for the shared DocumentManager Singleton instance.

        Returns:
            DocumentManager: Shared Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_storage_directories(self) -> None:
        """Creates knowledge storage directory paths if they do not exist."""
        try:
            os.makedirs(PROCESSED_CHUNKS_DIR, exist_ok=True)
        except Exception as err:
            logger.warning(f"Failed to create storage directory {PROCESSED_CHUNKS_DIR}: {err}")

    # ------------------------------------------------------------------
    # PUBLIC API METHODS
    # ------------------------------------------------------------------

    def process_document(self, filepath: str) -> ProcessingResult:
        """
        Primary pipeline method: validates file, selects format loader, extracts text, cleans text,
        extracts metadata, chunks document, and saves processed chunk artifacts.

        Args:
            filepath (str): Absolute or relative path to document file.

        Returns:
            ProcessingResult: ProcessingResult dataclass object.
        """
        start_t = time.time()
        logger.info(f"📄 Processing Document Ingestion | Path: '{filepath}'")

        # 1. Validate Document File
        val_res: ValidationResult = self.validator.validate_file(filepath)
        if not val_res.is_valid:
            logger.warning(f"⚠️ Document Validation Failed | Path: '{filepath}' | Reason: {val_res.reason}")
            return ProcessingResult(
                status=ProcessingStatus.VALIDATION_FAILED.value,
                doc_id="",
                chunks=[],
                execution_time_ms=(time.time() - start_t) * 1000.0,
                message=val_res.reason
            )

        if val_res.is_duplicate:
            file_hash = DocumentUtils.calculate_sha256(filepath)
            doc_id = DocumentUtils.generate_doc_id(Path(filepath).name, file_hash)
            existing_chunks = self.get_document_chunks(doc_id)
            return ProcessingResult(
                status=ProcessingStatus.DUPLICATE.value,
                doc_id=doc_id,
                chunks=existing_chunks,
                execution_time_ms=(time.time() - start_t) * 1000.0,
                message="Duplicate document detected. Returning cached chunks."
            )

        # 2. Select Loader & Extract Text
        file_type = DocumentUtils.validate_file_extension(Path(filepath).name)
        try:
            loader = LoaderFactory.get_loader(file_type)
            raw_text = loader.load(filepath)
        except Exception as load_err:
            logger.error(f"❌ Document Loader Failed for '{filepath}': {load_err}", exc_info=True)
            return ProcessingResult(
                status=ProcessingStatus.LOADER_FAILED.value,
                doc_id="",
                chunks=[],
                execution_time_ms=(time.time() - start_t) * 1000.0,
                message=f"Loader failed: {load_err}"
            )

        # 3. Clean & Normalize Text
        clean_res: CleaningResult = self.cleaner.clean_text(raw_text)
        if not clean_res.text:
            return ProcessingResult(
                status=ProcessingStatus.CLEANING_FAILED.value,
                doc_id="",
                chunks=[],
                execution_time_ms=(time.time() - start_t) * 1000.0,
                message="Document text cleaning resulted in empty string."
            )

        # 4. Extract Document Metadata
        metadata: DocumentMetadata = self.metadata_extractor.extract_metadata(filepath, clean_res.text)

        # 5. Generate Overlapping Recursive Chunks
        chunk_collection: ChunkCollection = self.chunker.create_chunks(metadata.file_id, clean_res.text, metadata)

        # 6. Register SHA256 Hash to prevent future duplicate processing
        self.validator.register_hash(metadata.hash_digest)

        # 7. Store Chunks in memory and save JSON artifact
        self._processed_chunks_store[metadata.file_id] = chunk_collection.chunks
        self._save_chunks_artifact(metadata.file_id, chunk_collection.chunks)

        elapsed_ms = (time.time() - start_t) * 1000.0
        logger.info(
            f"🎉 Document Processing Complete | Doc ID: '{metadata.file_id}' | "
            f"Chunks Generated: {len(chunk_collection.chunks)} | Elapsed: {elapsed_ms:.2f}ms"
        )

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS.value,
            doc_id=metadata.file_id,
            chunks=chunk_collection.chunks,
            execution_time_ms=elapsed_ms,
            message=f"Successfully processed document into {len(chunk_collection.chunks)} chunks."
        )

    def process_text(self, raw_text: str, filename: str = "custom_input.txt") -> ProcessingResult:
        """
        Processes a raw text string statement directly without requiring a file on disk.

        Args:
            raw_text (str): Input text statement.
            filename (str): Base filename identifier.

        Returns:
            ProcessingResult: ProcessingResult dataclass object.
        """
        start_t = time.time()

        if not raw_text or not raw_text.strip():
            return ProcessingResult(
                status=ProcessingStatus.VALIDATION_FAILED.value,
                doc_id="",
                chunks=[],
                execution_time_ms=0.0,
                message="Text statement is empty."
            )

        clean_res = self.cleaner.clean_text(raw_text)
        metadata = self.metadata_extractor.extract_metadata(filename, clean_res.text)
        chunk_collection = self.chunker.create_chunks(metadata.file_id, clean_res.text, metadata)

        self._processed_chunks_store[metadata.file_id] = chunk_collection.chunks
        elapsed_ms = (time.time() - start_t) * 1000.0

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS.value,
            doc_id=metadata.file_id,
            chunks=chunk_collection.chunks,
            execution_time_ms=elapsed_ms,
            message=f"Successfully processed raw text into {len(chunk_collection.chunks)} chunks."
        )

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """
        Retrieves processed Chunk list for a document doc_id.

        Args:
            doc_id (str): Document ID string.

        Returns:
            List[Chunk]: List of Chunk objects.
        """
        if doc_id in self._processed_chunks_store:
            return self._processed_chunks_store[doc_id]

        # Attempt to load from JSON artifact file on disk
        artifact_path = Path(PROCESSED_CHUNKS_DIR) / f"{doc_id}.json"
        if artifact_path.exists():
            try:
                with open(artifact_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chunks = [Chunk(**item) for item in data]
                    self._processed_chunks_store[doc_id] = chunks
                    return chunks
            except Exception as err:
                logger.warning(f"Failed to load chunk artifact {artifact_path}: {err}")

        return []

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """Alias for get_document_chunks."""
        return self.get_document_chunks(doc_id)

    def _save_chunks_artifact(self, doc_id: str, chunks: List[Chunk]) -> None:
        """Saves processed chunks as a JSON artifact file under knowledge/processed/chunks/."""
        try:
            artifact_path = Path(PROCESSED_CHUNKS_DIR) / f"{doc_id}.json"
            data = [c.to_dict() for c in chunks]
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"💾 Saved Chunk JSON Artifact: {artifact_path}")
        except Exception as err:
            logger.warning(f"Failed to save chunk JSON artifact for {doc_id}: {err}")

    def clear_processed_chunks(self) -> None:
        """Clears memory chunks store and hash registry."""
        self._processed_chunks_store.clear()
        self.validator.clear_hash_registry()
