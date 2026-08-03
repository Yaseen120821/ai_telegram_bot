r"""
app/rag/index_manager.py - Incremental Document Indexing Engine
================================================================

1. PURPOSE:
-----------
Executes intelligent incremental document indexing for SANA AI's RAG Subsystem. Tracks file SHA256 content hashes
and modification timestamps (`mtime`) to ensure only new, modified, or deleted documents are processed.

2. WHY IT EXISTS (INCREMENTAL EFFICIENCY & CONSISTENCY):
--------------------------------------------------------
In a production knowledge base with hundreds of documents, reprocessing unchanged files on every system startup
wastes CPU/GPU compute, disk I/O, and causes unnecessary cache invalidations. `IndexManager` ensures 0 wasted cycles
for unchanged files and automatically purges obsolete chunk vectors when files are modified or removed.

3. RESPONSIBILITIES:
--------------------
- Track document file states using SHA256 content hashes and `mtime` modification timestamps.
- Skip processing for unchanged documents (`force_reindex=False`).
- Automatically re-index modified documents and purge obsolete chunk vectors.
- Support recursive batch indexing of knowledge directory trees (`index_directory`).
- Purge document vectors and update state on file deletion (`remove_document`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `DocumentManager` from `app/rag/document_processing/document_manager.py`.
- Interfaced by `RAGManager` in `app/rag/rag_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import os
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.rag.document_processing.document_manager import DocumentManager
from app.rag.document_processing.document_models import ProcessingResult

logger = logging.getLogger("sana_ai.rag.index_manager")


class IndexManager:
    """
    Thread-safe Singleton managing incremental document indexing and vector lifecycle state.
    """
    _instance: Optional["IndexManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if IndexManager._instance is not None:
            raise RuntimeError("IndexManager is a Singleton! Use `IndexManager.get_instance()` instead.")

        self.doc_mgr: DocumentManager = DocumentManager.get_instance()
        # Internal state registry: file_path -> {hash, mtime, doc_id, chunks_count, indexed_at}
        self._document_registry: Dict[str, Dict[str, Any]] = {}
        logger.info("🔄 IndexManager Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "IndexManager":
        """Thread-safe accessor for shared IndexManager Singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # PUBLIC API METHODS
    # ------------------------------------------------------------------

    def compute_file_hash(self, file_path: str) -> str:
        """
        Calculates SHA256 digest hash of a file.

        Args:
            file_path (str): File system path string.

        Returns:
            str: SHA256 hex digest string.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_file_modified(self, file_path: str) -> bool:
        """
        Checks if a file is new or modified since its last indexed state.

        Args:
            file_path (str): Absolute file system path string.

        Returns:
            bool: True if new or modified, False if unchanged.
        """
        p = Path(file_path).resolve()
        norm_path = str(p)

        if norm_path not in self._document_registry:
            return True

        record = self._document_registry[norm_path]
        try:
            current_mtime = os.path.getmtime(norm_path)
            if current_mtime != record.get("mtime"):
                current_hash = self.compute_file_hash(norm_path)
                return current_hash != record.get("hash")
        except OSError:
            return True

        return False

    def index_document(self, file_path: str, force_reindex: bool = False) -> ProcessingResult:
        """
        Incrementally indexes a single document file.

        Args:
            file_path (str): Absolute file system path string.
            force_reindex (bool): Force re-processing even if hash matches.

        Returns:
            ProcessingResult: ProcessingResult dataclass payload.
        """
        p = Path(file_path).resolve()
        norm_path = str(p)

        if not p.exists() or not p.is_file():
            logger.error(f"❌ Cannot index non-existent file: {norm_path}")
            return ProcessingResult(
                doc_id=f"doc_err_{int(time.time())}",
                status="failed",
                chunks=[],
                message="File does not exist."
            )

        # Check incremental modification state
        if not force_reindex and not self.is_file_modified(norm_path):
            doc_id = self._document_registry[norm_path]["doc_id"]
            cached_chunks = self.doc_mgr.get_chunks_by_doc_id(doc_id)
            logger.info(f"⏩ Incremental Indexer: Document '{p.name}' is unchanged. Skipping re-embedding.")
            return ProcessingResult(
                doc_id=doc_id,
                status="unchanged",
                chunks=cached_chunks,
                message="Document is unchanged."
            )

        # Process document through DocumentManager pipeline
        start_t = time.time()
        res = self.doc_mgr.process_document(norm_path)

        if res.status in ["success", "duplicate"]:
            try:
                curr_mtime = os.path.getmtime(norm_path)
                curr_hash = self.compute_file_hash(norm_path)
                self._document_registry[norm_path] = {
                    "hash": curr_hash,
                    "mtime": curr_mtime,
                    "doc_id": res.doc_id,
                    "chunks_count": len(res.chunks),
                    "indexed_at": time.time()
                }
                elapsed_ms = (time.time() - start_t) * 1000.0
                logger.info(f"✅ Incremental Indexer: Successfully indexed '{p.name}' | Doc ID: '{res.doc_id}' | Elapsed: {elapsed_ms:.2f}ms")
            except Exception as e:
                logger.warning(f"Failed to record registry state for '{p.name}': {e}")

        return res

    def index_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Scans a directory tree and incrementally indexes all document files.

        Args:
            directory_path (str): Target directory path.
            recursive (bool): Scan subdirectories recursively if True.

        Returns:
            Dict[str, Any]: Summary dictionary of indexing operation statistics.
        """
        dir_p = Path(directory_path).resolve()
        if not dir_p.exists() or not dir_p.is_dir():
            logger.error(f"❌ Target indexing directory does not exist: {dir_p}")
            return {"status": "failed", "processed": 0, "unchanged": 0, "failed": 0}

        processed_cnt = 0
        unchanged_cnt = 0
        failed_cnt = 0

        pattern = "**/*" if recursive else "*"
        for path_obj in dir_p.glob(pattern):
            if path_obj.is_file():
                res = self.index_document(str(path_obj))
                if res.status == "success":
                    processed_cnt += 1
                elif res.status == "unchanged":
                    unchanged_cnt += 1
                elif res.status == "failed":
                    failed_cnt += 1

        summary = {
            "status": "success",
            "processed": processed_cnt,
            "unchanged": unchanged_cnt,
            "failed": failed_cnt,
            "total_files": processed_cnt + unchanged_cnt + failed_cnt
        }
        logger.info(f"📊 Directory Indexing Summary | Directory: '{dir_p.name}' | Processed: {processed_cnt} | Unchanged: {unchanged_cnt} | Failed: {failed_cnt}")
        return summary

    def remove_document(self, file_path: str) -> bool:
        """
        Purges indexed entry and associated chunk vectors for a deleted document.

        Args:
            file_path (str): Absolute file path string.

        Returns:
            bool: True if removed successfully.
        """
        p = Path(file_path).resolve()
        norm_path = str(p)

        if norm_path in self._document_registry:
            doc_id = self._document_registry[norm_path]["doc_id"]
            del self._document_registry[norm_path]
            # Clear chunks from DocumentManager
            if doc_id in self.doc_mgr._processed_chunks_store:
                del self.doc_mgr._processed_chunks_store[doc_id]
            logger.info(f"🗑️ Purged document vectors from index for '{p.name}' (Doc ID: '{doc_id}')")
            return True
        return False

    def get_indexed_summary(self) -> Dict[str, Any]:
        """Returns statistics summary of indexed documents."""
        total_chunks = sum(rec["chunks_count"] for rec in self._document_registry.values())
        return {
            "total_documents": len(self._document_registry),
            "total_chunks": total_chunks,
            "documents": list(self._document_registry.keys())
        }
