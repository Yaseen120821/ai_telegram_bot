"""
tests/test_rag_architecture.py - Comprehensive Diagnostic Test Suite for Chapter 8 Part 1
========================================================================================

Executes automated diagnostic verification across Chapter 8 Part 1 components:
1. RAG type enums (`DocumentType`, `RetrievalStatus`) & domain dataclass instantiation.
2. RAG configuration parameters & supported file extensions (`.pdf`, `.docx`, `.txt`, `.md`, `.py`).
3. RAG utility functions: query sanitization, file extension validation, ID generation, context formatting.
4. `RAGManager` Singleton facade public API execution and error handling.
5. Future architectural directory existence check (`document_processing`, `embeddings`, `retrieval`, `vector_store`).
"""

import sys
import logging
from pathlib import Path

# Setup Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import (
    RAGManager,
    RAGUtils,
    DocumentType,
    RetrievalStatus,
    DocumentMetadata,
    RetrievedChunk,
    RetrievalRequest,
    RetrievalResult,
    KnowledgeSource,
    SearchResult,
    KnowledgeContext,
    SUPPORTED_FILE_EXTENSIONS,
    MAX_RETRIEVED_DOCUMENTS
)

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_rag_architecture")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 8 Part 1."""
    logger.info("=== Starting SANA AI Chapter 8 Part 1 RAG Subsystem Diagnostic Tests ===")

    # ------------------------------------------------------------------
    # TEST 1: Future Directory Structure Verification
    # ------------------------------------------------------------------
    logger.info("\n--- 1. Testing Future Directory Architecture Existence ---")
    rag_base = PROJECT_ROOT / "app" / "rag"

    required_dirs = [
        "document_processing",
        "embeddings",
        "retrieval",
        "vector_store"
    ]

    for d in required_dirs:
        dir_path = rag_base / d
        init_file = dir_path / "__init__.py"
        logger.info(f"Directory 'app/rag/{d}': Exists? {dir_path.exists()} | Has __init__.py? {init_file.exists()}")
        assert dir_path.exists(), f"Directory app/rag/{d} must exist"
        assert init_file.exists(), f"Init file app/rag/{d}/__init__.py must exist"

    logger.info("✅ Future directory architecture verified!")

    # ------------------------------------------------------------------
    # TEST 2: RAG Types & Configuration Settings
    # ------------------------------------------------------------------
    logger.info("\n--- 2. Testing RAG Enums & Central Configuration ---")

    logger.info(f"Supported Extensions Count: {len(SUPPORTED_FILE_EXTENSIONS)}")
    assert ".pdf" in SUPPORTED_FILE_EXTENSIONS
    assert ".docx" in SUPPORTED_FILE_EXTENSIONS
    assert ".py" in SUPPORTED_FILE_EXTENSIONS

    assert DocumentType.PDF.value == "pdf"
    assert DocumentType.CODE.value == "code"
    assert RetrievalStatus.SUCCESS.value == "success"
    assert MAX_RETRIEVED_DOCUMENTS == 5

    logger.info("✅ Enums and Configuration parameters verified!")

    # ------------------------------------------------------------------
    # TEST 3: RAG Utility Functions
    # ------------------------------------------------------------------
    logger.info("\n--- 3. Testing RAG Utility Helpers ---")

    # Case A: Text Sanitization
    raw_q = "  What is   Retrieval-Augmented   Generation? \x00\x07 "
    clean_q = RAGUtils.clean_query_text(raw_q)
    logger.info(f"Query Sanitization: '{repr(raw_q)}' ──► '{clean_q}'")
    assert clean_q == "What is Retrieval-Augmented Generation?"

    # Case B: Extension Validation
    assert RAGUtils.validate_file_extension("handbook.pdf") == "pdf"
    assert RAGUtils.validate_file_extension("script.py") == "code"
    assert RAGUtils.validate_file_extension("notes.md") == "markdown"
    assert RAGUtils.validate_file_extension("unknown.xyz") == "unknown"

    # Case C: Deterministic ID Generation
    id_1 = RAGUtils.generate_unique_id("chunk", "Sample text passage content")
    id_2 = RAGUtils.generate_unique_id("chunk", "Sample text passage content")
    logger.info(f"Generated Unique Chunk ID: '{id_1}'")
    assert id_1 == id_2, "Same content must generate deterministic ID"

    # Case D: Context Formatting
    meta = DocumentMetadata(
        file_id="doc_101",
        filename="sana_architecture.pdf",
        file_type="pdf",
        file_path="/docs/sana_architecture.pdf",
        file_size_bytes=1024
    )
    chunk = RetrievedChunk(
        chunk_id="chunk_01",
        doc_id="doc_101",
        content="SANA AI uses RAG for dynamic external document knowledge retrieval.",
        score=0.92,
        metadata=meta
    )
    fmt_context = RAGUtils.format_knowledge_context([chunk])
    logger.info(f"Formatted Prompt Context Preview:\n{fmt_context}\n")
    assert "=== RELEVANT RETRIEVED KNOWLEDGE DOCUMENTS (RAG) ===" in fmt_context
    assert "sana_architecture.pdf" in fmt_context
    assert "92%" in fmt_context
    logger.info("✅ Utility functions verified!")

    # ------------------------------------------------------------------
    # TEST 4: RAGManager Singleton Facade API
    # ------------------------------------------------------------------
    logger.info("\n--- 4. Testing RAGManager Singleton Facade APIs ---")

    mgr_1 = RAGManager.get_instance()
    mgr_2 = RAGManager.get_instance()
    assert mgr_1 is mgr_2, "RAGManager must enforce Singleton pattern"

    # Test retrieval execution with sample chunk
    ret_res: RetrievalResult = mgr_1.retrieve("Explain RAG architecture")
    logger.info(f"Retrieval Result Status: '{ret_res.status}' | Found: {ret_res.total_found} chunks | Latency: {ret_res.execution_time_ms:.2f}ms")
    assert ret_res.status in (RetrievalStatus.NOT_FOUND.value, RetrievalStatus.SUCCESS.value)

    # Test context building and source extraction
    ctx = mgr_1.build_context("Explain RAG", [chunk])
    sources = mgr_1.get_sources([chunk])
    logger.info(f"Built Context Chunks Count: {ctx.chunks_count} | Sources Count: {len(sources)} | Source Name: '{sources[0].name}'")
    assert ctx.chunks_count == 1
    assert sources[0].name == "sana_architecture.pdf"
    assert mgr_1.validate_results(RetrievalResult(status=RetrievalStatus.SUCCESS.value, request_query="test", chunks=[chunk]))

    logger.info("✅ RAGManager facade APIs verified!")

    logger.info("\n🎉 ALL CHAPTER 8 PART 1 RAG SUBSYSTEM DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
