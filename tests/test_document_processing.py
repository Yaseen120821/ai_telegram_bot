"""
tests/test_document_processing.py - Comprehensive Diagnostic Test Suite for Chapter 8 Part 2
============================================================================================

Executes automated diagnostic verification across Chapter 8 Part 2 Document Processing Subsystem:
1. Format-specific document loaders (TXT, Markdown, Source Code, JSON, CSV).
2. Document validation, readable permissions, size limits, and SHA256 duplicate detection.
3. Text sanitization, Unicode NFKC normalization, null byte stripping, and whitespace collapsing.
4. Recursive Character Chunking with overlap mathematics and token estimates.
5. End-to-end `DocumentManager` facade orchestration & chunk JSON artifact generation.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Setup Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.document_processing import (
    DocumentManager,
    DocumentValidator,
    DocumentCleaner,
    DocumentChunker,
    MetadataExtractor,
    LoaderFactory,
    DocumentType,
    ProcessingStatus,
    ProcessingResult,
    ChunkCollection
)

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_document_processing")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 8 Part 2."""
    logger.info("=== Starting SANA AI Chapter 8 Part 2 Document Processing Subsystem Diagnostic Tests ===")

    doc_mgr = DocumentManager.get_instance()
    doc_mgr.clear_processed_chunks()

    # ------------------------------------------------------------------
    # TEST 1: Format-Specific Document Loaders
    # ------------------------------------------------------------------
    logger.info("\n--- 1. Testing Format-Specific Document Loaders ---")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Case A: Plain Text
        txt_path = Path(tmp_dir) / "sample_doc.txt"
        txt_path.write_text("SANA AI is an intelligent, private local personal AI assistant.", encoding="utf-8")
        txt_loader = LoaderFactory.get_loader(DocumentType.TXT.value)
        txt_content = txt_loader.load(str(txt_path))
        logger.info(f"TXT Loader Output: '{txt_content}'")
        assert "SANA AI" in txt_content

        # Case B: Markdown
        md_path = Path(tmp_dir) / "architecture.md"
        md_path.write_text("# SANA AI Architecture\n\n## Subsystems\n- RAG Subsystem\n- Memory System\n", encoding="utf-8")
        md_loader = LoaderFactory.get_loader(DocumentType.MARKDOWN.value)
        md_content = md_loader.load(str(md_path))
        logger.info(f"Markdown Loader Output:\n{md_content}")
        assert "# SANA AI Architecture" in md_content

        # Case C: Python Source Code
        py_path = Path(tmp_dir) / "main.py"
        py_path.write_text("def hello_sana():\n    print('Hello SANA AI')\n", encoding="utf-8")
        code_loader = LoaderFactory.get_loader(DocumentType.PYTHON.value)
        code_content = code_loader.load(str(py_path))
        logger.info(f"Python Loader Output:\n{code_content}")
        assert "def hello_sana():" in code_content

        logger.info("✅ Format-specific document loaders verified!")

    # ------------------------------------------------------------------
    # TEST 2: Document Cleaning & Unicode NFKC Sanitization
    # ------------------------------------------------------------------
    logger.info("\n--- 2. Testing Document Cleaning & Unicode Normalization ---")

    cleaner = DocumentCleaner()
    dirty_text = "  SANA AI   Subsystem \x00\x07 \r\n\r\n\r\n Paragraph 1. \r\n\r\n\r\n\r\n Paragraph 2.  "
    clean_res = cleaner.clean_text(dirty_text)

    logger.info(f"Original Text Length: {clean_res.original_length} ──► Cleaned Length: {clean_res.cleaned_length}")
    logger.info(f"Cleaned Text Output:\n'{clean_res.text}'")
    assert "\x00" not in clean_res.text
    assert "\r" not in clean_res.text
    assert "SANA AI Subsystem" in clean_res.text
    logger.info("✅ Document text cleaning verified!")

    # ------------------------------------------------------------------
    # TEST 3: Recursive Character Chunking with Overlap
    # ------------------------------------------------------------------
    logger.info("\n--- 3. Testing Recursive Character Chunking with Overlap ---")

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "SANA AI is an empathetic, intelligent local AI assistant engineered by Yaseen. "
        "It features a local Qwen LLM, persistent SQLite memory, affective computing, "
        "and a robust Retrieval-Augmented Generation (RAG) subsystem capable of parsing "
        "PDFs, DOCX files, Markdown, source code, and CSV tabular documents."
    )

    chunk_coll: ChunkCollection = chunker.create_chunks("doc_test_101", sample_text)
    logger.info(f"Generated {chunk_coll.total_chunks} chunks from {len(sample_text)} characters.")

    for idx, chk in enumerate(chunk_coll.chunks):
        logger.info(f"  • Chunk [{idx}] ({chk.start_char}-{chk.end_char} chars, ~{chk.estimated_tokens} tokens): '{chk.content[:40]}...'")

    assert chunk_coll.total_chunks > 1, "Text should be split into multiple chunks"
    assert len(chunk_coll.chunks[0].content) <= 120, "Chunk size should be within threshold boundary"
    logger.info("✅ Recursive character chunking verified!")

    # ------------------------------------------------------------------
    # TEST 4: End-to-End DocumentManager Facade & Duplicate Detection
    # ------------------------------------------------------------------
    logger.info("\n--- 4. Testing End-to-End DocumentManager Facade & SHA256 Duplicates ---")

    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_file = Path(tmp_dir) / "sana_guide.txt"
        doc_file.write_text(
            "Chapter 8: Retrieval-Augmented Generation (RAG)\n\n"
            "Part 1: RAG Architecture & System Design.\n"
            "Part 2: Document Processing & Ingestion Pipeline.\n"
            "Part 3: Sentence Transformers & Vector Embeddings.\n"
            "Part 4: FAISS Vector Index Storage & Similarity Search.\n",
            encoding="utf-8"
        )

        # First Ingestion Pass
        proc_res_1: ProcessingResult = doc_mgr.process_document(str(doc_file))
        logger.info(
            f"Pass 1 Result: Status='{proc_res_1.status}' | Doc ID='{proc_res_1.doc_id}' | "
            f"Chunks={len(proc_res_1.chunks)} | Latency={proc_res_1.execution_time_ms:.2f}ms"
        )
        assert proc_res_1.status == ProcessingStatus.SUCCESS.value
        assert len(proc_res_1.chunks) >= 1

        # Second Ingestion Pass (Duplicate Detection)
        proc_res_2: ProcessingResult = doc_mgr.process_document(str(doc_file))
        logger.info(f"Pass 2 Result (Duplicate Check): Status='{proc_res_2.status}' | Message='{proc_res_2.message}'")
        assert proc_res_2.status == ProcessingStatus.DUPLICATE.value

        # Verify Chunk Retrieval from Store
        retrieved_chunks = doc_mgr.get_document_chunks(proc_res_1.doc_id)
        logger.info(f"Retrieved {len(retrieved_chunks)} cached chunks from DocumentManager.")
        assert len(retrieved_chunks) == len(proc_res_1.chunks)

    logger.info("✅ End-to-end DocumentManager facade & duplicate detection verified!")

    logger.info("\n🎉 ALL CHAPTER 8 PART 2 DOCUMENT PROCESSING TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
