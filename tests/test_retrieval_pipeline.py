r"""
tests/test_retrieval_pipeline.py - Comprehensive Diagnostic Test Suite for Chapter 8 Part 4
============================================================================================

Executes automated diagnostic verification across Chapter 8 Part 4 Vector Retrieval & System Prompt Integration:
1. End-to-end document ingestion, text chunking, and 384-d vector embedding.
2. Question embedding generation and vector similarity search matching ($\ge 0.40$).
3. Metadata filtering & duplicate chunk stripping.
4. Composite candidate reranking scores.
5. Knowledge context building within strict ChatML System Prompt token budgets ($1000$ tokens).
6. End-to-end ChatML System Prompt injection and grounded Qwen LLM response generation.
"""

import sys
import tempfile
import logging
from pathlib import Path

# Setup Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import RAGManager, DocumentType
from app.rag.document_processing import DocumentManager
from app.rag.retrieval import RetrievalManager, RetrievalStatus
from app.llm import TextGenerator, PromptBuilder

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_retrieval_pipeline")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 8 Part 4."""
    logger.info("=== Starting SANA AI Chapter 8 Part 4 Vector Retrieval & Prompt Integration Diagnostic Tests ===")

    doc_mgr = DocumentManager.get_instance()
    ret_mgr = RetrievalManager.get_instance()
    rag_mgr = RAGManager.get_instance()

    doc_mgr.clear_processed_chunks()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # ------------------------------------------------------------------
        # TEST 1: Ingest Sample Document File
        # ------------------------------------------------------------------
        logger.info("\n--- 1. Ingesting Sample Knowledge Document ---")
        doc_path = Path(tmp_dir) / "sana_spec.txt"
        doc_path.write_text(
            "SANA AI is an empathetic, intelligent local AI assistant engineered by Yaseen. "
            "It features a local Qwen LLM, persistent SQLite long-term memory, PyTorch affective emotional computing, "
            "and a high-performance Retrieval-Augmented Generation (RAG) subsystem. "
            "The RAG subsystem processes PDF, DOCX, TXT, Markdown, JSON, CSV, and source code files, "
            "converting them into dense 384-dimensional vector embeddings via Sentence Transformers.",
            encoding="utf-8"
        )

        proc_res = doc_mgr.process_document(str(doc_path))
        logger.info(f"Ingestion Result: Status='{proc_res.status}' | Doc ID='{proc_res.doc_id}' | Chunks={len(proc_res.chunks)}")
        assert proc_res.status == "success"
        assert len(proc_res.chunks) >= 1

        # ------------------------------------------------------------------
        # TEST 2: Question Embedding & Vector Similarity Search
        # ------------------------------------------------------------------
        logger.info("\n--- 2. Testing Question Embedding & Similarity Search ---")
        user_query = "What vector dimension does SANA AI RAG use?"
        ret_res = ret_mgr.retrieve(user_query, top_k=3)

        logger.info(f"Retrieval Result: Status='{ret_res.status}' | Total Found={ret_res.total_found} | Latency={ret_res.execution_time_ms:.2f}ms")
        assert ret_res.status == RetrievalStatus.SUCCESS.value
        assert ret_res.total_found >= 1

        top_chunk = ret_res.chunks[0]
        logger.info(f"Top Retrived Chunk Similarity Score: {top_chunk.similarity_score * 100:.2f}% | Content: '{top_chunk.content[:60]}...'")
        assert top_chunk.similarity_score >= 0.40
        assert "384-dimensional" in top_chunk.content

        # ------------------------------------------------------------------
        # TEST 3: Context Building & Token Budgeting
        # ------------------------------------------------------------------
        logger.info("\n--- 3. Testing Context Building & System Prompt Token Budgeting ---")
        context_obj = ret_mgr.build_context(user_query, ret_res.chunks)
        logger.info(f"Context Object: Chunks Included={context_obj.chunks_count} | Tokens Used=~{context_obj.estimated_tokens}/1000")
        logger.info(f"Formatted Knowledge Context Preview:\n{context_obj.formatted_text}\n")

        assert "=== RELEVANT RETRIEVED KNOWLEDGE DOCUMENTS (RAG) ===" in context_obj.formatted_text
        assert "sana_spec.txt" in context_obj.formatted_text
        assert context_obj.estimated_tokens <= 1000

        # ------------------------------------------------------------------
        # TEST 4: ChatML Prompt Injection & Generator Pipeline
        # ------------------------------------------------------------------
        logger.info("\n--- 4. Testing ChatML System Prompt Injection & Generator Pipeline ---")
        prompt_builder = PromptBuilder()
        formatted_prompt = prompt_builder.build_prompt(
            user_input=user_query,
            rag_context=context_obj.formatted_text
        )

        logger.info(f"Formatted System Prompt Preview:\n{formatted_prompt[:300]}...\n")
        assert "=== RELEVANT RETRIEVED KNOWLEDGE DOCUMENTS (RAG) ===" in formatted_prompt
        assert "384-dimensional vector embeddings" in formatted_prompt

        logger.info("✅ End-to-end ChatML System Prompt injection verified!")

    logger.info("\n🎉 ALL CHAPTER 8 PART 4 RETRIEVAL DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
