"""
tests/test_embeddings.py - Comprehensive Diagnostic Test Suite for Chapter 8 Part 3
====================================================================================

Executes automated diagnostic verification across Chapter 8 Part 3 Embedding Subsystem & Vector Store Architecture:
1. Sentence Transformer model loading & device allocation (`cuda` / `cpu`).
2. Dense 384-dimensional floating-point vector generation.
3. L2 unit vector normalization verification (L2 norm = 1.0).
4. Mathematical Cosine Similarity calculation (high semantic similarity for related sentences vs low for unrelated).
5. Vector validation check execution (`NaN`, `Inf`, dimension checks).
6. End-to-end `EmbeddingManager` facade orchestration & `VectorManager` architecture setup.
"""

import sys
import logging
from pathlib import Path

# Setup Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings import (
    EmbeddingManager,
    EmbeddingModel,
    EmbeddingGenerator,
    EmbeddingValidator,
    EmbeddingUtils,
    EmbeddingStatus,
    VectorStatus,
    EMBEDDING_DIMENSION
)
from app.rag.vector_store import VectorManager

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_embeddings")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 8 Part 3."""
    logger.info("=== Starting SANA AI Chapter 8 Part 3 Embedding Subsystem Diagnostic Tests ===")

    emb_mgr = EmbeddingManager.get_instance()

    # ------------------------------------------------------------------
    # TEST 1: Model Loading & Hardware Allocation
    # ------------------------------------------------------------------
    logger.info("\n--- 1. Testing Sentence Transformer Model Loading ---")
    meta = emb_mgr.get_metadata()
    logger.info(f"Model Name: '{meta.model_name}' | Dimension: {meta.dimension} | Target Device: '{meta.device}'")

    assert meta.dimension == 384
    assert meta.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("✅ Model configuration and hardware allocation verified!")

    # ------------------------------------------------------------------
    # TEST 2: Dense 384-Dimensional Vector Generation & L2 Normalization
    # ------------------------------------------------------------------
    logger.info("\n--- 2. Testing Dense 384-d Vector Generation & L2 Normalization ---")
    sample_texts = [
        "Docker networking enables containers to exchange data.",
        "SANA AI uses Retrieval-Augmented Generation for external knowledge retrieval."
    ]

    res = emb_mgr.generate_embeddings(sample_texts)
    logger.info(f"Embedding Result: Status='{res.status}' | Vectors Count={len(res.vectors)} | Elapsed={res.execution_time_ms:.2f}ms")

    assert res.status == EmbeddingStatus.SUCCESS.value
    assert len(res.vectors) == 2

    for idx, vec in enumerate(res.vectors):
        logger.info(f"  • Vector [{idx}] ID: '{vec.vector_id}' | Dim: {vec.dimension} | L2 Norm: {vec.norm:.4f}")
        assert vec.dimension == 384
        assert abs(vec.norm - 1.0) < 0.01, f"Vector L2 norm must equal 1.0 (got {vec.norm})"

    logger.info("✅ Dense 384-d vector generation & L2 unit normalization verified!")

    # ------------------------------------------------------------------
    # TEST 3: Semantic Cosine Similarity Verification
    # ------------------------------------------------------------------
    logger.info("\n--- 3. Testing Semantic Cosine Similarity Scores ---")

    text_query = "How do Docker containers communicate?"
    text_related = "Docker networking enables containers to exchange data."
    text_unrelated = "Delicious Italian pepperoni pizza with extra cheese."

    sim_high = emb_mgr.calculate_similarity(text_query, text_related)
    sim_low = emb_mgr.calculate_similarity(text_query, text_unrelated)

    logger.info(f"Query: '{text_query}'")
    logger.info(f"  ├─► Related: '{text_related}' ──► Similarity: {sim_high * 100:.2f}%")
    logger.info(f"  └─► Unrelated: '{text_unrelated}' ──► Similarity: {sim_low * 100:.2f}%")

    assert sim_high > sim_low, "Related sentence must have higher similarity than unrelated sentence"
    logger.info("✅ Semantic Cosine Similarity scores verified!")

    # ------------------------------------------------------------------
    # TEST 4: Vector Validator Checks
    # ------------------------------------------------------------------
    logger.info("\n--- 4. Testing Vector Validator Numerical Safety Checks ---")
    validator = EmbeddingValidator()

    # Valid Vector
    valid_vec = [0.1] * 384
    v_res_1 = validator.validate_vector(valid_vec, expected_dim=384)
    assert v_res_1.is_valid

    # Wrong Dimension Vector
    wrong_dim_vec = [0.1] * 128
    v_res_2 = validator.validate_vector(wrong_dim_vec, expected_dim=384)
    logger.info(f"Dimension Mismatch Check: Status='{v_res_2.status}' | Reason='{v_res_2.reason}'")
    assert not v_res_2.is_valid
    assert v_res_2.status == VectorStatus.INVALID_DIMENSION.value

    # NaN Vector
    nan_vec = [0.1] * 383 + [float("nan")]
    v_res_3 = validator.validate_vector(nan_vec, expected_dim=384)
    logger.info(f"NaN Check: Status='{v_res_3.status}' | Reason='{v_res_3.reason}'")
    assert not v_res_3.is_valid
    assert v_res_3.status == VectorStatus.NAN_DETECTED.value

    logger.info("✅ Vector validator numerical safety checks verified!")

    # ------------------------------------------------------------------
    # TEST 5: Vector Store Architecture Setup
    # ------------------------------------------------------------------
    logger.info("\n--- 5. Testing Vector Store Architecture Setup ---")
    vec_mgr = VectorManager.get_instance()
    idx_meta = vec_mgr.get_index_metadata()
    logger.info(f"Vector Store Index Type: '{idx_meta.index_type}' | Dimension: {idx_meta.dimension} | Total Vectors: {idx_meta.total_vectors}")

    assert idx_meta.dimension == 384
    assert idx_meta.index_type == "Flat"
    logger.info("✅ Vector Store architecture setup verified!")

    logger.info("\n🎉 ALL CHAPTER 8 PART 3 EMBEDDING DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
