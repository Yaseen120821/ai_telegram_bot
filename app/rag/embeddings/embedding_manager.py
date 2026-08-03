"""
app/rag/embeddings/embedding_manager.py - Central Embedding Manager Facade
============================================================================

1. PURPOSE:
-----------
Acts as the central Thread-safe Singleton orchestrator facade for the Vector Embedding Subsystem.
Coordinates model loading, batch vector generation, numerical validation, and semantic similarity computation.

2. WHY IT EXISTS (FACADE PATTERN):
----------------------------------
Provides a unified public API for `RAGManager`, vector stores, and Telegram handlers.
Hides internal PyTorch CUDA allocations, HuggingFace transformers tokenization, Mean Pooling, and L2 norm calculations.

3. RESPONSIBILITIES:
--------------------
- Generate dense L2-normalized 384-d vectors for document chunks (`generate_embeddings`).
- Embed single text query statements (`embed_text`).
- Compute semantic Cosine Similarity scores between text statements (`calculate_similarity`).
- Validate generated vectors using `EmbeddingValidator`.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Orchestrates `EmbeddingModel`, `EmbeddingGenerator`, and `EmbeddingValidator`.
- Interfaced by `app/rag/rag_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
import threading
from typing import Optional, List, Dict, Any

from app.rag.embeddings.embedding_config import EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL, DEVICE
from app.rag.embeddings.embedding_types import EmbeddingStatus
from app.rag.embeddings.embedding_models import (
    EmbeddingVector,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingMetadata,
    EmbeddingStatistics
)
from app.rag.embeddings.embedding_model import EmbeddingModel
from app.rag.embeddings.embedding_generator import EmbeddingGenerator
from app.rag.embeddings.embedding_validator import EmbeddingValidator
from app.rag.embeddings.embedding_utils import EmbeddingUtils

logger = logging.getLogger("sana_ai.rag.embeddings.manager")


class EmbeddingManager:
    """
    Thread-safe Singleton facade for the Vector Embedding Subsystem.
    """
    _instance: Optional["EmbeddingManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if EmbeddingManager._instance is not None:
            raise RuntimeError(
                "EmbeddingManager is a Singleton! Use `EmbeddingManager.get_instance()` instead."
            )

        self.model_wrapper: EmbeddingModel = EmbeddingModel()
        self.generator: EmbeddingGenerator = EmbeddingGenerator(self.model_wrapper)
        self.validator: EmbeddingValidator = EmbeddingValidator()
        self.stats: EmbeddingStatistics = EmbeddingStatistics()

        logger.info("🧠 EmbeddingManager Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "EmbeddingManager":
        """
        Thread-safe accessor for the shared EmbeddingManager Singleton instance.

        Returns:
            EmbeddingManager: Shared Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # PUBLIC API METHODS
    # ------------------------------------------------------------------

    def generate_embeddings(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        normalize: bool = True
    ) -> EmbeddingResult:
        """
        Primary pipeline method: generates dense vectors for text passages and validates outputs.

        Args:
            texts (List[str]): List of text passage strings.
            doc_ids (Optional[List[str]]): List of document IDs.
            chunk_ids (Optional[List[str]]): List of chunk IDs.
            normalize (bool): Whether to apply L2 normalization.

        Returns:
            EmbeddingResult: EmbeddingResult dataclass object.
        """
        start_t = time.time()

        if not texts or len(texts) == 0:
            return EmbeddingResult(
                status=EmbeddingStatus.SUCCESS.value,
                vectors=[],
                dimension=EMBEDDING_DIMENSION,
                execution_time_ms=0.0,
                message="No text passages provided."
            )

        d_ids = doc_ids if doc_ids else [f"doc_{i}" for i in range(len(texts))]
        c_ids = chunk_ids if chunk_ids else [f"chunk_{i}" for i in range(len(texts))]

        req = EmbeddingRequest(texts=texts, doc_ids=d_ids, chunk_ids=c_ids, normalize=normalize)

        try:
            vectors = self.generator.generate_embeddings(req)
            
            # Validate every generated vector
            valid_vectors: List[EmbeddingVector] = []
            for vec in vectors:
                val_res = self.validator.validate_vector(vec.values, EMBEDDING_DIMENSION)
                if val_res.is_valid:
                    valid_vectors.append(vec)
                else:
                    logger.warning(f"⚠️ Rejecting invalid vector '{vec.vector_id}': {val_res.reason}")

            elapsed_ms = (time.time() - start_t) * 1000.0
            self.stats.total_vectors_generated += len(valid_vectors)

            logger.info(
                f"🎉 Embedding Generation Complete | Generated {len(valid_vectors)} dense vectors "
                f"[{EMBEDDING_DIMENSION}-d] | Elapsed: {elapsed_ms:.2f}ms"
            )

            return EmbeddingResult(
                status=EmbeddingStatus.SUCCESS.value,
                vectors=valid_vectors,
                dimension=EMBEDDING_DIMENSION,
                execution_time_ms=elapsed_ms,
                message=f"Successfully generated {len(valid_vectors)} embeddings."
            )

        except Exception as err:
            logger.error(f"❌ Exception in EmbeddingManager.generate_embeddings(): {err}", exc_info=True)
            return EmbeddingResult(
                status=EmbeddingStatus.FAILED.value,
                vectors=[],
                dimension=EMBEDDING_DIMENSION,
                execution_time_ms=(time.time() - start_t) * 1000.0,
                message=f"Embedding generation failed: {err}"
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single query string statement into a normalized 384-dimensional vector.

        Args:
            text (str): Query statement string.

        Returns:
            List[float]: 384-dimensional float vector.
        """
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIMENSION

        res = self.generate_embeddings([text])
        if res.vectors and len(res.vectors) > 0:
            return res.vectors[0].values
        return [0.0] * EMBEDDING_DIMENSION

    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculates semantic Cosine Similarity score between two text strings.

        Args:
            text_a (str): First text string.
            text_b (str): Second text string.

        Returns:
            float: Cosine similarity score (-1.0 to 1.0).
        """
        vec_a = self.embed_text(text_a)
        vec_b = self.embed_text(text_b)
        return EmbeddingUtils.cosine_similarity(vec_a, vec_b)

    def get_metadata(self) -> EmbeddingMetadata:
        """Returns EmbeddingMetadata container recording active model and hardware device."""
        return EmbeddingMetadata(
            model_name=DEFAULT_EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSION,
            device=self.model_wrapper.device,
            is_normalized=True
        )
