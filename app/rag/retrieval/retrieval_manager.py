"""
app/rag/retrieval/retrieval_manager.py - Central Retrieval Manager Facade
==========================================================================

1. PURPOSE:
-----------
Acts as the central Thread-safe Singleton orchestrator facade for the Vector Retrieval Subsystem.
Coordinates query embedding generation, candidate retrieval search, metadata filtering, duplicate removal,
composite reranking, and System Prompt knowledge context formatting.

2. WHY IT EXISTS (FACADE PATTERN):
----------------------------------
Provides a unified public API for `RAGManager` facade, Telegram bot handlers, and `PromptBuilder`.
Hides internal vector calculations, threshold filters, reranking formulas, and token budgeting.

3. RESPONSIBILITIES:
--------------------
- Process user search queries and return structured `RetrievalResult` objects (`retrieve`).
- Format ChatML System Prompt knowledge context blocks (`build_context`).
- Extract cited `KnowledgeSource` records (`get_sources`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `EmbeddingManager` from `app/rag/embeddings/embedding_manager.py`.
- Orchestrates `Retriever` and `ContextBuilder`.
- Interfaced by `app/rag/rag_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
import threading
from typing import Optional, List, Dict, Any

from app.rag.embeddings.embedding_manager import EmbeddingManager
from app.rag.retrieval.retrieval_config import DEFAULT_TOP_K, MIN_SIMILARITY_THRESHOLD
from app.rag.retrieval.retrieval_types import RetrievalStatus
from app.rag.retrieval.retrieval_models import (
    QueryRequest,
    RetrievedChunk,
    RetrievalResult,
    KnowledgeContext,
    ContextStatistics
)
from app.rag.retrieval.retriever import Retriever
from app.rag.retrieval.context_builder import ContextBuilder
from app.rag.rag_utils import RAGUtils

logger = logging.getLogger("sana_ai.rag.retrieval.manager")


class RetrievalManager:
    """
    Thread-safe Singleton facade for the Vector Retrieval Subsystem.
    """
    _instance: Optional["RetrievalManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if RetrievalManager._instance is not None:
            raise RuntimeError(
                "RetrievalManager is a Singleton! Use `RetrievalManager.get_instance()` instead."
            )

        self.embedding_mgr: EmbeddingManager = EmbeddingManager.get_instance()
        self.retriever: Retriever = Retriever()
        self.context_builder: ContextBuilder = ContextBuilder()

        logger.info("🔍 RetrievalManager Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "RetrievalManager":
        """
        Thread-safe accessor for the shared RetrievalManager Singleton instance.

        Returns:
            RetrievalManager: Shared Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # PUBLIC API METHODS
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Primary retrieval method: embeds user query into a 384-d vector, executes similarity search,
        applies filtering and composite ranking, and returns RetrievalResult payload.

        Args:
            query (str): User question statement.
            user_id (str): Telegram User ID string.
            top_k (int): Number of top candidate chunks to return.
            min_similarity (float): Similarity cutoff threshold score.
            filters (Optional[Dict[str, Any]]): Metadata filtering rules.

        Returns:
            RetrievalResult: RetrievalResult dataclass payload.
        """
        start_t = time.time()
        clean_q = RAGUtils.clean_query_text(query)

        if not clean_q:
            logger.warning("Empty search query provided to RetrievalManager.retrieve(). Returning EMPTY_QUERY status.")
            return RetrievalResult(
                status=RetrievalStatus.EMPTY_QUERY.value,
                query=query,
                chunks=[],
                execution_time_ms=0.0,
                total_found=0
            )

        try:
            # 1. Generate Query Vector Embedding
            query_vec = self.embedding_mgr.embed_text(clean_q)

            # 2. Execute Retrieval Pipeline (Similarity search + filtering + deduplication + ranking)
            candidates = self.retriever.retrieve_candidates(
                query_vector=query_vec,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters
            )

            elapsed_ms = (time.time() - start_t) * 1000.0
            status = RetrievalStatus.SUCCESS.value if candidates else RetrievalStatus.NOT_FOUND.value

            logger.info(
                f"🎯 Retrieval Query Executed | Query: '{clean_q[:30]}...' | "
                f"Status: '{status}' | Candidates Retrieved: {len(candidates)} | Elapsed: {elapsed_ms:.2f}ms"
            )

            return RetrievalResult(
                status=status,
                query=clean_q,
                chunks=candidates,
                execution_time_ms=elapsed_ms,
                total_found=len(candidates)
            )

        except Exception as err:
            logger.error(f"❌ Exception in RetrievalManager.retrieve() for '{clean_q}': {err}", exc_info=True)
            return RetrievalResult(
                status=RetrievalStatus.FAILED.value,
                query=clean_q,
                chunks=[],
                execution_time_ms=(time.time() - start_t) * 1000.0,
                total_found=0
            )

    def build_context(self, query: str, chunks: List[RetrievedChunk]) -> KnowledgeContext:
        """
        Formats candidate chunks into a structured ChatML System Prompt knowledge context block.

        Args:
            query (str): User question statement.
            chunks (List[RetrievedChunk]): List of retrieved candidate chunks.

        Returns:
            KnowledgeContext: KnowledgeContext object.
        """
        return self.context_builder.build_context(query, chunks)
