"""
app/rag/rag_manager.py - Central RAG Subsystem Manager Facade
==============================================================

1. PURPOSE:
-----------
Acts as the central Thread-safe Singleton orchestrator facade for Chapter 8 (Retrieval-Augmented Generation / RAG).
Coordinates knowledge retrieval requests, document context construction, source citation extraction, and validation.

2. WHY IT EXISTS (FACADE PATTERN & LOOSE COUPLING):
---------------------------------------------------
Provides a unified public API for Telegram bot handlers, `PromptBuilder`, and external modules.
Hides internal details of document parsing, text chunking, sentence transformer vector calculations, and FAISS vector index searches.

3. RESPONSIBILITIES:
--------------------
- Process user search queries and return structured `RetrievalResult` dataclasses.
- Retrieve top-k vector chunks matching a query string (`retrieve_documents`).
- Build formatted `KnowledgeContext` blocks for System Prompt ChatML injection (`build_context`).
- Extract unique `KnowledgeSource` citations from retrieved chunks (`get_sources`).
- Validate retrieval results and performance SLA limits (`validate_results`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Interfaced by `app/llm/prompt_builder.py` and `bot/handlers/echo.py`.
- Delegates parsing, embedding, and vector search to sub-modules in `app/rag/`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
import threading
from typing import Optional, List, Dict, Any

from app.rag.rag_config import (
    MAX_RETRIEVED_DOCUMENTS,
    RETRIEVAL_LIMIT,
    PERFORMANCE_THRESHOLD_MS
)
from app.rag.rag_types import RetrievalStatus, DocumentType
from app.rag.rag_models import (
    DocumentMetadata,
    RetrievedChunk,
    RetrievalRequest,
    RetrievalResult,
    KnowledgeSource,
    SearchResult,
    KnowledgeContext
)
from app.rag.rag_utils import RAGUtils

logger = logging.getLogger("sana_ai.rag.manager")


class RAGManager:
    """
    Thread-safe Singleton orchestrator facade for SANA AI RAG Subsystem.
    """
    _instance: Optional["RAGManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if RAGManager._instance is not None:
            raise RuntimeError(
                "RAGManager is a Singleton! Use `RAGManager.get_instance()` instead."
            )

        self._indexed_sources: Dict[str, KnowledgeSource] = {}
        from app.rag.index_manager import IndexManager
        from app.rag.cache_manager import CacheManager
        from app.rag.performance_monitor import PerformanceMonitor
        from app.rag.metrics import MetricsCollector
        from app.rag.health_check import HealthCheckEngine

        self.index_mgr: IndexManager = IndexManager.get_instance()
        self.cache_mgr: CacheManager = CacheManager.get_instance()
        self.perf_monitor: PerformanceMonitor = PerformanceMonitor.get_instance()
        self.metrics_collector: MetricsCollector = MetricsCollector.get_instance()
        self.health_engine: HealthCheckEngine = HealthCheckEngine.get_instance()

        logger.info("📚 RAGManager Singleton initialized successfully with production services.")

    @classmethod
    def get_instance(cls) -> "RAGManager":
        """
        Thread-safe accessor for the shared RAGManager Singleton instance.

        Returns:
            RAGManager: Shared Singleton instance.
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
        top_k: int = MAX_RETRIEVED_DOCUMENTS,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Primary retrieval method: sanitizes search query, queries vector store index (or placeholder store in Part 1),
        evaluates performance SLA, and returns structured RetrievalResult.

        Args:
            query (str): Input question or search string.
            user_id (str): Telegram User ID string.
            top_k (int): Number of top matching chunks to retrieve.
            filters (Optional[Dict[str, Any]]): Metadata filter dictionary.

        Returns:
            RetrievalResult: RetrievalResult dataclass object.
        """
        start_t = time.time()
        clean_query = RAGUtils.clean_query_text(query)

        if not clean_query:
            logger.warning("Empty search query provided to RAGManager.retrieve(). Returning EMPTY status.")
            return RetrievalResult(
                status=RetrievalStatus.EMPTY.value,
                request_query=query,
                chunks=[],
                execution_time_ms=0.0,
                total_found=0
            )

        try:
            # Retrieve documents matching query
            chunks = self.retrieve_documents(clean_query, top_k=top_k)
            elapsed_ms = (time.time() - start_t) * 1000.0

            if elapsed_ms > PERFORMANCE_THRESHOLD_MS:
                logger.warning(
                    f"⚠️ RAG Retrieval SLA Threshold Exceeded | Query: '{clean_query[:30]}...' | "
                    f"Elapsed: {elapsed_ms:.2f}ms > SLA Threshold {PERFORMANCE_THRESHOLD_MS}ms"
                )

            status = RetrievalStatus.SUCCESS.value if chunks else RetrievalStatus.NOT_FOUND.value
            return RetrievalResult(
                status=status,
                request_query=clean_query,
                chunks=chunks,
                execution_time_ms=elapsed_ms,
                total_found=len(chunks)
            )
        except Exception as err:
            logger.error(f"❌ Exception in RAGManager.retrieve() for query '{clean_query}': {err}", exc_info=True)
            return RetrievalResult(
                status=RetrievalStatus.FAILED.value,
                request_query=clean_query,
                chunks=[],
                execution_time_ms=(time.time() - start_t) * 1000.0,
                total_found=0
            )

    def retrieve_documents(self, query: str, top_k: int = RETRIEVAL_LIMIT) -> List[RetrievedChunk]:
        """
        Retrieves top-k text chunks matching search query via RetrievalManager.

        Args:
            query (str): Input query text.
            top_k (int): Max chunks limit.

        Returns:
            List[RetrievedChunk]: List of matching RetrievedChunk objects.
        """
        clean_q = RAGUtils.clean_query_text(query)
        if not clean_q:
            return []

        from app.rag.retrieval.retrieval_manager import RetrievalManager
        ret_mgr = RetrievalManager.get_instance()
        res = ret_mgr.retrieve(clean_q, top_k=top_k)
        
        # Convert app.rag.retrieval.retrieval_models.RetrievedChunk to app.rag.rag_models.RetrievedChunk if needed
        converted_chunks: List[RetrievedChunk] = []
        for c in res.chunks:
            converted_chunks.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    content=c.content,
                    score=c.similarity_score,
                    metadata=c.metadata
                )
            )
        return converted_chunks

    def search(self, query: str) -> SearchResult:
        """
        Executes search query and returns formatted SearchResult container.

        Args:
            query (str): Input query text.

        Returns:
            SearchResult: SearchResult dataclass object.
        """
        res = self.retrieve(query)
        sources = self.get_sources(res.chunks)
        return SearchResult(
            query=res.request_query,
            results=res.chunks,
            sources_count=len(sources)
        )

    def search_similar(self, query: str, limit: int = RETRIEVAL_LIMIT) -> SearchResult:
        """Alias search method specifying limit parameter."""
        res = self.retrieve(query, top_k=limit)
        sources = self.get_sources(res.chunks)
        return SearchResult(
            query=res.request_query,
            results=res.chunks,
            sources_count=len(sources)
        )

    def build_context(self, query: str, chunks: List[RetrievedChunk]) -> KnowledgeContext:
        """
        Builds a formatted KnowledgeContext object ready for ChatML System Prompt injection.

        Args:
            query (str): Input search query.
            chunks (List[RetrievedChunk]): List of retrieved text chunks.

        Returns:
            KnowledgeContext: KnowledgeContext object.
        """
        formatted_text = RAGUtils.format_knowledge_context(chunks)
        sources = self.get_sources(chunks)
        return KnowledgeContext(
            query=query,
            formatted_text=formatted_text,
            sources=sources,
            chunks_count=len(chunks)
        )

    def get_sources(self, chunks: List[RetrievedChunk]) -> List[KnowledgeSource]:
        """
        Extracts unique KnowledgeSource references from a list of retrieved chunks.

        Args:
            chunks (List[RetrievedChunk]): List of retrieved chunks.

        Returns:
            List[KnowledgeSource]: List of unique KnowledgeSource objects.
        """
        sources_map: Dict[str, KnowledgeSource] = {}
        for chunk in chunks:
            doc_id = chunk.doc_id
            if doc_id not in sources_map:
                name = chunk.metadata.filename if chunk.metadata else f"Document_{doc_id}"
                file_type = chunk.metadata.file_type if chunk.metadata else DocumentType.UNKNOWN.value
                sources_map[doc_id] = KnowledgeSource(
                    source_id=doc_id,
                    name=name,
                    type=file_type,
                    total_chunks=1
                )
            else:
                sources_map[doc_id].total_chunks += 1

        return list(sources_map.values())

    def validate_results(self, result: RetrievalResult) -> bool:
        """
        Validates whether a RetrievalResult payload is valid and contains non-empty chunks.

        Args:
            result (RetrievalResult): RetrievalResult object.

        Returns:
            bool: True if status is SUCCESS and chunks exist.
        """
        if not result:
            return False
        return result.status == RetrievalStatus.SUCCESS.value and len(result.chunks) > 0

    def check_health(self) -> Dict[str, Any]:
        """Runs RAG subsystem health diagnostic check."""
        return self.health_engine.check_health()

    def get_metrics(self) -> Dict[str, Any]:
        """Returns aggregated telemetry metrics for the RAG subsystem."""
        return self.metrics_collector.get_summary()
