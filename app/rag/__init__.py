"""
app/rag package initializer.
Exposes public API for SANA AI Retrieval-Augmented Generation (RAG) Subsystem.
"""

from app.rag.rag_config import (
    SUPPORTED_FILE_EXTENSIONS,
    MAX_RETRIEVED_DOCUMENTS,
    MAX_CHUNKS_PER_QUERY,
    RETRIEVAL_LIMIT,
    MIN_RELEVANCE_SCORE_THRESHOLD,
    PERFORMANCE_THRESHOLD_MS,
    INPUT_MAX_QUERY_CHARS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    VECTOR_STORE_DIR
)
from app.rag.rag_types import DocumentType, RetrievalStatus
from app.rag.rag_models import (
    DocumentMetadata,
    RetrievedChunk,
    ChunkReference,
    RetrievalRequest,
    RetrievalResult,
    KnowledgeSource,
    SearchResult,
    KnowledgeContext
)
from app.rag.rag_utils import RAGUtils
from app.rag.index_manager import IndexManager
from app.rag.cache_manager import CacheManager
from app.rag.performance_monitor import PerformanceMonitor
from app.rag.metrics import MetricsCollector
from app.rag.health_check import HealthCheckEngine
from app.rag.rag_manager import RAGManager

__all__ = [
    "SUPPORTED_FILE_EXTENSIONS",
    "MAX_RETRIEVED_DOCUMENTS",
    "MAX_CHUNKS_PER_QUERY",
    "RETRIEVAL_LIMIT",
    "MIN_RELEVANCE_SCORE_THRESHOLD",
    "PERFORMANCE_THRESHOLD_MS",
    "INPUT_MAX_QUERY_CHARS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "VECTOR_STORE_DIR",
    "DocumentType",
    "RetrievalStatus",
    "DocumentMetadata",
    "RetrievedChunk",
    "ChunkReference",
    "RetrievalRequest",
    "RetrievalResult",
    "KnowledgeSource",
    "SearchResult",
    "KnowledgeContext",
    "RAGUtils",
    "IndexManager",
    "CacheManager",
    "PerformanceMonitor",
    "MetricsCollector",
    "HealthCheckEngine",
    "RAGManager"
]
