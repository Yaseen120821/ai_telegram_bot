"""
app/rag/retrieval package initializer.
Exposes public API for RAG Vector Search, Metadata Filtering, Duplicate Removal, Ranking, Context Building, and RetrievalManager.
"""

from app.rag.retrieval.retrieval_config import (
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    MAX_CONTEXT_TOKENS,
    MAX_SOURCES_PER_PROMPT,
    ENABLE_RANKING,
    SIMILARITY_WEIGHT,
    FRESHNESS_WEIGHT,
    METADATA_MATCH_WEIGHT,
    ENABLE_METADATA_FILTER,
    ENABLE_DUPLICATE_FILTER
)
from app.rag.retrieval.retrieval_types import (
    SearchMode,
    SimilarityMetric,
    RankingMode,
    ContextStatus,
    RetrievalStatus
)
from app.rag.retrieval.retrieval_models import (
    QueryRequest,
    QueryEmbedding,
    RetrievedChunk,
    RetrievalResult,
    RankingResult,
    KnowledgeContext,
    ContextStatistics
)
from app.rag.retrieval.retrieval_utils import RetrievalUtils
from app.rag.retrieval.similarity_search import SimilaritySearchEngine
from app.rag.retrieval.metadata_filter import MetadataFilter
from app.rag.retrieval.duplicate_filter import DuplicateFilter
from app.rag.retrieval.ranking_engine import RankingEngine
from app.rag.retrieval.context_builder import ContextBuilder
from app.rag.retrieval.retriever import Retriever
from app.rag.retrieval.retrieval_manager import RetrievalManager

__all__ = [
    "DEFAULT_TOP_K",
    "MIN_SIMILARITY_THRESHOLD",
    "MAX_CONTEXT_TOKENS",
    "MAX_SOURCES_PER_PROMPT",
    "ENABLE_RANKING",
    "SIMILARITY_WEIGHT",
    "FRESHNESS_WEIGHT",
    "METADATA_MATCH_WEIGHT",
    "ENABLE_METADATA_FILTER",
    "ENABLE_DUPLICATE_FILTER",
    "SearchMode",
    "SimilarityMetric",
    "RankingMode",
    "ContextStatus",
    "RetrievalStatus",
    "QueryRequest",
    "QueryEmbedding",
    "RetrievedChunk",
    "RetrievalResult",
    "RankingResult",
    "KnowledgeContext",
    "ContextStatistics",
    "RetrievalUtils",
    "SimilaritySearchEngine",
    "MetadataFilter",
    "DuplicateFilter",
    "RankingEngine",
    "ContextBuilder",
    "Retriever",
    "RetrievalManager"
]
