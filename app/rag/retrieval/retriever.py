r"""
app/rag/retrieval/retriever.py - Candidate Retrieval Coordinator
==================================================================

1. PURPOSE:
-----------
Coordinates candidate retrieval operations (combining Similarity Search, Metadata Filtering, Duplicate Removal, and Reranking).

2. WHY IT EXISTS (COORDINATOR PATTERN):
----------------------------------------
Decouples retrieval execution workflow from facade managers and system prompt builders.

3. RESPONSIBILITIES:
--------------------
- Search candidate chunks using `SimilaritySearchEngine`.
- Apply `MetadataFilter` rules.
- Strip redundant text passages via `DuplicateFilter`.
- Rerank candidate list using `RankingEngine`.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Orchestrates `SimilaritySearchEngine`, `MetadataFilter`, `DuplicateFilter`, and `RankingEngine`.
- Used by `retrieval_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Dict, Any, Optional

from app.rag.retrieval.retrieval_config import (
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    ENABLE_METADATA_FILTER,
    ENABLE_DUPLICATE_FILTER,
    ENABLE_RANKING
)
from app.rag.retrieval.retrieval_models import QueryRequest, RetrievedChunk, RankingResult
from app.rag.retrieval.similarity_search import SimilaritySearchEngine
from app.rag.retrieval.metadata_filter import MetadataFilter
from app.rag.retrieval.duplicate_filter import DuplicateFilter
from app.rag.retrieval.ranking_engine import RankingEngine

logger = logging.getLogger("sana_ai.rag.retrieval.retriever")


class Retriever:
    """
    Candidate retrieval coordinator executing search, filtering, deduplication, and ranking.
    """

    def __init__(self) -> None:
        self.search_engine: SimilaritySearchEngine = SimilaritySearchEngine()
        self.metadata_filter: MetadataFilter = MetadataFilter()
        self.duplicate_filter: DuplicateFilter = DuplicateFilter()
        self.ranking_engine: RankingEngine = RankingEngine()

    def retrieve_candidates(
        self,
        query_vector: List[float],
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Executes candidate retrieval search pipeline.

        Args:
            query_vector (List[float]): 384-dimensional query vector.
            top_k (int): Number of top candidate chunks to return.
            min_similarity (float): Similarity threshold cutoff score.
            filters (Optional[Dict[str, Any]]): Metadata filtering rules.

        Returns:
            List[RetrievedChunk]: Ranked and deduplicated candidate chunks list.
        """
        if not query_vector:
            return []

        # 1. Similarity Search over chunk embeddings
        candidates = self.search_engine.search_similar(query_vector, top_k=top_k * 2, min_threshold=min_similarity)
        if not candidates:
            return []

        # 2. Apply Metadata Filter
        if ENABLE_METADATA_FILTER and filters:
            candidates = self.metadata_filter.apply_filters(candidates, filters)

        # 3. Apply Duplicate Filter
        if ENABLE_DUPLICATE_FILTER:
            candidates = self.duplicate_filter.remove_duplicates(candidates)

        # 4. Apply Reranking Engine
        if ENABLE_RANKING:
            rank_res: RankingResult = self.ranking_engine.rank_candidates(candidates)
            candidates = rank_res.ranked_chunks

        top_candidates = candidates[:top_k]
        logger.debug(f"🎯 Retriever returned {len(top_candidates)} candidate chunks.")
        return top_candidates
