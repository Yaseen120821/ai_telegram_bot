r"""
app/rag/retrieval/ranking_engine.py - Composite Candidate Reranking Engine
============================================================================

1. PURPOSE:
-----------
Reranks retrieved candidate chunks using a multi-factor composite scoring algorithm combining Cosine Similarity,
document freshness, and metadata matching.

2. WHY IT EXISTS (COMPOSITE RELEVANCE SCORING):
-----------------------------------------------
Pure cosine similarity can be misleading if an older, outdated file happens to match keywords slightly better.
`RankingEngine` calculates a composite score:

$$\text{Composite Score} = 0.70 \cdot \text{Similarity} + 0.15 \cdot \text{Freshness} + 0.15 \cdot \text{Metadata Match}$$

3. RESPONSIBILITIES:
--------------------
- Compute composite score for each candidate chunk.
- Rerank candidate list in descending order of composite score.
- Return structured `RankingResult` payload.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `SIMILARITY_WEIGHT`, `FRESHNESS_WEIGHT`, `METADATA_MATCH_WEIGHT` from `retrieval_config.py`.
- Used by `retriever.py`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
from typing import List

from app.rag.retrieval.retrieval_config import (
    SIMILARITY_WEIGHT,
    FRESHNESS_WEIGHT,
    METADATA_MATCH_WEIGHT
)
from app.rag.retrieval.retrieval_models import RetrievedChunk, RankingResult

logger = logging.getLogger("sana_ai.rag.retrieval.ranking")


class RankingEngine:
    """
    Composite candidate reranking engine.
    """

    def rank_candidates(self, chunks: List[RetrievedChunk]) -> RankingResult:
        """
        Calculates composite scores and reranks candidate chunks.

        Args:
            chunks (List[RetrievedChunk]): Input candidate chunks list.

        Returns:
            RankingResult: RankingResult dataclass container.
        """
        if not chunks:
            return RankingResult(ranked_chunks=[], top_score=0.0)

        now = time.time()
        for chunk in chunks:
            sim_score = chunk.similarity_score

            # Calculate Freshness Score (1.0 for fresh, degrading to 0.5 for 30+ day old files)
            freshness_score = 1.0
            if chunk.metadata and chunk.metadata.created_at:
                age_days = (now - chunk.metadata.created_at) / 86400.0
                freshness_score = max(0.5, 1.0 - (age_days / 60.0))

            # Calculate Metadata Quality Score
            metadata_score = 1.0 if chunk.metadata else 0.5

            # Composite Score Calculation
            composite = (
                (SIMILARITY_WEIGHT * sim_score) +
                (FRESHNESS_WEIGHT * freshness_score) +
                (METADATA_MATCH_WEIGHT * metadata_score)
            )

            chunk.composite_score = round(composite, 4)

        # Rerank candidates descending
        ranked = sorted(chunks, key=lambda c: c.composite_score, reverse=True)
        top_score = ranked[0].composite_score if ranked else 0.0

        logger.debug(f"📊 RankingEngine: Reranked {len(ranked)} candidates | Top Score: {top_score:.4f}")
        return RankingResult(ranked_chunks=ranked, top_score=top_score)
