"""
app/rag/retrieval/retrieval_utils.py - Retrieval Utility Helper Engine
========================================================================

1. PURPOSE:
-----------
Provides sorting, token estimation, similarity score validation, and citation formatting helper utilities for the Retrieval Subsystem.

2. WHY IT EXISTS:
-----------------
Decouples candidate sorting, token counting, and citation string generation from retrievers, rankers, and context builders.

3. RESPONSIBILITIES:
--------------------
- Sort candidate chunks by similarity or composite score (`sort_chunks_by_score`).
- Estimate token consumption for context strings (`estimate_context_tokens`).
- Validate similarity score bounds (`validate_similarity_score`).
- Format source citation strings (`format_source_citation`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `retriever.py`, `ranking_engine.py`, `context_builder.py`, and `retrieval_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List

from app.rag.retrieval.retrieval_models import RetrievedChunk

logger = logging.getLogger("sana_ai.rag.retrieval.utils")


class RetrievalUtils:
    """
    Utility helper class for candidate sorting, token estimation, and citation formatting.
    """

    @staticmethod
    def sort_chunks_by_score(
        chunks: List[RetrievedChunk],
        use_composite: bool = True
    ) -> List[RetrievedChunk]:
        """
        Sorts candidate chunks in descending order of similarity or composite score.

        Args:
            chunks (List[RetrievedChunk]): Input candidate chunks list.
            use_composite (bool): Sort by composite_score if True, else similarity_score.

        Returns:
            List[RetrievedChunk]: Sorted candidate chunks list.
        """
        if not chunks:
            return []

        if use_composite:
            return sorted(chunks, key=lambda c: c.composite_score, reverse=True)
        return sorted(chunks, key=lambda c: c.similarity_score, reverse=True)

    @staticmethod
    def estimate_context_tokens(text: str) -> int:
        """
        Estimates LLM token count for a context string (~4 chars per token).

        Args:
            text (str): Context string statement.

        Returns:
            int: Estimated token count.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def validate_similarity_score(score: float) -> bool:
        """
        Validates if similarity score is within valid numerical bounds (-1.0 to 1.0).

        Args:
            score (float): Similarity score float.

        Returns:
            bool: True if valid float.
        """
        import math
        if math.isnan(score) or math.isinf(score):
            return False
        return -1.0 <= score <= 1.0

    @staticmethod
    def format_source_citation(chunk: RetrievedChunk) -> str:
        """
        Formats a clean citation header for a retrieved chunk.

        Args:
            chunk (RetrievedChunk): RetrievedChunk object.

        Returns:
            str: Citation string (e.g. 'sana_architecture.pdf (Match: 92%)').
        """
        doc_name = chunk.metadata.filename if chunk.metadata else f"Document_{chunk.doc_id}"
        pct = int(chunk.similarity_score * 100) if chunk.similarity_score <= 1.0 else int(chunk.similarity_score)
        return f"{doc_name} (Relevance Match: {pct}%)"
