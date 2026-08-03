r"""
app/rag/retrieval/similarity_search.py - Vector Similarity Search Engine
===========================================================================

1. PURPOSE:
-----------
Executes vector similarity search comparing a 384-dimensional query vector against indexed document chunk embeddings.

2. WHY IT EXISTS (SEMANTIC NEAREST-NEIGHBOR MATCHING):
------------------------------------------------------
Compares query embeddings with chunk embeddings using Cosine Similarity:

$$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\|_2 \|\mathbf{c}\|_2}$$

Filters candidates falling below `MIN_SIMILARITY_THRESHOLD` (0.40) to prevent ungrounded AI hallucinations.

3. RESPONSIBILITIES:
--------------------
- Compare query vector against processed document chunk embeddings.
- Filter candidates meeting minimum similarity score threshold ($\ge 0.40$).
- Construct `RetrievedChunk` objects with calculated similarity scores.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `EmbeddingManager` from `app/rag/embeddings/embedding_manager.py`.
- Uses `DocumentManager` from `app/rag/document_processing/document_manager.py`.
- Called by `retriever.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Dict, Any

from app.rag.embeddings.embedding_manager import EmbeddingManager
from app.rag.embeddings.embedding_utils import EmbeddingUtils
from app.rag.document_processing.document_manager import DocumentManager
from app.rag.retrieval.retrieval_config import MIN_SIMILARITY_THRESHOLD, DEFAULT_TOP_K
from app.rag.retrieval.retrieval_models import RetrievedChunk

logger = logging.getLogger("sana_ai.rag.retrieval.search")


class SimilaritySearchEngine:
    """
    Vector similarity search engine computing Cosine Similarity across chunk embeddings.
    """

    def __init__(self) -> None:
        """Initializes dependencies on EmbeddingManager and DocumentManager."""
        self.embedding_mgr: EmbeddingManager = EmbeddingManager.get_instance()
        self.doc_mgr: DocumentManager = DocumentManager.get_instance()

    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = DEFAULT_TOP_K,
        min_threshold: float = MIN_SIMILARITY_THRESHOLD
    ) -> List[RetrievedChunk]:
        """
        Executes vector similarity search returning top-k candidates meeting threshold.

        Args:
            query_vector (List[float]): 384-dimensional query vector.
            top_k (int): Number of top vector candidates to return.
            min_threshold (float): Similarity threshold cutoff score.

        Returns:
            List[RetrievedChunk]: Candidates meeting threshold.
        """
        if not query_vector:
            return []

        candidates: List[RetrievedChunk] = []

        # Iterate over all processed chunks in DocumentManager memory/disk store
        for doc_id, chunk_list in self.doc_mgr._processed_chunks_store.items():
            for chunk in chunk_list:
                # Generate embedding for chunk content
                chunk_vec = self.embedding_mgr.embed_text(chunk.content)
                sim_score = EmbeddingUtils.cosine_similarity(query_vector, chunk_vec)

                if sim_score >= min_threshold:
                    candidate = RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=doc_id,
                        content=chunk.content,
                        similarity_score=sim_score,
                        composite_score=sim_score,
                        metadata=chunk.metadata
                    )
                    candidates.append(candidate)

        # Sort candidate list by similarity score descending
        sorted_candidates = sorted(candidates, key=lambda c: c.similarity_score, reverse=True)
        top_candidates = sorted_candidates[:top_k]

        logger.debug(
            f"🔍 Similarity Search Found {len(candidates)} candidates matching threshold {min_threshold} "
            f"──► Returning Top {len(top_candidates)}"
        )
        return top_candidates
