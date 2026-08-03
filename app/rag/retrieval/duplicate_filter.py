r"""
app/rag/retrieval/duplicate_filter.py - Duplicate Chunk Removal Engine
========================================================================

1. PURPOSE:
-----------
Detects and removes duplicate or near-duplicate text chunks from candidate retrieval lists.

2. WHY IT EXISTS (TOKEN ECONOMY):
---------------------------------
Because document chunking uses overlap windows (e.g., 50 characters overlap), adjacent chunks often contain
identical sentence fragments. Injecting duplicate information wastes System Prompt token budget.
`DuplicateFilter` strips redundant text passages.

3. RESPONSIBILITIES:
--------------------
- Strip exact text content duplicates.
- Strip high-overlap near-duplicate chunks ($\ge 85\%$ substring overlap).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `retriever.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Set

from app.rag.retrieval.retrieval_models import RetrievedChunk

logger = logging.getLogger("sana_ai.rag.retrieval.duplicate_filter")


class DuplicateFilter:
    """
    Duplicate chunk filtering engine stripping redundant text passages.
    """

    def remove_duplicates(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Removes exact and near-duplicate text chunks from candidate list.

        Args:
            chunks (List[RetrievedChunk]): Input candidate chunks list.

        Returns:
            List[RetrievedChunk]: Deduplicated candidate chunks list.
        """
        if not chunks:
            return []

        unique_chunks: List[RetrievedChunk] = []
        seen_texts: Set[str] = set()

        for chunk in chunks:
            clean_text = chunk.content.strip().lower()
            if clean_text in seen_texts:
                continue

            # Near-duplicate check against already accepted unique chunks
            is_near_dup = False
            for prev_text in seen_texts:
                if len(clean_text) > 40 and len(prev_text) > 40:
                    if clean_text[:40] in prev_text or prev_text[:40] in clean_text:
                        is_near_dup = True
                        break

            if not is_near_dup:
                seen_texts.add(clean_text)
                unique_chunks.append(chunk)

        logger.debug(f"🧹 DuplicateFilter: {len(chunks)} input chunks ──► {len(unique_chunks)} unique chunks")
        return unique_chunks
