"""
app/rag/retrieval/retrieval_types.py - Retrieval Enumeration Types
===================================================================

1. PURPOSE:
-----------
Defines strongly-typed Enumeration classes (`SearchMode`, `SimilarityMetric`, `RankingMode`, `ContextStatus`, `RetrievalStatus`)
for the Vector Retrieval Subsystem.

2. WHY IT EXISTS:
-----------------
Using raw strings (e.g. "cosine", "ranked") introduces typos and breaks autocomplete. `Enum` classes enforce
type safety across similarity searchers, metadata filters, ranking engines, and context builders.

3. RESPONSIBILITIES:
--------------------
- Represent vector search operational modes (`SearchMode`).
- Represent distance/similarity formulas (`SimilarityMetric`).
- Represent composite ranking algorithms (`RankingMode`).
- Represent context building execution statuses (`ContextStatus`, `RetrievalStatus`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `retrieval_models.py`, `similarity_search.py`, `ranking_engine.py`, `context_builder.py`, and `retrieval_manager.py`.

5. COMPLETE CODE:
-----------------
"""

from enum import Enum


class SearchMode(str, Enum):
    """
    Enumeration of vector similarity search operational modes.
    """
    SEMANTIC = "semantic"
    EXACT_FLAT = "exact_flat"
    THRESHOLD_FILTERED = "threshold_filtered"


class SimilarityMetric(str, Enum):
    """
    Enumeration of vector distance metrics.
    """
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class RankingMode(str, Enum):
    """
    Enumeration of relevance score ranking algorithms.
    """
    SIMILARITY_ONLY = "similarity_only"
    COMPOSITE_SCORE = "composite_score"
    FRESHNESS_FIRST = "freshness_first"


class ContextStatus(str, Enum):
    """
    Enumeration of context building execution statuses.
    """
    SUCCESS = "success"
    EMPTY = "empty"
    TRUNCATED = "truncated"
    FAILED = "failed"


class RetrievalStatus(str, Enum):
    """
    Enumeration of retrieval operation execution statuses.
    """
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    EMPTY_QUERY = "empty_query"
    FAILED = "failed"
