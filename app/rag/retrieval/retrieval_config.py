"""
app/rag/retrieval/retrieval_config.py - Retrieval Subsystem Configuration Store
================================================================================

1. PURPOSE:
-----------
Centralizes configuration parameters for the RAG Retrieval Subsystem, including top-k search counts,
minimum similarity thresholds, token budget limits, ranking weights, and filter toggles.

2. WHY IT EXISTS:
-----------------
Prevents magic numbers and hardcoded search limits throughout retrievers, filters, rankers, and context builders.
Enables developers to tune search precision, relevance thresholds, or prompt token budgets in a single file.

3. RESPONSIBILITIES:
--------------------
- Store default candidate retrieval count (`DEFAULT_TOP_K = 5`).
- Store relevance score cutoff (`MIN_SIMILARITY_THRESHOLD = 0.40`).
- Store system prompt context token budget (`MAX_CONTEXT_TOKENS = 1000`).
- Store composite ranking weights (`SIMILARITY_WEIGHT = 0.70`, `FRESHNESS_WEIGHT = 0.15`, `METADATA_MATCH_WEIGHT = 0.15`).
- Store maximum sources cited per prompt (`MAX_SOURCES_PER_PROMPT = 5`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `retriever.py`, `similarity_search.py`, `ranking_engine.py`, `context_builder.py`, and `retrieval_manager.py`.

5. COMPLETE CODE:
-----------------
"""

# Default Candidate Search Count & Similarity Cutoff Threshold
DEFAULT_TOP_K: int = 5
MIN_SIMILARITY_THRESHOLD: float = 0.40

# Maximum Token Budget for RAG Knowledge Context inside ChatML System Prompt
MAX_CONTEXT_TOKENS: int = 1000
MAX_SOURCES_PER_PROMPT: int = 5

# Composite Ranking Weights
ENABLE_RANKING: bool = True
SIMILARITY_WEIGHT: float = 0.70
FRESHNESS_WEIGHT: float = 0.15
METADATA_MATCH_WEIGHT: float = 0.15

# Filtering Toggles
ENABLE_METADATA_FILTER: bool = True
ENABLE_DUPLICATE_FILTER: bool = True
