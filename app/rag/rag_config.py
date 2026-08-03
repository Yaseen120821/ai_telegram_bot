"""
app/rag/rag_config.py - RAG Subsystem Configuration Store
==========================================================

1. PURPOSE:
-----------
Centralizes configuration parameters for the Retrieval-Augmented Generation (RAG) Subsystem, including
supported file types, retrieval limits, relevance score thresholds, performance SLAs, and embedding settings.

2. WHY IT EXISTS:
-----------------
Prevents magic strings and hardcoded parameters throughout document processing, embedding generation,
vector search, and prompt context building. Enables tuning retrieval performance in a single centralized file.

3. RESPONSIBILITIES:
--------------------
- Store supported document file extensions (`SUPPORTED_FILE_EXTENSIONS`).
- Store retrieval limits (`MAX_RETRIEVED_DOCUMENTS`, `MAX_CHUNKS_PER_QUERY`, `RETRIEVAL_LIMIT`).
- Store relevance score thresholds (`MIN_RELEVANCE_SCORE_THRESHOLD`).
- Store latency SLA threshold (`PERFORMANCE_THRESHOLD_MS = 300.0`).
- Store future embedding model and chunking defaults (`DEFAULT_EMBEDDING_MODEL`, `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/rag/rag_utils.py`, `app/rag/rag_manager.py`, and future parsing/vector store modules.

5. COMPLETE CODE:
-----------------
"""

from typing import Set
from pathlib import Path

# Supported Document File Extensions
SUPPORTED_FILE_EXTENSIONS: Set[str] = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
    ".html", ".htm", ".json", ".csv",
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".css"
}

# Maximum retrieved documents per query
MAX_RETRIEVED_DOCUMENTS: int = 5

# Maximum chunks returned per retrieval query
MAX_CHUNKS_PER_QUERY: int = 10

# Default retrieval result limit
RETRIEVAL_LIMIT: int = 5

# Minimum similarity score threshold (0.0 to 1.0) below which chunks are filtered out
MIN_RELEVANCE_SCORE_THRESHOLD: float = 0.40

# Maximum latency threshold (ms) before emitting performance warning log
PERFORMANCE_THRESHOLD_MS: float = 300.0

# Maximum query character length for sanitization
INPUT_MAX_QUERY_CHARS: int = 1000

# Future Embedding Model & Vector Store Defaults
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
VECTOR_STORE_DIR: str = "database/vector_store"
KNOWLEDGE_BASE_DIR: Path = Path("knowledge/documents")
