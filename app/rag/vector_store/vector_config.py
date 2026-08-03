"""
app/rag/vector_store/vector_config.py - Vector Store Subsystem Configuration Store
====================================================================================

1. PURPOSE:
-----------
Centralizes configuration parameters for local FAISS vector index storage and index persistence.

2. WHY IT EXISTS:
-----------------
Prevents hardcoded paths and parameters across FAISS index managers, vector managers, and retrievers.

3. RESPONSIBILITIES:
--------------------
- Store default FAISS index type (`DEFAULT_INDEX_TYPE = "Flat"`).
- Store index file path (`INDEX_FILE_PATH = "database/vector_store/sana_faiss.index"`).
- Store vector dimension (`VECTOR_DIMENSION = 384`).
- Store similarity metric (`DEFAULT_METRIC = "cosine"`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `faiss_manager.py`, `vector_manager.py`, and future Part 4 retrieval search modules.

5. COMPLETE CODE:
-----------------
"""

# FAISS Vector Index Configuration Defaults
DEFAULT_INDEX_TYPE: str = "Flat"
VECTOR_DIMENSION: int = 384
DEFAULT_METRIC: str = "cosine"

# Disk Persistence Directory & Index File Path
VECTOR_STORE_DIR: str = "database/vector_store"
INDEX_FILE_PATH: str = "database/vector_store/sana_faiss.index"
METADATA_FILE_PATH: str = "database/vector_store/sana_metadata.json"
