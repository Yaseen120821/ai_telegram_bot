"""
app/rag/document_processing/document_config.py - Document Processing Configuration Store
========================================================================================

1. PURPOSE:
-----------
Centralizes configuration parameters for the Document Processing & Ingestion Subsystem, including file size limits,
chunk size & overlap settings, encoding fallbacks, and storage paths.

2. WHY IT EXISTS:
-----------------
Prevents magic numbers and hardcoded parameters throughout loaders, cleaners, chunkers, and facade managers.
Enables developers to tune chunk sizes, overlap windows, or storage cutoffs in a single centralized file.

3. RESPONSIBILITIES:
--------------------
- Store max file size limits (`MAX_FILE_SIZE_BYTES = 50MB`).
- Store chunking defaults (`DEFAULT_CHUNK_SIZE = 500`, `DEFAULT_CHUNK_OVERLAP = 50`).
- Store character boundaries (`MIN_CHUNK_SIZE`, `MAX_CHUNK_SIZE`).
- Store text encoding fallbacks (`DEFAULT_ENCODING`, `FALLBACK_ENCODINGS`).
- Store storage directory paths (`KNOWLEDGE_BASE_DIR`, `PROCESSED_CHUNKS_DIR`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `document_validator.py`, `document_cleaner.py`, `document_chunker.py`, `document_manager.py`, and `loaders/`.

5. COMPLETE CODE:
-----------------
"""

from typing import List

# Maximum allowable file size for ingestion (50 MB)
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024

# Default Recursive Character Chunking Parameters
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
MIN_CHUNK_SIZE: int = 50
MAX_CHUNK_SIZE: int = 2000

# Default Text Encoding Settings & Fallbacks
DEFAULT_ENCODING: str = "utf-8"
FALLBACK_ENCODINGS: List[str] = ["utf-8", "latin-1", "cp1252", "ascii"]

# Storage Directory Paths
KNOWLEDGE_BASE_DIR: str = "knowledge/documents"
PROCESSED_CHUNKS_DIR: str = "knowledge/processed/chunks"
