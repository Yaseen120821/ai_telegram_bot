"""
app/rag/embeddings/embedding_config.py - Embedding Subsystem Configuration Store
==================================================================================

1. PURPOSE:
-----------
Centralizes configuration parameters for the Vector Embedding Subsystem, including model selection, vector dimension,
batch size, normalization flags, and hardware execution device (`cuda` or `cpu`).

2. WHY IT EXISTS:
-----------------
Prevents magic strings and hardcoded dimensions (`384`) throughout embedding generators, validators, and vector managers.
Enables tuning batch sizes or switching embedding models in a single centralized file.

3. RESPONSIBILITIES:
--------------------
- Store embedding model identifier (`DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`).
- Store expected vector dimension (`EMBEDDING_DIMENSION = 384`).
- Store batch size (`DEFAULT_BATCH_SIZE = 32`).
- Store vector normalization flag (`NORMALIZE_EMBEDDINGS = True`).
- Detect hardware acceleration device (`DEVICE = "cuda"` if `torch.cuda.is_available()` else `"cpu"`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `embedding_model.py`, `embedding_generator.py`, `embedding_validator.py`, and `embedding_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import torch

# Default Sentence Transformer Model & Dimension
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# Execution Batch Size & Maximum Token Sequence Length
DEFAULT_BATCH_SIZE: int = 32
MAX_SEQUENCE_LENGTH: int = 256

# Vector Normalization & Distance Metric Defaults
NORMALIZE_EMBEDDINGS: bool = True
DEFAULT_SIMILARITY_METRIC: str = "cosine"

# Hardware Execution Device Auto-Detection
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Model Cache Directory Path
MODEL_CACHE_DIR: str = "models/embeddings"
