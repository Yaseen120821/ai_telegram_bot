"""
app/rag/embeddings/embedding_types.py - Embedding Enumeration Types
====================================================================

1. PURPOSE:
-----------
Defines strongly-typed Enumeration classes (`EmbeddingModelType`, `SimilarityMetric`, `VectorStatus`, `EmbeddingStatus`)
for the Vector Embedding Subsystem.

2. WHY IT EXISTS:
-----------------
Using raw strings (e.g. "cosine", "valid") introduces typos and breaks autocomplete. `Enum` classes enforce
type safety across embedding models, generators, validators, and vector store managers.

3. RESPONSIBILITIES:
--------------------
- Represent supported embedding model architectures (`EmbeddingModelType`).
- Represent mathematical distance & similarity metrics (`SimilarityMetric`).
- Represent vector validation statuses (`VectorStatus`).
- Represent execution statuses of embedding generation (`EmbeddingStatus`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `embedding_models.py`, `embedding_validator.py`, `embedding_generator.py`, and `embedding_manager.py`.

5. COMPLETE CODE:
-----------------
"""

from enum import Enum


class EmbeddingModelType(str, Enum):
    """
    Enumeration of supported Sentence Transformer model architectures.
    """
    MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    BGE_SMALL_EN = "BAAI/bge-small-en-v1.5"
    E5_SMALL_V2 = "intfloat/e5-small-v2"
    MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"
    CUSTOM = "custom"


class SimilarityMetric(str, Enum):
    """
    Enumeration of mathematical vector distance metrics.
    """
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class VectorStatus(str, Enum):
    """
    Enumeration of vector validation check results.
    """
    VALID = "valid"
    INVALID_DIMENSION = "invalid_dimension"
    NAN_DETECTED = "nan_detected"
    INF_DETECTED = "inf_detected"
    ZERO_VECTOR = "zero_vector"
    EMPTY = "empty"


class EmbeddingStatus(str, Enum):
    """
    Enumeration of embedding generation execution statuses.
    """
    PENDING = "pending"
    SUCCESS = "success"
    MODEL_LOAD_FAILED = "model_load_failed"
    VALIDATION_FAILED = "validation_failed"
    FAILED = "failed"
