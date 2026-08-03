"""
app/rag/embeddings package initializer.
Exposes public API for Sentence Transformer Model Loading, Vector Generator, Validator, and EmbeddingManager.
"""

from app.rag.embeddings.embedding_config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    DEFAULT_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    NORMALIZE_EMBEDDINGS,
    DEFAULT_SIMILARITY_METRIC,
    DEVICE,
    MODEL_CACHE_DIR
)
from app.rag.embeddings.embedding_types import (
    EmbeddingModelType,
    SimilarityMetric,
    VectorStatus,
    EmbeddingStatus
)
from app.rag.embeddings.embedding_models import (
    EmbeddingVector,
    VectorRecord,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingMetadata,
    EmbeddingStatistics,
    ValidationResult
)
from app.rag.embeddings.embedding_utils import EmbeddingUtils
from app.rag.embeddings.embedding_validator import EmbeddingValidator
from app.rag.embeddings.embedding_model import EmbeddingModel
from app.rag.embeddings.embedding_generator import EmbeddingGenerator
from app.rag.embeddings.embedding_manager import EmbeddingManager

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "DEFAULT_BATCH_SIZE",
    "MAX_SEQUENCE_LENGTH",
    "NORMALIZE_EMBEDDINGS",
    "DEFAULT_SIMILARITY_METRIC",
    "DEVICE",
    "MODEL_CACHE_DIR",
    "EmbeddingModelType",
    "SimilarityMetric",
    "VectorStatus",
    "EmbeddingStatus",
    "EmbeddingVector",
    "VectorRecord",
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingMetadata",
    "EmbeddingStatistics",
    "ValidationResult",
    "EmbeddingUtils",
    "EmbeddingValidator",
    "EmbeddingModel",
    "EmbeddingGenerator",
    "EmbeddingManager"
]
