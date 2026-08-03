"""
app/rag/vector_store package initializer.
Exposes public API for Vector Database Architecture & FAISS Manager.
"""

from app.rag.vector_store.vector_config import (
    DEFAULT_INDEX_TYPE,
    VECTOR_DIMENSION,
    DEFAULT_METRIC,
    VECTOR_STORE_DIR,
    INDEX_FILE_PATH,
    METADATA_FILE_PATH
)
from app.rag.vector_store.vector_types import VectorIndexType, StoreStatus
from app.rag.vector_store.vector_models import VectorIndexMetadata, IndexSearchRequest, IndexSearchResult
from app.rag.vector_store.vector_utils import VectorStoreUtils
from app.rag.vector_store.faiss_manager import FAISSIndexManager
from app.rag.vector_store.vector_manager import VectorManager

__all__ = [
    "DEFAULT_INDEX_TYPE",
    "VECTOR_DIMENSION",
    "DEFAULT_METRIC",
    "VECTOR_STORE_DIR",
    "INDEX_FILE_PATH",
    "METADATA_FILE_PATH",
    "VectorIndexType",
    "StoreStatus",
    "VectorIndexMetadata",
    "IndexSearchRequest",
    "IndexSearchResult",
    "VectorStoreUtils",
    "FAISSIndexManager",
    "VectorManager"
]
