"""
app/rag/vector_store/vector_types.py - Vector Store Enumeration Types
======================================================================
"""

from enum import Enum


class VectorIndexType(str, Enum):
    """
    Enumeration of supported FAISS index algorithm types.
    """
    FLAT_IP = "FlatIP"        # Exact Inner Product / Cosine
    FLAT_L2 = "FlatL2"        # Exact Euclidean L2
    IVF_FLAT = "IVFFlat"      # Inverted File Index with Voronoi cells
    HNSW = "HNSW"              # Hierarchical Navigable Small World graph


class StoreStatus(str, Enum):
    """
    Enumeration of vector index storage execution statuses.
    """
    READY = "ready"
    EMPTY = "empty"
    INDEXING = "indexing"
    SAVED = "saved"
    FAILED = "failed"
