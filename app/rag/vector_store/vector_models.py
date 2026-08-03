"""
app/rag/vector_store/vector_models.py - Vector Store Domain Data Models
========================================================================
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class VectorIndexMetadata:
    """
    Metadata recording FAISS vector index attributes and count stats.

    Attributes:
        index_type (str): FAISS index type name string.
        dimension (int): Vector dimension count.
        total_vectors (int): Total vectors indexed.
        index_path (str): File path string to saved index file.
    """
    index_type: str
    dimension: int = 384
    total_vectors: int = 0
    index_path: str = ""


@dataclass
class IndexSearchRequest:
    """
    Search request payload querying vector store.

    Attributes:
        query_vector (List[float]): Dense query vector.
        top_k (int): Number of nearest neighbors to retrieve.
        filters (Dict[str, Any]): Optional metadata filter dictionary.
    """
    query_vector: List[float]
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexSearchResult:
    """
    Search output payload returned by vector store.

    Attributes:
        indices (List[int]): Internal vector index position integers.
        scores (List[float]): Similarity scores.
        metadata_list (List[Dict[str, Any]]): Associated chunk metadata dicts.
    """
    indices: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata_list: List[Dict[str, Any]] = field(default_factory=list)
