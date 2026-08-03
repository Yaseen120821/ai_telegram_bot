"""
app/rag/embeddings/embedding_models.py - Embedding Domain Data Models
======================================================================

1. PURPOSE:
-----------
Provides strongly-typed `@dataclass` containers (`EmbeddingVector`, `VectorRecord`, `EmbeddingRequest`, `EmbeddingResult`,
`EmbeddingMetadata`, `EmbeddingStatistics`, `ValidationResult`) for transferring dense vectors and metadata across the embedding pipeline.

2. WHY IT EXISTS:
-----------------
Encapsulating dense float arrays, L2 norm values, dimensions, and execution metrics inside dataclasses
enforces type safety and provides clean dictionary serialization (`to_dict()`).

3. RESPONSIBILITIES:
--------------------
- Represent a dense vector payload (`EmbeddingVector`).
- Represent a vector storage record (`VectorRecord`).
- Represent structured embedding generation queries (`EmbeddingRequest`).
- Represent execution outputs (`EmbeddingResult`).
- Represent model metadata & statistics (`EmbeddingMetadata`, `EmbeddingStatistics`).
- Represent vector validation check outputs (`ValidationResult`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `embedding_validator.py`, `embedding_generator.py`, `embedding_manager.py`, and `vector_store/`.

5. COMPLETE CODE:
-----------------
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class EmbeddingVector:
    """
    Represents a dense numerical vector generated for a text passage.

    Attributes:
        vector_id (str): Unique vector identifier.
        doc_id (str): Parent document identifier.
        chunk_id (str): Associated text chunk identifier.
        values (List[float]): Dense floating-point array (e.g. 384 dimensions).
        dimension (int): Vector dimension count.
        norm (float): L2 vector norm magnitude.
        is_normalized (bool): True if L2 unit vector normalized.
    """
    vector_id: str
    doc_id: str
    chunk_id: str
    values: List[float]
    dimension: int = 384
    norm: float = 1.0
    is_normalized: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Converts EmbeddingVector into a dictionary."""
        return {
            "vector_id": self.vector_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "values": self.values,
            "dimension": self.dimension,
            "norm": self.norm,
            "is_normalized": self.is_normalized
        }


@dataclass
class VectorRecord:
    """
    Storage payload passed to local FAISS or vector database index.

    Attributes:
        id (str): Vector record ID string.
        vector (List[float]): Dense vector floats.
        metadata (Dict[str, Any]): Metadata attribute mapping.
    """
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRequest:
    """
    Structured query payload requesting vector embedding generation.

    Attributes:
        texts (List[str]): List of text passage strings.
        doc_ids (List[str]): List of associated document IDs.
        chunk_ids (List[str]): List of associated chunk IDs.
        normalize (bool): Whether to apply L2 normalization.
    """
    texts: List[str]
    doc_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    normalize: bool = True


@dataclass
class EmbeddingResult:
    """
    Output payload of vector embedding generation.

    Attributes:
        status (str): EmbeddingStatus string ('success', 'failed').
        vectors (List[EmbeddingVector]): Generated EmbeddingVector objects.
        dimension (int): Vector dimension count.
        execution_time_ms (float): Execution duration in milliseconds.
        message (str): Informational result summary message.
    """
    status: str
    vectors: List[EmbeddingVector] = field(default_factory=list)
    dimension: int = 384
    execution_time_ms: float = 0.0
    message: str = "Embedding completed"


@dataclass
class EmbeddingMetadata:
    """
    Metadata recording model properties and device allocation.

    Attributes:
        model_name (str): Sentence transformer model identifier string.
        dimension (int): Vector dimension count.
        device (str): Execution hardware device ('cuda' or 'cpu').
        is_normalized (bool): True if L2 normalization enabled.
    """
    model_name: str
    dimension: int = 384
    device: str = "cpu"
    is_normalized: bool = True


@dataclass
class EmbeddingStatistics:
    """
    Aggregate metrics across generated vector embeddings.

    Attributes:
        total_vectors_generated (int): Aggregate vectors count.
        total_tokens_embedded (int): Aggregate tokens embedded.
        average_latency_ms (float): Average per-batch execution duration.
    """
    total_vectors_generated: int = 0
    total_tokens_embedded: int = 0
    average_latency_ms: float = 0.0


@dataclass
class ValidationResult:
    """
    Result payload of vector validation check.

    Attributes:
        status (str): VectorStatus string value.
        is_valid (bool): True if vector passed all mathematical checks.
        reason (str): Explanation if invalid.
    """
    status: str
    is_valid: bool
    reason: str = "Vector is mathematically valid"
