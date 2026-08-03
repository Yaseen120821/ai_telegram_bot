"""
app/rag/retrieval/retrieval_models.py - Retrieval Subsystem Data Models
========================================================================

1. PURPOSE:
-----------
Provides strongly-typed `@dataclass` containers (`QueryRequest`, `QueryEmbedding`, `RetrievedChunk`, `RetrievalResult`,
`RankingResult`, `KnowledgeContext`, `ContextStatistics`) for transferring search queries, candidate chunks, ranked outputs,
and system prompt knowledge blocks across the retrieval pipeline.

2. WHY IT EXISTS:
-----------------
Encapsulating query vectors, similarity scores, metadata filters, composite ranking metrics, and prompt context blocks inside dataclasses
enforces type safety and provides clean dictionary serialization (`to_dict()`).

3. RESPONSIBILITIES:
--------------------
- Represent user retrieval query inputs (`QueryRequest`).
- Represent embedded query vectors (`QueryEmbedding`).
- Represent retrieved candidate text chunks with similarity & ranking scores (`RetrievedChunk`).
- Represent retrieval execution payloads (`RetrievalResult`).
- Represent reranked chunk lists (`RankingResult`).
- Represent ChatML system prompt knowledge context blocks (`KnowledgeContext`).
- Represent context token statistics (`ContextStatistics`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `DocumentMetadata` from `app/rag/document_processing/document_models.py`.
- Used by `retriever.py`, `similarity_search.py`, `ranking_engine.py`, `context_builder.py`, and `retrieval_manager.py`.

5. COMPLETE CODE:
-----------------
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from app.rag.document_processing.document_models import DocumentMetadata
from app.rag.rag_models import KnowledgeSource


@dataclass
class QueryRequest:
    """
    Structured retrieval search query request.

    Attributes:
        query (str): Input question statement.
        user_id (str): Telegram User ID string.
        top_k (int): Number of top vector candidates to retrieve.
        min_similarity (float): Similarity threshold cutoff score.
        filters (Dict[str, Any]): Metadata filtering rules.
    """
    query: str
    user_id: str = "default"
    top_k: int = 5
    min_similarity: float = 0.40
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryEmbedding:
    """
    Embedded search query representation.

    Attributes:
        query (str): Input query text.
        vector (List[float]): 384-dimensional query embedding vector.
        dimension (int): Vector dimension count.
    """
    query: str
    vector: List[float]
    dimension: int = 384


@dataclass
class RetrievedChunk:
    """
    Retrieved text chunk candidate containing similarity match score and composite rank score.

    Attributes:
        chunk_id (str): Unique chunk identifier.
        doc_id (str): Parent document file_id.
        content (str): Text passage content.
        similarity_score (float): Cosine similarity score (0.0 to 1.0).
        composite_score (float): Reranked composite score.
        metadata (Optional[DocumentMetadata]): Parent document metadata object.
    """
    chunk_id: str
    doc_id: str
    content: str
    similarity_score: float
    composite_score: float = 0.0
    metadata: Optional[DocumentMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts RetrievedChunk into a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "composite_score": self.composite_score,
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


@dataclass
class RetrievalResult:
    """
    Output payload of retrieval pipeline execution.

    Attributes:
        status (str): RetrievalStatus string value ('success', 'not_found', 'failed').
        query (str): Input query statement.
        chunks (List[RetrievedChunk]): List of retrieved and ranked chunks.
        execution_time_ms (float): Search duration in milliseconds.
        total_found (int): Total candidates meeting threshold.
    """
    status: str
    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    execution_time_ms: float = 0.0
    total_found: int = 0


@dataclass
class RankingResult:
    """
    Output payload of composite ranking engine.

    Attributes:
        ranked_chunks (List[RetrievedChunk]): Ordered list of chunks by composite score.
        top_score (float): Highest score in ranking list.
    """
    ranked_chunks: List[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0


@dataclass
class KnowledgeContext:
    """
    Formatted RAG Knowledge Context payload ready for ChatML System Prompt injection.

    Attributes:
        query (str): User question statement.
        formatted_text (str): Formatted ChatML System Prompt knowledge context block.
        sources (List[KnowledgeSource]): Cited document sources.
        chunks_count (int): Number of chunks formatted.
        estimated_tokens (int): Estimated LLM token count of formatted text block.
    """
    query: str
    formatted_text: str
    sources: List[KnowledgeSource] = field(default_factory=list)
    chunks_count: int = 0
    estimated_tokens: int = 0


@dataclass
class ContextStatistics:
    """
    Token budget metrics across formatted RAG context blocks.

    Attributes:
        total_chunks_retrieved (int): Candidates retrieved.
        chunks_included (int): Candidates formatted within token budget.
        tokens_used (int): Total tokens consumed by knowledge context block.
    """
    total_chunks_retrieved: int = 0
    chunks_included: int = 0
    tokens_used: int = 0
