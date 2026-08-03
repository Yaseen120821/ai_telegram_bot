"""
app/rag/rag_models.py - RAG Domain Data Models
===============================================

1. PURPOSE:
-----------
Provides strongly-typed `@dataclass` containers (`DocumentMetadata`, `RetrievedChunk`, `ChunkReference`,
`RetrievalRequest`, `RetrievalResult`, `KnowledgeSource`, `SearchResult`, `KnowledgeContext`) for transferring
RAG document metadata, vector chunks, and search results across the application.

2. WHY IT EXISTS:
-----------------
Encapsulating document metadata, chunk text, vector similarity scores, and retrieval context inside dataclasses
enforces type safety, eliminates dictionary key mismatches, and provides clean dictionary serialization (`to_dict()`).

3. RESPONSIBILITIES:
--------------------
- Represent document file metadata (`DocumentMetadata`).
- Represent a single extracted text vector chunk (`RetrievedChunk`).
- Represent source citations & page references (`ChunkReference`).
- Represent structured retrieval queries (`RetrievalRequest`).
- Represent comprehensive retrieval outputs (`RetrievalResult`).
- Represent active knowledge base sources (`KnowledgeSource`).
- Represent formatted RAG search results (`SearchResult`).
- Represent injected system prompt knowledge context (`KnowledgeContext`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/rag/rag_utils.py`, `app/rag/rag_manager.py`, `app/llm/prompt_builder.py`, and future vector store modules.

5. COMPLETE CODE:
-----------------
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentMetadata:
    """
    Metadata recording document file attributes, file path, size, and cryptographic hash.

    Attributes:
        file_id (str): Unique document identifier.
        filename (str): Base filename string.
        file_type (str): Document format type (e.g. 'pdf', 'docx', 'txt').
        file_path (str): Full absolute file path.
        file_size_bytes (int): File size in bytes.
        created_at (float): Epoch timestamp of ingestion.
        hash_digest (str): SHA256 / MD5 hash digest of file contents.
    """
    file_id: str
    filename: str
    file_type: str
    file_path: str
    file_size_bytes: int
    created_at: float = field(default_factory=time.time)
    hash_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts DocumentMetadata into a dictionary."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "created_at": self.created_at,
            "hash_digest": self.hash_digest
        }


@dataclass
class RetrievedChunk:
    """
    Represents a single extracted text chunk retrieved from the vector knowledge index.

    Attributes:
        chunk_id (str): Unique chunk identifier.
        doc_id (str): Parent document file_id.
        content (str): Text passage content.
        score (float): Vector similarity score (0.0 to 1.0).
        metadata (Optional[DocumentMetadata]): Parent document metadata object.
    """
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: Optional[DocumentMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts RetrievedChunk into a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


@dataclass
class ChunkReference:
    """
    Represents a source citation reference for answer verification.

    Attributes:
        doc_name (str): Source document name.
        page_or_line (str): Page number or line range.
        similarity_score (float): Relevance match score.
    """
    doc_name: str
    page_or_line: str
    similarity_score: float


@dataclass
class RetrievalRequest:
    """
    Represents a structured knowledge retrieval query request.

    Attributes:
        query (str): Input question or query string.
        user_id (str): Telegram User ID string.
        top_k (int): Number of top vector chunks to retrieve.
        filters (Dict[str, Any]): Optional metadata filtering conditions.
    """
    query: str
    user_id: str
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Represents the output payload of a RAG retrieval query.

    Attributes:
        status (str): Retrieval status string ('success', 'not_found', 'failed').
        request_query (str): Input query text.
        chunks (List[RetrievedChunk]): List of retrieved text chunks.
        execution_time_ms (float): Search duration in milliseconds.
        total_found (int): Total relevant chunks found.
    """
    status: str
    request_query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    execution_time_ms: float = 0.0
    total_found: int = 0


@dataclass
class KnowledgeSource:
    """
    Represents an active document source in the knowledge base.

    Attributes:
        source_id (str): Unique knowledge source identifier.
        name (str): Display name of the document or URL.
        type (str): DocumentType string.
        total_chunks (int): Number of indexed chunks.
    """
    source_id: str
    name: str
    type: str
    total_chunks: int = 0


@dataclass
class SearchResult:
    """
    High-level search output payload.

    Attributes:
        query (str): Query statement.
        results (List[RetrievedChunk]): List of matching chunks.
        sources_count (int): Number of unique document sources represented.
    """
    query: str
    results: List[RetrievedChunk] = field(default_factory=list)
    sources_count: int = 0


@dataclass
class KnowledgeContext:
    """
    Formatted knowledge context payload ready for ChatML System Prompt injection.

    Attributes:
        query (str): Input query statement.
        formatted_text (str): Formatted context block text.
        sources (List[KnowledgeSource]): List of cited knowledge sources.
        chunks_count (int): Number of chunks formatted.
    """
    query: str
    formatted_text: str
    sources: List[KnowledgeSource] = field(default_factory=list)
    chunks_count: int = 0
