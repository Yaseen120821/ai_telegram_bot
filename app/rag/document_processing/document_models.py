"""
app/rag/document_processing/document_models.py - Document Domain Data Models
============================================================================

1. PURPOSE:
-----------
Provides strongly-typed `@dataclass` containers (`Document`, `DocumentMetadata`, `Chunk`, `ChunkCollection`,
`DocumentStatistics`, `ProcessingResult`, `ValidationResult`, `CleaningResult`) for transferring documents, text chunks,
and processing metadata across the ingestion pipeline.

2. WHY IT EXISTS:
-----------------
Encapsulating document properties, cleaned text, chunk offsets, token estimates, and validation results inside dataclasses
enforces type safety, eliminates dictionary key mismatches, and provides clean dictionary serialization (`to_dict()`).

3. RESPONSIBILITIES:
--------------------
- Represent complete document objects (`Document`).
- Represent file & extraction metadata (`DocumentMetadata`).
- Represent a single extracted text chunk (`Chunk`).
- Represent a collection of chunks for a document (`ChunkCollection`).
- Represent aggregate ingestion metrics (`DocumentStatistics`).
- Represent processing outputs (`ProcessingResult`).
- Represent validation & cleaning outputs (`ValidationResult`, `CleaningResult`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `document_validator.py`, `document_cleaner.py`, `document_chunker.py`, `document_manager.py`, and `loaders/`.

5. COMPLETE CODE:
-----------------
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentMetadata:
    """
    Metadata recording file properties, size, word/char counts, token estimates, and hash digest.

    Attributes:
        file_id (str): Unique document identifier.
        filename (str): Base filename string.
        file_type (str): Format type string (e.g. 'pdf', 'docx', 'txt').
        file_path (str): Absolute file path string.
        file_size_bytes (int): Size in bytes.
        word_count (int): Calculated word count.
        char_count (int): Calculated character count.
        estimated_tokens (int): Estimated LLM token count (~4 chars per token).
        hash_digest (str): SHA256 cryptographic hash digest of file.
        created_at (float): Epoch timestamp of ingestion.
    """
    file_id: str
    filename: str
    file_type: str
    file_path: str
    file_size_bytes: int
    word_count: int = 0
    char_count: int = 0
    estimated_tokens: int = 0
    hash_digest: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Converts DocumentMetadata into a dictionary."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "hash_digest": self.hash_digest,
            "created_at": self.created_at
        }


@dataclass
class Document:
    """
    Represents a full loaded and cleaned document object.

    Attributes:
        doc_id (str): Unique document identifier.
        raw_text (str): Extracted raw text from loader.
        cleaned_text (str): Sanitized, Unicode-normalized text.
        metadata (DocumentMetadata): Associated DocumentMetadata object.
    """
    doc_id: str
    raw_text: str
    cleaned_text: str
    metadata: DocumentMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Converts Document into a dictionary."""
        return {
            "doc_id": self.doc_id,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "metadata": self.metadata.to_dict()
        }


@dataclass
class Chunk:
    """
    Represents a single extracted text chunk passage.

    Attributes:
        chunk_id (str): Unique chunk identifier.
        doc_id (str): Parent document file_id.
        content (str): Text passage content.
        chunk_index (int): 0-indexed position in document.
        start_char (int): Starting character offset in parent document.
        end_char (int): Ending character offset in parent document.
        estimated_tokens (int): Estimated LLM token count.
        hash_digest (str): MD5 hash digest of chunk content.
        metadata (Optional[DocumentMetadata]): Parent document metadata.
    """
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    estimated_tokens: int = 0
    hash_digest: str = ""
    metadata: Optional[DocumentMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts Chunk into a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "estimated_tokens": self.estimated_tokens,
            "hash_digest": self.hash_digest,
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


@dataclass
class ChunkCollection:
    """
    Collection of chunks generated for a parent document.

    Attributes:
        doc_id (str): Parent document identifier.
        chunks (List[Chunk]): List of Chunk objects.
        total_chunks (int): Total chunks count.
    """
    doc_id: str
    chunks: List[Chunk] = field(default_factory=list)
    total_chunks: int = 0


@dataclass
class DocumentStatistics:
    """
    Aggregate metrics across processed knowledge base documents.

    Attributes:
        total_documents (int): Number of documents processed.
        total_chunks (int): Number of text chunks generated.
        total_bytes (int): Aggregate file size in bytes.
        total_words (int): Aggregate word count.
    """
    total_documents: int = 0
    total_chunks: int = 0
    total_bytes: int = 0
    total_words: int = 0


@dataclass
class ValidationResult:
    """
    Result payload of document validation.

    Attributes:
        is_valid (bool): True if file passed all validation checks.
        reason (str): Explanation if invalid.
        is_duplicate (bool): True if SHA256 hash matches an existing document.
    """
    is_valid: bool
    reason: str = "Passed validation"
    is_duplicate: bool = False


@dataclass
class CleaningResult:
    """
    Result payload of document text cleaning.

    Attributes:
        original_length (int): Pre-cleaning character count.
        cleaned_length (int): Post-cleaning character count.
        text (str): Cleaned text string.
    """
    original_length: int
    cleaned_length: int
    text: str


@dataclass
class ProcessingResult:
    """
    Final output payload of document ingestion pipeline.

    Attributes:
        status (str): ProcessingStatus string value ('success', 'failed', 'duplicate').
        doc_id (str): Unique document ID string.
        chunks (List[Chunk]): Generated list of Chunk objects.
        execution_time_ms (float): Execution duration in milliseconds.
        message (str): Informational result summary message.
    """
    status: str
    doc_id: str
    chunks: List[Chunk] = field(default_factory=list)
    execution_time_ms: float = 0.0
    message: str = "Processing completed"
