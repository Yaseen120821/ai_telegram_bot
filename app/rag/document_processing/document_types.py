"""
app/rag/document_processing/document_types.py - Document Processing Enumeration Types
======================================================================================

1. PURPOSE:
-----------
Defines strongly-typed Enumeration classes (`DocumentType`, `ProcessingStatus`, `ChunkingStrategy`) for document processing.

2. WHY IT EXISTS:
-----------------
Using raw strings (e.g. "processed", "recursive") introduces typos and breaks autocomplete. `Enum` classes enforce
type safety across loaders, cleaners, chunkers, and facade managers.

3. RESPONSIBILITIES:
--------------------
- Represent supported document formats (`DocumentType`).
- Represent execution statuses of document ingestion (`ProcessingStatus`).
- Represent chunking algorithms (`ChunkingStrategy`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `document_models.py`, `document_chunker.py`, `document_manager.py`, and `loaders/`.

5. COMPLETE CODE:
-----------------
"""

from enum import Enum


class DocumentType(str, Enum):
    """
    Enumeration of supported document file formats.
    """
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    PYTHON = "python"
    JAVA = "java"
    CPP = "cpp"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CODE = "code"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """
    Enumeration of document ingestion execution statuses.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    LOADER_FAILED = "loader_failed"
    CLEANING_FAILED = "cleaning_failed"
    CHUNKING_FAILED = "chunking_failed"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class ChunkingStrategy(str, Enum):
    """
    Enumeration of supported document chunking algorithms.
    """
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"
    TOKEN_BASED = "token_based"
