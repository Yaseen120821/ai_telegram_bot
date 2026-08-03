"""
app/rag/rag_types.py - RAG Enumeration Types
=============================================

1. PURPOSE:
-----------
Defines strongly-typed Enumeration classes (`DocumentType`, `RetrievalStatus`) for the RAG Subsystem.

2. WHY IT EXISTS:
-----------------
Using string constants (e.g. "pdf", "success") introduces typos and breaks autocomplete. `Enum` classes enforce
type safety and explicit categorization across document ingestion, indexing, and vector retrieval.

3. RESPONSIBILITIES:
--------------------
- Represent supported document formats (`DocumentType`).
- Represent execution statuses of retrieval operations (`RetrievalStatus`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/rag/rag_models.py`, `app/rag/rag_utils.py`, `app/rag/rag_manager.py`, and future parsing modules.

5. COMPLETE CODE:
-----------------
"""

from enum import Enum


class DocumentType(str, Enum):
    """
    Enumeration of supported document file formats in SANA AI RAG Subsystem.
    """
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    CODE = "code"
    UNKNOWN = "unknown"


class RetrievalStatus(str, Enum):
    """
    Enumeration of retrieval operation execution statuses.
    """
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    EMPTY = "empty"
    FAILED = "failed"
