"""
app/memory/memory_models.py - Memory Domain Data Models
=========================================================

1. PURPOSE:
-----------
Provides strongly-typed `@dataclass` containers (`MemoryItem`, `ExtractionResult`, and `SearchResult`) for
transferring memory records between SQLite storage, classification, search, and retrieval modules.

2. WHY IT EXISTS:
-----------------
Passing raw tuples or dictionaries causes KeyError and index mismatch bugs. Dataclasses enforce explicit
typing, default value initialization, and seamless dictionary conversion (`to_dict()`).

3. RESPONSIBILITIES:
--------------------
- Represent stored memory records (`MemoryItem`).
- Represent candidate classification outputs (`ExtractionResult`).
- Represent ranked memory search results (`SearchResult`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/memory/memory_classifier.py`, `app/memory/memory_store.py`, `app/memory/memory_retriever.py`,
  `app/memory/memory_manager.py`, and `app/memory/sqlite_manager.py`.

5. COMPLETE CODE:
-----------------
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


def get_current_iso_timestamp() -> str:
    """Generates a standard ISO 8601 UTC timestamp string (e.g. '2026-07-31T15:35:00Z')."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryItem:
    """
    Represents a persistent long-term memory record stored in SQLite.

    Attributes:
        user_id (str): Telegram User ID string.
        category (str): Taxonomy category (profile, preference, goal, project, etc.).
        memory_key (str): Normalized property key (e.g. 'name', 'favorite_language').
        memory_value (str): Stored fact value (e.g. 'Yaseen', 'Python').
        id (Optional[int]): Primary key autoincrement ID in SQLite.
        importance (int): Importance weight score from 1 (trivial) to 10 (critical).
        confidence (float): Confidence metric from 0.0 to 1.0.
        created_at (str): ISO 8601 creation timestamp string.
        updated_at (str): ISO 8601 last-modified timestamp string.
        last_accessed (str): ISO 8601 last-retrieved timestamp string.
        access_count (int): Retrieval hit frequency counter.
        is_active (int): Soft delete flag (1 = Active, 0 = Deleted).
        source (str): Provenance source ('telegram', 'manual', etc.).
    """
    user_id: str
    category: str
    memory_key: str
    memory_value: str
    id: Optional[int] = None
    importance: int = 5
    confidence: float = 0.95
    created_at: str = field(default_factory=get_current_iso_timestamp)
    updated_at: str = field(default_factory=get_current_iso_timestamp)
    last_accessed: str = field(default_factory=get_current_iso_timestamp)
    access_count: int = 0
    is_active: int = 1
    source: str = "telegram"

    def to_dict(self) -> Dict[str, Any]:
        """Converts MemoryItem into a dictionary representation."""
        return {
            "id": self.id,
            "user_id": str(self.user_id),
            "category": self.category,
            "memory_key": self.memory_key,
            "memory_value": self.memory_value,
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "is_active": self.is_active,
            "source": self.source
        }


@dataclass
class ExtractionResult:
    """
    Represents the output of the MemoryClassifier fact extraction.

    Attributes:
        is_candidate (bool): True if statement contains a durable user fact.
        category (str): Taxonomy category string.
        memory_key (str): Extracted property key string.
        memory_value (str): Extracted fact value string.
        confidence (float): Classification confidence (0.0 to 1.0).
        importance (int): Importance priority score (1 to 10).
    """
    is_candidate: bool
    category: str = "custom"
    memory_key: str = ""
    memory_value: str = ""
    confidence: float = 0.0
    importance: int = 5


@dataclass
class SearchResult:
    """
    Represents ranked search outputs from MemoryRetriever.

    Attributes:
        items (List[MemoryItem]): List of matching MemoryItem records.
        total_count (int): Total records found matching the search criteria.
        query (str): The search query or filter string applied.
    """
    items: List[MemoryItem] = field(default_factory=list)
    total_count: int = 0
    query: str = ""
