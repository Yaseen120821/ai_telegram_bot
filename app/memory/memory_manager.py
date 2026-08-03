"""
app/memory/memory_manager.py - Central Long-Term Memory Manager
================================================================

1. PURPOSE:
-----------
Acts as the central Singleton orchestrator for the Long-Term Memory System (`app/memory/`).
Coordinates fact classification, storage CRUD, duplicate detection, ranking search, and prompt memory retrieval.

2. WHY IT EXISTS (ARCHITECTURE FACADE & CACHE MANAGEMENT):
-----------------------------------------------------------
Provides a clean, unified Python API for the rest of the application. Automatically triggers cache invalidation
on `MemoryRetriever` whenever memory modifications occur (`save_memory`, `update_memory`, `delete_memory`, `clear_user_memory`).

3. RESPONSIBILITIES:
--------------------
- Bootstrap SQLite database schema on startup via `DatabaseInitializer`.
- Process incoming user text and persist extracted facts asynchronously.
- Provide clean CRUD API: `save_memory`, `retrieve_memories`, `search_memory`, `update_memory`, `delete_memory`,
  `memory_exists`, `get_memory_by_key`, `get_memories_by_category`, `clear_user_memory`.
- Invalidate in-memory cache on mutations to prevent stale data retrieval.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Orchestrates `SQLiteManager`, `DatabaseInitializer`, `MemoryStore`, `MemoryClassifier`, and `MemoryRetriever`.
- Called by Telegram bot handlers and system verification suites.

5. COMPLETE CODE:
-----------------
"""

import logging
import threading
from typing import List, Optional

from app.memory.sqlite_manager import SQLiteManager
from app.memory.database_initializer import DatabaseInitializer
from app.memory.memory_models import MemoryItem, ExtractionResult, SearchResult
from app.memory.memory_store import MemoryStore
from app.memory.memory_classifier import MemoryClassifier
from app.memory.memory_retriever import MemoryRetriever

logger = logging.getLogger("sana_ai.memory.manager")


class MemoryManager:
    """
    Thread-safe Singleton orchestrator for SANA AI persistent long-term memory.
    """
    _instance: Optional["MemoryManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if MemoryManager._instance is not None:
            raise RuntimeError(
                "MemoryManager is a Singleton! Use `MemoryManager.get_instance()` instead."
            )
        
        self.sqlite_manager: SQLiteManager = SQLiteManager()
        # Initialize SQLite database schema on boot
        DatabaseInitializer.initialize_database(self.sqlite_manager)

        self.memory_store: MemoryStore = MemoryStore(db_manager=self.sqlite_manager)
        self.memory_classifier: MemoryClassifier = MemoryClassifier()
        self.memory_retriever: MemoryRetriever = MemoryRetriever(memory_store=self.memory_store)
        
        logger.info("🧠 MemoryManager Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "MemoryManager":
        """
        Thread-safe accessor for the MemoryManager Singleton instance.

        Returns:
            MemoryManager: Shared Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # PUBLIC API METHODS & CACHE INVALIDATION
    # ------------------------------------------------------------------

    def save_memory(self, item: MemoryItem) -> MemoryItem:
        """
        Saves or updates a MemoryItem in SQLite storage and invalidates the user's cache.

        Args:
            item (MemoryItem): MemoryItem record to persist.

        Returns:
            MemoryItem: Saved MemoryItem object.
        """
        saved = self.memory_store.save_memory(item)
        self.memory_retriever.invalidate_cache(str(item.user_id))
        return saved

    def retrieve_memories(self, user_id: str, query: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """
        Retrieves top N ranked memories stored for user_id.

        Args:
            user_id (str): Telegram User ID string.
            query (Optional[str]): Optional user query string for relevance ranking.
            limit (int): Maximum records threshold (default: 10).

        Returns:
            List[MemoryItem]: List of ranked MemoryItem objects.
        """
        return self.memory_retriever.retrieve_by_user(str(user_id), query=query, limit=limit)

    def search_memory(self, user_id: str, query: str, limit: int = 10) -> SearchResult:
        """
        Searches stored user memories matching a keyword query string.

        Args:
            user_id (str): Telegram User ID string.
            query (str): Search filter string.
            limit (int): Maximum records threshold.

        Returns:
            SearchResult: SearchResult object containing matching items.
        """
        return self.memory_retriever.search_memories(str(user_id), query=query, limit=limit)

    def update_memory(self, item: MemoryItem) -> MemoryItem:
        """
        Updates an existing memory record in SQLite and invalidates cache.

        Args:
            item (MemoryItem): MemoryItem to update.

        Returns:
            MemoryItem: Updated MemoryItem.
        """
        return self.save_memory(item)

    def delete_memory(self, user_id: str, memory_key: str) -> bool:
        """
        Soft-deletes a specific memory entry by key for user_id and invalidates cache.

        Args:
            user_id (str): Telegram User ID string.
            memory_key (str): Memory key string to soft-delete.

        Returns:
            bool: True if soft-deleted successfully.
        """
        success = self.memory_store.delete_memory(str(user_id), memory_key)
        if success:
            self.memory_retriever.invalidate_cache(str(user_id))
        return success

    def memory_exists(self, user_id: str, memory_key: str) -> bool:
        """
        Checks whether a specific memory key exists for user_id.

        Args:
            user_id (str): Telegram User ID string.
            memory_key (str): Memory key string.

        Returns:
            bool: True if key exists in memory storage.
        """
        item = self.get_memory_by_key(str(user_id), memory_key)
        return item is not None

    def get_memory_by_key(self, user_id: str, memory_key: str) -> Optional[MemoryItem]:
        """
        Fetches a specific memory record by key for user_id.

        Args:
            user_id (str): Telegram User ID string.
            memory_key (str): Memory key string.

        Returns:
            Optional[MemoryItem]: Matching MemoryItem record or None.
        """
        all_memories = self.memory_retriever.get_user_memories_cached(str(user_id))
        key_norm = memory_key.strip().lower()
        for item in all_memories:
            if item.memory_key.lower() == key_norm and item.is_active == 1:
                return item
        return None

    def get_memories_by_category(self, user_id: str, category: str) -> List[MemoryItem]:
        """
        Retrieves user memories belonging to a specific taxonomy category.

        Args:
            user_id (str): Telegram User ID string.
            category (str): Taxonomy category string.

        Returns:
            List[MemoryItem]: Filtered MemoryItem records.
        """
        return self.memory_retriever.retrieve_by_category(str(user_id), category=category)

    def clear_user_memory(self, user_id: str) -> int:
        """
        Permanently hard-deletes all long-term memory records for user_id and invalidates cache.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            int: Number of deleted memory rows.
        """
        logger.info(f"🗑️ Wiping all persistent long-term memories for User ID: {user_id}")
        count = self.memory_store.clear_all_user_memories(str(user_id))
        self.memory_retriever.invalidate_cache(str(user_id))
        return count

    def clear_user_memories(self, user_id: str) -> int:
        """Alias method for clear_user_memory for backward compatibility."""
        return self.clear_user_memory(user_id)

    def process_and_store_user_message(self, user_id: str, user_text: str) -> Optional[MemoryItem]:
        """
        Classifies incoming user text and persists extracted facts into SQLite.

        Args:
            user_id (str): Telegram User ID string.
            user_text (str): Incoming user statement.

        Returns:
            Optional[MemoryItem]: Saved MemoryItem if a fact was extracted, else None.
        """
        extraction: ExtractionResult = self.memory_classifier.classify_statement(user_text)

        if not extraction.is_candidate or not extraction.memory_key or not extraction.memory_value:
            return None

        item = MemoryItem(
            user_id=str(user_id),
            category=extraction.category,
            memory_key=extraction.memory_key,
            memory_value=extraction.memory_value,
            confidence=extraction.confidence,
            importance=extraction.importance
        )

        saved_item = self.save_memory(item)
        logger.info(f"✅ Memory Saved for User ID {user_id}: [{saved_item.category}] {saved_item.memory_key} = '{saved_item.memory_value}'")
        return saved_item

    def get_memory_context_for_prompt(self, user_id: str, query: Optional[str] = None) -> str:
        """
        Retrieves formatted long-term memory context string for System Prompt injection.

        Args:
            user_id (str): Telegram User ID string.
            query (Optional[str]): Optional user prompt string for relevance scoring.

        Returns:
            str: Formatted memory block string.
        """
        return self.memory_retriever.get_formatted_memory_context(str(user_id), query=query)
