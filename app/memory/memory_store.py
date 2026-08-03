"""
app/memory/memory_store.py - Long-Term Memory CRUD & Storage Engine
====================================================================

1. PURPOSE:
-----------
Executes high-level CRUD (Create, Read, Update, Delete) operations for persistent `MemoryItem` objects
against SQLite using `SQLiteManager`.

2. WHY IT EXISTS:
-----------------
Encapsulates all database query construction and mapping logic. Ensures `MemoryItem` records are cleanly
persisted, retrieved, updated (upserted), or soft/hard deleted.

3. RESPONSIBILITIES:
--------------------
- Save or update (upsert) memory records in SQLite.
- Retrieve active long-term memories for a user (ordered by importance and recency).
- Update memory `last_accessed` timestamp and `access_count` on retrieval.
- Soft-delete or hard-delete memory records for a user (`/forget`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `MemoryItem` from `app/memory/memory_models.py`.
- Uses `SQLiteManager` from `app/memory/sqlite_manager.py`.
- Called by `app/memory/memory_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from app.memory.memory_models import MemoryItem, get_current_iso_timestamp
from app.memory.sqlite_manager import SQLiteManager

logger = logging.getLogger("sana_ai.memory.store")


class MemoryStore:
    """
    CRUD repository for persistent MemoryItem objects matching Part 2 SQLite schema.
    """

    def __init__(self, db_manager: Optional[SQLiteManager] = None) -> None:
        """
        Initializes MemoryStore.

        Args:
            db_manager (Optional[SQLiteManager]): Active SQLiteManager instance.
        """
        self.db_manager: SQLiteManager = db_manager or SQLiteManager()

    def save_memory(self, item: MemoryItem) -> MemoryItem:
        """
        Saves or updates a MemoryItem in SQLite using UPSERT logic.

        Args:
            item (MemoryItem): Memory item to persist.

        Returns:
            MemoryItem: Updated MemoryItem instance.
        """
        now_iso = get_current_iso_timestamp()
        sql = """
        INSERT INTO memories (
            user_id, category, memory_key, memory_value, importance, confidence,
            created_at, updated_at, last_accessed, access_count, is_active, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1, ?)
        ON CONFLICT(user_id, category, memory_key) DO UPDATE SET
            memory_value = excluded.memory_value,
            importance = excluded.importance,
            confidence = excluded.confidence,
            updated_at = ?,
            is_active = 1;
        """
        params = (
            str(item.user_id),
            item.category.lower(),
            item.memory_key.lower(),
            item.memory_value.strip(),
            item.importance,
            item.confidence,
            item.created_at or now_iso,
            now_iso,
            now_iso,
            item.source or "telegram",
            now_iso
        )

        self.db_manager.execute_update(sql, params)
        logger.info(
            f"💾 Persisted Memory | User: {item.user_id} | [{item.category}] {item.memory_key}: '{item.memory_value}'"
        )
        return item

    def get_memories_by_user(self, user_id: str, category: Optional[str] = None) -> List[MemoryItem]:
        """
        Retrieves active persistent long-term memories for a user (is_active = 1).

        Args:
            user_id (str): Telegram User ID string.
            category (Optional[str]): Optional category filter.

        Returns:
            List[MemoryItem]: List of active MemoryItem records.
        """
        u_id = str(user_id)
        if category:
            sql = """
            SELECT id, user_id, category, memory_key, memory_value, importance, confidence,
                   created_at, updated_at, last_accessed, access_count, is_active, source
            FROM memories
            WHERE user_id = ? AND category = ? AND is_active = 1
            ORDER BY importance DESC, updated_at DESC;
            """
            params = (u_id, category.lower())
        else:
            sql = """
            SELECT id, user_id, category, memory_key, memory_value, importance, confidence,
                   created_at, updated_at, last_accessed, access_count, is_active, source
            FROM memories
            WHERE user_id = ? AND is_active = 1
            ORDER BY importance DESC, updated_at DESC;
            """
            params = (u_id,)

        rows = self.db_manager.execute_query(sql, params)
        memories: List[MemoryItem] = []

        for row in rows:
            memories.append(
                MemoryItem(
                    id=row["id"],
                    user_id=row["user_id"],
                    category=row["category"],
                    memory_key=row["memory_key"],
                    memory_value=row["memory_value"],
                    importance=row["importance"],
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    last_accessed=row["last_accessed"],
                    access_count=row["access_count"],
                    is_active=row["is_active"],
                    source=row["source"]
                )
            )

        logger.debug(f"Retrieved {len(memories)} active memories for User ID: {u_id}")
        return memories

    def delete_memory(self, user_id: str, memory_key: str) -> bool:
        """
        Soft-deletes a specific memory entry by setting is_active = 0.

        Args:
            user_id (str): Telegram User ID string.
            memory_key (str): Memory key property to delete.

        Returns:
            bool: True if record was updated to soft-deleted status.
        """
        sql = "UPDATE memories SET is_active = 0, updated_at = ? WHERE user_id = ? AND memory_key = ?;"
        now_iso = get_current_iso_timestamp()
        affected = self.db_manager.execute_update(sql, (now_iso, str(user_id), memory_key.lower()))
        return affected > 0

    def clear_all_user_memories(self, user_id: str) -> int:
        """
        Permanently hard-deletes all stored long-term memories for a user (invoked by `/forget`).

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            int: Number of deleted memory records.
        """
        sql = "DELETE FROM memories WHERE user_id = ?;"
        affected = self.db_manager.execute_update(sql, (str(user_id),))
        logger.info(f"🗑️ Hard-deleted all long-term memories for User ID {user_id} ({affected} rows removed).")
        return affected
