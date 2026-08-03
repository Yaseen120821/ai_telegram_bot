"""
app/memory/database_initializer.py - Professional Database Schema Bootstrapper & Versioning
=============================================================================================

1. PURPOSE:
-----------
Bootstraps the SQLite database (`database/sana_memory.db`) on application startup. Ensures required directory,
table structures (`memories`, `schema_info`), indexes, and schema version records exist before receiving traffic.

2. WHY IT EXISTS:
-----------------
Prevents runtime "no such table: memories" crashes. Automatically handles initial schema creation, index tuning,
and prepares the architecture for future database migrations (schema versioning).

3. RESPONSIBILITIES:
--------------------
- Check database file existence and automatically create `database/` folder if missing.
- Create `memories` table matching Part 2 schema columns (`user_id`, `category`, `memory_key`, `memory_value`, etc.).
- Create `schema_info` table for database versioning tracking.
- Create performance indexes on `user_id`, `category`, `memory_key`, `importance`, and compound `(user_id, category)`.
- Enforce `UNIQUE(user_id, category, memory_key) ON CONFLICT REPLACE` (UPSERT logic).
- Log every step of database initialization.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `SQLiteManager` from `app/memory/sqlite_manager.py`.
- Triggered by `MemoryManager` on startup.

5. COMPLETE CODE:
-----------------
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from app.memory.sqlite_manager import SQLiteManager

logger = logging.getLogger("sana_ai.memory.initializer")

CURRENT_SCHEMA_VERSION = 1


class DatabaseInitializer:
    """
    Production-grade SQLite schema initializer and versioning bootstrapper.
    """

    @staticmethod
    def initialize_database(db_manager: Optional[SQLiteManager] = None) -> None:
        """
        Initializes SQLite database tables, schema versioning, and indexes.

        Args:
            db_manager (Optional[SQLiteManager]): Active SQLiteManager instance.
        """
        manager = db_manager or SQLiteManager()
        logger.info(f"⚙️ Initializing SQLite Database at: '{manager.db_path}'")

        # 1. Schema Versioning Table DDL
        create_schema_info_sql = """
        CREATE TABLE IF NOT EXISTS schema_info (
            version INTEGER PRIMARY KEY,
            installed_at TEXT NOT NULL
        );
        """

        # 2. Main Long-Term Memories Table DDL (Part 2 Schema)
        create_memories_sql = """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            memory_value TEXT NOT NULL,
            importance INTEGER DEFAULT 5,
            confidence REAL DEFAULT 0.95,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            source TEXT DEFAULT 'telegram',
            UNIQUE(user_id, category, memory_key) ON CONFLICT REPLACE
        );
        """

        # 3. Performance Index DDL Statements
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);",
            "CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(memory_key);",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_category ON memories(user_id, category);"
        ]

        try:
            # Create schema_info versioning table
            manager.execute_update(create_schema_info_sql)
            logger.info("✅ Schema versioning table 'schema_info' ready.")

            # Create main memories table
            manager.execute_update(create_memories_sql)
            logger.info("✅ Main long-term memory table 'memories' ready.")

            # Create all required performance indexes
            for idx_sql in indexes_sql:
                manager.execute_update(idx_sql)
            logger.info("✅ Performance indexes created successfully.")

            # Register schema version if not present
            version_check = manager.execute_query("SELECT version FROM schema_info WHERE version = ?;", (CURRENT_SCHEMA_VERSION,))
            if not version_check:
                now_iso = datetime.now(timezone.utc).isoformat()
                manager.execute_update(
                    "INSERT INTO schema_info (version, installed_at) VALUES (?, ?);",
                    (CURRENT_SCHEMA_VERSION, now_iso)
                )
                logger.info(f"📌 Registered Database Schema Version v{CURRENT_SCHEMA_VERSION}.")

            logger.info("🎉 Database Initialization Complete | Status: OPERATIONAL")

        except Exception as err:
            logger.critical(f"❌ Failed to initialize database schema: {err}", exc_info=True)
            raise err
