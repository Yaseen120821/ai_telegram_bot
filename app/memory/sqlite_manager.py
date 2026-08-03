"""
app/memory/sqlite_manager.py - Professional Thread-Safe SQLite Database Manager
===================================================================================

1. PURPOSE:
-----------
Manages low-level SQLite database operations against `database/sana_memory.db`. Handles parameterized query execution,
transaction commits, automatic rollbacks on failure, error recovery, and Write-Ahead Logging (WAL) concurrency.

2. WHY IT EXISTS (SINGLE RESPONSIBILITY PRINCIPLE):
----------------------------------------------------
Isolates ALL database communication logic inside a single manager module. Higher layers (classifier, retriever, store)
never write raw SQL or interact directly with `sqlite3` connection objects. Everything communicates through `SQLiteManager`.

3. RESPONSIBILITIES:
--------------------
- Safely open and close SQLite database connections.
- Execute parameterized SELECT queries using placeholders (`?`) to prevent SQL Injection attacks.
- Execute INSERT, UPDATE, and DELETE queries inside strict `with conn:` transaction blocks.
- Automatically execute `conn.rollback()` on transaction failure to prevent database corruption.
- Handle database locking (`OperationalError`), file permission errors (`PermissionError`), and corruption.
- Enable Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) for high-concurrency read/write operations.
- Provide comprehensive logging for connection lifecycle, query execution, commits, rollbacks, and errors.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/memory/database_initializer.py` and `app/memory/memory_store.py`.

5. COMPLETE CODE:
-----------------
"""

import os
import sqlite3
import logging
import threading
from typing import List, Tuple, Any, Optional

logger = logging.getLogger("sana_ai.memory.sqlite_manager")

DEFAULT_DB_PATH = "database/sana_memory.db"


class SQLiteManager:
    """
    Production-grade thread-safe SQLite Manager enforcing SQL injection protection,
    transaction safety (commit/rollback), WAL mode, and comprehensive logging.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """
        Initializes SQLiteManager.

        Args:
            db_path (str): Relative or absolute path to SQLite database file.
        """
        self.db_path: str = os.path.abspath(db_path)
        self._lock: threading.Lock = threading.Lock()
        self._ensure_db_dir()

    def _ensure_db_dir(self) -> None:
        """Automatically creates the parent directory if it does not exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"📁 Created database directory at: '{db_dir}'")
            except PermissionError as p_err:
                logger.critical(f"❌ Permission denied creating directory '{db_dir}': {p_err}")
                raise p_err

    def get_connection(self) -> sqlite3.Connection:
        """
        Opens a fresh SQLite connection configured with WAL mode, foreign keys, and Row factory.

        Returns:
            sqlite3.Connection: Configured connection object.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for high-concurrency read/write operations
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            logger.debug(f"🔌 Connection opened to SQLite database: '{self.db_path}'")
            return conn
        except sqlite3.OperationalError as op_err:
            logger.error(f"❌ SQLite Connection Failure (Database locked or inaccessible): {op_err}")
            raise op_err
        except sqlite3.DatabaseError as db_err:
            logger.critical(f"❌ Corrupted or Invalid SQLite Database File: {db_err}")
            raise db_err

    def execute_query(self, sql: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        """
        Executes a parameterized SELECT query using positional placeholders (`?`).

        SECURITY: Always pass parameters via `params` tuple to prevent SQL Injection attacks.

        Args:
            sql (str): Parameterized SELECT SQL query string.
            params (Tuple[Any, ...]): Query parameters tuple.

        Returns:
            List[sqlite3.Row]: Matching result rows.
        """
        with self._lock:
            conn: Optional[sqlite3.Connection] = None
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                logger.debug(f"🔍 Executed SELECT Query [Rows returned: {len(rows)} | SQL: {sql[:60]}...]")
                return rows
            except sqlite3.Error as sql_err:
                logger.error(f"❌ SQLite Query Execution Error: {sql_err} | SQL: '{sql}' | Params: {params}")
                raise sql_err
            finally:
                if conn:
                    conn.close()
                    logger.debug("🔌 Connection closed after SELECT query.")

    def execute_update(self, sql: str, params: Tuple[Any, ...] = ()) -> int:
        """
        Executes an INSERT, UPDATE, or DELETE query within an automatic commit/rollback transaction block.

        SECURITY: Always pass parameters via `params` tuple to prevent SQL Injection attacks.

        Args:
            sql (str): Parameterized UPDATE/INSERT/DELETE SQL string.
            params (Tuple[Any, ...]): Parameters tuple.

        Returns:
            int: Number of affected rows or last inserted row ID.
        """
        with self._lock:
            conn: Optional[sqlite3.Connection] = None
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                affected = cursor.rowcount if cursor.rowcount != -1 else cursor.lastrowid
                logger.debug(f"💾 Executed UPDATE Query [Affected/RowID: {affected} | Transaction Committed]")
                return affected
            except sqlite3.Error as sql_err:
                if conn:
                    conn.rollback()
                    logger.warning(f"⚠️ Transaction Rolled Back due to Error: {sql_err}")
                logger.error(f"❌ SQLite Transaction Failure: {sql_err} | SQL: '{sql}' | Params: {params}")
                raise sql_err
            finally:
                if conn:
                    conn.close()
                    logger.debug("🔌 Connection closed after transaction.")

    def vacuum(self) -> None:
        """Executes SQLite VACUUM command to rebuild database file and reclaim unused space."""
        with self._lock:
            try:
                with self.get_connection() as conn:
                    conn.execute("VACUUM;")
                    logger.info("🧹 Executed VACUUM on SQLite database file.")
            except sqlite3.Error as err:
                logger.error(f"Failed to execute VACUUM: {err}")

    def integrity_check(self) -> bool:
        """
        Runs SQLite PRAGMA integrity_check to verify database health.

        Returns:
            bool: True if database integrity is ok.
        """
        rows = self.execute_query("PRAGMA integrity_check;")
        if rows and rows[0][0] == "ok":
            logger.info("✅ Database integrity check: OK")
            return True
        logger.error(f"❌ Database integrity check FAILED: {rows}")
        return False
