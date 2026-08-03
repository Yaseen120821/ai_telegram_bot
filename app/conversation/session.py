"""
app/conversation/session.py - Per-User Session Management Layer
================================================================

1. PURPOSE:
-----------
Manages isolated per-user chat sessions (`Session`) and provides a thread-safe registry (`SessionManager`)
mapping Telegram User IDs to their respective active `Session` instances.

2. WHY IT EXISTS:
-----------------
Telegram is a multi-user platform. Multiple users can send messages to the bot concurrently.
Without session isolation, User A's private messages could leak into User B's conversation prompt!
`SessionManager` isolates every user by their unique Telegram User ID (`user_id`).

3. RESPONSIBILITIES:
--------------------
- Maintain metadata per user (Telegram ID, creation timestamp, last accessed timestamp).
- Map `user_id` -> `Session` instance.
- Provide thread-safe `get_or_create_session`, `clear_session`, and session activity updates.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Holds a `ConversationHistory` instance from `app/conversation/history.py`.
- Used by `app/conversation/conversation_manager.py` to route updates per user.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
import threading
from typing import Dict, Optional, List
from app.conversation.history import ConversationHistory

logger = logging.getLogger("sana_ai.conversation.session")


class Session:
    """
    Represents an active chat session for a single Telegram user.
    """

    def __init__(self, user_id: int) -> None:
        """
        Initializes a Session for a Telegram user.

        Args:
            user_id (int): Unique Telegram User ID.
        """
        self.user_id: int = user_id
        self.created_at: float = time.time()
        self.last_accessed: float = time.time()
        self.history: ConversationHistory = ConversationHistory()

    def touch(self) -> None:
        """Updates last_accessed timestamp to current time."""
        self.last_accessed = time.time()


class SessionManager:
    """
    Thread-safe registry managing user sessions mapped by Telegram user_id.
    """

    def __init__(self) -> None:
        """Initializes empty session dictionary and thread lock."""
        self._sessions: Dict[int, Session] = {}
        self._lock: threading.Lock = threading.Lock()

    def get_or_create_session(self, user_id: int) -> Session:
        """
        Retrieves existing session for user_id or creates a new Session if absent.

        Args:
            user_id (int): Telegram User ID.

        Returns:
            Session: Active user Session object.
        """
        with self._lock:
            if user_id not in self._sessions:
                logger.info(f"🆕 Creating new isolated Session for Telegram User ID: {user_id}")
                self._sessions[user_id] = Session(user_id=user_id)
            
            session = self._sessions[user_id]
            session.touch()
            return session

    def get_session(self, user_id: int) -> Optional[Session]:
        """
        Retrieves active session for user_id without creating a new one if missing.

        Args:
            user_id (int): Telegram User ID.

        Returns:
            Optional[Session]: Active Session if found, else None.
        """
        with self._lock:
            session = self._sessions.get(user_id)
            if session:
                session.touch()
            return session

    def clear_session(self, user_id: int) -> bool:
        """
        Clears history and removes session for the specified user_id.

        Args:
            user_id (int): Telegram User ID.

        Returns:
            bool: True if session existed and was cleared, False otherwise.
        """
        with self._lock:
            if user_id in self._sessions:
                session = self._sessions.pop(user_id)
                session.history.clear()
                logger.info(f"🧹 Cleared and removed Session for User ID: {user_id}")
                return True
            logger.warning(f"Attempted to clear non-existent Session for User ID: {user_id}")
            return False

    def get_active_user_ids(self) -> List[int]:
        """
        Returns list of all active Telegram User IDs.

        Returns:
            List[int]: Active user IDs.
        """
        with self._lock:
            return list(self._sessions.keys())

    def total_active_sessions(self) -> int:
        """
        Returns count of currently active user sessions.

        Returns:
            int: Active session count.
        """
        with self._lock:
            return len(self._sessions)
