"""
app/conversation/history.py - Conversation Data Structure & Message Objects
=============================================================================

1. PURPOSE:
-----------
Defines strongly-typed data structures for representing individual chat messages (`Message`)
and maintaining a chronological, thread-safe sequence of messages (`ConversationHistory`).

2. WHY IT EXISTS:
-----------------
Raw untyped strings or unstructured lists make it easy to mix up speaker roles ("user" vs "assistant"),
lose track of timestamps, or suffer from race conditions when multiple threads read/write history.
Wrapping chat history inside dedicated dataclasses guarantees type safety, consistent ChatML formatting,
and thread-safe mutations.

3. RESPONSIBILITIES:
--------------------
- Store message metadata (role, text content, Unix epoch timestamp).
- Convert `Message` objects into standard `{"role": "...", "content": "..."}` dictionary lists.
- Provide safe `add_user_message`, `add_assistant_message`, `clear`, and `get_messages` operations.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Managed inside `app/conversation/session.py` (`Session` object holds a `ConversationHistory` instance).
- Passed to `app/conversation/context_window.py` for history trimming.
- Serialized output passed to `app/llm/prompt_builder.py`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
import threading
from typing import List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger("sana_ai.conversation.history")


@dataclass
class Message:
    """
    Represents a single message turn in a conversation.

    Attributes:
        role (str): Role of the message sender ("user", "assistant", or "system").
        content (str): Text body of the message.
        timestamp (float): Unix epoch timestamp when message was created.
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, str]:
        """
        Serializes Message into a dictionary format compatible with HuggingFace ChatML templates.

        Returns:
            Dict[str, str]: Dictionary containing 'role' and 'content'.
        """
        return {
            "role": self.role,
            "content": self.content
        }


class ConversationHistory:
    """
    Thread-safe container holding chronological conversation messages for a single session.
    """

    def __init__(self) -> None:
        """Initializes an empty conversation history with a thread lock."""
        self._messages: List[Message] = []
        self._lock: threading.Lock = threading.Lock()

    def add_user_message(self, content: str) -> Message:
        """
        Appends a user message to history.

        Args:
            content (str): Text prompt sent by the user.

        Returns:
            Message: Instantiated and appended Message object.
        """
        msg = Message(role="user", content=content.strip())
        with self._lock:
            self._messages.append(msg)
        logger.debug(f"Added User message [Length: {len(content)} chars | Total: {len(self._messages)}]")
        return msg

    def add_assistant_message(self, content: str) -> Message:
        """
        Appends an assistant message to history.

        Args:
            content (str): AI generated response text.

        Returns:
            Message: Instantiated and appended Message object.
        """
        msg = Message(role="assistant", content=content.strip())
        with self._lock:
            self._messages.append(msg)
        logger.debug(f"Added Assistant message [Length: {len(content)} chars | Total: {len(self._messages)}]")
        return msg

    def get_messages(self) -> List[Message]:
        """
        Returns a shallow copy of all Message objects in chronological order.

        Returns:
            List[Message]: Copy of current messages list.
        """
        with self._lock:
            return list(self._messages)

    def set_messages(self, messages: List[Message]) -> None:
        """
        Replaces the current messages list (used during history trimming).

        Args:
            messages (List[Message]): Replacement list of Message objects.
        """
        with self._lock:
            self._messages = list(messages)

    def to_dict_list(self) -> List[Dict[str, str]]:
        """
        Converts all stored messages into a list of role/content dictionaries.

        Returns:
            List[Dict[str, str]]: Serialized messages list for prompt construction.
        """
        with self._lock:
            return [msg.to_dict() for msg in self._messages]

    def clear(self) -> None:
        """Empties all messages from history."""
        with self._lock:
            count = len(self._messages)
            self._messages.clear()
        logger.info(f"Cleared conversation history ({count} messages removed).")

    def message_count(self) -> int:
        """
        Returns total number of messages currently stored.

        Returns:
            int: Message count.
        """
        with self._lock:
            return len(self._messages)
