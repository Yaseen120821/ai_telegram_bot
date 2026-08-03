"""
app/conversation/context_window.py - Context Window Trimming & Token Budget Manager
=====================================================================================

1. PURPOSE:
-----------
Manages conversation history bounds to fit within the local Qwen LLM's context window.
Trims older messages when history exceeds a configurable maximum limit (`max_messages`).

2. WHY IT EXISTS:
-----------------
Language Models have finite context limits (e.g. 2048 / 4096 tokens). As a conversation progresses,
sending thousands of tokens on every message slows down inference, increases VRAM usage, and will
eventually cause Out Of Memory (OOM) crashes. `ContextWindow` ensures prompt size stays bounded.

3. RESPONSIBILITIES:
--------------------
- Inspect total message count in a conversation session.
- Apply sliding-window trimming to retain only the most recent `N` messages (default: 10 messages).
- Preserve user/assistant alignment so trimmed history starts with a `user` role message.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Invocated by `app/conversation/conversation_manager.py` before passing history to `app/llm/prompt_builder.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Dict
from app.conversation.history import ConversationHistory, Message

logger = logging.getLogger("sana_ai.conversation.context_window")

DEFAULT_MAX_MESSAGES = 10  # 10 messages = 5 conversation turns (5 User + 5 Assistant)


class ContextWindow:
    """
    Manages sliding window trimming for conversation histories.
    """

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        """
        Initializes ContextWindow with maximum message threshold.

        Args:
            max_messages (int): Maximum number of recent messages to retain.
        """
        self.max_messages: int = max(2, max_messages)

    def get_trimmed_history(self, history: ConversationHistory) -> List[Dict[str, str]]:
        """
        Retrieves serialized history dictionaries trimmed to fit within max_messages threshold.

        Args:
            history (ConversationHistory): History object containing session messages.

        Returns:
            List[Dict[str, str]]: List of role/content message dicts ready for prompt construction.
        """
        all_messages: List[Message] = history.get_messages()

        if len(all_messages) <= self.max_messages:
            return [msg.to_dict() for msg in all_messages]

        # Slide window to retain only the latest max_messages
        trimmed_messages = all_messages[-self.max_messages:]

        # Ensure trimmed list begins with a 'user' message to maintain turn symmetry
        if trimmed_messages and trimmed_messages[0].role == "assistant":
            trimmed_messages = trimmed_messages[1:]

        logger.info(
            f"✂️ Context Window Trimmed: Reduced from {len(all_messages)} to {len(trimmed_messages)} messages."
        )

        return [msg.to_dict() for msg in trimmed_messages]
