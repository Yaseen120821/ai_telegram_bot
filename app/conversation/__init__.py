"""
app/conversation package initializer.
Exposes public API for conversation management, session tracking, and context trimming.
"""

from app.conversation.history import Message, ConversationHistory
from app.conversation.session import Session, SessionManager
from app.conversation.context_window import ContextWindow
from app.conversation.conversation_manager import ConversationManager

__all__ = [
    "Message",
    "ConversationHistory",
    "Session",
    "SessionManager",
    "ContextWindow",
    "ConversationManager"
]
