"""
app/personality package initializer.
Exposes public API for identity, behavior, response style, rules, system prompts, and personality management.
"""

from app.personality.identity import Identity
from app.personality.behavior import Behavior
from app.personality.response_style import ResponseStyle
from app.personality.rules import Rules
from app.personality.system_prompt import SystemPromptBuilder
from app.personality.templates import PromptTemplate
from app.personality.personality_manager import PersonalityManager

__all__ = [
    "Identity",
    "Behavior",
    "ResponseStyle",
    "Rules",
    "SystemPromptBuilder",
    "PromptTemplate",
    "PersonalityManager"
]
