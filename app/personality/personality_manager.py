"""
app/personality/personality_manager.py - Central Personality Manager
======================================================================

1. PURPOSE:
-----------
Acts as the central Singleton orchestrator for the Personality Layer. Loads, compiles, caches,
and manages the active System Prompt and personality configuration for SANA AI.

2. WHY IT EXISTS:
-----------------
Provides a single source of truth for the assistant's personality across the entire application.
Instead of rebuilding system prompt strings on every request, `PersonalityManager` pre-caches the compiled
system prompt string, reducing latency and memory allocations.

3. RESPONSIBILITIES:
--------------------
- Maintain active `SystemPromptBuilder` instance.
- Pre-cache compiled master System Prompt string.
- Provide clean `get_system_prompt()` API to `PromptBuilder`.
- Prepare architectural slots for future mode switching (e.g. Developer Mode, Teacher Mode).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `SystemPromptBuilder` from `app/personality/system_prompt.py`.
- Called by `app/llm/prompt_builder.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
import threading
from typing import Optional, Dict, Any

from app.personality.system_prompt import SystemPromptBuilder
from app.personality.identity import Identity
from app.personality.behavior import Behavior
from app.personality.response_style import ResponseStyle
from app.personality.rules import Rules

logger = logging.getLogger("sana_ai.personality.manager")


class PersonalityManager:
    """
    Thread-safe Singleton manager for SANA AI personality and System Prompt generation.
    """
    _instance: Optional["PersonalityManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if PersonalityManager._instance is not None:
            raise RuntimeError(
                "PersonalityManager is a Singleton! Use `PersonalityManager.get_instance()` instead."
            )
        
        self.system_prompt_builder: SystemPromptBuilder = SystemPromptBuilder()
        self._cached_system_prompt: str = self.system_prompt_builder.build_system_prompt()
        self.active_mode: str = "default"
        logger.info("🎭 PersonalityManager initialized with default SANA AI personality persona.")

    @classmethod
    def get_instance(cls) -> "PersonalityManager":
        """
        Thread-safe accessor for the PersonalityManager Singleton instance.

        Returns:
            PersonalityManager: Shared Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_system_prompt(self) -> str:
        """
        Returns the active compiled System Prompt string.

        Returns:
            str: Master System Prompt string.
        """
        return self._cached_system_prompt

    def reload_personality(
        self,
        identity: Optional[Identity] = None,
        behavior: Optional[Behavior] = None,
        response_style: Optional[ResponseStyle] = None,
        rules: Optional[Rules] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Recompiles and updates the cached System Prompt with new personality components.

        Args:
            identity (Optional[Identity]): New identity component.
            behavior (Optional[Behavior]): New behavior component.
            response_style (Optional[ResponseStyle]): New response style component.
            rules (Optional[Rules]): New rules component.
            custom_instructions (Optional[str]): Custom mode instructions.

        Returns:
            str: Newly compiled System Prompt string.
        """
        with self._lock:
            self.system_prompt_builder = SystemPromptBuilder(
                identity=identity,
                behavior=behavior,
                response_style=response_style,
                rules=rules
            )
            self._cached_system_prompt = self.system_prompt_builder.build_system_prompt(
                custom_instructions=custom_instructions
            )
            logger.info("🔄 Personality recompiled and system prompt updated.")
            return self._cached_system_prompt

    def get_adaptive_communication_style(self, emotion_context: Optional[Any] = None) -> str:
        """
        Generates dynamic System Prompt communication guidance based on user's active EmotionContext,
        intensity, confidence, and emotional trend.

        Args:
            emotion_context (Optional[Any]): EmotionContext object from app.emotion.

        Returns:
            str: Formatted system prompt communication directive string.
        """
        if not emotion_context or not hasattr(emotion_context, "current_emotion"):
            return ""

        current_emotion = emotion_context.current_emotion.lower() if emotion_context.current_emotion else "neutral"
        dominant_trend = getattr(emotion_context, "dominant_trend", "neutral")

        try:
            from app.emotion import EmotionUtils, DEFAULT_MIN_CONFIDENCE
            empathy_prompt = EmotionUtils.get_empathy_prompt(current_emotion)

            # Check latest history entry for confidence score
            confidence = 0.95
            intensity = "medium"
            if hasattr(emotion_context, "history") and emotion_context.history:
                latest_res = emotion_context.history[-1]
                confidence = getattr(latest_res, "confidence", 0.95)
                intensity = getattr(latest_res, "intensity", "medium")

            # Low confidence fallback rule: default to neutral
            if confidence < DEFAULT_MIN_CONFIDENCE:
                logger.debug(f"ℹ️ Low emotion confidence ({confidence:.2f} < {DEFAULT_MIN_CONFIDENCE}). Defaulting to neutral communication.")
                return ""

            guidance = (
                f"=== USER EMOTIONAL STATE & ADAPTIVE COMMUNICATION GUIDANCE ===\n"
                f"• Current Primary Emotion: {current_emotion.capitalize()} (Confidence: {confidence:.2f}, Intensity: {intensity.capitalize()})\n"
                f"• Conversational Emotional Trend: {dominant_trend.capitalize()}\n"
                f"• Directive: {empathy_prompt}"
            )
            return guidance
        except Exception as err:
            logger.warning(f"Failed to generate adaptive communication style: {err}")
            return ""
