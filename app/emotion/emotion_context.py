"""
app/emotion/emotion_context.py - Ephemeral RAM Emotion Timeline & Transition Tracker
=====================================================================================

1. PURPOSE:
-----------
Tracks per-user emotional state history across conversation turns in RAM, calculates state transitions (e.g. `Joy -> Stress`),
computes dominant mood trends over time, and provides high-level `EmotionSummary` records.

2. WHY IT EXISTS (EPHEMERAL RAM STATE TRACKING):
------------------------------------------------
Single-message emotion classification lacks conversational context. `EmotionContextTracker` preserves emotional continuity
in RAM for instant access (< 0.01ms), enabling SANA AI to perceive whether a user's mood is shifting, stabilizing, or degrading.

3. RESPONSIBILITIES:
--------------------
- Maintain sliding window history (`MAX_EMOTION_HISTORY_TURNS = 10`) per user_id.
- Record `previous_emotion`, `current_emotion`, and `EmotionTransition` events.
- Calculate dominant mood trend across recent dialogue turns.
- Generate `EmotionSummary` objects detailing emotional metrics.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `EmotionContext`, `EmotionResult`, `EmotionTransition`, and `EmotionSummary` from `app/emotion/emotion_models.py`.
- Called by `app/emotion/emotion_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
from collections import Counter
from typing import Dict, Optional, List

from app.emotion.emotion_models import EmotionContext, EmotionResult, EmotionTransition, EmotionSummary
from app.emotion.emotion_types import EmotionType
from app.emotion.emotion_config import MAX_EMOTION_HISTORY_TURNS

logger = logging.getLogger("sana_ai.emotion.context")


class EmotionContextTracker:
    """
    Per-user ephemeral RAM emotional timeline and transition tracking engine.
    """

    def __init__(self) -> None:
        """Initializes user emotion context store dictionary."""
        self._contexts: Dict[str, EmotionContext] = {}

    def get_user_context(self, user_id: str) -> EmotionContext:
        """
        Retrieves or initializes EmotionContext for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            EmotionContext: Active EmotionContext object.
        """
        u_id = str(user_id)
        if u_id not in self._contexts:
            self._contexts[u_id] = EmotionContext(user_id=u_id)
        return self._contexts[u_id]

    def add_emotion(self, user_id: str, result: EmotionResult) -> EmotionContext:
        """
        Appends a new EmotionResult to the user's emotional timeline, updates previous/current state,
        records transitions, and recalculates dominant mood trend.

        Args:
            user_id (str): Telegram User ID string.
            result (EmotionResult): Newly classified EmotionResult object.

        Returns:
            EmotionContext: Updated EmotionContext object.
        """
        context = self.get_user_context(user_id)

        # 1. Update previous and current emotion states
        context.previous_emotion = context.current_emotion
        context.current_emotion = result.primary_emotion
        context.last_updated = time.time()

        # 2. Record transition if emotion state changed
        if context.previous_emotion and context.previous_emotion != context.current_emotion:
            transition = EmotionTransition(
                from_emotion=context.previous_emotion,
                to_emotion=context.current_emotion,
                timestamp=context.last_updated
            )
            context.transitions.append(transition)
            logger.info(
                f"🔀 Emotion Transition Detected | User {user_id}: '{context.previous_emotion}' ──► '{context.current_emotion}'"
            )

        # 3. Update history with sliding window limit
        context.history.append(result)
        if len(context.history) > MAX_EMOTION_HISTORY_TURNS:
            context.history.pop(0)

        # 4. Calculate dominant mood trend across recent non-neutral turns
        non_neutral = [e.primary_emotion for e in context.history if e.primary_emotion != EmotionType.NEUTRAL.value]
        if non_neutral:
            counts = Counter(non_neutral)
            context.dominant_trend = counts.most_common(1)[0][0]
        else:
            context.dominant_trend = EmotionType.NEUTRAL.value

        logger.debug(
            f"📈 Updated Emotion Context | User: {user_id} | Current: '{context.current_emotion}' | "
            f"Previous: '{context.previous_emotion}' | Trend: '{context.dominant_trend}' | History: {len(context.history)}"
        )
        return context

    def get_latest_transition(self, user_id: str) -> Optional[EmotionTransition]:
        """
        Retrieves the most recent EmotionTransition record for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            Optional[EmotionTransition]: Latest transition or None.
        """
        context = self.get_user_context(user_id)
        return context.transitions[-1] if context.transitions else None

    def get_emotional_summary(self, user_id: str) -> EmotionSummary:
        """
        Generates a high-level EmotionSummary object for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            EmotionSummary: Summary object.
        """
        context = self.get_user_context(user_id)
        duration = time.time() - context.start_time

        return EmotionSummary(
            user_id=str(user_id),
            current_emotion=context.current_emotion,
            previous_emotion=context.previous_emotion,
            dominant_trend=context.dominant_trend,
            total_turns=len(context.history),
            transitions_count=len(context.transitions),
            session_duration_seconds=duration
        )

    def clear_user_context(self, user_id: str) -> None:
        """
        Clears stored emotion context timeline for user_id.

        Args:
            user_id (str): Telegram User ID string.
        """
        u_id = str(user_id)
        if u_id in self._contexts:
            del self._contexts[u_id]
            logger.debug(f"🗑️ Cleared Ephemeral Emotion Context Timeline for User ID: {u_id}")
