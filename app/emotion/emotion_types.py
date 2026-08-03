"""
app/emotion/emotion_types.py - Complete Emotion & Intensity Enumerations
=========================================================================

1. PURPOSE:
-----------
Defines strongly-typed Enumeration classes (`EmotionType` and `EmotionIntensity`) for multi-class emotional state
classification and intensity scoring across 20 distinct emotional categories.

2. WHY IT EXISTS:
-----------------
Centralizes discrete emotion category labels (Joy, Sadness, Anger, Fear, Surprise, Love, Curiosity, Optimism, Disappointment,
Confusion, Stress, Calm, Pride, Excitement, Hope, Motivation, Frustration, Embarrassment, Gratitude, Neutral).
Prevents string literal typos and enforces type validation across the system.

3. RESPONSIBILITIES:
--------------------
- Define valid discrete emotional taxonomy categories.
- Define intensity levels (Low, Medium, High, Extreme).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/emotion/emotion_models.py`, `app/emotion/emotion_detector.py`, `app/emotion/emotion_classifier.py`,
  `app/emotion/emotion_context.py`, `app/emotion/emotion_memory.py`, and `app/emotion/emotion_manager.py`.

5. COMPLETE CODE:
-----------------
"""

from enum import Enum


class EmotionType(str, Enum):
    """
    Complete 20-category discrete emotional taxonomy.
    """
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    LOVE = "love"
    CURIOSITY = "curiosity"
    OPTIMISM = "optimism"
    DISAPPOINTMENT = "disappointment"
    CONFUSION = "confusion"
    STRESS = "stress"
    CALM = "calm"
    PRIDE = "pride"
    EXCITEMENT = "excitement"
    HOPE = "hope"
    MOTIVATION = "motivation"
    FRUSTRATION = "frustration"
    EMBARRASSMENT = "embarrassment"
    GRATITUDE = "gratitude"
    NEUTRAL = "neutral"


class EmotionIntensity(str, Enum):
    """
    Emotional intensity level classification.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
