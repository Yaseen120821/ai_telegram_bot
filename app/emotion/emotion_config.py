"""
app/emotion/emotion_config.py - Emotion Intelligence Configuration Store
========================================================================

1. PURPOSE:
-----------
Centralizes configuration parameters for the Emotion Subsystem, including confidence thresholds,
intensity cutoffs, emotion importance scoring (1-10 scale), memory persistence thresholds,
emotion-to-empathy prompt mapping templates, and timeline history limits.

2. WHY IT EXISTS:
-----------------
Prevents magic numbers, duplicate prompt templates, and hardcoded threshold values throughout the emotion layer.
Enables developers to tune importance scales or adjust storage cutoffs in a single centralized file.

3. RESPONSIBILITIES:
--------------------
- Store confidence thresholds for emotion classification.
- Store intensity thresholds (LOW, MEDIUM, HIGH, EXTREME).
- Store 1-10 emotion importance scoring map (`EMOTION_IMPORTANCE_MAP`).
- Store minimum importance threshold for SQLite persistence (`MIN_MEMORY_IMPORTANCE_THRESHOLD = 6`).
- Map discrete emotional states to System Prompt empathy directives.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/emotion/emotion_classifier.py`, `app/emotion/emotion_context.py`, `app/emotion/emotion_memory.py`,
  and `app/emotion/emotion_manager.py`.

5. COMPLETE CODE:
-----------------
"""

from typing import Dict

# Minimum confidence score (0.0 to 1.0) below which an emotion defaults to NEUTRAL
DEFAULT_MIN_CONFIDENCE: float = 0.45

# Intensity calculation confidence thresholds
EXTREME_INTENSITY_THRESHOLD: float = 0.85
HIGH_INTENSITY_THRESHOLD: float = 0.70
MEDIUM_INTENSITY_THRESHOLD: float = 0.55

# Maximum turn history tracked in EmotionContext timeline
MAX_EMOTION_HISTORY_TURNS: int = 10

# Minimum importance score (1 to 10 scale) required to persist emotional event to SQLite
MIN_MEMORY_IMPORTANCE_THRESHOLD: int = 6

# Maximum latency threshold (ms) before emitting performance warning log
PERFORMANCE_THRESHOLD_MS: float = 200.0

# Maximum user input character length for emotion sanitization & tokenization
INPUT_MAX_CHARS: int = 2000

# Hardware device preference ('cuda' or 'cpu')
DEVICE_PREFERENCE: str = "cuda"

# Default Transformer model checkpoint
MODEL_CHECKPOINT: str = "bhadresh-savani/distilbert-base-uncased-emotion"

# Emotion Importance Scoring System (1 to 10 Scale)
EMOTION_IMPORTANCE_MAP: Dict[str, int] = {
    "neutral": 1,
    "joy": 2,
    "excited": 2,
    "excitement": 2,
    "calm": 2,
    "calmness": 2,
    "curiosity": 3,
    "surprise": 4,
    "optimism": 4,
    "gratitude": 4,
    "embarrassment": 5,
    "frustration": 5,
    "confusion": 4,
    "hope": 5,
    "motivation": 5,
    "pride": 5,
    "love": 6,
    "disappointment": 6,
    "fear": 7,
    "anxiety": 7,
    "anger": 8,
    "sadness": 8,
    "stress": 9,
    "grief": 10
}

# Mapping from EmotionType values to System Prompt Empathy Directives
EMPATHY_PROMPT_MAP: Dict[str, str] = {
    "joy": (
        "[EMPATHY DIRECTIVE]: The user is feeling joyful and happy! "
        "Adopt a warm, cheerful, positive, and celebratory tone."
    ),
    "sadness": (
        "[EMPATHY DIRECTIVE]: The user is experiencing sadness or disappointment. "
        "Adopt a gentle, deeply supportive, empathetic, and encouraging tone. Avoid cold or blunt technical jargon."
    ),
    "anger": (
        "[EMPATHY DIRECTIVE]: The user is expressing anger. "
        "Maintain a calm, polite, respectful, and reassuring tone. Focus on de-escalation and constructive guidance."
    ),
    "frustration": (
        "[EMPATHY DIRECTIVE]: The user is feeling frustrated with a technical or general problem. "
        "Acknowledge the difficulty patiently and provide clear, reassuring, step-by-step solutions."
    ),
    "excited": (
        "[EMPATHY DIRECTIVE]: The user is feeling excited! "
        "Match their high energy with an enthusiastic, encouraging, and supportive response."
    ),
    "fear": (
        "[EMPATHY DIRECTIVE]: The user is expressing fear or concern. "
        "Adopt a steady, reassuring, calm, and comforting tone."
    ),
    "anxiety": (
        "[EMPATHY DIRECTIVE]: The user is feeling anxious or worried. "
        "Be patient, comforting, and clear. Offer reassuring structure to help alleviate stress."
    ),
    "confusion": (
        "[EMPATHY DIRECTIVE]: The user is feeling confused. "
        "Break down your explanation into simple, logical, step-by-step bullet points without overwhelming details."
    ),
    "curiosity": (
        "[EMPATHY DIRECTIVE]: The user is highly curious and eager to learn! "
        "Provide insightful, educational, and engaging explanations."
    ),
    "disappointment": (
        "[EMPATHY DIRECTIVE]: The user is feeling disappointed. "
        "Offer empathetic validation and constructive, encouraging next steps."
    ),
    "motivation": (
        "[EMPATHY DIRECTIVE]: The user is feeling motivated and driven! "
        "Encourage their momentum and provide clear, actionable guidance."
    ),
    "stress": (
        "[EMPATHY DIRECTIVE]: The user is experiencing stress. "
        "Be concise, organized, calm, and helpful to reduce cognitive load."
    ),
    "calmness": (
        "[EMPATHY DIRECTIVE]: The user is calm and relaxed. "
        "Maintain a balanced, clear, and thoughtful conversational tone."
    ),
    "hope": (
        "[EMPATHY DIRECTIVE]: The user is feeling hopeful. "
        "Reinforce their optimism with positive, encouraging feedback."
    ),
    "pride": (
        "[EMPATHY DIRECTIVE]: The user is feeling proud of an achievement. "
        "Acknowledge their accomplishment with genuine praise and validation."
    ),
    "neutral": ""  # No extra empathy override needed for neutral prompts
}
