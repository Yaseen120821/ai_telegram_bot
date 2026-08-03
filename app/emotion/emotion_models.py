"""
app/emotion/emotion_models.py - Emotion Domain Data Models
===========================================================

1. PURPOSE:
-----------
Provides strongly-typed `@dataclass` containers (`EmotionProbability`, `PredictionResult`, `EmotionResult`,
`EmotionTransition`, `EmotionContext`, `EmotionSummary`, `EmotionMemoryRecord`, `EmotionStatistics`,
`EmotionMetadata`, `EmotionHistoryEntry`) for transferring emotion payloads across the emotion subsystem.

2. WHY IT EXISTS:
-----------------
Encapsulating emotion predictions, context state transitions, and memory storage decisions inside dataclasses
prevents schema mismatches, enforces type safety, and enables clean dictionary conversion (`to_dict()`).

3. RESPONSIBILITIES:
--------------------
- Represent single emotion probability scores (`EmotionProbability`).
- Represent raw inference prediction outputs (`PredictionResult`).
- Represent finalized emotion classification results (`EmotionResult`).
- Represent state transitions between consecutive turns (`EmotionTransition`).
- Represent active user context timeline (`EmotionContext`).
- Represent high-level summary of a user's emotional state (`EmotionSummary`).
- Represent memory storage evaluation records (`EmotionMemoryRecord`).
- Represent aggregate emotional statistics (`EmotionStatistics`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/emotion/emotion_detector.py`, `app/emotion/emotion_classifier.py`, `app/emotion/emotion_context.py`,
  `app/emotion/emotion_memory.py`, and `app/emotion/emotion_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class EmotionProbability:
    """
    Represents a single emotion label and its calculated Softmax probability score.

    Attributes:
        emotion (str): Emotion category label string.
        probability (float): Softmax probability (0.0 to 1.0).
    """
    emotion: str
    probability: float


@dataclass
class EmotionMetadata:
    """
    Metadata recording model inference parameters and hardware device details.

    Attributes:
        model_name (str): Transformer model checkpoint identifier string.
        device (str): Execution hardware device ('cuda' or 'cpu').
        latency_ms (float): Inference duration in milliseconds.
    """
    model_name: str
    device: str
    latency_ms: float


@dataclass
class PredictionResult:
    """
    Represents raw inference prediction outputs emitted by the EmotionDetector engine.

    Attributes:
        top_emotion (str): Primary predicted emotion label string.
        top_score (float): Top Softmax probability score (0.0 to 1.0).
        raw_scores (Dict[str, float]): Map of all emotion labels to Softmax probabilities.
        metadata (Optional[EmotionMetadata]): Optional model execution metadata.
    """
    top_emotion: str
    top_score: float
    raw_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[EmotionMetadata] = None


@dataclass
class EmotionResult:
    """
    Represents the verified output of emotion classification for a user statement.

    Attributes:
        primary_emotion (str): Predominant emotion category string (e.g. 'joy', 'sadness').
        intensity (str): Emotional intensity level ('low', 'medium', 'high', 'extreme').
        confidence (float): Classification confidence score (0.0 to 1.0).
        secondary_emotions (Dict[str, float]): Map of secondary detected emotion probabilities.
        empathy_directive (str): Formatted System Prompt empathy instruction text.
        timestamp (float): Detection epoch timestamp.
    """
    primary_emotion: str
    intensity: str = "medium"
    confidence: float = 0.95
    secondary_emotions: Dict[str, float] = field(default_factory=dict)
    empathy_directive: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Converts EmotionResult into a dictionary representation."""
        return {
            "primary_emotion": self.primary_emotion,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "secondary_emotions": self.secondary_emotions,
            "empathy_directive": self.empathy_directive,
            "timestamp": self.timestamp
        }


@dataclass
class EmotionTransition:
    """
    Represents an emotional state transition between two consecutive turns.

    Attributes:
        from_emotion (str): Previous emotion state string.
        to_emotion (str): Current emotion state string.
        timestamp (float): Epoch timestamp of transition.
    """
    from_emotion: str
    to_emotion: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class EmotionHistoryEntry:
    """
    Historical timeline record of a user statement and its classified emotion.

    Attributes:
        statement (str): Original user message text.
        emotion_result (EmotionResult): Classified EmotionResult object.
        timestamp (float): Detection epoch timestamp.
    """
    statement: str
    emotion_result: EmotionResult
    timestamp: float = field(default_factory=time.time)


@dataclass
class EmotionContext:
    """
    Tracks a user's recent emotional timeline, transition history, and mood trend.

    Attributes:
        user_id (str): Telegram User ID string.
        current_emotion (str): Most recently detected primary emotion.
        previous_emotion (Optional[str]): Emotion state from preceding turn.
        dominant_trend (str): Most frequent emotion across recent turns.
        history (List[EmotionResult]): Sliding window list of recent EmotionResult records.
        transitions (List[EmotionTransition]): List of recorded state transitions.
        start_time (float): Session start epoch timestamp.
        last_updated (float): Last updated epoch timestamp.
    """
    user_id: str
    current_emotion: str = "neutral"
    previous_emotion: Optional[str] = None
    dominant_trend: str = "neutral"
    history: List[EmotionResult] = field(default_factory=list)
    transitions: List[EmotionTransition] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class EmotionSummary:
    """
    High-level conversational summary of a user's emotional state.

    Attributes:
        user_id (str): Telegram User ID string.
        current_emotion (str): Current active emotion string.
        previous_emotion (Optional[str]): Preceding emotion string.
        dominant_trend (str): Most frequent emotion across session history.
        total_turns (int): Total recorded turns in context timeline.
        transitions_count (int): Number of emotion transitions detected.
        session_duration_seconds (float): Active emotion session duration in seconds.
    """
    user_id: str
    current_emotion: str
    previous_emotion: Optional[str]
    dominant_trend: str
    total_turns: int
    transitions_count: int
    session_duration_seconds: float


@dataclass
class EmotionMemoryRecord:
    """
    Record detailing an emotional event storage decision.

    Attributes:
        user_id (str): Telegram User ID string.
        emotion (str): Emotion label string.
        importance (int): Calculated importance score (1 to 10 scale).
        trigger_text (str): Input trigger user message.
        stored (bool): True if stored to SQLite DB via MemoryManager.
    """
    user_id: str
    emotion: str
    importance: int
    trigger_text: str
    stored: bool


@dataclass
class EmotionStatistics:
    """
    Aggregate emotional statistics for a user over time.

    Attributes:
        user_id (str): Telegram User ID string.
        emotion_counts (Dict[str, int]): Count of turns per detected emotion category.
        total_analyzed (int): Total number of messages analyzed.
    """
    user_id: str
    emotion_counts: Dict[str, int] = field(default_factory=dict)
    total_analyzed: int = 0
