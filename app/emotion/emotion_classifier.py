"""
app/emotion/emotion_classifier.py - Emotion Result Classification Engine
========================================================================

1. PURPOSE:
-----------
Processes raw predictions from `EmotionDetector` and maps them into structured `EmotionResult` instances,
calculating emotional intensity levels and assigning formatted System Prompt empathy directives.

2. WHY IT EXISTS:
-----------------
Decouples raw model prediction output from emotion taxonomy classification. Applies minimum confidence cutoffs,
calculates intensity levels (`low`, `medium`, `high`, `extreme`), and attaches empathy prompt directives.

3. RESPONSIBILITIES:
--------------------
- Call `EmotionDetector.get_instance().predict_emotion()`.
- Filter predictions below confidence threshold (defaulting to `neutral`).
- Calculate intensity using `EmotionUtils.calculate_intensity()`.
- Attach system prompt empathy directives via `EmotionUtils.get_empathy_prompt()`.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `EmotionDetector` from `app/emotion/emotion_detector.py`.
- Uses `EmotionResult` from `app/emotion/emotion_models.py`.
- Called by `app/emotion/emotion_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import Optional

from app.emotion.emotion_detector import EmotionDetector
from app.emotion.emotion_models import EmotionResult, PredictionResult
from app.emotion.emotion_types import EmotionType, EmotionIntensity
from app.emotion.emotion_utils import EmotionUtils
from app.emotion.emotion_config import DEFAULT_MIN_CONFIDENCE

logger = logging.getLogger("sana_ai.emotion.classifier")


class EmotionClassifier:
    """
    Classifies raw emotion predictions into verified, structured EmotionResult records.
    """

    def __init__(self, detector: Optional[EmotionDetector] = None) -> None:
        """
        Initializes EmotionClassifier.

        Args:
            detector (Optional[EmotionDetector]): EmotionDetector Singleton instance.
        """
        self.detector: EmotionDetector = detector or EmotionDetector.get_instance()

    def classify_emotion(self, text: str) -> EmotionResult:
        """
        Analyzes input text statement and returns a structured EmotionResult.

        Args:
            text (str): Input text statement from user.

        Returns:
            EmotionResult: Structured EmotionResult dataclass instance.
        """
        pred: PredictionResult = self.detector.predict_emotion(text)

        # Fallback to NEUTRAL if prediction confidence is below cutoff threshold
        if pred.top_score < DEFAULT_MIN_CONFIDENCE or pred.top_emotion == EmotionType.NEUTRAL.value:
            primary_emotion = EmotionType.NEUTRAL.value
            confidence = 0.95
            intensity = EmotionIntensity.LOW.value
        else:
            primary_emotion = pred.top_emotion
            confidence = EmotionUtils.normalize_confidence(pred.top_score)
            intensity = EmotionUtils.calculate_intensity(confidence)

        empathy_directive = EmotionUtils.get_empathy_prompt(primary_emotion)

        logger.info(
            f"🎯 Classified Emotion | Primary: '{primary_emotion}' | Intensity: '{intensity}' | "
            f"Confidence: {confidence:.2f} | Latency: {pred.metadata.latency_ms if pred.metadata else 0.0:.2f}ms"
        )

        return EmotionResult(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence,
            secondary_emotions=pred.raw_scores,
            empathy_directive=empathy_directive
        )
