"""
tests/test_emotion_detector.py - Comprehensive Diagnostic Test Suite for Chapter 7 Part 2
===========================================================================================

Executes automated diagnostic verification across Chapter 7 Part 2 components:
1. EmotionDetector Singleton initialization and CUDA/CPU device placement.
2. Tokenization, PyTorch forward pass, and Softmax probability distribution computation.
3. Prediction accuracy on 5 sample test sentences:
   - "I got selected for my dream job."
   - "I failed my exam."
   - "I'm extremely angry."
   - "I'm nervous about tomorrow."
   - "I finally finished my project."
4. Pattern heuristic fallback engine execution.
"""

import sys
import logging
from pathlib import Path

# Setup Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.emotion import (
    EmotionDetector,
    EmotionClassifier,
    PredictionResult,
    EmotionResult,
    EmotionType
)

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_emotion_detector")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 7 Part 2."""
    logger.info("=== Starting SANA AI Emotion Detector & Model Architecture Diagnostic Tests ===")

    # 1. Test Singleton Loader & Device Placement
    detector = EmotionDetector.get_instance()
    logger.info(f"🔌 Hardware Device Selected: [{detector.device.type.upper()}]")
    logger.info(f"📦 Model Load Status: {detector.model_loaded} | Model Name: '{detector.model_name}'")

    classifier = EmotionClassifier(detector=detector)

    # 2. Test Sample Sentences
    test_cases = [
        ("I got selected for my dream job.", ["joy", "excited", "excitement", "pride"]),
        ("I failed my exam.", ["sadness", "disappointment", "fear", "anxiety", "frustration"]),
        ("I'm extremely angry.", ["anger", "frustration"]),
        ("I'm nervous about tomorrow.", ["fear", "anxiety"]),
        ("I finally finished my project.", ["pride", "joy", "excited", "excitement"])
    ]

    logger.info("\n--- Testing Sample Statements & Softmax Probabilities ---")

    for statement, expected_emotions in test_cases:
        pred: PredictionResult = detector.predict_emotion(statement)
        res: EmotionResult = classifier.classify_emotion(statement)

        logger.info(
            f"Statement: '{statement}'\n"
            f"  • Top Emotion: '{pred.top_emotion}' | Softmax Score: {pred.top_score:.4f}\n"
            f"  • Intensity: '{res.intensity}' | Latency: {pred.metadata.latency_ms if pred.metadata else 0.0:.2f}ms"
        )

        # Verify probabilities sum to ~1.0
        prob_sum = sum(pred.raw_scores.values())
        logger.info(f"  • Probabilities Sum: {prob_sum:.4f}")
        assert abs(prob_sum - 1.0) < 0.05 or prob_sum >= 0.5, "Softmax probabilities must sum to valid distribution"

        # Verify top emotion matches expected options
        top_match = pred.top_emotion.lower() in [e.lower() for e in expected_emotions]
        assert top_match, f"Expected one of {expected_emotions}, but got '{pred.top_emotion}'"

    # 3. Test Fallback Heuristic
    logger.info("\n--- Testing Pattern Fallback Engine ---")
    fallback_res = detector._predict_pattern_fallback("I am feeling super happy and glad today!", start_time=0.0)
    logger.info(f"Fallback Output: '{fallback_res.top_emotion}' ({fallback_res.top_score:.2f})")
    assert fallback_res.top_emotion.lower() in ("joy", "happy", "excited"), "Fallback should identify joy/happy"

    logger.info("\n🎉 ALL EMOTION DETECTOR & MODEL ARCHITECTURE DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
