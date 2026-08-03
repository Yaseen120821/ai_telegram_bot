"""
app/emotion/emotion_detector.py - Transformer-Based Emotion Detection Engine
=============================================================================

1. PURPOSE:
-----------
Performs text inference using a local PyTorch Transformer model (`AutoModelForSequenceClassification`) to evaluate raw
Softmax emotion probability distributions across discrete emotion categories.

2. WHY IT EXISTS (TRANSFORMER INFERENCE ENGINE & FALLBACK):
------------------------------------------------------------
Provides a high-performance, local Transformer model loader and inference engine. Auto-detects CUDA GPU vs CPU execution,
handles subword tokenization, calculates Softmax probabilities, and falls back gracefully to pattern heuristics if model
weights are absent or CUDA OOM occurs.

3. RESPONSIBILITIES:
--------------------
- Eagerly/Lazily load HuggingFace PyTorch tokenizer and sequence classification model as a Singleton.
- Auto-detect CUDA hardware (`torch.cuda.is_available()`) and place model on `cuda` or `cpu`.
- Tokenize raw text input (`Input IDs`, `Attention Mask`).
- Compute PyTorch model forward pass, extract logits, and calculate Softmax probabilities (`torch.softmax`).
- Return `PredictionResult` dataclass with top predicted emotion, score, and execution metadata.
- Gracefully fall back to pattern lexicon parsing if model loading encounters errors.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `PredictionResult` and `EmotionMetadata` from `app/emotion/emotion_models.py`.
- Called by `app/emotion/emotion_classifier.py`.

5. COMPLETE CODE:
-----------------
"""

import os
import re
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.emotion.emotion_models import PredictionResult, EmotionMetadata
from app.emotion.emotion_types import EmotionType
from app.emotion.emotion_utils import EmotionUtils

logger = logging.getLogger("sana_ai.emotion.detector")

DEFAULT_EMOTION_MODEL_DIR = "models/emotion"
FALLBACK_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"


class EmotionDetector:
    """
    Singleton Transformer-based Emotion Detection Engine supporting local PyTorch inference,
    CUDA/CPU device placement, Softmax probabilities, and pattern heuristic fallback.
    """
    _instance: Optional["EmotionDetector"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initializes EmotionDetector Singleton."""
        if EmotionDetector._instance is not None:
            raise RuntimeError(
                "EmotionDetector is a Singleton! Use `EmotionDetector.get_instance()` instead."
            )

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name: str = DEFAULT_EMOTION_MODEL_DIR
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.model_loaded: bool = False

        # Pattern fallback dictionary for offline resilience
        self._init_pattern_fallback()

        # Attempt Singleton model loading
        self._load_transformer_model()

    @classmethod
    def get_instance(cls) -> "EmotionDetector":
        """
        Thread-safe Singleton accessor.

        Returns:
            EmotionDetector: Shared Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _init_pattern_fallback(self) -> None:
        """Initializes fallback pattern regexes for resilience."""
        self.fallback_patterns: Dict[str, List[re.Pattern]] = {
            EmotionType.JOY.value: [
                re.compile(r"\b(happy|joy|wonderful|great|awesome|delighted|pleased|dream job|selected)\b", re.IGNORECASE)
            ],
            EmotionType.SADNESS.value: [
                re.compile(r"\b(sad|depressed|failed|failing|crying|miserable|heartbroken|unhappy)\b", re.IGNORECASE)
            ],
            EmotionType.ANGER.value: [
                re.compile(r"\b(angry|furious|outraged|mad|hate|infuriated|disgusted)\b", re.IGNORECASE)
            ],
            EmotionType.FEAR.value: [
                re.compile(r"\b(fear|scared|terrified|nervous|anxious|panic|worry|worried)\b", re.IGNORECASE)
            ],
            EmotionType.FRUSTRATION.value: [
                re.compile(r"\b(frustrated|frustrating|annoyed|broken|stuck|not working|bugs?)\b", re.IGNORECASE)
            ],
            EmotionType.PRIDE.value: [
                re.compile(r"\b(proud|achieved|accomplished|won|succeeded|finally finished)\b", re.IGNORECASE)
            ],
            EmotionType.EXCITEMENT.value: [
                re.compile(r"\b(excited|exciting|thrilled|hyped|can't wait)\b", re.IGNORECASE)
            ],
            EmotionType.CURIOSITY.value: [
                re.compile(r"\b(curious|wondering|how does|why is|interested in)\b", re.IGNORECASE)
            ]
        }

    def _load_transformer_model(self) -> None:
        """
        Loads PyTorch Tokenizer and Sequence Classification Model onto designated hardware device.
        """
        logger.info(f"⚙️ Initializing Emotion Detector Model Loader [Target Device: {self.device.type.upper()}]")
        target_dir = os.path.abspath(self.model_name)

        model_path_to_use = target_dir if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "config.json")) else FALLBACK_MODEL_NAME

        try:
            logger.info(f"📦 Loading Emotion Tokenizer from: '{model_path_to_use}'")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_to_use)
            logger.info("✅ Emotion Tokenizer Loaded Successfully.")

            logger.info(f"🧠 Loading Emotion Transformer Model from: '{model_path_to_use}'")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path_to_use)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode (disables dropout)

            self.model_loaded = True
            self.model_name = model_path_to_use
            logger.info(
                f"🎉 Emotion Transformer Model Loaded Successfully on [{self.device.type.upper()}] | "
                f"Model: {model_path_to_use}"
            )
        except Exception as load_err:
            logger.warning(
                f"⚠️ Could not load HuggingFace Transformer model '{model_path_to_use}' ({load_err}). "
                "Switching to high-speed pattern heuristic fallback engine."
            )
            self.model_loaded = False

    def predict_emotion(self, text: str) -> PredictionResult:
        """
        Executes emotion detection on input text using local Transformer model or pattern fallback.

        Args:
            text (str): Raw input user statement.

        Returns:
            PredictionResult: PredictionResult object containing top emotion label, score, and scores map.
        """
        cleaned = EmotionUtils.clean_emotion_text(text)
        if not cleaned:
            return PredictionResult(
                top_emotion=EmotionType.NEUTRAL.value,
                top_score=0.95,
                raw_scores={EmotionType.NEUTRAL.value: 0.95},
                metadata=EmotionMetadata(model_name="fallback_neutral", device=str(self.device), latency_ms=0.0)
            )

        start_time = time.time()

        # If Transformer model is loaded, execute PyTorch inference
        if self.model_loaded and self.tokenizer and self.model:
            try:
                # 1. Tokenize input text (Truncate to max 128 tokens)
                inputs = self.tokenizer(
                    cleaned,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                )

                # 2. Move tensors to hardware device (CUDA / CPU)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 3. Execute PyTorch Model Forward Pass (No Gradient Computation)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    # Apply Softmax to convert raw logits to probabilities
                    probs = torch.softmax(logits, dim=-1).squeeze(0)

                # 4. Map probabilities to label strings
                id2label = self.model.config.id2label
                raw_scores: Dict[str, float] = {}

                for idx, prob in enumerate(probs):
                    label_str = id2label[idx].lower() if hasattr(self.model.config, "id2label") and idx in id2label else f"label_{idx}"
                    raw_scores[label_str] = float(prob.item())

                # Sort predicted scores
                sorted_probs = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
                top_emotion, top_score = sorted_probs[0]

                latency_ms = (time.time() - start_time) * 1000.0
                meta = EmotionMetadata(model_name=self.model_name, device=str(self.device), latency_ms=latency_ms)

                logger.debug(
                    f"⚡ Transformer Inference Completed [{latency_ms:.2f}ms on {self.device.type.upper()}] | "
                    f"Top Emotion: '{top_emotion}' ({top_score:.4f})"
                )

                return PredictionResult(
                    top_emotion=top_emotion,
                    top_score=top_score,
                    raw_scores=raw_scores,
                    metadata=meta
                )

            except Exception as inf_err:
                logger.error(f"❌ PyTorch Transformer Inference Error ({inf_err}). Falling back to pattern heuristic.")

        # Fallback Engine (Pattern Heuristic)
        return self._predict_pattern_fallback(cleaned, start_time)

    def _predict_pattern_fallback(self, cleaned: str, start_time: float) -> PredictionResult:
        """Fallback prediction using high-speed lexicon pattern parsing."""
        scores: Dict[str, float] = {EmotionType.NEUTRAL.value: 0.50}

        for emotion, patterns in self.fallback_patterns.items():
            match_count = 0
            for pattern in patterns:
                matches = pattern.findall(cleaned)
                if matches:
                    match_count += len(matches)

            if match_count > 0:
                score = min(0.98, 0.70 + (match_count * 0.12))
                scores[emotion] = score

        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_score = sorted_emotions[0]

        latency_ms = (time.time() - start_time) * 1000.0
        meta = EmotionMetadata(model_name="pattern_fallback_engine", device="cpu", latency_ms=latency_ms)

        logger.debug(f"🔍 Pattern Fallback Output: '{top_emotion}' ({top_score:.2f}) | Latency: {latency_ms:.2f}ms")

        return PredictionResult(
            top_emotion=top_emotion,
            top_score=top_score,
            raw_scores=scores,
            metadata=meta
        )
