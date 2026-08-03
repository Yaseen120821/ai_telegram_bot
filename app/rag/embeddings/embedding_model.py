"""
app/rag/embeddings/embedding_model.py - Sentence Transformer Model Loader
========================================================================

1. PURPOSE:
-----------
Loads and manages HuggingFace Sentence Transformer models (`sentence-transformers/all-MiniLM-L6-v2`) on GPU (`cuda`) or CPU.

2. WHY IT EXISTS (RESOURCE MANAGEMENT & INFERENCE ACCELERATION):
----------------------------------------------------------------
Sentence Transformer models require PyTorch CUDA allocations and memory management. `EmbeddingModel` encapsulates
model loading, tokenization pipeline instantiation, device allocation, and thread-safe reference caching.

3. RESPONSIBILITIES:
--------------------
- Safely load model checkpoint (`all-MiniLM-L6-v2`).
- Transfer model weights to `cuda` if PyTorch CUDA is enabled, else `cpu`.
- Provide underlying tokenizer and transformer model references.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `DEFAULT_EMBEDDING_MODEL`, `DEVICE` from `embedding_config.py`.
- Used by `embedding_generator.py` and `embedding_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
import threading
from typing import Optional, Any

import torch
from app.rag.embeddings.embedding_config import DEFAULT_EMBEDDING_MODEL, DEVICE, EMBEDDING_DIMENSION

logger = logging.getLogger("sana_ai.rag.embeddings.model")


class EmbeddingModel:
    """
    Model loader and wrapper for Sentence Transformer models.
    """
    _instance: Optional["EmbeddingModel"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = DEVICE) -> None:
        """
        Initializes embedding model.

        Args:
            model_name (str): HuggingFace model repo string.
            device (str): Execution hardware device string ('cuda' or 'cpu').
        """
        self.model_name: str = model_name
        self.device: str = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._is_loaded: bool = False

        logger.info(f"🧠 EmbeddingModel wrapper initialized | Model: '{self.model_name}' | Device target: '{self.device}'")

    def load_model(self) -> bool:
        """
        Loads Sentence Transformer model weights onto target hardware device.

        Returns:
            bool: True if model loaded successfully.
        """
        if self._is_loaded and self._model is not None:
            return True

        with self._lock:
            if self._is_loaded and self._model is not None:
                return True

            logger.info(f"⏳ Loading Sentence Transformer Model '{self.model_name}' onto device '{self.device}'...")
            try:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    self._is_loaded = True
                    logger.info(f"🎉 SentenceTransformer model loaded successfully on '{self.device}'!")
                    return True
                except ImportError:
                    logger.warning("sentence-transformers package not available. Loading via HuggingFace AutoModel...")
                    from transformers import AutoTokenizer, AutoModel
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
                    self._model.eval()
                    self._is_loaded = True
                    logger.info(f"🎉 AutoModel transformer loaded successfully on '{self.device}'!")
                    return True
            except Exception as err:
                logger.error(f"❌ Failed to load embedding model '{self.model_name}': {err}", exc_info=True)
                self._is_loaded = False
                return False

    @property
    def is_loaded(self) -> bool:
        """Returns True if model is loaded and ready."""
        return self._is_loaded

    @property
    def model(self) -> Optional[Any]:
        """Returns active model instance."""
        return self._model

    @property
    def dimension(self) -> int:
        """Returns expected vector dimension count."""
        return EMBEDDING_DIMENSION
