"""
app/llm/model_loader.py - Singleton Model & Tokenizer Loader
==============================================================

1. PURPOSE:
-----------
Loads and retains the Qwen language model (`AutoModelForCausalLM`) and tokenizer (`AutoTokenizer`)
from local storage (`models/qwen`) into system memory (RAM/VRAM) as a thread-safe Singleton.

2. WHY IT EXISTS:
-----------------
- Loading a Large Language Model involves deserializing weight files (e.g. `model.safetensors`),
  allocating GPU/CPU memory, and initializing tensor data structures.
- This process takes 3-15 seconds depending on hardware.
- If we reloaded the model on every user message, latency would explode and memory would quickly overflow.
- Singleton pattern guarantees the model is loaded EXACTLY ONCE when the application starts.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Reads model path from `bot/config.py` or default path `models/qwen`.
- Called by `app/llm/generator.py` to retrieve model and tokenizer references.
- Triggered during bot startup in `bot/bot.py`.

4. HARDWARE OPTIMIZATION:
-------------------------
- Checks `torch.cuda.is_available()`:
  - If CUDA (GPU) is available:
    * Device set to `cuda`
    * Data type set to `torch.float16` or `torch.bfloat16` (halves memory usage, accelerates matrix math)
  - If CUDA is unavailable (CPU fallback):
    * Device set to `cpu`
    * Data type set to `torch.float32` (standard CPU floating point precision)

5. COMMON MISTAKES:
-------------------
- Instantiating `ModelLoader()` directly instead of using `ModelLoader.get_instance()`.
- Forgetting `local_files_only=True`, which might cause HuggingFace to attempt downloading from internet.
- Not specifying `torch_dtype`, causing double RAM allocation during float32 cast.
"""

import os
import sys
import time
import logging
import threading
from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("sana_ai.llm.model_loader")


class ModelLoader:
    """
    Thread-safe Singleton class for loading and holding Qwen LLM and Tokenizer instances.
    """
    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor to enforce Singleton pattern."""
        if ModelLoader._instance is not None:
            raise RuntimeError(
                "ModelLoader is a Singleton! Use `ModelLoader.get_instance()` instead of instantiating directly."
            )
        
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_path: str = "models/qwen"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """
        Thread-safe accessor for the ModelLoader singleton instance.

        Returns:
            ModelLoader: Shared singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self, model_path: str = "models/qwen") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the Qwen model and tokenizer into memory if not already loaded.

        Args:
            model_path (str): Relative or absolute path to local model folder.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer tuple.

        Raises:
            FileNotFoundError: If model_path directory does not exist or lacks config.json.
            Exception: Wraps any underlying PyTorch or HuggingFace initialization error.
        """
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            logger.debug("ModelLoader: Model already resident in memory. Skipping reload.")
            return self.model, self.tokenizer

        with self._lock:
            # Double-check inside lock to prevent race condition across threads
            if self.is_loaded and self.model is not None and self.tokenizer is not None:
                return self.model, self.tokenizer

            self.model_path = model_path
            abs_model_path = os.path.abspath(model_path)

            if not os.path.exists(abs_model_path):
                error_msg = f"Model path directory not found at: '{abs_model_path}'"
                logger.critical(error_msg)
                raise FileNotFoundError(error_msg)

            config_file = os.path.join(abs_model_path, "config.json")
            if not os.path.exists(config_file):
                error_msg = f"Missing 'config.json' in model path: '{abs_model_path}'"
                logger.critical(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info(f"⏳ Starting Qwen LLM load from: '{abs_model_path}'")
            logger.info(f"🎯 Target Device: '{self.device.upper()}' | Precision: '{self.torch_dtype}'")

            start_time = time.perf_counter()

            try:
                # 1. Load Tokenizer
                logger.info("Loading Tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    abs_model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )

                # Ensure pad token is defined (Qwen uses eos_token as pad_token if not explicit)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                # 2. Load Causal Language Model
                logger.info("Loading Causal LM Weights into memory...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    abs_model_path,
                    dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    local_files_only=True,
                    trust_remote_code=True
                )

                if self.device == "cpu":
                    self.model.to("cpu")

                self.model.eval()  # Set model to evaluation mode (disables dropout layers)
                self.is_loaded = True

                elapsed_time = time.perf_counter() - start_time
                logger.info(
                    f"✅ Qwen Model & Tokenizer loaded successfully in {elapsed_time:.2f}s "
                    f"[Device: {self.device.upper()}, Mode: EVAL]"
                )

                return self.model, self.tokenizer

            except Exception as exc:
                logger.critical(f"❌ Failed to load Qwen model from '{abs_model_path}': {exc}", exc_info=True)
                raise exc

    def get_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Retrieves the active model and tokenizer, loading them if not already loaded.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Active model and tokenizer instance.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return self.load_model(self.model_path)
        return self.model, self.tokenizer
