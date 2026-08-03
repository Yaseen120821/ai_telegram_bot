"""
app/vision/vision_models/vision_model_manager.py - Vision Model Lifecycle Manager
==================================================================================
Manages model initializations, device assignments (CPU/CUDA), memory warm-ups,
and resource cleanups for Florence-2 Base and EasyOCR engines.
"""

import logging
import torch
from typing import Optional
from PIL import Image as PILImage

from app.vision.vision_models.vision_config import get_vision_model_config, VisionModelConfig
from app.vision.vision_models.florence2_model import Florence2Model

logger = logging.getLogger("sana_ai.vision.models.manager")


class VisionModelManager:
    """Singleton manager for Vision AI neural network model lifecycle."""

    _instance: Optional["VisionModelManager"] = None

    def __init__(self, config: Optional[VisionModelConfig] = None):
        self.config = config or get_vision_model_config()
        self.florence_model = Florence2Model(self.config)

    @classmethod
    def get_instance(cls) -> "VisionModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize_models(self) -> bool:
        """Initializes model weights and performs warm-up inference."""
        logger.info("Initializing Vision Model Subsystem...")
        success = self.florence_model.load_model()
        if success:
            self._warmup_inference()
        return success

    def _warmup_inference(self) -> None:
        """Runs a dummy 64x64 warm-up image to initialize CUDA kernels/caches."""
        try:
            dummy_img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
            self.florence_model.run_task(dummy_img, task=self.florence_model.config.model_name)
            logger.info("Vision model warm-up inference completed successfully.")
        except Exception as e:
            logger.debug(f"Warm-up inference notice: {e}")

    def get_florence_model(self) -> Florence2Model:
        """Returns loaded Florence2Model instance."""
        if not self.florence_model.is_loaded:
            self.initialize_models()
        return self.florence_model

    def shutdown(self) -> None:
        """Unloads vision models from memory."""
        self.florence_model.unload()
