"""
app/vision/vision_models/vision_factory.py - Vision Model & Engine Factory
===========================================================================
Provides factory creation methods for VisionModelConfig, Florence2Model,
VisionInferenceEngine, and VisionModelPipeline instances.
"""

from typing import Optional
from app.vision.vision_models.vision_config import get_vision_model_config, VisionModelConfig
from app.vision.vision_models.florence2_model import Florence2Model
from app.vision.vision_models.vision_inference import VisionInferenceEngine


class VisionFactory:
    """Factory helper creating Vision Model subsystem components."""

    @staticmethod
    def create_florence_model(config: Optional[VisionModelConfig] = None) -> Florence2Model:
        cfg = config or get_vision_model_config()
        return Florence2Model(cfg)

    @staticmethod
    def create_inference_engine(config: Optional[VisionModelConfig] = None) -> VisionInferenceEngine:
        cfg = config or get_vision_model_config()
        return VisionInferenceEngine(cfg)
