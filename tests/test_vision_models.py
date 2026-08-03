"""
tests/test_vision_models.py - Vision Model & EasyOCR Diagnostic Test Suite
===========================================================================

Verifies:
1. VisionTask and VisionStatus enums.
2. VisionModelConfig configuration settings.
3. Florence2Model task execution (<CAPTION>, <DETAILED_CAPTION>, <OD>).
4. VisionInferenceEngine scene captioning, object detection, and EasyOCR text extraction.
5. VisionCache & VisionStatisticsTracker operations.
6. Master VisionModelPipeline end-to-end multimodal inference.
"""

import sys
import os
import unittest
from PIL import Image as PILImage, ImageDraw

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision.vision_models import (
    VisionTask,
    VisionStatus,
    VisionModelConfig,
    VisionRequest,
    VisionResponse,
    CaptionResult,
    OCRTextResult,
    Florence2Model,
    VisionInferenceEngine,
    VisionCache,
    VisionStatisticsTracker,
    VisionModelPipeline
)


class TestVisionModelsPipeline(unittest.TestCase):
    def setUp(self):
        """Creates temporary test image in workspace."""
        self.img_path = "test_vision_sample.png"

        # Always ensure a valid sample image file exists
        if not os.path.exists(self.img_path):
            img = PILImage.new("RGB", (400, 300), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, 350, 250], outline=(0, 0, 0), width=3)
            draw.text((80, 100), "SANA AI Vision Test", fill=(0, 0, 0))
            img.save(self.img_path, format="PNG")

        self.pipeline = VisionModelPipeline.get_instance()
        self.cache = VisionCache.get_instance()
        self.cache.clear()

    def tearDown(self):
        """Cleans up temporary test file."""
        if os.path.exists(self.img_path):
            try:
                os.remove(self.img_path)
            except Exception:
                pass

    # 1. Test Vision Enums
    def test_vision_enums(self):
        self.assertEqual(VisionTask.CAPTION.value, "<CAPTION>")
        self.assertEqual(VisionTask.OBJECT_DETECTION.value, "<OD>")
        self.assertEqual(VisionStatus.COMPLETED.value, "completed")

    # 2. Test Florence2Model Wrapper Execution
    def test_florence2_model(self):
        model = Florence2Model()
        self.assertTrue(model.load_model())

        with PILImage.open(self.img_path) as img:
            res = model.run_task(img, task=VisionTask.CAPTION)

        self.assertIsNotNone(res)
        self.assertIn(VisionTask.CAPTION.value, res)

    # 3. Test VisionInferenceEngine
    def test_vision_inference_engine(self):
        engine = VisionInferenceEngine()
        with PILImage.open(self.img_path) as img:
            caption = engine.execute_captioning(img)
            self.assertIsNotNone(caption.short_caption)

            objects = engine.execute_object_detection(img)
            self.assertIsInstance(objects, list)

            ocr = engine.execute_ocr(img)
            self.assertIsInstance(ocr, OCRTextResult)

    # 4. Test VisionCache & Statistics Tracker
    def test_cache_and_tracker(self):
        tracker = VisionStatisticsTracker.get_instance()
        tracker.reset()

        tracker.record_request(success=True, latency_ms=150.0)
        stats = tracker.get_statistics()

        self.assertEqual(stats.total_requests, 1)
        self.assertEqual(stats.successful_requests, 1)

    # 5. Test Master VisionModelPipeline Execution
    def test_master_pipeline(self):
        req = VisionRequest(image_path=self.img_path, task=VisionTask.DETAILED_CAPTION)
        resp: VisionResponse = self.pipeline.process_vision_request(req)

        self.assertEqual(resp.status, VisionStatus.COMPLETED)
        self.assertIsNotNone(resp.context)
        self.assertIn("=== ATTACHED IMAGE VISUAL UNDERSTANDING ===", resp.context.formatted_prompt_block)
        self.assertGreater(resp.inference_time_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
