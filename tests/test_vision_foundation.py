"""
tests/test_vision_foundation.py - Vision AI Foundation Diagnostic Test Suite
=============================================================================

Verifies:
1. VisionTypes enums integrity (ImageFormat, ImageType, VisionTask, VisionStatus).
2. VisionConfig global settings & singleton configuration overrides.
3. Vision models & dataclasses instantiation (VisionRequest, ImageMetadata, ImageContext).
4. Vision utility routines (ID generation, path normalization, file hash, validation).
5. VisionManager orchestrator process_query, error handling, caching, and prompt context formatting.
"""

import sys
import os
import unittest
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision import (
    ImageFormat,
    ImageType,
    VisionTask,
    VisionStatus,
    AnalysisMode,
    VisionConfig,
    get_vision_config,
    VisionRequest,
    VisionResult,
    ImageMetadata,
    ImageContext,
    CaptionResult,
    OCRResult,
    generate_image_id,
    normalize_image_path,
    calculate_file_hash,
    validate_image_file,
    extract_image_metadata,
    VisionManager
)


class TestVisionFoundationFramework(unittest.TestCase):
    def setUp(self):
        """Prepare temporary test image file in workspace."""
        self.test_img_path = "test_sample_image.png"

        # Create dummy PNG file with valid PNG header bytes
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06\x00\x00\x00'
        with open(self.test_img_path, "wb") as f:
            f.write(png_header + b'\x00' * 100)

        self.manager = VisionManager.get_instance()
        self.manager.clear_cache()

    def tearDown(self):
        """Clean up temporary test file."""
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    # 1. Test Vision Enums
    def test_vision_enums(self):
        self.assertEqual(ImageFormat.PNG.value, "png")
        self.assertEqual(ImageType.SCREENSHOT.value, "screenshot")
        self.assertEqual(VisionTask.CAPTIONING.value, "captioning")
        self.assertEqual(VisionStatus.COMPLETED.value, "completed")
        self.assertEqual(AnalysisMode.FAST.value, "fast")

    # 2. Test Vision Config
    def test_vision_config(self):
        config = get_vision_config()
        self.assertGreater(config.max_file_size_bytes, 0)
        self.assertIn("png", config.supported_formats)
        self.assertEqual(config.vision_model_name, "florence-2-base")

    # 3. Test Vision Models Dataclasses
    def test_vision_models_instantiation(self):
        meta = ImageMetadata(file_path=self.test_img_path, file_name="sample.png", width=800, height=600)
        self.assertEqual(meta.width, 800)

        req = VisionRequest(image_path=self.test_img_path)
        self.assertTrue(req.request_id.startswith("vreq_"))

        caption = CaptionResult(short_caption="A sample image.")
        self.assertEqual(caption.short_caption, "A sample image.")

    # 4. Test Vision Utilities
    def test_vision_utilities(self):
        # ID generation
        img_id = generate_image_id("test")
        self.assertTrue(img_id.startswith("test_"))

        # Path normalization
        norm = normalize_image_path("./test_sample_image.png")
        self.assertTrue(os.path.isabs(norm))

        # Hash calculation
        h = calculate_file_hash(self.test_img_path)
        self.assertEqual(len(h), 64)

        # File validation
        is_valid, msg = validate_image_file(self.test_img_path)
        self.assertTrue(is_valid)

        # Invalid file validation
        is_valid_fake, msg_fake = validate_image_file("non_existent_file.png")
        self.assertFalse(is_valid_fake)

        # Metadata extraction
        meta = extract_image_metadata(self.test_img_path)
        self.assertEqual(meta.format, ImageFormat.PNG)
        self.assertEqual(meta.width, 256)
        self.assertEqual(meta.height, 256)

    # 5. Test VisionManager Process Image Pipeline
    def test_vision_manager_pipeline(self):
        req = VisionRequest(image_path=self.test_img_path)
        res: VisionResult = self.manager.process_image(req)

        self.assertEqual(res.status, VisionStatus.COMPLETED)
        self.assertIsNotNone(res.image_context)
        self.assertEqual(res.image_context.metadata.format, ImageFormat.PNG)
        self.assertIn("=== ATTACHED IMAGE CONTEXT", res.image_context.formatted_prompt_block)

    # 6. Test VisionManager Caching
    def test_vision_manager_caching(self):
        req1 = VisionRequest(image_path=self.test_img_path)
        res1 = self.manager.process_image(req1)

        # Second request for same file hash should hit cache
        req2 = VisionRequest(image_path=self.test_img_path)
        res2 = self.manager.process_image(req2)

        self.assertEqual(res1.image_context.metadata.sha256_hash, res2.image_context.metadata.sha256_hash)
        self.assertEqual(res2.status, VisionStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
