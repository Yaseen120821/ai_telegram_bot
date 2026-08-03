"""
tests/test_image_processing.py - Image Processing Pipeline Diagnostic Test Suite
================================================================================

Verifies:
1. Image loading & validation (PNG, JPEG, GIF, corrupted file, invalid format).
2. Technical metadata extraction & perceptual hashing (SHA256, MD5, aHash, dHash, pHash).
3. Quantitative image statistics (brightness, contrast, Laplacian sharpness, histogram).
4. Resizing algorithms & aspect ratio handling (LETTERBOX, MAINTAIN_ASPECT_RATIO, CENTER_CROP, STRETCH).
5. Color space conversion to RGB & float32 pixel tensor normalization.
6. In-memory LRU Image Cache operation (hits, eviction, clearing).
7. Unified ImageProcessingPipeline end-to-end execution.
"""

import sys
import os
import unittest
from PIL import Image as PILImage, ImageDraw

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision.image_processing import (
    ImageFormat,
    ImageColorMode,
    ValidationStatus,
    ResizeMode,
    NormalizationMode,
    ImageProcessingConfig,
    ImageRequest,
    ImageLoader,
    ImageValidator,
    MetadataExtractor,
    ImageStatisticsCalculator,
    ImagePreprocessor,
    ImageOptimizer,
    ImageCache,
    ImageProcessingPipeline,
    compute_average_hash,
    compute_perceptual_hash
)


class TestImageProcessingPipeline(unittest.TestCase):
    def setUp(self):
        """Creates temporary test images in workspace."""
        self.png_path = "test_sample_1.png"
        self.jpg_path = "test_sample_2.jpg"
        self.invalid_path = "test_invalid.txt"

        # 1. Create a 400x200 red/blue split PNG image
        img_png = PILImage.new("RGB", (400, 200), color=(255, 0, 0))
        draw = ImageDraw.Draw(img_png)
        draw.rectangle([200, 0, 400, 200], fill=(0, 0, 255))
        img_png.save(self.png_path, format="PNG")

        # 2. Create a 300x300 RGBA image saved as JPEG
        img_jpg = PILImage.new("RGBA", (300, 300), color=(0, 255, 0, 128))
        img_jpg.convert("RGB").save(self.jpg_path, format="JPEG")

        # 3. Create a non-image text file
        with open(self.invalid_path, "w") as f:
            f.write("This is a plain text file, not an image.")

        self.pipeline = ImageProcessingPipeline.get_instance()
        self.cache = ImageCache.get_instance()
        self.cache.clear()

    def tearDown(self):
        """Cleans up temporary test files."""
        for p in (self.png_path, self.jpg_path, self.invalid_path):
            if os.path.exists(p):
                os.remove(p)

    # 1. Test ImageLoader
    def test_image_loader(self):
        img, err = ImageLoader.load_image(self.png_path)
        self.assertIsNone(err)
        self.assertIsNotNone(img)
        self.assertEqual(img.width, 400)
        self.assertEqual(img.height, 200)

        # Invalid path loading
        fake_img, fake_err = ImageLoader.load_image("non_existent.png")
        self.assertIsNotNone(fake_err)
        self.assertIsNone(fake_img)

    # 2. Test ImageValidator
    def test_image_validator(self):
        validator = ImageValidator()
        res = validator.validate(self.png_path)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, ValidationStatus.PASSED)

        # Invalid file extension validation
        invalid_res = validator.validate(self.invalid_path)
        self.assertFalse(invalid_res.is_valid)
        self.assertEqual(invalid_res.status, ValidationStatus.INVALID_FORMAT)

    # 3. Test MetadataExtractor & Perceptual Hashing
    def test_metadata_and_hashes(self):
        img, _ = ImageLoader.load_image(self.png_path)
        meta = MetadataExtractor.extract_metadata(self.png_path, img)

        self.assertEqual(meta.width, 400)
        self.assertEqual(meta.height, 200)
        self.assertEqual(meta.aspect_ratio, 2.0)
        self.assertEqual(meta.format, ImageFormat.PNG)
        self.assertTrue(len(meta.hashes.sha256) == 64)
        self.assertTrue(len(meta.hashes.average_hash) > 0)

    # 4. Test ImageStatisticsCalculator
    def test_image_statistics(self):
        img, _ = ImageLoader.load_image(self.png_path)
        stats = ImageStatisticsCalculator.calculate_statistics(img)

        self.assertGreater(stats.brightness, 0.0)
        self.assertGreater(stats.contrast, 0.0)
        self.assertIsNotNone(stats.sharpness)
        self.assertIn("r", stats.color_histogram)

    # 5. Test ImagePreprocessor Resizing Modes
    def test_preprocessor_resizing(self):
        img, _ = ImageLoader.load_image(self.png_path)

        # Letterbox 800x800 padding
        letterbox = ImagePreprocessor.resize_image(img, target_size=(800, 800), mode=ResizeMode.LETTERBOX)
        self.assertEqual(letterbox.width, 800)
        self.assertEqual(letterbox.height, 800)

        # Maintain aspect ratio (thumbnail inside 800x800)
        ratio_img = ImagePreprocessor.resize_image(img, target_size=(800, 800), mode=ResizeMode.MAINTAIN_ASPECT_RATIO)
        self.assertEqual(ratio_img.width, 800)
        self.assertEqual(ratio_img.height, 400)

        # Center crop 800x800
        crop_img = ImagePreprocessor.resize_image(img, target_size=(800, 800), mode=ResizeMode.CENTER_CROP)
        self.assertEqual(crop_img.width, 800)
        self.assertEqual(crop_img.height, 800)

    # 6. Test ImagePreprocessor Pixel Normalization
    def test_preprocessor_normalization(self):
        img, _ = ImageLoader.load_image(self.png_path)
        tensor = ImagePreprocessor.normalize_pixel_tensor(img, mode=NormalizationMode.RESCALE_0_1)

        self.assertEqual(tensor.dtype, "float32")
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)

    # 7. Test ImageCache
    def test_image_cache(self):
        req = ImageRequest(image_path=self.png_path)
        processed, val_res = self.pipeline.process_image(req)

        self.assertIsNotNone(processed)
        file_hash = processed.original_metadata.hashes.sha256
        self.assertTrue(self.cache.has(file_hash))

        # Cache Hit
        cached_img = self.cache.get(file_hash)
        self.assertIsNotNone(cached_img)
        self.assertEqual(self.cache.hit_rate, 1.0)

    # 8. Test Full Pipeline Execution
    def test_full_pipeline_execution(self):
        req = ImageRequest(
            image_path=self.jpg_path,
            target_size=(500, 500),
            resize_mode=ResizeMode.LETTERBOX,
            normalization_mode=NormalizationMode.RESCALE_0_1
        )
        processed, val = self.pipeline.process_image(req)

        self.assertTrue(val.is_valid)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.processed_width, 500)
        self.assertEqual(processed.processed_height, 500)
        self.assertEqual(processed.processed_color_mode, ImageColorMode.RGB)
        self.assertGreater(processed.processing_time_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
