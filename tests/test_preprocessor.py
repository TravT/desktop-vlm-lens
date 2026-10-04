"""
Unit tests for image preprocessor and coordinate parsing.
"""

import unittest
from PIL import Image
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import preprocessor


class TestPreprocessor(unittest.TestCase):

    def test_aspect_ratio_downsampling_landscape(self):
        # 4K image 3840x2160
        img = Image.new("RGB", (3840, 2160), color="blue")
        scaled, meta = preprocessor.preprocess_image(img, max_dim=1024)

        self.assertEqual(scaled.width, 1024)
        self.assertEqual(scaled.height, int(2160 * (1024 / 3840)))
        self.assertEqual(meta["original_width"], 3840)
        self.assertEqual(meta["processed_width"], 1024)

    def test_aspect_ratio_downsampling_portrait(self):
        # Mobile portrait 1080x2400
        img = Image.new("RGB", (1080, 2400), color="red")
        scaled, meta = preprocessor.preprocess_image(img, max_dim=1024)

        self.assertEqual(scaled.height, 1024)
        self.assertEqual(scaled.width, int(1080 * (1024 / 2400)))

    def test_no_upscaling_small_image(self):
        # Small icon 300x200 should NOT be upscaled
        img = Image.new("RGB", (300, 200), color="green")
        scaled, meta = preprocessor.preprocess_image(img, max_dim=1024)

        self.assertEqual(scaled.width, 300)
        self.assertEqual(scaled.height, 200)

    def test_roi_cropping_normalized(self):
        img = Image.new("RGB", (1000, 1000), color="white")
        # Crop center box [250, 250, 750, 750] (0-1000 normalized)
        scaled, meta = preprocessor.preprocess_image(img, max_dim=1024, roi_crop=[250, 250, 750, 750])

        self.assertTrue(meta["crop_applied"])
        self.assertEqual(scaled.width, 500)
        self.assertEqual(scaled.height, 500)

    def test_grounding_coordinate_parsing(self):
        # Simulate Qwen-VL response with normalized coordinates
        model_out = "The submit button is located at [700, 400, 750, 600]."
        orig_w = 1920
        orig_h = 1080

        coords = preprocessor.parse_grounding_coordinates(model_out, orig_w, orig_h)
        self.assertEqual(len(coords), 1)

        c = coords[0]
        self.assertEqual(c["normalized_box"], [700, 400, 750, 600])
        # Expected pixel calculations:
        # ymin = 700 * 1080 / 1000 = 756
        # xmin = 400 * 1920 / 1000 = 768
        # ymax = 750 * 1080 / 1000 = 810
        # xmax = 600 * 1920 / 1000 = 1152
        self.assertEqual(c["pixel_box"], [756, 768, 810, 1152])
        # Click center: x = (768 + 1152) / 2 = 960, y = (756 + 810) / 2 = 783
        self.assertEqual(c["click_x"], 960)
        self.assertEqual(c["click_y"], 783)

    def test_base64_encoding(self):
        img = Image.new("RGB", (100, 100), color="yellow")
        b64 = preprocessor.encode_image_base64(img)
        self.assertTrue(b64.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(b64), 100)


if __name__ == "__main__":
    unittest.main()
