"""
Unit tests for capture and image loading module.
"""

import unittest
import tempfile
import sys
from pathlib import Path
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import capture


class TestCapture(unittest.TestCase):

    def test_load_existing_image_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            img = Image.new("RGB", (640, 480), color="orange")
            img.save(tmp_path)

            loaded_img, meta = capture.load_image_file(tmp_path)
            self.assertIsNotNone(loaded_img)
            self.assertEqual(meta["source"], "file")
            self.assertEqual(meta["original_width"], 640)
            self.assertEqual(meta["original_height"], 480)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        loaded_img, meta = capture.load_image_file("/path/to/nonexistent/image.png")
        self.assertIsNone(loaded_img)
        self.assertIn("error", meta)


if __name__ == "__main__":
    unittest.main()
