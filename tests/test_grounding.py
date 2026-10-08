"""Qwen2.5-VL answers in absolute pixels [x1, y1, x2, y2] of the 28-multiple image it saw.

Regression for the bug found 2026-10-08: the parser read those numbers as normalized 0-1000
[ymin, xmin, ymax, xmax], so click targets were 208-395 px off on a mock login page.
"""

import json
import sys
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "src"))

import preprocessor as pp

FIXTURE = json.loads((PKG / "tests" / "fixtures" / "grounding_cases.json").read_text(encoding="utf-8"))


class TestQwenInputSize(unittest.TestCase):

    def test_measured_sizes_and_token_counts(self):
        for case in FIXTURE["input_sizes"]:
            w, h = case["sent"]
            got = pp.qwen_input_size(w, h, min_tokens=256)
            self.assertEqual(list(got), case["model_size"], case["sent"])
            self.assertEqual(pp.expected_prompt_tokens(got), case["prompt_tokens"], case["sent"])

    def test_sizes_are_multiples_of_28_and_respect_the_floor(self):
        for w, h in [(100, 100), (1280, 800), (800, 1280), (4032, 3024), (64, 900)]:
            ow, oh = pp.qwen_input_size(w, h, min_tokens=256)
            self.assertEqual((ow % 28, oh % 28), (0, 0))
            self.assertGreaterEqual((ow // 28) * (oh // 28), 256)

    def test_max_tokens_caps_the_grid(self):
        ow, oh = pp.qwen_input_size(1024, 640, min_tokens=256, max_tokens=512)
        self.assertLessEqual((ow // 28) * (oh // 28), 512)
        self.assertGreater((ow // 28) * (oh // 28), 400)

    def test_mock_page_is_seen_at_1036_by_644(self):
        self.assertEqual(pp.qwen_input_size(1024, 640, min_tokens=256), (1036, 644))


class TestModelSizeResolution(unittest.TestCase):

    def test_matching_token_count_keeps_the_computed_size(self):
        size, source = pp.resolve_model_size(1024, 640, prompt_tokens=851 + 31 + 12, min_tokens=256)
        self.assertEqual((size, source), ((1036, 644), "computed"))

    def test_no_token_count_means_computed(self):
        self.assertEqual(pp.resolve_model_size(1024, 640, prompt_tokens=None, min_tokens=256)[1], "computed")

    def test_server_with_a_lower_image_cap_is_detected_from_prompt_tokens(self):
        truth = pp.qwen_input_size(1024, 640, min_tokens=256, max_tokens=512)
        tokens = pp.expected_prompt_tokens(truth) + 10          # a short prompt on top of the template
        size, source = pp.resolve_model_size(1024, 640, prompt_tokens=tokens, min_tokens=256)
        self.assertEqual(source, "inferred")
        self.assertEqual(size, truth)


class TestAbsoluteGrounding(unittest.TestCase):
    PAGE = FIXTURE["mock_login_page"]

    def parse(self, raw, crop_info=None, canvas=None):
        size = pp.qwen_input_size(*self.PAGE["sent"], min_tokens=256)
        cw, ch = canvas or self.PAGE["canvas"]
        return pp.parse_grounding_coordinates(raw, cw, ch, crop_info, model_size=size)

    def test_click_targets_match_the_known_elements(self):
        for el in self.PAGE["elements"]:
            (m,) = self.parse(el["raw"])
            tx, ty = el["true_centre"]
            self.assertLessEqual(abs(m["click_x"] - tx), 5, el["name"])
            self.assertLessEqual(abs(m["click_y"] - ty), 5, el["name"])

    def test_xyxy_fields_are_consistent(self):
        (m,) = self.parse(self.PAGE["elements"][0]["raw"])
        x1, y1, x2, y2 = m["box_xyxy_pixels"]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertEqual((m["click_x"], m["click_y"]), (round((x1 + x2) / 2), round((y1 + y2) / 2)))
        self.assertEqual(m["pixel_box"], [y1, x1, y2, x2])     # y-first field keeps its meaning

    def test_reversed_corners_are_normalized(self):
        (m,) = self.parse("[654, 478, 382, 437]")
        x1, y1, x2, y2 = m["box_xyxy_pixels"]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)

    def test_json_bbox_2d_is_read_as_xyxy(self):
        raw = '```json\n[{"bbox_2d": [382, 437, 654, 478], "label": "button"}]\n```'
        (m,) = self.parse(raw)
        self.assertLessEqual(abs(m["click_x"] - 640), 5)

    def test_values_outside_the_model_image_are_clamped(self):
        (m,) = self.parse("[0, 0, 5000, 5000]")
        x1, y1, x2, y2 = m["box_xyxy_pixels"]
        self.assertEqual((x1, y1), (0, 0))
        self.assertLessEqual(x2, self.PAGE["canvas"][0])
        self.assertLessEqual(y2, self.PAGE["canvas"][1])

    def test_prose_answer_gives_no_boxes(self):
        self.assertEqual(self.parse("The Sign in button is below the password field."), [])

    def test_roi_crop_is_remapped_to_the_full_canvas(self):
        # crop of the canvas starting at (400, 300), 480x300; the image sent is the crop itself (no downscale)
        crop = {"applied": True, "crop_width": 480, "crop_height": 300, "x_offset": 400, "y_offset": 300}
        size = pp.qwen_input_size(480, 300, min_tokens=256)
        # model says the element spans the middle fifth of the crop it saw
        mw, mh = size
        raw = f"[{int(mw * .4)}, {int(mh * .4)}, {int(mw * .6)}, {int(mh * .6)}]"
        (m,) = pp.parse_grounding_coordinates(raw, 1280, 800, crop, model_size=size)
        self.assertTrue(m["remapped_from_crop"])
        self.assertAlmostEqual(m["click_x"], 400 + 480 * .5, delta=4)
        self.assertAlmostEqual(m["click_y"], 300 + 300 * .5, delta=4)


class TestLegacyNormalizedMode(unittest.TestCase):
    """Other model families (moondream) answer 0-1000 [ymin, xmin, ymax, xmax]; no model_size means legacy."""

    def test_normalized_box_still_works(self):
        (c,) = pp.parse_grounding_coordinates("at [700, 400, 750, 600].", 1920, 1080)
        self.assertEqual(c["normalized_box"], [700, 400, 750, 600])
        self.assertEqual(c["pixel_box"], [756, 768, 810, 1152])
        self.assertEqual((c["click_x"], c["click_y"]), (960, 783))
        self.assertEqual(c["box_xyxy_pixels"], [768, 756, 1152, 810])


if __name__ == "__main__":
    unittest.main()
