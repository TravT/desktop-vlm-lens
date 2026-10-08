"""Tool-level behaviour of the MCP server with a faked VLM: grounding accuracy, retry, budgets, frames."""

import base64
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "src"))

import frames
import preprocessor as pp
import server
import vlm_client

FIXTURE = json.loads((PKG / "tests" / "fixtures" / "grounding_cases.json").read_text(encoding="utf-8"))
PAGE = FIXTURE["mock_login_page"]


def mock_page(path: Path) -> Path:
    img = Image.new("RGB", tuple(PAGE["canvas"]), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((500, 550, 780, 586), fill=(30, 100, 220))
    img.save(path)
    return path


def decode(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))


def ok(content, tokens=None):
    return {"status": "success", "content": content, "duration_sec": 0.1,
            "prompt_tokens": tokens if tokens is not None else pp.expected_prompt_tokens((1036, 644)) + 12,
            "completion_tokens": 12, "server": "http://127.0.0.1:8085", "model_family": "qwen"}


def json_block(result) -> dict:
    text = result["content"][0]["text"]
    return json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.page = str(mock_page(Path(self.tmp.name) / "page.png"))
        server.FRAMES.clear()

    def ground(self, replies, **args):
        calls = []

        def fake(uri, prompt, **kw):
            calls.append({"image": decode(uri), "prompt": prompt, **kw})
            return replies[min(len(calls) - 1, len(replies) - 1)]

        with mock.patch.object(vlm_client, "query_vlm", side_effect=fake):
            result = server.handle_tool_call("ground_ui_element", {"image_path": self.page, **args})
        return result, calls


class TestGroundingTool(ToolTestCase):

    def test_click_targets_match_the_known_elements(self):
        for el in PAGE["elements"]:
            result, _ = self.ground([ok(el["raw"])], element_description=el["name"])
            self.assertFalse(result["isError"], result)
            click = json_block(result)["primary_click"]
            self.assertLessEqual(abs(click["x"] - el["true_centre"][0]), 5, el["name"])
            self.assertLessEqual(abs(click["y"] - el["true_centre"][1]), 5, el["name"])

    def test_image_is_sent_at_1024_and_prompt_asks_for_pixel_xyxy(self):
        _, calls = self.ground([ok(PAGE["elements"][0]["raw"])], element_description="Sign in button")
        self.assertEqual(calls[0]["image"].size, (1024, 640))
        self.assertIn("[x1, y1, x2, y2]", calls[0]["prompt"])
        self.assertIn("pixel", calls[0]["prompt"])
        self.assertNotIn("normalized to 1000", calls[0]["prompt"])

    def test_prose_reply_gets_one_strict_retry(self):
        result, calls = self.ground([ok("It is below the form."), ok(PAGE["elements"][0]["raw"])],
                                    element_description="Sign in button")
        self.assertEqual(len(calls), 2)
        self.assertIn("only", calls[1]["prompt"].lower())
        self.assertIsNotNone(json_block(result)["primary_click"])

    def test_two_prose_replies_stop_after_one_retry(self):
        result, calls = self.ground([ok("I cannot tell.")], element_description="Sign in button")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(json_block(result)["primary_click"])
        self.assertIn("No bounding boxes", result["content"][0]["text"])

    def test_server_with_lower_token_cap_is_recovered_from_prompt_tokens(self):
        capped = pp.qwen_input_size(1024, 640, 256, 512)            # what a server with --image-max-tokens 512 saw
        raw = f"[{capped[0] // 2 - 10}, {capped[1] // 2 - 5}, {capped[0] // 2 + 10}, {capped[1] // 2 + 5}]"
        result, _ = self.ground([ok(raw, tokens=pp.expected_prompt_tokens(capped) + 12)], element_description="centre dot")
        data = json_block(result)
        self.assertEqual(data["size_source"], "inferred")
        self.assertLessEqual(abs(data["primary_click"]["x"] - 640), 6)
        self.assertLessEqual(abs(data["primary_click"]["y"] - 400), 6)

    def test_result_reports_model_view_and_scale(self):
        result, _ = self.ground([ok(PAGE["elements"][0]["raw"])], element_description="Sign in button")
        data = json_block(result)
        self.assertEqual(data["model_input_size"], [1036, 644])
        self.assertEqual(data["size_source"], "computed")
        self.assertAlmostEqual(data["canvas_px_per_model_px"], 1280 / 1036, places=2)

    def test_no_server_is_a_clear_error(self):
        err = {"status": "error", "error_kind": "no_server", "error": "No VLM server is available. Tried: x",
               "duration_sec": 0.0, "model_family": "qwen"}
        result, calls = self.ground([err], element_description="Sign in button")
        self.assertTrue(result["isError"])
        self.assertIn("No VLM server", result["content"][0]["text"])
        self.assertEqual(len(calls), 1)         # no pointless retry when there is no server


class TestClickSpaces(ToolTestCase):

    def test_window_capture_adds_screen_coordinates(self):
        img = Image.open(self.page)
        meta = {"source": "active_window", "left": 300, "top": 120,
                "original_width": img.width, "original_height": img.height}
        with mock.patch.object(server.capture, "capture_active_window", return_value=(img, meta)), \
                mock.patch.object(vlm_client, "query_vlm", return_value=ok(PAGE["elements"][0]["raw"])):
            result = server.handle_tool_call("ground_ui_element", {"element_description": "Sign in button"})
        data = json_block(result)
        self.assertEqual(data["click_space"], "window")
        self.assertEqual(data["primary_click_screen"]["x"], data["primary_click"]["x"] + 300)
        self.assertEqual(data["primary_click_screen"]["y"], data["primary_click"]["y"] + 120)

    def test_file_has_no_screen_coordinates(self):
        result, _ = self.ground([ok(PAGE["elements"][0]["raw"])], element_description="Sign in button")
        data = json_block(result)
        self.assertEqual(data["click_space"], "image")
        self.assertIsNone(data.get("primary_click_screen"))


class TestBudgets(ToolTestCase):

    def inspect_size(self, tool, **args):
        sizes = []

        def fake(uri, prompt, **kw):
            sizes.append(decode(uri).size)
            return ok("ok")

        with mock.patch.object(vlm_client, "query_vlm", side_effect=fake):
            server.handle_tool_call(tool, {"image_path": self.page, **args})
        return sizes[0]

    def test_detail_levels_map_to_pixel_budgets(self):
        self.assertEqual(self.inspect_size("inspect_image_file", prompt="p", detail="scene")[0], 512)
        self.assertEqual(self.inspect_size("inspect_image_file", prompt="p", detail="read")[0], 768)
        self.assertEqual(self.inspect_size("inspect_image_file", prompt="p", detail="precise")[0], 1024)

    def test_default_budgets_per_tool(self):
        self.assertEqual(server.DEFAULT_DETAIL["ground_ui_element"], "precise")
        self.assertEqual(self.inspect_size("transcribe_screen_text")[0], 768)
        self.assertEqual(self.inspect_size("inspect_image_file", prompt="p")[0], 1024)

    def test_unknown_detail_is_an_error(self):
        with mock.patch.object(vlm_client, "query_vlm", return_value=ok("ok")):
            result = server.handle_tool_call("inspect_image_file", {"image_path": self.page, "prompt": "p", "detail": "bogus"})
        self.assertTrue(result["isError"])

    def test_env_overrides_a_budget(self):
        with mock.patch.dict("os.environ", {"VLM_READ_PX": "640"}):
            self.assertEqual(self.inspect_size("inspect_image_file", prompt="p", detail="read")[0], 640)


class TestFrameReuse(unittest.TestCase):

    def setUp(self):
        server.FRAMES.clear()
        self.img = Image.new("RGB", (1280, 800), "white")
        self.meta = {"source": "fullscreen", "original_width": 1280, "original_height": 800}

    def run_tool(self, args, cap):
        with mock.patch.object(server.capture, "capture_fullscreen", cap), \
                mock.patch.object(vlm_client, "query_vlm", return_value=ok("a login form")):
            return server.handle_tool_call("capture_and_inspect", {"prompt": "what is this?", "target": "fullscreen", **args})

    def test_capture_returns_a_frame_id_that_skips_recapture(self):
        cap = mock.Mock(return_value=(self.img, self.meta))
        first = self.run_tool({}, cap)
        fid = re.search(r"frame_id\*\*: `([0-9a-f]+)`", first["content"][0]["text"]).group(1)
        second = self.run_tool({"frame_id": fid}, cap)
        self.assertFalse(second["isError"])
        self.assertEqual(cap.call_count, 1)

    def test_same_frame_gives_identical_image_bytes(self):
        uris = []
        cap = mock.Mock(return_value=(self.img, self.meta))
        with mock.patch.object(server.capture, "capture_fullscreen", cap), \
                mock.patch.object(vlm_client, "query_vlm", side_effect=lambda uri, p, **k: uris.append(uri) or ok("x")):
            server.handle_tool_call("capture_and_inspect", {"prompt": "a", "target": "fullscreen"})
            fid = next(iter(server.FRAMES._frames))
            server.handle_tool_call("capture_and_inspect", {"prompt": "b", "frame_id": fid})
        self.assertEqual(uris[0], uris[1])      # byte-identical, so the server's prompt cache can hit

    def test_unknown_frame_id_is_a_clear_error(self):
        result = self.run_tool({"frame_id": "deadbeef"}, mock.Mock(return_value=(self.img, self.meta)))
        self.assertTrue(result["isError"])
        self.assertIn("frame_id", result["content"][0]["text"])


class TestFrameStore(unittest.TestCase):

    def test_lru_eviction(self):
        fs = frames.FrameStore(max_frames=2)
        a, b, c = (fs.put(i, {}) for i in range(3))
        self.assertIsNone(fs.get(a))
        self.assertIsNotNone(fs.get(b))
        self.assertIsNotNone(fs.get(c))

    def test_ttl_expiry(self):
        now = [0.0]
        fs = frames.FrameStore(ttl_sec=10, clock=lambda: now[0])
        fid = fs.put("img", {})
        now[0] = 5
        self.assertIsNotNone(fs.get(fid))
        now[0] = 20
        self.assertIsNone(fs.get(fid))
        self.assertEqual(len(fs), 0)

    def test_nothing_is_written_to_disk(self):
        src = (PKG / "src" / "frames.py").read_text(encoding="utf-8")
        for word in ("open(", ".save(", "tempfile", "write_"):
            self.assertNotIn(word, src)


class TestProcessLifecycle(unittest.TestCase):

    def test_server_exits_when_stdin_closes(self):
        p = subprocess.Popen([sys.executable, str(PKG / "src" / "server.py")],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n")
        p.stdin.flush()
        self.assertIn("desktop-vlm-lens", p.stdout.readline())
        p.stdin.close()
        try:
            self.assertEqual(p.wait(timeout=5), 0)
        finally:
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    unittest.main()
