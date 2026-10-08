"""Backend resolution: standalone, loopback-only by default, clear errors when nothing is available."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "src"))

import vlm_client as vc


class FakeResp:
    def __init__(self, status=200, text=""):
        self.status_code = status
        self.text = text


def fake_get(ready_urls=(), loading_urls=()):
    """requests.get stand-in: records every URL it is asked for."""
    seen = []

    def _get(url, timeout=None, **_):
        seen.append(url)
        base = url[: -len("/health")] if url.endswith("/health") else url
        if base in ready_urls:
            return FakeResp(200)
        if base in loading_urls:
            return FakeResp(503, "Loading model")
        raise vc.requests.exceptions.ConnectionError("down")

    _get.seen = seen
    return _get


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLM_")}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class TestLoopbackGuard(unittest.TestCase):

    def test_is_loopback_url(self):
        for url in ("http://127.0.0.1:8085", "http://localhost:8085", "http://[::1]:8085", "http://127.1.2.3"):
            self.assertTrue(vc.is_loopback_url(url), url)
        for url in ("http://203.0.113.7:8090", "http://198.51.100.20", "http://vlm.intranet.example", "https://example.com"):
            self.assertFalse(vc.is_loopback_url(url), url)

    def test_remote_url_is_refused_without_opt_in_and_never_contacted(self):
        get = fake_get(ready_urls=("http://203.0.113.7:8090",))
        with clean_env(VLM_SERVER_URL="http://203.0.113.7:8090"), mock.patch.object(vc.requests, "get", get):
            res = vc.resolve_server()
        self.assertIsNone(res.url)
        self.assertIn("VLM_ALLOW_REMOTE", res.error)
        self.assertEqual(get.seen, [])

    def test_remote_url_is_used_with_opt_in(self):
        get = fake_get(ready_urls=("http://203.0.113.7:8090",))
        with clean_env(VLM_SERVER_URL="http://203.0.113.7:8090", VLM_ALLOW_REMOTE="1"), \
                mock.patch.object(vc.requests, "get", get):
            res = vc.resolve_server()
        self.assertEqual((res.url, res.source), ("http://203.0.113.7:8090", "env"))


class TestResolution(unittest.TestCase):

    def test_running_local_server_is_used(self):
        get = fake_get(ready_urls=("http://127.0.0.1:8085",))
        with clean_env(), mock.patch.object(vc.requests, "get", get):
            res = vc.resolve_server()
        self.assertEqual((res.url, res.source, res.error), ("http://127.0.0.1:8085", "running", None))

    def test_loading_server_counts_as_present(self):
        get = fake_get(loading_urls=("http://127.0.0.1:8085",))
        with clean_env(), mock.patch.object(vc.requests, "get", get):
            self.assertEqual(vc.resolve_server().url, "http://127.0.0.1:8085")

    def test_explicit_dead_loopback_url_is_an_error_that_names_it(self):
        get = fake_get()
        with clean_env(VLM_SERVER_URL="http://127.0.0.1:9999"), mock.patch.object(vc.requests, "get", get):
            res = vc.resolve_server()
        self.assertIsNone(res.url)
        self.assertIn("127.0.0.1:9999", res.error)

    def test_nothing_available_is_a_structured_error_and_stays_on_loopback(self):
        get = fake_get()
        with clean_env(), mock.patch.object(vc.requests, "get", get), \
                mock.patch.object(vc, "find_local_binary", return_value=None):
            res = vc.resolve_server()
        self.assertIsNone(res.url)
        self.assertIn("No VLM server", res.error)
        self.assertIn("127.0.0.1:8085", res.error)
        self.assertTrue(res.tried)
        self.assertTrue(all(vc.is_loopback_url(u) for u in get.seen), get.seen)

    def test_local_binary_and_model_trigger_auto_spawn(self):
        get = fake_get()
        with clean_env(), mock.patch.object(vc.requests, "get", get), \
                mock.patch.object(vc, "find_local_binary", return_value=Path("/x/llama-server")), \
                mock.patch.object(vc, "find_local_model", return_value=(Path("/m/a.gguf"), Path("/m/mmproj-a-f16.gguf"))), \
                mock.patch.object(vc, "auto_spawn_llama_server", return_value=True) as spawn:
            res = vc.resolve_server()
        spawn.assert_called_once()
        self.assertEqual((res.url, res.source), ("http://127.0.0.1:8085", "spawned"))

    def test_shipped_source_has_no_homelab_or_private_addresses(self):
        """Air-gap invariant (ADR-41): no homelab hostname, tailnet or LAN address in anything that ships."""
        import re
        bad = re.compile(r"home\.arpa|\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+|\b192\.168\.\d+\.\d+|\b10\.\d+\.\d+\.\d+")
        for path in list((PKG / "src").glob("*.py")) + list((PKG / "scripts").glob("*.py")):
            self.assertIsNone(bad.search(path.read_text(encoding="utf-8")), path.name)

    def test_query_vlm_reports_no_server_without_posting(self):
        with clean_env(), mock.patch.object(vc.requests, "get", fake_get()), \
                mock.patch.object(vc, "find_local_binary", return_value=None), \
                mock.patch.object(vc.requests, "post") as post:
            res = vc.query_vlm("data:image/jpeg;base64,AAAA", "hi")
        post.assert_not_called()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_kind"], "no_server")
        self.assertIn("No VLM server", res["error"])

    def test_query_vlm_refuses_a_remote_server_url_argument(self):
        with clean_env(), mock.patch.object(vc.requests, "post") as post:
            res = vc.query_vlm("data:image/jpeg;base64,AAAA", "hi", server_url="http://198.51.100.20:8085")
        post.assert_not_called()
        self.assertEqual(res["error_kind"], "remote_refused")


class TestModelDiscovery(unittest.TestCase):

    def find(self, names):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        models = Path(tmp.name) / "models"
        models.mkdir()
        for n in names:
            (models / n).write_bytes(b"x")
        with mock.patch.object(vc, "PACKAGE_ROOT", Path(tmp.name)):
            return vc.find_local_model()

    def test_stock_layout_has_a_separate_projector(self):
        model, proj = self.find(["Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"])
        self.assertEqual(model.name, "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")
        self.assertEqual(proj.name, "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf")

    def test_single_file_gguf_falls_back_to_embedded_projector(self):
        model, proj = self.find(["qwen2.5vl-3b.gguf"])
        self.assertEqual(model, proj)

    def test_projector_is_never_picked_as_the_model(self):
        model, proj = self.find(["mmproj-only-f16.gguf"])
        self.assertIsNone(model)

    def test_quantized_projector_is_flagged(self):
        self.assertIn("F16", vc.mmproj_warning(Path("mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")))
        self.assertIsNone(vc.mmproj_warning(Path("mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf")))
        self.assertIsNone(vc.mmproj_warning(Path("mmproj-Qwen2.5-VL-3B-Instruct.gguf")))   # fetch_models.py name


class TestTokenLimits(unittest.TestCase):

    def test_defaults_and_env(self):
        with clean_env():
            self.assertEqual(vc.image_token_limits(), (256, None))
        with clean_env(VLM_MIN_IMAGE_TOKENS="300", VLM_MAX_IMAGE_TOKENS="700"):
            self.assertEqual(vc.image_token_limits(), (300, 700))


if __name__ == "__main__":
    unittest.main()
