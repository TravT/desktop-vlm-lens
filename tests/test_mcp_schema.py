"""Tool definitions must follow the MCP spec (inputSchema), or clients such as Claude Code drop the whole tool list."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent


def list_tools():
    msgs = "\n".join(json.dumps(m) for m in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))
    out = subprocess.run([sys.executable, str(PKG / "src" / "server.py")], input=msgs + "\n",
                         capture_output=True, text=True, timeout=20).stdout
    return next(json.loads(l)["result"]["tools"] for l in out.splitlines() if json.loads(l).get("id") == 2)


class TestToolSchemas(unittest.TestCase):

    def test_every_tool_has_an_object_input_schema(self):
        tools = list_tools()
        self.assertEqual(len(tools), 4)
        for t in tools:
            self.assertIn("inputSchema", t, t["name"])
            self.assertEqual(t["inputSchema"]["type"], "object", t["name"])
            self.assertIn("properties", t["inputSchema"], t["name"])

    def test_required_fields_exist_in_properties(self):
        for t in list_tools():
            schema = t["inputSchema"]
            for field in schema.get("required", []):
                self.assertIn(field, schema["properties"], (t["name"], field))

    def test_valid_for_the_official_sdk_when_installed(self):
        try:
            from mcp.types import Tool
        except ImportError:
            self.skipTest("mcp SDK not installed")
        for t in list_tools():
            Tool.model_validate({k: v for k, v in t.items() if k != "parameters"})


if __name__ == "__main__":
    unittest.main()
