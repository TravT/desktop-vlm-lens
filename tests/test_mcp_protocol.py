import subprocess
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
python_exe = sys.executable

p = subprocess.Popen(
    [python_exe, str(PACKAGE_ROOT / "src" / "server.py")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# 1. Initialize
init_msg = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client"}
    }
})
p.stdin.write(init_msg + "\n")
p.stdin.flush()
line1 = p.stdout.readline()
resp1 = json.loads(line1)
server_info = resp1.get("result", {}).get("serverInfo", {})
print(f"[OK] MCP Server Initialized: {server_info.get('name')} v{server_info.get('version')}")

# 2. Tools list
tools_msg = json.dumps({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
})
p.stdin.write(tools_msg + "\n")
p.stdin.flush()
line2 = p.stdout.readline()
resp2 = json.loads(line2)
tool_list = resp2.get("result", {}).get("tools", [])
print(f"[OK] Available MCP Tools ({len(tool_list)}):")
for t in tool_list:
    has_crop = "crop_bbox" in t.get("parameters", {}).get("properties", {})
    print(f"  • {t['name']}: {t.get('description')[:60]}... (RoI Crop: {'YES' if has_crop else 'NO'})")

p.kill()
