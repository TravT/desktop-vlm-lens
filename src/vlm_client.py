"""
VLM Client and Self-Healing Auto-Spawn Manager.
Connects to a llama-server (OpenAI-compatible /v1/chat/completions) on THIS machine.

The lens is a standalone tool: by default it only ever talks to a loopback address. A remote server
must be named explicitly (VLM_SERVER_URL) and allowed explicitly (VLM_ALLOW_REMOTE=1); there is no
discovery of other machines and no fallback to one.
"""

import os
import sys
import time
import socket
import ipaddress
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_SEC = int(os.environ.get("VLM_TIMEOUT_SEC", "120"))
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8085
LOCAL_URL = f"http://{LOCAL_HOST}:{LOCAL_PORT}"
DEFAULT_MIN_IMAGE_TOKENS = 256
SPAWN_MAX_IMAGE_TOKENS = 1280   # room for a 1024 px image (about 1030 tokens); the client's max_dim sets the real cost


@dataclass
class ServerResolution:
    """Outcome of looking for a VLM server: a URL, or an error that says what was tried."""
    url: Optional[str] = None
    source: Optional[str] = None            # env | running | spawned
    error: Optional[str] = None
    error_kind: Optional[str] = None        # no_server | remote_refused
    tried: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def is_loopback_url(url: str) -> bool:
    """True when the URL points at this machine."""
    host = urllib.parse.urlparse(url if "//" in url else f"//{url}").hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def probe_health(base_url: str, timeout: float = 0.8) -> str:
    """'ready' (200), 'loading' (503 while the model loads) or 'down'. Read-only GET, nothing is started."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    except Exception:
        return "down"
    if r.status_code == 200:
        return "ready"
    if r.status_code == 503:
        return "loading"
    return "down"


def image_token_limits() -> Tuple[int, Optional[int]]:
    """(min, max) image tokens the server is assumed to run with; max None means no cap.

    The grounding parser needs the same numbers the server uses to know what size the model saw.
    """
    lo, hi = DEFAULT_MIN_IMAGE_TOKENS, None
    cfg = _user_config()
    if cfg.get("image_min_tokens"):
        lo = int(cfg["image_min_tokens"])
    if cfg.get("image_max_tokens"):
        hi = int(cfg["image_max_tokens"])
    if os.environ.get("VLM_MIN_IMAGE_TOKENS"):
        lo = int(os.environ["VLM_MIN_IMAGE_TOKENS"])
    if os.environ.get("VLM_MAX_IMAGE_TOKENS"):
        hi = int(os.environ["VLM_MAX_IMAGE_TOKENS"])
    return lo, hi


def _user_config() -> Dict[str, Any]:
    config_path = PACKAGE_ROOT / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def resolve_server() -> ServerResolution:
    """
    Finds a VLM server on this machine, in this order:
    1. VLM_SERVER_URL (must be loopback unless VLM_ALLOW_REMOTE=1)
    2. a server already listening on 127.0.0.1:8085
    3. auto-spawn of a local llama-server, when a binary and a model are present
    Otherwise returns an error that lists what was tried. Nothing else is contacted.
    """
    res = ServerResolution()
    env_url = (os.environ.get("VLM_SERVER_URL") or "").strip()

    if env_url:
        if not is_loopback_url(env_url) and os.environ.get("VLM_ALLOW_REMOTE") != "1":
            res.error_kind = "remote_refused"
            res.error = (f"VLM_SERVER_URL points to a non-local address ({env_url}). The lens is standalone and "
                         f"only talks to this machine unless VLM_ALLOW_REMOTE=1 is set.")
            return res
        state = probe_health(env_url)
        res.tried.append({"url": env_url, "reason": "VLM_SERVER_URL: " + ("up" if state != "down" else "not answering")})
        if state != "down":
            res.url, res.source = env_url, "env"
            return res
        res.error_kind = "no_server"
        res.error = f"No VLM server answered at VLM_SERVER_URL ({env_url}). Start it, or unset VLM_SERVER_URL."
        return res

    state = probe_health(LOCAL_URL, timeout=0.3)
    res.tried.append({"url": LOCAL_URL, "reason": "local server: " + ("up" if state != "down" else "not listening")})
    if state != "down":
        res.url, res.source = LOCAL_URL, "running"
        return res

    server_bin = find_local_binary()
    model_file, mmproj_file = find_local_model()
    if server_bin and model_file:
        warn = mmproj_warning(mmproj_file) if mmproj_file else None
        if warn:
            res.notes.append(warn)
        if auto_spawn_llama_server(LOCAL_HOST, LOCAL_PORT):
            res.url, res.source = LOCAL_URL, "spawned"
            return res
        res.tried.append({"url": LOCAL_URL, "reason": f"auto-spawn of {server_bin.name} did not become ready"})
    else:
        missing = "llama-server binary" if not server_bin else "model file"
        res.tried.append({"url": LOCAL_URL, "reason": f"auto-spawn skipped: no {missing} under the lens"})

    res.error_kind = "no_server"
    res.error = (f"No VLM server is available. Tried: " + "; ".join(f"{t['url']} ({t['reason']})" for t in res.tried) +
                 f". Start a llama-server with a vision model on {LOCAL_URL}, "
                 f"or put a llama-server binary and a model under the lens (python scripts/fetch_models.py).")
    return res


def resolve_best_server_url() -> str:
    """URL of the server to use, or '' when none is available (see resolve_server for the reason)."""
    return resolve_server().url or ""


def is_port_open(host: str = "127.0.0.1", port: int = 8085, timeout: float = 0.5) -> bool:
    """Fast check whether local server port is active."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def is_server_ready(host: str = "127.0.0.1", port: int = 8085, timeout: float = 0.5) -> bool:
    """Checks whether the VLM server is fully initialized and returning HTTP 200 on /health."""
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False



def find_local_binary() -> Optional[Path]:
    """Discovers llama-server binary in local bin/ directory, Ollama lib, or PATH."""
    bin_dir = PACKAGE_ROOT / "bin"
    candidates = []
    if sys.platform == "win32":
        candidates.extend([
            bin_dir / "llama-server.exe",
            bin_dir / "build" / "bin" / "llama-server.exe",
        ])
        for p in bin_dir.glob("**/llama-server.exe"):
            candidates.append(p)
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "lib" / "ollama" / "llama-server.exe")
    else:
        candidates.extend([bin_dir / "llama-server", bin_dir / "llama-server.exe"])
        for p in bin_dir.glob("**/llama-server"):
            candidates.append(p)

    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c

    # Check system PATH
    import shutil
    path_hit = shutil.which("llama-server")
    if path_hit:
        return Path(path_hit)

    return None


def find_local_model() -> tuple[Optional[Path], Optional[Path]]:
    """Finds the GGUF model and its vision projector in models/.

    Stock llama.cpp layout: the model plus a separate `mmproj-*.gguf`, which must stay F16 (a quantized
    projector breaks spatial grounding). A single GGUF with no projector file is treated as an
    Ollama-format file whose projector is embedded.
    """
    models_dir = PACKAGE_ROOT / "models"
    if not models_dir.exists():
        return None, None

    model_file = None
    mmproj_file = None

    for f in sorted(models_dir.glob("*.gguf")):
        if "mmproj" in f.name.lower():
            mmproj_file = f
        else:
            model_file = f

    # Ollama-format single-file GGUF (projector embedded in the model file)
    if model_file and not mmproj_file:
        mmproj_file = model_file

    return model_file, mmproj_file


def mmproj_warning(mmproj_file: Path) -> Optional[str]:
    """Warns when a projector file name says it is quantized (grounding needs F16)."""
    name = mmproj_file.name.lower()
    if "mmproj" in name and any(q in name for q in ("q4", "q5", "q6", "q8", "int8")):
        return f"Projector {mmproj_file.name} looks quantized; keep the projector F16 or UI grounding degrades."
    return None


def auto_spawn_llama_server(host: str = "127.0.0.1", port: int = 8085) -> bool:
    """Spawns local llama-server with full GPU offload in the background if binary and model are present."""
    server_bin = find_local_binary()
    model_file, mmproj_file = find_local_model()

    if not server_bin or not model_file or not mmproj_file:
        return False

    # Hardware-adaptive launch parameters and token budgeting (ADR-43)
    threads = 4
    threads_batch = 8
    ngl = 99
    # Same limits the grounding parser assumes (image_token_limits); no cap means room for a 1024 px image.
    min_tokens, max_cap = image_token_limits()
    max_tokens = max_cap or SPAWN_MAX_IMAGE_TOKENS
    background_priority = True

    # 1. Check user config.json first
    config_path = PACKAGE_ROOT / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if user_cfg.get("device") == "cpu":
                ngl = 0
            if user_cfg.get("cpu_threads"):
                threads = int(user_cfg["cpu_threads"])
            if user_cfg.get("background_priority") is not None:
                background_priority = bool(user_cfg["background_priority"])
        except Exception:
            pass

    # 2. Check cached hardware profile
    profile_path = PACKAGE_ROOT / ".hardware_profile.json"
    if profile_path.exists():
        try:
            import json
            with open(profile_path, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            opt = p_data.get("optimal_args", {})
            threads = opt.get("threads", threads)
            threads_batch = opt.get("threads_batch", threads_batch)
            if ngl != 0:  # Don't override explicit CPU mode ngl=0
                ngl = opt.get("ngl", ngl)
            if opt.get("background_priority") is not None:
                background_priority = bool(opt["background_priority"])
        except Exception:
            pass

    if "VLM_THREADS" in os.environ:
        try:
            threads = int(os.environ["VLM_THREADS"])
        except ValueError:
            pass
    if "VLM_NGL" in os.environ:
        try:
            ngl = int(os.environ["VLM_NGL"])
        except ValueError:
            pass

    cmd = [
        str(server_bin),
        "-m", str(model_file),
        "--mmproj", str(mmproj_file),
        "-c", "4096",
        "--image-max-tokens", str(max_tokens),
        "--image-min-tokens", str(min_tokens),
        "-ngl", str(ngl),
        "--port", str(port),
        "--host", host,
        "-t", str(threads),
        "-tb", str(threads_batch)
    ]

    try:
        working_dir = str(server_bin.parent)
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            flags = CREATE_NO_WINDOW
            if background_priority:
                flags |= BELOW_NORMAL_PRIORITY_CLASS

            subprocess.Popen(
                cmd,
                cwd=working_dir,
                creationflags=flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen(
                cmd,
                cwd=working_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

        # Wait up to 40s for server readiness
        for _ in range(80):
            time.sleep(0.5)
            if is_server_ready(host, port):
                return True
    except Exception:
        pass

    return False


def detect_model_family() -> str:
    """Detects active model family from local model filename: 'qwen', 'moondream', 'smolvlm'."""
    model_file, _ = find_local_model()
    if not model_file:
        return "qwen"
    name = model_file.name.lower()
    if "smol" in name:
        return "smolvlm"
    elif "moondream" in name:
        return "moondream"
    return "qwen"


def query_vlm(
    image_b64_uri: str,
    prompt: str,
    server_url: Optional[str] = None,
    max_tokens: int = 250,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Sends multimodal inference query to the VLM server.
    """
    family = detect_model_family()
    if server_url:
        if not is_loopback_url(server_url) and os.environ.get("VLM_ALLOW_REMOTE") != "1":
            return {
                "status": "error",
                "error_kind": "remote_refused",
                "error": f"Refusing non-local VLM server '{server_url}': the lens is standalone (set VLM_ALLOW_REMOTE=1 to allow it).",
                "duration_sec": 0.0,
                "model_family": family
            }
    else:
        resolution = resolve_server()
        if not resolution.url:
            return {
                "status": "error",
                "error_kind": resolution.error_kind or "no_server",
                "error": resolution.error,
                "tried": resolution.tried,
                "duration_sec": 0.0,
                "model_family": family
            }
        server_url = resolution.url

    # Socket open but model still loading: wait for readiness
    parsed = urllib.parse.urlparse(server_url)
    host = parsed.hostname or LOCAL_HOST
    port = parsed.port or LOCAL_PORT
    if is_loopback_url(server_url) and is_port_open(host, port) and not is_server_ready(host, port):
        for _ in range(60):
            time.sleep(0.5)
            if is_server_ready(host, port):
                break

    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "qwen2.5-vl" if family == "qwen" else family,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64_uri}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    t0 = time.time()
    try:
        retries = 30
        resp = None
        while retries > 0:
            resp = requests.post(endpoint, json=payload, timeout=DEFAULT_TIMEOUT_SEC)
            if resp.status_code == 503 and "Loading model" in resp.text:
                time.sleep(1.0)
                retries -= 1
                continue
            break
        dur = round(time.time() - t0, 2)

        if resp.status_code == 200:
            data = resp.json()
            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "status": "success",
                "content": choice.strip(),
                "duration_sec": dur,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "server": server_url,
                "model_family": family
            }
        else:
            return {
                "status": "error",
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                "duration_sec": dur,
                "model_family": family
            }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "error": f"Cannot connect to VLM server at '{server_url}'. Ensure llama-server is running on port {port}.",
            "duration_sec": round(time.time() - t0, 2),
            "model_family": family
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "duration_sec": round(time.time() - t0, 2),
            "model_family": family
        }
