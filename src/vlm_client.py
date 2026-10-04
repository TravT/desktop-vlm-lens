"""
VLM Client and Self-Healing Auto-Spawn Manager.
Connects to native llama-server (OpenAI-compatible /v1/chat/completions).
Transparently launches local llama-server binary on Windows or Linux if port is offline.
"""

import os
import sys
import time
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import requests

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_SEC = int(os.environ.get("VLM_TIMEOUT_SEC", "120"))


def resolve_best_server_url() -> str:
    """
    Auto-discovers the best available VLM endpoint:
    1. VLM_SERVER_URL env var if explicitly configured
    2. Local port 8085 if already listening
    3. If local model and binary exist, auto-spawn on 127.0.0.1:8085 (SPLIT-SECOND GPU INFERENCE)
    4. Fall back to homelab vlm-native.home.arpa only if no local model exists
    """
    env_url = os.environ.get("VLM_SERVER_URL")
    if env_url:
        return env_url

    if is_server_ready("127.0.0.1", 8085, timeout=0.3):
        return "http://127.0.0.1:8085"

    # Prioritize local auto-spawn with GPU offload before remote fallback
    server_bin = find_local_binary()
    model_file, _ = find_local_model()
    if server_bin and model_file:
        if auto_spawn_llama_server("127.0.0.1", 8085):
            return "http://127.0.0.1:8085"

    try:
        r = requests.get("http://vlm-native.home.arpa/health", timeout=1.0)
        if r.status_code == 200:
            return "http://vlm-native.home.arpa"
    except Exception:
        pass

    return "http://127.0.0.1:8085"


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
    """Discovers GGUF model and vision projector mmproj in models/ directory."""
    models_dir = PACKAGE_ROOT / "models"
    if not models_dir.exists():
        return None, None

    model_file = None
    mmproj_file = None

    for f in models_dir.glob("*.gguf"):
        if "mmproj" in f.name.lower():
            mmproj_file = f
        else:
            model_file = f

    # In unified Ollama-format GGUFs (e.g. Qwen2.5-VL), mmproj is embedded in the model GGUF itself
    if model_file and not mmproj_file:
        mmproj_file = model_file

    return model_file, mmproj_file


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
    max_tokens = int(os.environ.get("VLM_MAX_IMAGE_TOKENS", "512"))
    min_tokens = int(os.environ.get("VLM_MIN_IMAGE_TOKENS", "256"))
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
            if user_cfg.get("image_max_tokens"):
                max_tokens = int(user_cfg["image_max_tokens"])
            if user_cfg.get("image_min_tokens"):
                min_tokens = int(user_cfg["image_min_tokens"])
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
    if not server_url:
        server_url = resolve_best_server_url()

    # Parse host & port for liveness check
    import urllib.parse
    parsed = urllib.parse.urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8085

    # Auto-spawn if localhost and currently offline or loading
    if host in ("127.0.0.1", "localhost") and not is_server_ready(host, port):
        if not is_port_open(host, port):
            auto_spawn_llama_server(host, port)
        else:
            # Socket open but model loading, wait for readiness
            for _ in range(60):
                time.sleep(0.5)
                if is_server_ready(host, port):
                    break

    family = detect_model_family()
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
