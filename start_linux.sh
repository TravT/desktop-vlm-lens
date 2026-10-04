#!/usr/bin/env bash
# ==============================================================================
# Desktop VLM Lens - Linux 1-Click Launcher
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
elif [ -f "../../.venv/bin/activate" ]; then
    source "../../.venv/bin/activate"
fi

exec python3 src/server.py
