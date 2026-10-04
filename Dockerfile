# ==============================================================================
# Desktop VLM Lens: Portable Perception MCP Server Container
# Standalone, zero-cloud visual perception oracle for text-only LLMs
# ==============================================================================
FROM python:3.12.8-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python3", "src/server.py"]
