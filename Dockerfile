# EWAT — Docker image for reproducible runs
#
# Step 10 fix 10.1 (audit 2026-05-26): the project previously had no
# containerisation, making system-level dependency versioning (CUDA, libgomp,
# system Python) untraceable. This Dockerfile produces a deterministic build
# environment that mirrors the development setup.
#
# Usage
# -----
# Build:
#     docker build -t ewat:latest .
#
# Run unit tests:
#     docker run --rm -v $(pwd):/workspace ewat:latest \
#         pytest tests/unit/ -q
#
# Run pipeline (mount data + outputs):
#     docker run --rm -v $(pwd):/workspace -w /workspace ewat:latest \
#         python -m experiments.precursor.train ...
#
# Tip: pin the image digest in CI for bit-reproducible builds.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# System libs required by PyTorch CPU, scientific stack, OWL reasoner, etc.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        git \
        libgomp1 \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (cache-friendly)
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Copy the source (kept after deps so code changes don't bust the layer cache)
COPY src /workspace/src
COPY scripts /workspace/scripts
COPY experiments /workspace/experiments
COPY tests /workspace/tests
COPY configs /workspace/configs

# Default: run unit tests as a smoke check (override with `docker run … <cmd>`).
CMD ["pytest", "tests/unit/", "-q"]
