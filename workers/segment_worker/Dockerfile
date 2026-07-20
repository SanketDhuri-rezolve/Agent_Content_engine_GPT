# Stage 2 (segment_worker) Celery worker image.
#
# Step 1: plain Python 3.11-slim, NO CUDA base image and NO GPU deps
# installed. All model calls in this stage run through mock adapters (see
# workers/segment_worker/adapters/) by default. See Dockerfile.gpu in this
# same directory for the Step 2 GPU variant (CUDA base image + the project's
# `gpu` extra) — do not add GPU deps to this file.
#
# Build context must be the repo root (this stage imports models/, storage/,
# config/, orchestrator/ alongside workers/), e.g.:
#   docker build -f workers/segment_worker/Dockerfile -t segment-worker .

FROM python:3.11-slim

WORKDIR /app

# Minimal build deps only — no CUDA/cuDNN, no GPU userspace libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY config ./config
COPY models ./models
COPY storage ./storage
COPY orchestrator ./orchestrator
COPY workers ./workers

# Installs only the project's default (non-GPU) dependency set. The `gpu`
# extra is intentionally NOT installed here (see header note above).
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "orchestrator.celery_app", "worker", "-Q", "segment_worker", "--loglevel=info"]
