# Stage 4 (LLM-as-judge scorer) worker image.
#
# CPU-only. Step 1 wires in MockScorer (workers/scorer/adapters.py), which has
# no GPU/model dependency at all — it's pure Python hashing.
#
# Step 2's Gemma4Scorer is a plain httpx POST to an already-deployed Gemma 4
# inference endpoint (network call), not a local model load. So this image
# likely stays CPU-only even after Step 2 flips `get_scorer()` over to
# Gemma4Scorer — do NOT add the `gpu` extra here unless a future step moves
# scoring to a locally-hosted model.
#
# ffmpeg: Gemma4Scorer's multimodal path crops/saves each span's actual
# video+audio clip via workers/segment_worker/adapters/_media.py's shared
# ffmpeg helpers (this image COPYs the whole workers/ tree, so that module is
# available here too) — needed to attach real keyframe images/audio to the
# scoring call, not just the aggregate feature_vector.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY config/ ./config/
COPY models/ ./models/
COPY storage/ ./storage/
COPY orchestrator/ ./orchestrator/
COPY workers/ ./workers/

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "orchestrator.celery_app", "worker", "--loglevel=info", "-Q", "scorer"]
