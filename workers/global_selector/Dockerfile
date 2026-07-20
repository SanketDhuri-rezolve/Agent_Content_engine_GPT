# Stage 3.5 (global moment selection) worker image — opt-in via
# config.Settings.use_global_memory_pipeline, see workers/global_selector/tasks.py.
#
# CPU-only, same reasoning as workers/scorer/Dockerfile: both
# Gemma4GlobalSelector (text-only httpx POST) and the concurrent
# Gemma4Scorer calls this task makes on the selected winners are network
# calls to an already-deployed endpoint, not a local model load. ffmpeg is
# needed for the same reason as the scorer image — this task also invokes
# workers.scorer.adapters.Gemma4Scorer (via score_candidates_concurrently)
# on the selected winners, which crops/saves real clips.

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

CMD ["celery", "-A", "orchestrator.celery_app", "worker", "--loglevel=info", "-Q", "global_selector"]
