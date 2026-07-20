# Stage 6 (ranker) worker — CPU-only, pure-Python + Postgres.
# No GPU deps here (torch/faster-whisper/pyannote live under the pyproject.toml
# [project.optional-dependencies] "gpu" extra and are never installed for this
# worker — see CLAUDE.md Step 1 constraint).

FROM python:3.11-slim

WORKDIR /app

# libpq5 covers psycopg2's runtime shared-lib needs even though
# psycopg2-binary vendors its own libpq (belt-and-braces for slim base images).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Monorepo-style build context: this worker only runs code from
# config/, models/, storage/, orchestrator/, workers/, but pyproject.toml's
# [tool.setuptools] packages list spans the whole repo, so the full context
# is copied in.
COPY . .

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "orchestrator.celery_app", "worker", "-Q", "ranker", "--loglevel=info"]
