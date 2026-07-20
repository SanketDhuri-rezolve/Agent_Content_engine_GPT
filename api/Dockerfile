FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Project metadata + source. Copied explicitly (rather than a blanket `COPY . .`)
# so the image only ever contains the non-GPU API surface — no ai_layer/worker
# GPU extras get pulled in here (see pyproject.toml [project.optional-dependencies] "gpu").
COPY pyproject.toml ./
COPY api ./api
COPY orchestrator ./orchestrator
COPY workers ./workers
COPY models ./models
COPY storage ./storage
COPY config ./config

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
