#!/usr/bin/env bash
# Quick UP/DOWN health check for this pipeline's runtime dependencies —
# Postgres, Redis, the FastAPI/uvicorn API, and the vLLM/Gemma4 endpoint.
# Reads DATABASE_URL/REDIS_URL/GEMMA4_ENDPOINT_URL from a .env file if one
# exists (so it checks whatever the app itself would actually connect to,
# including the docker-vs-bare-metal port differences scripts/setup_gpu_pod.sh
# writes), falling back to config.Settings' own defaults otherwise.
#
# Usage:
#   ./scripts/check_infra.sh [path-to-.env]   # default: ./.env
#   API_URL=http://localhost:8001 ./scripts/check_infra.sh   # override anything

set -uo pipefail

ENV_FILE="${1:-.env}"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

DATABASE_URL="${DATABASE_URL:-postgresql+psycopg2://postgres:postgres@localhost:5432/movie_highlights}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
API_URL="${API_URL:-http://localhost:8000}"
GEMMA4_ENDPOINT_URL="${GEMMA4_ENDPOINT_URL:-}"

parse_host_port() {
    python3 -c "
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1].replace('postgresql+psycopg2', 'postgresql'))
print(u.hostname or 'localhost', u.port or sys.argv[2])
" "$1" "$2"
}

read -r PG_HOST PG_PORT <<<"$(parse_host_port "$DATABASE_URL" 5432)"
read -r REDIS_HOST REDIS_PORT <<<"$(parse_host_port "$REDIS_URL" 6379)"

FAILED_COUNT=0

check() {
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '[UP  ] %s\n' "$label"
    else
        printf '[DOWN] %s\n' "$label"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
}

echo "================ INFRA HEALTH CHECK ================"
check "Postgres    (localhost:$PG_PORT)" pg_isready -h "$PG_HOST" -p "$PG_PORT"
check "Redis       (localhost:$REDIS_PORT)" bash -c "redis-cli -h '$REDIS_HOST' -p '$REDIS_PORT' ping | grep -qi PONG"
check "API/uvicorn ($API_URL)" curl -sf "$API_URL/docs"
if [ -n "$GEMMA4_ENDPOINT_URL" ]; then
    GEMMA4_BASE="${GEMMA4_ENDPOINT_URL%/v1/chat/completions}"
    check "vLLM/Gemma4 ($GEMMA4_BASE)" curl -sf "$GEMMA4_BASE/v1/models"
else
    echo "[SKIP] vLLM/Gemma4 — GEMMA4_ENDPOINT_URL not set in $ENV_FILE/environment"
fi
echo "======================================================"
exit "$FAILED_COUNT"
