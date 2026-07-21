#!/usr/bin/env bash
# One-shot, best-effort GPU pod bootstrap: vLLM + Gemma4 serving, this
# pipeline's real (Step 2) adapters, and local infra — see
# CLAUDE.md's "Deploying on RunPod" section for the operational gotchas this
# encodes (HF_HUB_DISABLE_XET, flash-attn --no-build-isolation, PEP 668, etc).
#
# Every step is independent: a failure is logged and the script moves on
# instead of aborting, so one bad step never blocks unrelated ones. A
# pass/fail/skip summary prints at the end; per-step logs are kept for
# whatever failed.
#
# Usage:
#   HF_TOKEN=hf_your_token_here ./scripts/setup_gpu_pod.sh
#
# HF_TOKEN is read from the environment ONLY — never hardcode a real token
# in this file or any committed script. Get one at
# https://huggingface.co/settings/tokens. If a real token was ever pasted
# into a terminal/chat you don't fully trust, rotate it there.
#
# Override any of these via env vars before running if your layout differs:
#   WORKSPACE_DIR, PIPELINE_DIR, VLLM_VENV_DIR, GEMMA4_MODEL, GEMMA4_PORT,
#   PIP_INSTALL_EXTRAS (default "gpu" — set to "" to match the plain
#   `pip install -e .` reference, but then USE_REAL_TRANSCRIBER/etc. below
#   won't actually have their real adapters installed).

set -uo pipefail  # deliberately NOT -e — every step must survive a failure

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PIPELINE_DIR="${PIPELINE_DIR:-$WORKSPACE_DIR/movie-highlight-pipeline}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-$WORKSPACE_DIR/vllm-venv}"
GEMMA4_MODEL="${GEMMA4_MODEL:-google/gemma-4-12B-it-qat-w4a16-ct}"
GEMMA4_PORT="${GEMMA4_PORT:-8002}"
PIP_INSTALL_EXTRAS="${PIP_INSTALL_EXTRAS-gpu}"
LOG_DIR="$(mktemp -d /tmp/pipeline_setup.XXXXXX)"

# CLAUDE.md: hf-xet hung indefinitely (0 bytes, no error, no timeout) on a
# real RunPod pod pulling faster-whisper's weights — disable it globally
# before anything downloads from the HF Hub.
export HF_HUB_DISABLE_XET=1

STEP_NAMES=()
STEP_STATUSES=()

log_step_result() { STEP_NAMES+=("$1"); STEP_STATUSES+=("$2"); }

run_step() {
    local name="$1"; shift
    local slug; slug="$(echo "$name" | tr -c 'A-Za-z0-9' '_')"
    local log_file="$LOG_DIR/${slug}.log"
    echo "==> $name"
    if bash -c "$*" >"$log_file" 2>&1; then
        log_step_result "$name" "OK"
    else
        local code=$?
        log_step_result "$name" "FAILED (exit $code) — log: $log_file"
        echo "    FAILED (exit $code) — last lines of log:"
        tail -n 8 "$log_file" | sed 's/^/    | /'
    fi
}

skip_step() {
    log_step_result "$1" "SKIPPED ($2)"
    echo "==> $1 — SKIPPED ($2)"
}

mkdir -p "$WORKSPACE_DIR"

# 0. GPU/driver check
run_step "0. nvidia-smi (GPU/driver check)" "nvidia-smi"

# 1. Clean stale caches
run_step "1. clean pip/huggingface caches" \
    "rm -rf '$WORKSPACE_DIR/.cache/pip' '$WORKSPACE_DIR/.cache/huggingface'"

# 2. Hugging Face auth
if [ -n "${HF_TOKEN:-}" ]; then
    run_step "2. huggingface_hub install + auth login" \
        "(pip install -q -U huggingface_hub --break-system-packages 2>/dev/null || pip install -q -U huggingface_hub) && hf auth login --token '$HF_TOKEN'"
else
    skip_step "2. huggingface_hub install + auth login" "HF_TOKEN not set — export HF_TOKEN=hf_... before running"
fi

# 3. vLLM venv + install
run_step "3a. create vLLM venv" "python3 -m venv '$VLLM_VENV_DIR'"
export PATH="$VLLM_VENV_DIR/bin:$PATH"
run_step "3b. pip install -U vllm" "'$VLLM_VENV_DIR/bin/pip' install -q -U vllm"

# 4. Pin torch/torchvision to cu128
run_step "4a. pin torch==2.11.0+cu128" \
    "'$VLLM_VENV_DIR/bin/pip' install -q --no-deps torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
run_step "4b. pin torchvision==0.26.0+cu128" \
    "'$VLLM_VENV_DIR/bin/pip' install -q --no-deps torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128"

# 5. Audio support
run_step "5. pip install librosa" "'$VLLM_VENV_DIR/bin/pip' install -q librosa"

# 6. Sanity check
run_step "6. torch/CUDA sanity check" \
    "'$VLLM_VENV_DIR/bin/python' -c \"import torch; assert torch.cuda.is_available(); print('torch OK:', torch.__version__, torch.version.cuda)\""

# 7. Launch Gemma4 via vLLM (background — model load takes minutes, so this
# step only confirms the process didn't die immediately, not that it's
# actually serving yet).
echo "==> 7. launch vLLM Gemma4 server (background)"
nohup "$VLLM_VENV_DIR/bin/vllm" serve "$GEMMA4_MODEL" \
    --host 0.0.0.0 --port "$GEMMA4_PORT" \
    --max-model-len 8192 --gpu-memory-utilization 0.6 \
    >/tmp/vllm.log 2>&1 &
VLLM_PID=$!
disown "$VLLM_PID" 2>/dev/null || true
sleep 5
if kill -0 "$VLLM_PID" 2>/dev/null; then
    log_step_result "7. launch vLLM Gemma4 server" \
        "OK (pid $VLLM_PID, still loading — tail -f /tmp/vllm.log; a 12B model can take several minutes)"
else
    log_step_result "7. launch vLLM Gemma4 server" "FAILED (process exited immediately — see /tmp/vllm.log)"
fi

# 8. Pipeline deps (inside the already-handed-off folder)
if [ -d "$PIPELINE_DIR" ]; then
    run_step "8a. create pipeline venv" "python3 -m venv '$PIPELINE_DIR/.venv'"
    if [ -n "$PIP_INSTALL_EXTRAS" ]; then
        # --no-build-isolation: flash-attn's build needs the already-installed
        # torch visible, which pip's isolated build env otherwise hides
        # (CLAUDE.md's "Deploying on RunPod" section).
        run_step "8b. pip install -e .[$PIP_INSTALL_EXTRAS] (--no-build-isolation)" \
            "'$PIPELINE_DIR/.venv/bin/pip' install -q -e '$PIPELINE_DIR[$PIP_INSTALL_EXTRAS]' --no-build-isolation"
    else
        run_step "8b. pip install -e ." "'$PIPELINE_DIR/.venv/bin/pip' install -q -e '$PIPELINE_DIR'"
    fi
else
    skip_step "8. pipeline deps" "$PIPELINE_DIR not found — set PIPELINE_DIR to the repo's actual path"
fi

# 9. .env — upsert rather than overwrite, so any other keys already in .env
# (DATABASE_URL, REDIS_URL, etc. — see .env.example) survive a re-run.
if [ -d "$PIPELINE_DIR" ]; then
    ENV_FILE="$PIPELINE_DIR/.env"
    (
        touch "$ENV_FILE" &&
        for kv in \
            "GEMMA4_ENDPOINT_URL=http://localhost:${GEMMA4_PORT}/v1/chat/completions" \
            "GEMMA4_TIMEOUT_SECONDS=60.0" \
            "USE_REAL_TRANSCRIBER=true"
        do
            key="${kv%%=*}"
            grep -q "^${key}=" "$ENV_FILE" 2>/dev/null \
                && sed -i.bak "s|^${key}=.*|${kv}|" "$ENV_FILE" \
                || echo "$kv" >>"$ENV_FILE"
        done && rm -f "$ENV_FILE.bak"
    ) && log_step_result "9. write .env" "OK ($ENV_FILE)" \
      || log_step_result "9. write .env" "FAILED"
else
    skip_step "9. write .env" "$PIPELINE_DIR not found"
fi

# 10. Local infra — docker-compose.local.yml is what the reference names;
# this repo only ships docker-compose.yml, so fall back to that. NOTE
# (CLAUDE.md): a RunPod Pod cannot run nested docker-compose at all (the Pod
# itself is already one container) — expect this to fail there and use the
# bare-metal service start path from CLAUDE.md's "Deploying on RunPod"
# section instead.
if [ -d "$PIPELINE_DIR" ]; then
    if [ -f "$PIPELINE_DIR/docker-compose.local.yml" ]; then
        COMPOSE_FILE="docker-compose.local.yml"
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
    run_step "10. docker-compose up -d ($COMPOSE_FILE)" \
        "cd '$PIPELINE_DIR' && docker-compose -f '$COMPOSE_FILE' up -d"
else
    skip_step "10. docker-compose up -d" "$PIPELINE_DIR not found"
fi

# --- Summary -----------------------------------------------------------------
echo
echo "================ SETUP SUMMARY ================"
FAILED_COUNT=0
for i in "${!STEP_NAMES[@]}"; do
    status="${STEP_STATUSES[$i]}"
    case "$status" in
        OK*) icon="OK  " ;;
        SKIPPED*) icon="SKIP" ;;
        *) icon="FAIL"; FAILED_COUNT=$((FAILED_COUNT + 1)) ;;
    esac
    printf '[%s] %-45s %s\n' "$icon" "${STEP_NAMES[$i]}" "$status"
done
echo "================================================="
if [ "$FAILED_COUNT" -eq 0 ]; then
    echo "All steps completed (some may have been skipped — see above)."
else
    echo "$FAILED_COUNT step(s) failed — see per-step logs under $LOG_DIR"
fi
exit "$FAILED_COUNT"
