# Movie Highlight Pipeline — CLAUDE.md

## What this is
Horizontally-sharded pipeline that turns a ~2hr movie into ranked highlight moments in under 4 minutes wall-clock. A 2-hour film is split into N overlapping segments, each processed independently and concurrently (Celery chord), then reduced into one globally-ranked output.

This project is **separate from AdStitch** (the cricket ad-stitching backend) — no shared code, no shared conventions beyond both being Python/FastAPI/Celery. Do not assume AdStitch's status-code conventions, Supabase usage, or directory layout apply here.

**No git repo** — per explicit user instruction, this directory is not (and should not be) initialized as a git repository.

## Build order (do not skip ahead)
- **Step 1** (done): scaffold everything GPU/model-dependent as mocked adapters. Zero GPU, zero cost, runs on a laptop. FastAPI + Postgres models + Celery chord wiring + reducer logic against mocked segment output. Verified end-to-end: `docker-compose up -d --build`, then `POST /jobs` → all 6 stages ran as real Celery tasks across real containers → ranked results returned from `GET /jobs/{id}/results`. 100 pytest tests pass against a real Postgres.
- **Step 2** (done — **all 6 adapters confirmed on real GPU hardware**, RunPod pod): `TransNetV2ShotDetector`, `FasterWhisperTranscriber`, `PyannoteDiarizer`, `InternVideo2VisualEmbedder`, `OpenCVMotionFeatureExtractor`, `OpenSmileAudioFeatureExtractor` all ran against a real video clip and produced real output. 7 real bugs found and fixed along the way (4 in InternVideo2, 1 in faster-whisper's download path, 2 in pyannote). See "Step 2: real adapters" section below. Only run at the adapter level, not yet through the full Celery pipeline/Postgres on GPU hardware — see "Real-video validation setup".
- **Step 3** (done, run against Step 1's mocks — re-run against Step 2's now-validated real adapters for the actual 8-vs-12 decision): `scripts/measure_latency.py` submits a real job, polls to completion, and reports per-stage + cumulative timing. See "Step 3: latency harness" section below — the numbers there come from mocked adapters and are NOT the real per-stage costs to use for the 8-vs-12 decision.
- **Step 4** (done): failure injection testing — kill/delay a segment worker mid-run, confirm the system degrades gracefully instead of hanging. All 3 failure modes verified live (not just unit-tested): a segment crashing, a segment soft-timing-out, and a worker being SIGKILLed mid-task. See "Step 4: failure injection testing" section below — it found and fixed 2 real bugs no unit test caught.
- **Step 5** (proposed, NOT started — no code written): character/cast-aware metadata layer (who's on screen, TMDb lookup, OCR cast cards). See "Step 5 (proposed): character/cast metadata" below. Deliberately deferred — see that section for why.

**Steps 1, 2, and 4 are done and GPU/real-hardware-validated where applicable.** Step 3 needs re-running against Step 2's real adapters (not just mocks) before the 8-vs-12 segment decision. Step 5 is a documented future direction only, not started.

## Local port conflicts on this dev machine
This machine already runs a native Postgres (port 5432), another project's Redis (6379), and a dev server on 8000. `docker-compose.yml` publishes Postgres on **5433**, Redis on **6380**, and the API on **8001** instead — container-internal ports are untouched (in-network service-to-service calls still use 5432/6379/8000 via the `postgres`/`redis`/`api` hostnames). `.env.example` matches. If you move this repo to a machine without these conflicts, the published ports can be changed back to the defaults, or left as-is — either works.

## Key decisions / gotchas

### Segment count is NOT decided
8 vs 12 segments is an open question pending Phase 3 latency measurements — **do not hardcode a production default**. `config.Settings.provisional_dev_segment_count` exists only for local dev/tests and is explicitly documented as an unconfirmed placeholder. `JobCreateRequest.segment_count` is optional and explicit; production call sites should require it, not silently fall back.

### Timestamps are global, never segment-local
`Segment`, `CandidateSpan`, `ScoredSpan` all store timestamps relative to the full film, not the segment. This is load-bearing for the reducer's dedup logic across segment boundaries.

### Fixed Celery task contract
Every stage is a Celery task with a fixed dotted path and JSON-dict-shaped input/output (see `models/schemas.py` for the corresponding Pydantic shapes):

| Task | Queue | Input → Output |
|---|---|---|
| `workers.segment_worker.tasks.run_segment_worker` | `segment_worker` | `SegmentTaskPayload` → `SegmentWorkerOutput` |
| `workers.span_builder.tasks.build_spans` | `span_builder` | `SegmentWorkerOutput` → candidate-span dict |
| `workers.scorer.tasks.score_spans` | `scorer` | candidate-span dict → `SegmentPipelineResult` |
| `workers.reducer.tasks.reduce_job` | `reducer` | chord result list + `job_id` → `ReducerOutput` |
| `workers.ranker.tasks.rank_and_persist` | `ranker` | `ReducerOutput` + `job_id` → ranked results |

Orchestrator wiring: `chord([chain(run_segment_worker.s(p), build_spans.s(), score_spans.s()) for p in payloads])(reduce_job.s(job_id) | rank_and_persist.s(job_id))`.

### Reducer is the highest-risk component
`workers/reducer/` does three things in order: (1) validate/classify segments as usable vs. degraded, raising `InsufficientSegmentsError` if too few usable segments remain (threshold: `config.Settings.reducer_min_segments_required_fraction`); (2) dedupe boundary-touching spans across adjacent segments by timestamp-overlap; (3) z-score normalize `raw_score` → `normalized_score`, handling the zero-variance edge case explicitly. It has the most extensive test coverage in the repo, including an adversarial test pass — see `tests/reducer/`.

### Every model call is behind an adapter interface
TransNetV2, faster-whisper, pyannote, InternVideo2, Gemma 4 (and even the video-duration probe) each get an ABC + a `Mock*` implementation used in Step 1, plus now a real implementation (see below). Step 2 swaps in the real implementation without touching any caller. Do not call a real model/network endpoint from Step 1 code.

### Object storage / vector storage / graph storage are all behind interfaces
- `storage/object_storage.py` — local filesystem (dev), S3-compatible (RunPod volume), Azure Blob (prod). Backend chosen via `config.Settings.object_storage_backend`.
- `storage/qdrant_client.py` — in-memory store for local dev (`environment == "local"`), real Qdrant otherwise.
- `storage/neo4j_stub.py` — **stub only**, `NoOpGraphStore` is the only implementation. Not wired to a real Neo4j instance yet; do not add one without being asked.

### Environment
- `AZURE_WISHPER_*`-style typos from AdStitch do **not** apply here — this is a fresh env var namespace, see `config/settings.py` / `.env.example`.
- Local dev: `docker-compose up -d postgres redis` (qdrant is optional — code defaults to the in-memory vector store locally).

## Bugs found and fixed during Step 1 integration review
The 7 Step-1 modules (orchestrator, 5 workers, api, infra) were built in parallel against a fixed contract, then reviewed and exercised end-to-end for real. Three real integration bugs surfaced this way — none would have been caught by any single module's own unit tests, only by running the whole chain:
1. **Reducer dedup was silently disabled.** `orchestrator/pipeline.py` built `build_spans.s()` with no arguments, but `build_spans` needs `overlap_start`/`overlap_end` passed explicitly (`SegmentWorkerOutput` doesn't carry them). Without the fix, `touches_boundary` would always be `False` in the real pipeline, so the reducer's entire boundary-dedup logic — despite being heavily unit- and adversarially-tested in isolation — would never actually engage. Fixed by binding `overlap_start=payload["overlap_start"], overlap_end=payload["overlap_end"]` at chain-construction time.
2. **Ranker crashed on real span data.** `workers/ranker/logic.py`'s cosine similarity assumed every `feature_vector` value was a plain float, but `span_builder`'s real output includes non-scalar entries (`keyframe_refs`: list[str], `visual_embedding_mean`: list[float]) — `TypeError: float() argument must be a string or a real number, not 'list'`, discovered only when running a real job through Docker Compose (the two modules' own unit tests each used feature vectors that happened not to trigger it). Fixed by filtering to scalar numeric entries before computing similarity — see `_scalar_items` in `workers/ranker/logic.py`.
3. **`pip install .` failed inside every worker image.** `pyproject.toml`'s `[tool.setuptools] packages = [...]` was a static list including `api`, but worker Dockerfiles only `COPY` the packages they actually need (no `api/`) — setuptools hard-errors on any listed package directory that doesn't exist. Fixed by switching to `[tool.setuptools.packages.find]`, which only registers packages actually present in the build context.

Plus two smaller fixes: `docker-compose.yml` was missing a worker service for the `orchestrator` queue that `run_job` actually dispatches to (added one, reusing `api/Dockerfile`'s image since it needs the same non-GPU deps); `scripts/smoke_test.py` guessed a `create_segments` signature that didn't match what `orchestrator/state.py` landed with (fixed to call `splitter.compute_segment_plan` first, matching the real signature).

**Lesson for future sessions**: when building this pipeline's stages in parallel against a fixed contract, always follow up with a real end-to-end run (not just each module's own unit tests) before considering a step done — the bugs above only exist at the seams between modules, invisible to any single module's test suite.

## Step 2: real adapters

Every real adapter is opt-in per `config.Settings.use_real_*` (all default `False`) — `workers/segment_worker/tasks.py`'s `_default_*` factory functions pick `Real*` only when the matching flag is set, otherwise `Mock*`, so Step 1 behavior is completely unaffected until you deliberately flip these on a GPU box. The Gemma 4 scorer is the one exception: it's driven by whether `gemma4_endpoint_url` is set at all (`workers/scorer/adapters.get_scorer()`), not a separate boolean.

**All 6 real segment_worker adapters are now confirmed working on real GPU hardware** (RunPod pod: first an A40 48GB, later tests on the same setup — see "Real-video validation" below for the actual run and "N real bugs found" sections for what broke and how it was fixed). This is the single biggest validation milestone in the project so far — every adapter that was previously only import/syntax-checked has now actually produced real output from a real video.

| Adapter | Real class | Package(s) | Status |
|---|---|---|---|
| ShotDetector | `TransNetV2ShotDetector` (shot_detector.py) | `transnetv2-pytorch>=1.0.5` | **Confirmed on real GPU hardware.** Detected 4 real shot boundaries in a 30s real video clip. Unofficial single-maintainer PyTorch port, bundles weights in the wheel — no official TransNetV2 PyPI package exists. |
| Transcriber | `FasterWhisperTranscriber` (transcriber.py) | `faster-whisper>=1.1.0,<2` | **Confirmed on real GPU hardware.** Ran successfully end-to-end (see "5 real bugs" list — the `hf-xet` hang was found via this adapter specifically). |
| Diarizer | `PyannoteDiarizer` (diarizer.py) | `pyannote.audio>=4.0.0,!=4.0.6`, `soundfile>=0.12.1` | **Confirmed on real GPU hardware**, after fixing 2 real bugs (torchcodec/`libnvrtc` + pyannote 4.0's `DiarizeOutput` return type — see below). Still requires a HuggingFace token that has accepted the gated model's terms (`PYANNOTE_AUTH_TOKEN`/`HUGGINGFACE_TOKEN`/`HF_TOKEN`). Defaults to `pyannote/speaker-diarization-community-1`. |
| VisualEmbedder | `InternVideo2VisualEmbedder` (visual_embedder.py) | `transformers>=4.40,<5`, `torch>=2.1`, `pillow`, `einops`, `timm`, `flash-attn>=2.5`, `huggingface_hub>=0.34.0`, `hf_transfer>=0.1.4` | **Confirmed on real GPU hardware** (see "4 real bugs" below) — was the lowest-confidence adapter going in, now the most-validated. `OpenGVLab/InternVideo2-Stage2_6B`, ~24GB of weights, needs a 24GB+ GPU comfortably. Produced real 512-dim embeddings from real keyframes. |
| MotionFeatureExtractor | `OpenCVMotionFeatureExtractor` (motion_features.py) | `opencv-python-headless>=4.9,<6` | **Confirmed on real GPU hardware** — real frame-diff features from a real video clip, no issues found. Plain frame-diff, no model weights, samples at 3fps. |
| AudioFeatureExtractor | `OpenSmileAudioFeatureExtractor` (audio_features.py) | `opensmile>=2.6.0` | **Confirmed on real GPU hardware** — real eGeMAPS features from real audio, no issues found. `rms_energy_mean`/`loudness_peak_db` are documented approximations (eGeMAPS has no exact equivalents); `pitch_mean_hz` conversion is exact math. |
| DurationProbe | `FfprobeDurationProbe` (orchestrator/splitter.py) | none (needs the `ffmpeg` binary on PATH, not a pip package) | Not yet exercised in the real-video test (the test passed `total_duration_seconds` explicitly) — straightforward `ffprobe` subprocess call, low risk. |
| Scorer | `Gemma4Scorer` (workers/scorer/adapters.py) | none (plain `httpx` POST, already a core dep) | Was already fully implemented in Step 1 — still just needs a real `GEMMA4_ENDPOINT_URL` to test against; not exercised this round. |

### Real-video validation setup
Ran directly on a RunPod GPU pod (not through the full Celery pipeline — each adapter's `detect`/`transcribe`/`diarize`/`extract`/`embed` method called directly against a 30-second clip of a real, freely-licensed video (Big Buck Bunny, direct MP4 URL — `ffmpeg` streams it, no local download needed). Zero speaker turns and a single ambiguous transcript line ("To be continued...") are the CORRECT results for this clip, not bugs — it's a mostly-wordless animated short. Postgres/Redis/the full pipeline were not exercised in this round; this validated the six model adapters in isolation.

Everything here was run on a pod with **no Docker** — RunPod Pods don't support nested `docker-compose`, so Postgres/Redis/the app were installed as plain background processes directly on the pod (see "Deploying on RunPod" below) rather than through `Dockerfile.gpu`. `Dockerfile.gpu` itself is still untested as an actual Docker build — the fixes below were applied to the underlying Python code and to `Dockerfile.gpu`'s `ENV`/dependencies in parallel, but only the bare-metal path has been GPU-verified so far.

### InternVideo2: 4 real bugs found on first GPU run

Deployed on a RunPod pod (A40, 48GB VRAM, everything — Postgres/Redis/API/workers — running directly on the one pod as plain background processes, not docker-compose, since RunPod Pods don't support nested Docker Compose). Getting `InternVideo2VisualEmbedder()._get_model()` to actually load took fixing four distinct, previously-unknown problems, all now fixed in code (not just worked around by hand) and all specific to this checkpoint's third-party `trust_remote_code=True` remote source, not to our own adapter code:

1. **`transformers` v5 broke the model's own imports.** `pyproject.toml` only had a floor (`transformers>=4.40`), so pip installed `5.13.1`. The remote code does `from transformers.modeling_utils import (PreTrainedModel, apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer)` — the three non-`PreTrainedModel` helpers were relocated to `transformers.pytorch_utils` at some point and dropped from `modeling_utils` entirely (confirmed still missing even after downgrading to `transformers==4.57.6` — the relocation happened within the 4.x series, not just at the v5 boundary). Fixed two ways: pinned `transformers>=4.40,<5` in `pyproject.toml`, AND added a defensive monkey-patch in `_get_model()` that copies all three symbols from `pytorch_utils` into `modeling_utils` if missing — belt-and-suspenders, since the exact version where they vanish isn't pinned down.
2. **`HF_HUB_ENABLE_HF_TRANSFER=1` was set in the environment but `hf_transfer` wasn't installed** — hard-fails instead of falling back to a normal download. Added `hf_transfer>=0.1.4` to the `gpu` extra.
3. **The model's text encoder loads via `BertTokenizer.from_pretrained("bert-large-uncased", local_files_only=True, ...)`** — hardcoded `local_files_only=True`, so on a fresh machine with nothing cached it fails (and fails with a confusing `TypeError: stat: ... NoneType`, not a clear "not found"). Fixed by pre-fetching `bert-large-uncased` in `_get_model()` before constructing the model, so it's already in the local HF cache by the time the remote code asks for it "locally only".
4. **The model's BERT config loads via a bare relative path**, `BertConfig.from_json_file("configs/config_bert_large.json")` — only works if the current process's cwd happens to already contain that exact relative path (an assumption baked in from the original GitHub repo's layout, not valid when loaded via `AutoModel.from_pretrained` from an arbitrary directory). This matches a live, still-open HF discussion on this exact checkpoint (`discussions/4`) that Step 2's original research flagged as an untested risk. The file IS a real repo file on the HF Hub (confirmed) — transformers' dynamic-module loader just doesn't fetch it automatically (only `.py` files). Fixed by fetching it via `huggingface_hub.hf_hub_download` and writing it to that same relative path under the current working directory before model construction, if not already present.

None of these four were catchable without an actual GPU + real model load — Step 2's original review was import/syntax-only. This is the same lesson as Steps 1/3/4: cross-boundary assumptions (here: a third-party model's own code assumptions about its runtime environment) only surface when you actually run the thing.

### faster-whisper: 1 real bug (`hf-xet` hang)

Getting `FasterWhisperTranscriber` to actually download its weights hit a fifth bug, environment-level rather than code-level: **HuggingFace's `hf-xet` fast-download backend hung indefinitely** — 0 bytes transferred, no error, no timeout — fetching `mobiuslabsgmbh/faster-whisper-large-v3-turbo` from this RunPod pod. `hf_transfer` (used for InternVideo2's download, item 2 above) was unaffected — this is `xet`-specific, likely a network-path issue between this pod and `xet`'s specific backend infrastructure. Fixed by setting `HF_HUB_DISABLE_XET=1` (added to `Dockerfile.gpu`'s `ENV` and `.env.example`). **Caveat**: disabling `xet` globally then made `InternVideo2`'s (already-cached) weights re-download from scratch via the slower classic HTTP path on the next run, since that repo apparently transfers via `xet` rather than `hf_transfer` — a real but tolerable trade-off (~7 minutes instead of instant-from-cache), not a regression worth chasing further right now.

### PyannoteDiarizer: 2 more real bugs found

1. **`torchcodec` (pyannote's default audio-reading backend) failed to load its CUDA extension** — `OSError: libnvrtc.so.13: cannot open shared object file`, a version mismatch between the installed torch build (`2.8.0+cu128`) and this pod's available NVRTC library. Rather than chase the exact CUDA/torchcodec version pairing (fragile, environment-specific), fixed by bypassing `torchcodec` entirely: `diarizer.py` now reads the extracted WAV file itself via `soundfile` (a plain `libsndfile` wrapper with zero CUDA dependency) and hands pyannote a raw `{"waveform": tensor, "sample_rate": int}` dict instead of a file path — exactly the workaround pyannote's own error message suggests. Added `soundfile>=0.12.1` to the `gpu` extra.
2. **pyannote.audio 4.0's `community-1` pipeline returns a different result type than expected.** Calling the pipeline returns a `DiarizeOutput` wrapper, not the classic `Annotation` directly — `diarization.itertracks(...)` failed with `AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'`. Confirmed via direct inspection: the real `Annotation` (with `.itertracks()`) lives at `DiarizeOutput.exclusive_speaker_diarization` (non-overlapping turns — matches our `SpeakerTurn` schema, which models one speaker per turn) or `.speaker_diarization` (overlap-aware, not used here). Fixed by reading `.exclusive_speaker_diarization` before calling `.itertracks()`.

### Deploying on RunPod (bare-metal, no Docker) — operational notes

A RunPod **Pod** cannot run `docker-compose` (the Pod itself is already a single Docker container — no nested Docker). For adapter-level GPU validation, everything ran as plain background processes directly on the pod instead:
- No systemd (`systemctl` fails with "System has not been booted with systemd") — use `service <name> start` instead (works fine for `ssh`, `postgresql`, `redis-server` — all still have SysV-style init scripts even without systemd running).
- SSH needs host keys generated manually on a fresh pod image: `ssh-keygen -A` before `service ssh start` ("no hostkeys available" otherwise).
- `pip install` may hit PEP 668's "externally-managed-environment" protection depending on the image/pip version — add `--break-system-packages` (safe on a throwaway GPU pod).
- `flash-attn` needs `--no-build-isolation` (see Step 2 adapter table) — its build script needs the already-installed `torch` visible, which pip's isolated build env hides by default.
- `HF_HUB_DISABLE_XET=1` should be set globally (see faster-whisper bug above) before installing/running anything that pulls HF Hub weights.
- Gated HF models (pyannote) need the token owner's account to have manually accepted terms on the model's HuggingFace page — a valid token alone returns a 403 `GatedRepoError` otherwise.

After both fixes, `PyannoteDiarizer` ran successfully end-to-end against the real video clip (0 speaker turns detected — correct for this content, see "Real-video validation setup" above).

**Model-loading is cached at the class level, not per-instance** — this matters because `SegmentWorkerAdapters()` (and therefore each `Real*` adapter) is reconstructed fresh on every single `run_segment_worker` task call. `TransNetV2ShotDetector`, `FasterWhisperTranscriber`, and `InternVideo2VisualEmbedder` already used class-level `_MODEL_CACHE` dicts as originally written. Two others (`PyannoteDiarizer`, `OpenSmileAudioFeatureExtractor`) originally cached only on `self` — fixed during review to class-level caches (`_PIPELINE_CACHE`, `_SMILE_CACHE`) so their models/configs load once per worker process instead of on every segment. Without this, Step 3's latency numbers would be dominated by repeated model-load overhead rather than actual inference.

**`workers/segment_worker/adapters/_media.py`** is the shared ffmpeg/ffprobe extraction layer every real adapter depends on (`extracted_frame`, `extracted_frames_uniform`, `extracted_audio_wav`, `probe_duration_seconds`) — requires the `ffmpeg`/`ffprobe` binaries on PATH (installed via apt in `Dockerfile.gpu`, not a pip package). It also defines the **keyframe_ref contract**: a real ShotDetector's keyframe_ref is always `f"kf_t{timestamp:.3f}"` (`make_keyframe_ref`/`parse_keyframe_timestamp`) — this is how `TransNetV2ShotDetector` and `InternVideo2VisualEmbedder` agree on keyframe identity statelessly, without any shared image cache between them.

**`workers/segment_worker/Dockerfile.gpu`** — new CUDA-base-image variant (existing `Dockerfile` stays Step-1-only, CPU, no GPU deps). Untested; installs `torch` first from the CUDA 12.4 wheel index, then `.[gpu]` with `--no-build-isolation` (works around flash-attn's most common build failure). Not wired into `docker-compose.yml` yet — that still runs the Step 1 CPU stack by default; swap in `Dockerfile.gpu` for the `segment_worker` service when you have GPU hardware to test against (e.g. RunPod, matching this project's original brief).

## Step 3: latency harness

`scripts/measure_latency.py` submits a real job via the API, polls `GET /jobs/{id}` to a terminal status, and reports per-stage + cumulative timing. Requires the stack already running (`docker-compose up -d --build`). Usage:

```
python scripts/measure_latency.py \
    --source-video-url https://example.com/movie.mp4 \
    --total-duration-seconds 7200 \
    --segment-count 8 \
    --sla-target-seconds 240
```

**How per-stage timing works**: every process (`api`, every Celery worker) now runs `config.logging.configure_logging()` at import time (wired into `orchestrator/celery_app.py` and `api/main.py`), which configures structlog's `JSONRenderer` — every `stage.started`/`stage.completed`/`stage.failed` log line becomes a single parseable JSON object instead of structlog's default human-readable console text. The harness runs `docker compose logs --no-log-prefix --since=<window>` across all 7 logged services and parses each line.

**Two bugs found and fixed while first running this for real** (same lesson as Step 1: things built/reasoned about in isolation have integration bugs only a live run surfaces):
1. **Celery hijacks the root logger.** Despite `configure_logging()`'s own `logging.basicConfig`, Celery wraps every message in its own `"[timestamp: LEVEL/ProcessName] "` prefix — so a log line looks like `[2026-...: WARNING/ForkPoolWorker-8] {"job_id": ...}`, not a bare JSON object. Fixed the parser to locate the embedded `{...}` within the line rather than assume the whole line is JSON.
2. **`span_builder`/`scorer` log lines don't carry `job_id`** — by design (their task payloads genuinely don't include it, only `segment_id`/`sequence_index` — see the fixed Celery task contract above), so a naive job_id-only filter silently dropped those two stages' timing entirely. Fixed with a two-pass correlation: pass 1 uses `segment_worker`'s lines (which DO carry both `job_id` and `segment_id`) to learn which `segment_id`s belong to the job; pass 2 matches every other line by `segment_id` membership in that set.

Also fixed: the SLA verdict now only reports PASS/FAIL when the job actually reached `status=completed` — a job that times out mid-run (still `running_segments` when polling stops) reports `INCOMPLETE`, never a misleading `PASS` just because elapsed time happened to be under budget.

**Sample output against Step 1's mocked adapters** (8 segments, illustrative only — this measures pipeline/orchestration overhead, NOT real model latency, since every adapter here is a mock returning near-instantly):

```
Harness-measured wall-clock (submit -> terminal status): 2.40s
SLA verdict: PASS (2.40s vs 240s budget)

Per-stage timing (from docker compose logs):
  stage                 count     min(s)    mean(s)     max(s)
  orchestrator.run_job      1     0.0281     0.0281     0.0281
  ranker                    1     0.0430     0.0430     0.0430
  reducer                   1     0.0004     0.0004     0.0004
  scorer                    8     0.0002     0.0004     0.0009
  segment_worker            8     0.0187     0.0263     0.0334
  span_builder              8     0.0006     0.0013     0.0031
```

**Do not use these numbers for the 8-vs-12 segment decision** — re-run this exact harness once Step 2's real adapters are validated on GPU hardware (swap `workers/segment_worker/Dockerfile.gpu` in for the `segment_worker` service, set the `use_real_*` flags), then compare `--segment-count 8` vs `--segment-count 12` runs against real per-stage numbers.

## Step 4: failure injection testing

### The subtlety the brief's wording doesn't spell out
"Kill/delay a segment worker mid-run, confirm the reducer's timeout+fallback path degrades gracefully" sounds like one failure mode, but it's actually TWO, with very different consequences:

1. **A catchable failure or delay** (an adapter throws, or a task hits Celery's own soft time limit and raises `SoftTimeLimitExceeded` inside the task). The task still returns SOMETHING (a fail-soft dict), so the chord completes normally and `workers/reducer/tasks.py`'s `reduce_job` (the chord callback) runs as usual — its own `classify_segments`/`InsufficientSegmentsError` logic (built and adversarially tested in Step 1) handles this correctly.
2. **A worker is genuinely lost** (SIGKILL, OOM, node death) with no other worker of that queue available to redeliver the task to. No result for that chord member EVER arrives. **`reduce_job` never runs at all** — there is no Python code executing that could time out, no matter how good its fallback logic is. This is the actual "hang forever" risk, and it requires an INDEPENDENT bound outside the chord itself, not better reducer logic.

### What was built
- **Celery config** (`orchestrator/celery_app.py`): `task_acks_late=True` + `task_reject_on_worker_lost=True` — if a worker dies mid-task, the message is requeued for another worker rather than silently lost (defense in depth; doesn't by itself prevent a hang if no replacement worker ever appears).
- **Time limits** (`config.Settings.segment_worker_soft_time_limit_seconds`/`_hard_time_limit_seconds`, provisional, not GPU-measured): a soft limit is caught inside `run_segment_worker` and produces `SegmentStatus.timeout` (distinct from `.failed` — see `workers/segment_worker/tasks.py`); a hard limit kills the task's worker child process outright — no code can catch that, which is exactly case 2 above.
- **Job-level watchdog** (`orchestrator/watchdog.py`): `run_job` schedules a one-shot `check_job_timeout(job_id)` at `sla_target_seconds + config.Settings.watchdog_grace_period_seconds` after job creation. If the job hasn't reached `completed`/`failed` by then, the watchdog force-fails it and marks any still-`pending`/`running` segments `SegmentStatus.timeout` (`orchestrator/state.py`'s `mark_stuck_segments_timeout`). Deliberately does NOT try to reconstruct a partial result from whatever segments happened to finish — that would need persisting each segment's individual Celery task id and querying the result backend directly, real extra plumbing for a case ("chord got stuck") that's rare and for which "retry the whole job" is a perfectly reasonable response.
- **Failure injection hook** (`workers/segment_worker/tasks.py`'s `_apply_failure_injection`, safety-gated behind `config.Settings.enable_failure_injection` — defaults `False`, verified by its own test that the marker has zero effect when disabled): a `__inject_failure=crash` or `__inject_failure=delay:<seconds>` query param on `source_video_url` triggers a controllable, reproducible failure instead of waiting for a real one. Every segment of a job shares the same `source_video_url` (it's the whole film's URL), so a bare directive applies to ALL segments uniformly — append `:seq<N>` (e.g. `delay:20:seq1`) to target just one segment's `sequence_index`, which is what makes it possible to demonstrate "some segments fail, job still degrades gracefully" rather than only all-or-nothing.

### Live verification (not just unit tests) — 3 scenarios, all run against the real docker-compose stack
1. **Targeted crash** (`?__inject_failure=crash:seq2`, 4 segments): job completed with 3/4 results; segment 2 correctly shows `status=failed` in the DB with a clear error.
2. **Soft timeout** (`?__inject_failure=delay:7:seq1` with a 5s soft limit, 4 segments): job completed with 3/4 results; segment 1 correctly shows `status=timeout`.
3. **Hard kill** (`?__inject_failure=delay:30:seq0`, soft/hard limits raised to 90s/120s so only the kill itself could end the task, `sla_target_seconds=5`, `watchdog_grace_period_seconds=5`): killed the `segment_worker` container ~1s into segment 0's 30s sleep via `docker compose kill segment_worker`, did NOT restart it. The job sat in `running_segments` for exactly 10s (5+5, matching the scheduled countdown to the second — confirmed via the watchdog's own `watchdog.scheduled`/`watchdog.job_force_failed` log lines), then the watchdog force-failed it. Restarting `segment_worker` afterward and submitting a fresh job confirmed the system fully self-heals (4/4 results, normal completion) — this was a bounded, recoverable failure of one job, not a systemic problem.

### 2 real bugs found live, neither caught by any unit test
1. **Segment DB rows never left `pending`.** `segment_worker`/`span_builder`/`scorer` are deliberately stateless (pure JSON in/out — see their own module docstrings) and never touch Postgres; nothing was updating `Segment.status`/`.error` after `create_segments` set them all to `pending`. Every segment stayed `pending` forever in the DB even after a job completed successfully. Fixed by adding `orchestrator/state.py`'s `sync_segment_statuses_from_pipeline_results`, called from `reduce_job` (the one place with the full per-segment picture) on both its success and failure paths. **Known residual limitation**: if the chord itself gets stuck (scenario 2 above), `reduce_job` never runs, so this sync never happens either — the watchdog's bulk `mark_stuck_segments_timeout` is the only thing that touches those rows, and it marks EVERY still-pending/running segment as `timeout`, including any that actually finished successfully at the Celery level moments before the kill (confirmed in the hard-kill test above: segment 1, never delayed, almost certainly finished in milliseconds, but still shows `timeout` in the DB because the job it was part of never got to use that result). This is accurate at the job-outcome level (that segment's work was never consumed by anything) even though it's not literally true at the task level — fixing this fully would mean persisting per-segment Celery task IDs and querying the result backend directly, which is real added complexity for a rare case.
2. **`scorer` silently downgraded a timeout into a false "completed".** `workers/scorer/tasks.py`'s fail-soft check only tested `upstream_status == SegmentStatus.failed.value` — it didn't know about `SegmentStatus.timeout`, which `span_builder` already propagates correctly (`span_builder`'s own `_FAILED_UPSTREAM_STATUSES = {"failed", "timeout"}`). A timed-out segment reached `scorer` with `candidate_spans=[]` and fell through to the "span_builder legitimately found zero candidates" path, which is a `completed` status — so a segment that actually timed out silently reported success with zero spans, all the way through to a `completed` DB row. Found by live test 2 above (the DB showed all 4 segments `completed` when segment 1 should have been `timeout`) — fixed by making `scorer` recognize the same status set `span_builder` uses, and propagate the upstream status UNCHANGED instead of hardcoding `.failed`.

**Lesson, same as Step 1 and Step 3**: cross-module contracts (here: which statuses count as "not usable", and whether every stage in a chain agrees) are exactly the kind of thing unit tests miss when each module is tested in isolation with hand-built fixtures that happen not to trigger the mismatch. A live run through the real stack is what actually catches it.

## Step 5 (proposed): character/cast metadata

**Status: design discussion only, NOT started, no code written.** Captured here so the thinking isn't lost, not as a commitment to build it next.

### The idea
Movies/web series often show cast intro cards ("Robert Downey Jr. as Tony Stark"), character/location/timeline text overlays, or episode/chapter titles — either as on-screen text (OCR-able) or inferable from external metadata (TMDb). Knowing who's on screen, and being able to track a character across the film, is genuinely valuable for search and highlight generation ("show every Tony Stark scene", "show scenes in Mumbai") — this is a real technique used in production media-indexing systems, not a novelty idea.

### Proposed shape
A **Metadata Bootstrap Engine** that runs once per job, before or alongside the main 6-stage pipeline, combining multiple sources so it degrades gracefully when any one is missing:
- Container metadata (language, duration) — cheap, already covered by `FfprobeDurationProbe`.
- Whisper spoken-language detection — already have the transcriber adapter, this is close to free.
- OCR on cast/title cards (PaddleOCR/EasyOCR) — a **new adapter**, same pattern as the existing six (ABC + Mock + Real, same `_media.py`-style frame extraction).
- Face detection + embedding + clustering, to find recurring characters even when nothing on screen names them — a **new, nontrivial CV subsystem** (needs an actual face embedding model, e.g. ArcFace/InsightFace, not just "clustering" as a step).
- Transcript NER (character names mentioned in dialogue) — an LLM pass over the transcript, could reuse the Gemma 4 endpoint already wired for scoring.
- Optional external lookup (TMDb primary; IMDb/Wikipedia as fallback for missing cast/character data) — **new external dependency**, its own network/rate-limit/attribution considerations.

Output would be a per-job metadata document (language, genre, cast/character list with embeddings, episode/season/timeline if applicable) that the rest of the pipeline — and eventually the reducer/ranker — could condition on.

### Why this is deferred, not just "later"
- **It's a new Step, not a patch to Steps 1-4.** The current data model (`Job`/`Segment`/`CandidateSpan`/`ScoredSpan`/`HighlightResult`) has zero concept of a character, cast member, or scene-to-character mapping. Doing this properly means new tables, a new pipeline stage, and a new adapter set — comparable in scope to Steps 1-2 combined, not a quick add-on.
- **Movie identification is the actual hard problem**, not TMDb lookup itself. TMDb only helps once you know *which* movie/episode you're looking at — filenames are unreliable, embedded metadata is often absent, and fingerprinting is a project of its own. For regional/independent/UGC content specifically, expect to land in the "not found, AI-only" branch often — meaning the AI fallback isn't really a fallback for that traffic, it's the primary path.
- **Face clustering/recognition is its own subsystem**, not a bullet point — a real face embedding model, clustering, and matching clusters to names is comparable in effort to any one of the existing six adapters.
- **TMDb coverage for Indian regional content is decent for well-known titles but inconsistent for new/small/independent releases** (by TMDb's own crowd-sourced nature) — the hybrid design already accounts for this, but it means the "primary path" will miss a meaningful fraction of real traffic for this product.
- **Legal/commercial-use terms for TMDb's API need checking** before depending on it in a shipped product (attribution requirements, rate limits).
- The bigger, more urgent unknown right now is that **Step 2's real adapters have never been GPU-validated** — building a new feature on top of an unvalidated core is premature.

### If/when this gets picked up
Recommended sequencing, cheapest and most independently-useful first:
1. TMDb lookup + OCR cast-card text extraction (no new heavy model training, reuses the OCR-adapter pattern already established).
2. Transcript NER via the existing Gemma 4 endpoint.
3. Face detection/embedding/clustering as a distinct, later phase — the hardest and most expensive part; character-aware search can ship off OCR+TMDb text alone before faces are needed at all.

## Open questions for the user
- Confirm segment count (8 vs 12) — run `scripts/measure_latency.py` with `--segment-count 8` and `--segment-count 12` against Step 2's real adapters on GPU hardware and compare; do not decide from the mocked baseline above.
- Confirm the provisional Step 4 time limits (`segment_worker_soft_time_limit_seconds=120`, `_hard_time_limit_seconds=180`, `watchdog_grace_period_seconds=60`) once real GPU model latency is known from Step 2/3 — these were picked as reasonable placeholders, not measured.
- Confirm dedup overlap threshold used by the reducer (documented inline in `workers/reducer/logic.py` — currently a first-pass judgment call, revisit once real embeddings exist in Step 2).
- Confirm the Step 2 choices flagged above once you have GPU hardware: InternVideo2 model variant (6B may be overkill/slow per-keyframe), pyannote model version (community-1 vs 3.1), and whether TransNetV2's single-maintainer PyPI port is acceptable long-term vs. vendoring the official weights directly.
