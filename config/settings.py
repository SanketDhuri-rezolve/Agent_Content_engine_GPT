from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-driven config. Values differ per environment (local/runpod/prod)
    via the .env file loaded or real environment variables — no per-environment
    code branches."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["local", "runpod", "prod"] = "local"

    # Postgres (job/segment state)
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/movie_highlights"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "highlight_embeddings"

    # Neo4j (stub only — not implemented, interface reserved for future graph storage)
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None

    # Object storage — backend is swappable; see storage/object_storage.py
    object_storage_backend: Literal["local", "s3", "azure_blob", "runpod_volume"] = "local"
    object_storage_bucket: str = "movie-highlights"
    object_storage_endpoint_url: str | None = None
    object_storage_access_key: str | None = None
    object_storage_secret_key: str | None = None
    object_storage_local_root: str = "./.local_object_storage"

    # Segment sharding — NOT DECIDED. Do not treat this as a production default.
    # The user will confirm 8 vs 12 (or another value) after Phase 3 latency
    # measurements (see scripts/measure_latency.py). Every call site that needs
    # a segment count MUST take it as an explicit parameter (job submission
    # request, test fixture, etc.) rather than silently falling back to this
    # value in production code paths. This setting exists only so local dev /
    # Step 1 tests have *something* to pass without hardcoding a magic number
    # inline at every call site.
    provisional_dev_segment_count: int = Field(
        default=4,
        description="UNCONFIRMED placeholder for local dev/tests only — not a production default. See CLAUDE.md.",
    )
    segment_overlap_seconds: float = 5.0

    # SLA
    default_sla_target_seconds: int = 240

    # Gemma 4 scorer endpoint (already deployed — treat as existing inference endpoint)
    gemma4_endpoint_url: str | None = None
    gemma4_api_key: str | None = None
    gemma4_timeout_seconds: float = 30.0
    # Must match the `served_model_name` the endpoint's OpenAI-compatible API
    # actually registers the model under — vLLM's `vllm serve <repo_id>`
    # serves it under the full HF repo id by default, NOT a short alias like
    # "gemma-4", confirmed via a real 404 on this exact mismatch.
    gemma4_model_name: str = "google/gemma-4-12B-it-qat-w4a16-ct"

    # Reducer behavior
    reducer_segment_timeout_seconds: float = 90.0
    reducer_min_segments_required_fraction: float = 0.6

    # Step 4: failure handling. Celery time limits on run_segment_worker —
    # provisional, not yet measured against real GPU workloads (Step 2).
    # Soft limit raises a catchable SoftTimeLimitExceeded inside the task
    # (caught by the existing fail-soft try/except, same as any other
    # exception); hard limit forcibly kills the task's worker child process,
    # which the task itself cannot catch — see CLAUDE.md's Step 4 section.
    segment_worker_soft_time_limit_seconds: float = 300.0
    segment_worker_hard_time_limit_seconds: float = 360.0

    # Job-level watchdog: guards against a chord that never completes at all
    # (e.g. a worker is killed mid-task with no other worker available to
    # redeliver to) — the reducer's own fallback logic only ever runs once the
    # chord callback actually fires, so a permanently-lost chord member would
    # otherwise hang the job forever regardless of how good that logic is.
    # orchestrator.pipeline.run_job schedules a one-shot
    # orchestrator.watchdog.check_job_timeout(job_id) this many seconds after
    # the job's own sla_target_seconds — if the job hasn't reached a terminal
    # status by then, the watchdog force-fails it rather than waiting forever.
    watchdog_grace_period_seconds: float = 60.0

    # Test-only failure injection — see workers/segment_worker/tasks.py's
    # _apply_failure_injection. Safety-gated: defaults to False so a magic
    # query string can NEVER accidentally trigger simulated failures outside
    # an explicit test run, even if it somehow appeared in a real
    # source_video_url.
    enable_failure_injection: bool = False

    # Step 2 real-model opt-ins. Every one of these defaults to False so
    # Step 1's behavior (100% mocked, zero GPU, zero network) is completely
    # unaffected unless a GPU environment explicitly flips it on. Each real
    # adapter class is only ever imported lazily inside the corresponding
    # Mock*/Real* selection — importing this settings module, or any adapter
    # module, never imports torch/faster_whisper/pyannote/cv2/etc. See
    # workers/segment_worker/tasks.py's SegmentWorkerAdapters and
    # orchestrator/splitter.py's get_duration_probe() for where these are read.
    use_real_duration_probe: bool = False
    use_real_shot_detector: bool = False
    use_real_transcriber: bool = False
    use_real_diarizer: bool = False
    use_real_visual_embedder: bool = False
    use_real_motion_extractor: bool = False
    use_real_audio_extractor: bool = False
    use_real_face_detector: bool = False
    use_real_object_detector: bool = False

    # Whisper+Gemma4-only mode: skip shot_detector, diarizer, visual_embedder,
    # face_detector, motion_extractor, audio_extractor, object_detector
    # ENTIRELY (not even their Mock stand-ins run) — only Transcriber
    # (real Whisper) and the Scorer (real Gemma4) do any work per segment.
    # Gemma4Scorer samples its own keyframe images directly from the cropped
    # clip (see Gemma4Scorer._build_multimodal_content), so it does not need
    # ShotDetector's keyframes at all — none of the skipped adapters' output
    # is required for Gemma4Scorer's real multimodal scoring call. Default
    # True: measured on real GPU hardware that Face Detector+Embed alone was
    # 54% of total per-segment time (345.6s/640.7s across a 4-segment job)
    # while its output wasn't even wired to anything downstream yet — see
    # CLAUDE.md.
    enable_secondary_adapters: bool = False

    # Concurrency cap for Gemma4 HTTP calls fired at once (scorer's per-span
    # rich-schema calls, and the global-memory pipeline's per-winner detail
    # calls below). Previously these ran in a plain sequential for-loop —
    # measured on the real 13-min test clip at ~26s/span average because
    # vLLM's continuous batching was never exercised (one request in flight
    # at a time). Bound by --gpu-memory-utilization/--max-model-len on the
    # vLLM side, not just this number — raise cautiously.
    gemma4_max_concurrent_requests: int = 6

    # Global-memory architecture (opt-in, default False so the existing
    # per-span pipeline — segment_worker -> span_builder -> scorer, every
    # candidate span scored independently — is completely unaffected until
    # explicitly enabled). When True:
    #   1. segment_worker also runs a lightweight TEXT-ONLY Gemma4 call per
    #      chunk producing a compact local_memory (characters/events/objects/
    #      threads) — see workers/segment_worker/adapters/memory_extractor.py.
    #   2. The per-segment chain drops workers.scorer.tasks.score_spans —
    #      span_builder's output goes straight to the chord callback.
    #   3. workers.global_selector.tasks.select_and_score_job runs ONCE per
    #      job as the new chord callback: merges every segment's local_memory
    #      into one global movie memory, asks a single text-only Gemma4 call
    #      to pick the top global_selector_top_n candidates across the WHOLE
    #      job (full-story context, not just spans that look exciting in
    #      isolation), then scores ONLY those winners with the real
    #      multimodal Gemma4Scorer (concurrently) before handing off to the
    #      unchanged reducer/ranker.
    # This trades "score every candidate" for "score only what a
    # global-context pass selected" — see the architecture discussion this
    # setting exists to make testable.
    use_global_memory_pipeline: bool = False
    global_selector_top_n: int = 10
    global_selector_timeout_seconds: float = 45.0

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
