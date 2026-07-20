"""FastAPI app for the movie-highlight-pipeline job API.

Three endpoints, per the fixed cross-stage contract:
  POST /jobs               — shard a video into segments, persist Job +
                              Segment rows, enqueue the orchestrator pipeline.
  GET  /jobs/{job_id}      — job status lookup.
  GET  /jobs/{job_id}/results — ranked HighlightResult rows for a job (may be
                              empty if the job hasn't completed yet).

Step 1 note: orchestrator.splitter / orchestrator.state / orchestrator.pipeline
are being built in parallel against this same contract. This module imports
them at the exact dotted paths specified by the contract regardless of
whether they exist yet at the time this file is written.
"""

import shutil
import uuid
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from config import get_settings
from config.logging import configure_logging
from models.db import get_db
from models.enums import JobStatus
from models.orm import HighlightResult, Job, ScoredSpan
from models.schemas import (
    HighlightResultOut,
    JobCreateRequest,
    JobResponse,
    JobResultsResponse,
    UploadResponse,
)
from orchestrator import pipeline, splitter, state

configure_logging()

logger = structlog.get_logger()

app = FastAPI(title="Movie Highlight Pipeline API")

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Serves the local object storage root (where segment_worker/scorer write
# highlight clips, see storage/object_storage.py's LocalFilesystemStorage)
# over HTTP at /clips, so the demo UI's <video> tags can actually play them.
# Dev-only concern — only meaningful when object_storage_backend=local; a
# real S3/Azure Blob deployment would generate its own accessible URLs
# instead (see JobResultsResponse.clip_url), not this static mount.
_OBJECT_STORAGE_ROOT = Path(get_settings().object_storage_local_root).resolve()
_OBJECT_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=str(_OBJECT_STORAGE_ROOT)), name="clips")

_UPLOADS_DIR = _OBJECT_STORAGE_ROOT / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/demo", include_in_schema=False)
def demo_ui() -> FileResponse:
    """Quick, no-build-step demo UI — a single static HTML page (vanilla JS,
    no framework) that calls this same app's /jobs endpoints. Served from
    here (not a separate static file server) specifically so it's same-origin
    with the API and needs zero CORS configuration. Not part of the API
    contract (excluded from the OpenAPI schema) — this is a demo aid, not a
    product frontend; see CLAUDE.md for why a real UI is a separate, later
    decision."""
    return FileResponse(_STATIC_DIR / "demo.html")


@app.post("/uploads", response_model=UploadResponse, status_code=201)
async def upload_video(file: UploadFile) -> UploadResponse:
    """Saves an uploaded video file under the local uploads dir and returns
    a plain filesystem path usable directly as JobCreateRequest.
    source_video_url — ffmpeg's -i flag takes a bare path (see tonight's
    real-video test), not a file:// URI, so this deliberately does NOT go
    through storage.object_storage's url_for() (which prefixes file://).
    Dev-only: assumes object_storage_backend=local, same as the /clips
    static mount above; a real deployment would need a real upload target
    (S3 presigned URL, etc.), not a bare local save."""
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    dest_path = _UPLOADS_DIR / f"{uuid.uuid4()}{suffix}"

    with dest_path.open("wb") as dest:
        shutil.copyfileobj(file.file, dest)

    from workers.segment_worker.adapters._media import probe_duration_seconds

    total_duration_seconds = probe_duration_seconds(str(dest_path))

    logger.info(
        "stage.completed",
        stage="upload",
        source_video_url=str(dest_path),
        total_duration_seconds=total_duration_seconds,
    )

    return UploadResponse(
        source_video_url=str(dest_path),
        total_duration_seconds=total_duration_seconds,
    )


@app.post("/jobs", response_model=JobResponse, status_code=201)
def submit_job(payload: JobCreateRequest, db: Session = Depends(get_db)) -> Job:
    settings = get_settings()

    segment_count = payload.segment_count or settings.provisional_dev_segment_count
    sla_target_seconds = payload.sla_target_seconds or settings.default_sla_target_seconds

    total_duration_seconds = payload.total_duration_seconds
    if total_duration_seconds is None:
        total_duration_seconds = splitter.get_duration_probe().probe(payload.source_video_url)

    segment_plan = splitter.compute_segment_plan(
        total_duration_seconds,
        segment_count,
        settings.segment_overlap_seconds,
    )

    job = state.create_job(
        db,
        source_video_url=payload.source_video_url,
        total_segments=len(segment_plan),
        sla_target_seconds=sla_target_seconds,
    )
    state.create_segments(db, job, segment_plan)

    # Reflect that the job is about to be handed to Celery before we actually
    # enqueue it, and commit so the Job/Segment rows are durably visible to
    # the pipeline task (which — in eager test mode — runs synchronously
    # in-process the moment .delay() is called below).
    job.status = JobStatus.queued
    db.commit()
    db.refresh(job)

    pipeline.run_job.delay(str(job.id))

    logger.info(
        "stage.completed",
        job_id=str(job.id),
        stage="job_submission",
        total_segments=job.total_segments,
    )

    return job


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/jobs/{job_id}/results", response_model=JobResultsResponse)
def get_job_results(job_id: uuid.UUID, db: Session = Depends(get_db)) -> JobResultsResponse:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    rows = (
        db.query(HighlightResult, ScoredSpan)
        .join(ScoredSpan, HighlightResult.span_id == ScoredSpan.id)
        .filter(HighlightResult.job_id == job_id)
        .order_by(HighlightResult.rank)
        .all()
    )

    results = [
        HighlightResultOut(
            rank=highlight.rank,
            span_id=span.id,
            start_ts=span.start_ts,
            end_ts=span.end_ts,
            transcript_excerpt=span.transcript_excerpt,
            final_score=highlight.final_score,
            justification=span.justification,
            clip_url=span.clip_url,
            rich_data=span.rich_data,
        )
        for highlight, span in rows
    ]

    return JobResultsResponse(job_id=job.id, status=job.status, results=results)
