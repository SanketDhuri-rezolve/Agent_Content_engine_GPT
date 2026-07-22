"""Tests for the FastAPI job submission/status/results endpoints.

These endpoints read and write real Job/Segment/HighlightResult rows, so the
whole module is skipped when Postgres is not reachable (see
tests/conftest.py::postgres_available). orchestrator.pipeline.run_job.delay
is monkeypatched to a no-op so POST /jobs does not require a running Celery
worker/broker or a fully-implemented downstream pipeline.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

import orchestrator.pipeline as pipeline_module
from api.main import app


@pytest.fixture(autouse=True)
def _require_postgres(postgres_available):
    if not postgres_available:
        pytest.skip(
            "Postgres not reachable at config.Settings.database_url — "
            "start docker-compose to run these API tests"
        )


@pytest.fixture(autouse=True)
def _no_op_run_job(monkeypatch):
    """These are HTTP-contract tests, not full-pipeline integration tests —
    prevent POST /jobs from actually dispatching the segment_worker chord."""
    monkeypatch.setattr(pipeline_module.run_job, "delay", lambda *args, **kwargs: None)


@pytest.fixture
def client():
    return TestClient(app)


def test_create_job_returns_valid_job_response(client, db_session):
    payload = {
        "source_video_url": "file:///tmp/fake_movie.mp4",
        "total_duration_seconds": 600.0,
        "segment_count": 4,
    }

    resp = client.post("/jobs", json=payload)

    assert resp.status_code in (200, 201)
    body = resp.json()
    assert body["source_video_url"] == payload["source_video_url"]
    assert body["total_segments"] == payload["segment_count"]
    assert body["sla_target_seconds"] > 0
    # Valid UUID and a status string (JobStatus enum value).
    uuid.UUID(body["id"])
    assert isinstance(body["status"], str) and body["status"]


def test_create_job_derives_segment_count_from_segment_duration(client, db_session):
    payload = {
        "source_video_url": "file:///tmp/fake_movie_duration.mp4",
        "total_duration_seconds": 700.0,
        "segment_duration_seconds": 120.0,
    }

    resp = client.post("/jobs", json=payload)

    assert resp.status_code in (200, 201)
    body = resp.json()
    # ceil(700/120) == 6 — see TestComputeSegmentCountFromDuration in
    # tests/test_splitter.py for the underlying unit-level assertions.
    assert body["total_segments"] == 6


def test_segment_duration_seconds_takes_priority_over_segment_count(client, db_session):
    payload = {
        "source_video_url": "file:///tmp/fake_movie_priority.mp4",
        "total_duration_seconds": 700.0,
        "segment_count": 4,
        "segment_duration_seconds": 120.0,
    }

    resp = client.post("/jobs", json=payload)

    assert resp.status_code in (200, 201)
    assert resp.json()["total_segments"] == 6


def test_zero_segment_duration_seconds_rejected_by_schema(client, db_session):
    # gt=0 on the Pydantic field rejects this before it ever reaches
    # orchestrator.splitter.compute_segment_count_from_duration.
    payload = {
        "source_video_url": "file:///tmp/fake_movie_bad_duration.mp4",
        "total_duration_seconds": 700.0,
        "segment_duration_seconds": 0.0,
    }

    resp = client.post("/jobs", json=payload)

    assert resp.status_code == 422


def test_negative_total_duration_seconds_returns_clean_422_not_500(client, db_session):
    # total_duration_seconds has no Pydantic-level positivity constraint (it
    # can legitimately arrive via an external probe, not just the request
    # body) — this exercises api.main.submit_job's own try/except around the
    # splitter calls, not schema validation.
    payload = {
        "source_video_url": "file:///tmp/fake_movie_negative_duration.mp4",
        "total_duration_seconds": -10.0,
        "segment_duration_seconds": 60.0,
    }

    resp = client.post("/jobs", json=payload)

    assert resp.status_code == 422


def test_get_job_unknown_id_returns_404(client, db_session):
    resp = client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_job_by_id_after_create(client, db_session):
    payload = {
        "source_video_url": "file:///tmp/fake_movie_2.mp4",
        "total_duration_seconds": 400.0,
        "segment_count": 2,
    }
    create_resp = client.post("/jobs", json=payload)
    assert create_resp.status_code in (200, 201)
    job_id = create_resp.json()["id"]

    resp = client.get(f"/jobs/{job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == job_id
    assert body["total_segments"] == payload["segment_count"]


def test_get_job_results_empty_for_new_job(client, db_session):
    payload = {
        "source_video_url": "file:///tmp/fake_movie_3.mp4",
        "total_duration_seconds": 300.0,
        "segment_count": 2,
    }
    create_resp = client.post("/jobs", json=payload)
    assert create_resp.status_code in (200, 201)
    job_id = create_resp.json()["id"]

    resp = client.get(f"/jobs/{job_id}/results")

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["results"] == []
