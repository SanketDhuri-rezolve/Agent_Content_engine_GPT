"""Unit tests for workers.scorer — pure logic + MockScorer, no DB/network."""

import uuid

from models.enums import SegmentStatus
from workers.scorer.adapters import MockScorer, ScorerError
from workers.scorer.tasks import score_candidates_concurrently, score_spans


def _candidate_span(segment_id, start_ts=10.0, end_ts=20.0, excerpt="a great line of dialogue"):
    return {
        "segment_id": str(segment_id),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "transcript_excerpt": excerpt,
        "feature_vector": {"motion": 0.5},
        "touches_boundary": False,
    }


def test_mock_scorer_is_deterministic():
    scorer = MockScorer()
    span = _candidate_span(uuid.uuid4())

    score_a, justification_a, model_a, clip_url_a, rich_data_a = scorer.score(span)
    score_b, justification_b, model_b, clip_url_b, rich_data_b = scorer.score(span)

    assert score_a == score_b
    assert justification_a == justification_b
    assert model_a == model_b
    assert clip_url_a is None
    assert clip_url_b is None
    assert rich_data_a == rich_data_b
    assert rich_data_a["moment_id"]
    assert 0.0 <= score_a <= 1.0


def test_score_candidates_concurrently_preserves_input_order():
    """Concurrency fix (score_candidates_concurrently, used by both
    score_spans and workers.global_selector.tasks.select_and_score_job)
    must return results indexed by ORIGINAL position, not completion order —
    downstream code assumes span_index in the returned tuples lines up with
    the input list it submitted."""
    segment_id = uuid.uuid4()
    spans = [_candidate_span(segment_id, start_ts=float(i), end_ts=float(i) + 1, excerpt=f"span-{i}") for i in range(8)]

    results = score_candidates_concurrently(MockScorer(), spans, max_concurrent_requests=4)

    assert [span_index for span_index, _, _ in results] == list(range(8))
    for span_index, scored, error in results:
        assert error is None
        assert scored is not None
        assert scored.transcript_excerpt == f"span-{span_index}"


def test_score_candidates_concurrently_reports_per_span_failures():
    good = _candidate_span(uuid.uuid4(), excerpt="fine")
    bad = _candidate_span(uuid.uuid4(), excerpt="boom")

    def flaky_score(self, span):
        if span["transcript_excerpt"] == "boom":
            raise ScorerError("injected failure for test")
        return MockScorer.score(self, span)

    scorer = MockScorer()
    scorer.score = flaky_score.__get__(scorer, MockScorer)

    results = score_candidates_concurrently(scorer, [good, bad], max_concurrent_requests=2)

    assert results[0][1] is not None and results[0][2] is None
    assert results[1][1] is None and results[1][2] == "injected failure for test"


def test_score_spans_produces_one_scored_span_per_candidate():
    segment_id = uuid.uuid4()
    span_builder_output = {
        "segment_id": str(segment_id),
        "sequence_index": 2,
        "status": SegmentStatus.completed.value,
        "candidate_spans": [
            _candidate_span(segment_id, start_ts=0.0, end_ts=5.0, excerpt="short"),
            _candidate_span(
                segment_id, start_ts=5.0, end_ts=12.0, excerpt="a much longer line of dialogue here"
            ),
        ],
        "error": None,
    }

    result = score_spans(span_builder_output)

    assert result["segment_id"] == str(segment_id)
    assert result["sequence_index"] == 2
    assert result["status"] == SegmentStatus.completed.value
    assert result["error"] is None
    assert len(result["scored_spans"]) == 2
    for scored_span in result["scored_spans"]:
        assert scored_span["llm_model_version"]
        assert isinstance(scored_span["raw_score"], float)


def test_score_spans_with_empty_candidates_returns_empty_scored_spans():
    segment_id = uuid.uuid4()
    span_builder_output = {
        "segment_id": str(segment_id),
        "sequence_index": 0,
        "status": SegmentStatus.completed.value,
        "candidate_spans": [],
        "error": None,
    }

    result = score_spans(span_builder_output)

    assert result["status"] == SegmentStatus.completed.value
    assert result["scored_spans"] == []
    assert result["error"] is None


def test_score_spans_propagates_upstream_failure_without_scoring():
    segment_id = uuid.uuid4()
    span_builder_output = {
        "segment_id": str(segment_id),
        "sequence_index": 1,
        "status": SegmentStatus.failed.value,
        "candidate_spans": [],
        "error": "span_builder blew up",
    }

    result = score_spans(span_builder_output)

    assert result["status"] == SegmentStatus.failed.value
    assert result["error"] == "span_builder blew up"
    assert result["scored_spans"] == []


def test_partial_span_scoring_failure_degrades_gracefully(monkeypatch):
    """Fail-soft policy: one bad span among several does NOT fail the
    segment — it's dropped and logged, the segment ships with fewer
    scored_spans than candidate_spans."""
    segment_id = uuid.uuid4()
    good_span = _candidate_span(segment_id, start_ts=1.0, end_ts=2.0, excerpt="fine")
    bad_span = _candidate_span(segment_id, start_ts=3.0, end_ts=4.0, excerpt="boom")

    real_score = MockScorer.score

    def flaky_score(self, span):
        if span["transcript_excerpt"] == "boom":
            raise ScorerError("injected failure for test")
        return real_score(self, span)

    monkeypatch.setattr(MockScorer, "score", flaky_score)

    span_builder_output = {
        "segment_id": str(segment_id),
        "sequence_index": 3,
        "status": SegmentStatus.completed.value,
        "candidate_spans": [good_span, bad_span],
        "error": None,
    }

    result = score_spans(span_builder_output)

    assert result["status"] == SegmentStatus.completed.value
    assert result["error"] is None
    assert len(result["scored_spans"]) == 1
    assert result["scored_spans"][0]["transcript_excerpt"] == "fine"


def test_score_spans_propagates_upstream_timeout_without_scoring():
    """Step 4 regression test: a segment_worker SoftTimeLimitExceeded produces
    SegmentStatus.timeout, propagated through span_builder with empty
    candidate_spans (see workers.span_builder.tasks._FAILED_UPSTREAM_STATUSES).
    Before this fix, score_spans only checked for status=="failed" — a
    timeout fell through to the "empty candidates is fine" path and silently
    came out as status=="completed", which a live Step 4 test caught (the
    DB ended up saying a timed-out segment succeeded)."""
    segment_id = uuid.uuid4()
    span_builder_output = {
        "segment_id": str(segment_id),
        "sequence_index": 1,
        "status": SegmentStatus.timeout.value,
        "candidate_spans": [],
        "error": "segment_worker soft time limit (5.0s) exceeded",
    }

    result = score_spans(span_builder_output)

    assert result["status"] == SegmentStatus.timeout.value
    assert result["error"] == "segment_worker soft time limit (5.0s) exceeded"
    assert result["scored_spans"] == []


def test_all_spans_failing_degrades_segment_to_failed(monkeypatch):
    """Fail-soft policy edge case: if EVERY candidate span fails to score,
    there is nothing useful for the reducer to consume, so the segment as a
    whole is degraded to `failed` rather than silently shipping an empty
    scored_spans list that would look identical to "no candidates found"."""
    segment_id = uuid.uuid4()
    bad_span = _candidate_span(segment_id, start_ts=3.0, end_ts=4.0, excerpt="boom")

    def always_fail(self, span):
        raise ScorerError("injected failure for test")

    monkeypatch.setattr(MockScorer, "score", always_fail)

    span_builder_output = {
        "segment_id": str(segment_id),
        "sequence_index": 4,
        "status": SegmentStatus.completed.value,
        "candidate_spans": [bad_span],
        "error": None,
    }

    result = score_spans(span_builder_output)

    assert result["status"] == SegmentStatus.failed.value
    assert result["scored_spans"] == []
    assert result["error"] is not None
