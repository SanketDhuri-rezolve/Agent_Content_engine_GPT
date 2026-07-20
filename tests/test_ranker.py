"""Unit tests for workers.ranker.logic.mmr_rerank — pure logic, no DB, no
Celery. Spans are hand-built dicts shaped like models.schemas.ReducedSpan.
"""

import uuid

import pytest

from workers.ranker.logic import _cosine_similarity, mmr_rerank


def _span(
    *,
    normalized_score: float,
    feature_vector: dict,
    start_ts: float = 0.0,
    end_ts: float = 1.0,
    segment_id: str | None = None,
    raw_score: float | None = None,
) -> dict:
    """Builds a ReducedSpan-shaped dict (ScoredSpanPayload + normalized_score)."""
    return {
        "segment_id": segment_id or str(uuid.uuid4()),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "transcript_excerpt": "some line of dialogue",
        "feature_vector": feature_vector,
        "touches_boundary": False,
        "raw_score": raw_score if raw_score is not None else normalized_score,
        "justification": "test fixture",
        "llm_model_version": "mock-scorer-v0",
        "normalized_score": normalized_score,
    }


class TestMmrRerankBasics:
    def test_empty_spans_returns_empty_list(self):
        assert mmr_rerank([], top_k=5) == []

    def test_zero_top_k_returns_empty_list(self):
        spans = [_span(normalized_score=0.9, feature_vector={"x": 1.0})]
        assert mmr_rerank(spans, top_k=0) == []

    def test_negative_top_k_returns_empty_list(self):
        spans = [_span(normalized_score=0.9, feature_vector={"x": 1.0})]
        assert mmr_rerank(spans, top_k=-1) == []

    def test_does_not_mutate_input(self):
        spans = [
            _span(normalized_score=0.9, feature_vector={"x": 1.0}, start_ts=0.0, end_ts=1.0),
            _span(normalized_score=0.5, feature_vector={"y": 1.0}, start_ts=2.0, end_ts=3.0),
        ]
        original_len = len(spans)
        original_keys = [set(s.keys()) for s in spans]

        mmr_rerank(spans, top_k=2)

        assert len(spans) == original_len
        assert [set(s.keys()) for s in spans] == original_keys
        assert "final_score" not in spans[0]
        assert "final_score" not in spans[1]


class TestTopKTrimming:
    def test_top_k_smaller_than_input_length_trims_correctly(self):
        spans = [
            _span(normalized_score=score, feature_vector={f"dim{i}": 1.0}, start_ts=float(i), end_ts=float(i) + 1)
            for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])
        ]

        result = mmr_rerank(spans, top_k=2)

        assert len(result) == 2
        # Every returned item is augmented with a final_score.
        assert all("final_score" in span for span in result)

    def test_top_k_larger_than_input_returns_all_spans(self):
        spans = [
            _span(normalized_score=0.9, feature_vector={"a": 1.0}, start_ts=0.0, end_ts=1.0),
            _span(normalized_score=0.5, feature_vector={"b": 1.0}, start_ts=1.0, end_ts=2.0),
        ]

        result = mmr_rerank(spans, top_k=10)

        assert len(result) == len(spans)


class TestMmrDiversity:
    def test_near_duplicate_spans_are_not_both_picked_near_top(self):
        """Naive top-k-by-score would pick A and B (the two highest scores,
        which happen to be near-duplicates with an identical feature_vector).
        MMR should instead trade B out for the lower-scoring but distinct C,
        proving diversity actually changes the outcome."""
        span_a = _span(
            normalized_score=0.95, feature_vector={"x": 1.0, "y": 0.0}, start_ts=0.0, end_ts=5.0
        )
        span_b_near_duplicate = _span(
            normalized_score=0.90, feature_vector={"x": 1.0, "y": 0.0}, start_ts=5.5, end_ts=10.0
        )
        span_c_distinct = _span(
            normalized_score=0.60, feature_vector={"x": 0.0, "y": 1.0}, start_ts=50.0, end_ts=55.0
        )
        spans = [span_a, span_b_near_duplicate, span_c_distinct]

        # Sanity check: naive top-2-by-score would pick A and B.
        naive_top_2 = sorted(spans, key=lambda s: s["normalized_score"], reverse=True)[:2]
        assert naive_top_2 == [span_a, span_b_near_duplicate]

        result = mmr_rerank(spans, top_k=2, lambda_param=0.7)

        assert len(result) == 2
        result_start_ts = {span["start_ts"] for span in result}
        assert span_a["start_ts"] in result_start_ts
        assert span_c_distinct["start_ts"] in result_start_ts
        assert span_b_near_duplicate["start_ts"] not in result_start_ts

    def test_first_pick_is_always_highest_relevance(self):
        span_a = _span(normalized_score=0.95, feature_vector={"x": 1.0}, start_ts=0.0, end_ts=1.0)
        span_b = _span(normalized_score=0.5, feature_vector={"y": 1.0}, start_ts=1.0, end_ts=2.0)

        result = mmr_rerank([span_b, span_a], top_k=2, lambda_param=0.7)

        assert result[0]["start_ts"] == span_a["start_ts"]
        # First pick has no already-selected spans to penalize against, so
        # its final_score is exactly lambda_param * normalized_score.
        assert result[0]["final_score"] == pytest.approx(0.7 * 0.95)

    def test_identical_vectors_are_maximally_similar_and_get_penalized(self):
        span_a = _span(normalized_score=0.9, feature_vector={"x": 1.0, "y": 2.0}, start_ts=0.0, end_ts=1.0)
        span_b = _span(normalized_score=0.9, feature_vector={"x": 1.0, "y": 2.0}, start_ts=1.0, end_ts=2.0)

        result = mmr_rerank([span_a, span_b], top_k=2, lambda_param=0.5)

        # Second pick is penalized by full similarity (1.0) against the first.
        assert result[1]["final_score"] == pytest.approx(0.5 * 0.9 - 0.5 * 1.0)


class TestCosineSimilarityHelper:
    def test_disjoint_keys_are_treated_as_zero_similarity(self):
        assert _cosine_similarity({"x": 1.0}, {"z": 1.0}) == 0.0

    def test_identical_vectors_have_similarity_one(self):
        assert _cosine_similarity({"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 2.0}) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self):
        assert _cosine_similarity({"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}) == pytest.approx(0.0)

    def test_empty_vectors_do_not_raise(self):
        assert _cosine_similarity({}, {"x": 1.0}) == 0.0
        assert _cosine_similarity({}, {}) == 0.0

    def test_non_scalar_entries_are_ignored_not_crashed_on(self):
        # workers.span_builder.logic.build_candidate_spans's real
        # feature_vector output includes non-scalar entries alongside scalar
        # ones (keyframe_refs: list[str], visual_embedding_mean: list[float])
        # — regression test for a live TypeError: float() argument must be a
        # string or a real number, not 'list' that crashed the whole
        # rank_and_persist task against real (non-test-fixture) span data.
        vector_a = {"shot_count": 3.0, "keyframe_refs": ["kf_0", "kf_1"], "visual_embedding_mean": [0.1, 0.2, 0.3]}
        vector_b = {"shot_count": 3.0, "keyframe_refs": ["kf_5", "kf_6"], "visual_embedding_mean": [0.9, 0.8, 0.7]}

        assert _cosine_similarity(vector_a, vector_b) == pytest.approx(1.0)

    def test_purely_non_scalar_vectors_have_zero_similarity(self):
        vector_a = {"keyframe_refs": ["kf_0"]}
        vector_b = {"keyframe_refs": ["kf_1"]}

        assert _cosine_similarity(vector_a, vector_b) == 0.0
