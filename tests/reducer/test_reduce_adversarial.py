"""Adversarial tests for workers.reducer — written to actively try to BREAK
`workers/reducer/logic.py` / `workers/reducer/tasks.py` beyond the cases the
first pass (tests/reducer/test_reduce_logic.py) already covers.

Two real bugs were found and FIXED in workers/reducer/logic.py while writing
this file (not just documented as failing tests) — see the module docstrings
in workers/reducer/logic.py for the full "why", and the final adversarial
report for a summary:

1. `dedupe_boundary_spans` used a single mutable "current absorbs the next
   matching span" pairwise accumulator. For a 3-(or-more)-segment chain of
   genuine duplicates (A~B adjacent+overlapping, B~C adjacent+overlapping,
   but A/C not themselves adjacent), whether the chain fully collapsed to
   one surviving span depended on which span happened to win the FIRST
   pairwise raw_score comparison — order-dependent data loss (a real
   duplicate highlight could leak through to the final ranked output).
   Fixed with a union-find transitive-closure grouping instead.

2. The "minimal shape guard" in `reduce_segment_results` only checked for
   *key presence* (`"raw_score" not in raw_span`), not for the value
   actually being a finite number. A scored span with `raw_score=None` (or
   a non-numeric string) crashed `normalize_scores`'s `float(...)` call
   with an unhandled TypeError/ValueError, which `reduce_job` does catch —
   but that turns ONE malformed span into a hard failure for the ENTIRE
   job, defeating the documented "skip a malformed span, don't fail the
   whole reduce" intent. Worse, `raw_score=nan`/`inf` didn't crash at all —
   it silently poisoned the population mean/std, producing
   `normalized_score = nan` for EVERY span in the job (see
   `test_normalize_scores_...` below), i.e. a "confident, complete-looking"
   ranking built on garbage. Fixed by validating finiteness, not just
   presence.

A third, narrower numerical-soundness gap was also fixed in
`normalize_scores` directly: even with every individual `raw_score` finite,
sufficiently extreme magnitudes can overflow the intermediate mean/variance
sums to +/-inf, which then yields NaN in the z-score formula. Guarded with a
per-span finite-or-neutral-0.0 fallback (see test below).
"""

import math
import uuid

import pytest

from models.enums import SegmentStatus
from workers.reducer.logic import (
    InsufficientSegmentsError,
    classify_segments,
    dedupe_boundary_spans,
    normalize_scores,
    reduce_segment_results,
)


def _scored_span(
    segment_id,
    *,
    start_ts: float = 0.0,
    end_ts: float = 1.0,
    raw_score: float = 0.5,
    touches_boundary: bool = False,
    excerpt: str = "a line of dialogue",
    **overrides,
) -> dict:
    span = {
        "segment_id": str(segment_id),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "transcript_excerpt": excerpt,
        "feature_vector": {},
        "touches_boundary": touches_boundary,
        "raw_score": raw_score,
        "justification": "test fixture",
        "llm_model_version": "mock-scorer-v0",
    }
    span.update(overrides)
    return span


def _segment_result(
    segment_id,
    sequence_index: int,
    *,
    status: str = SegmentStatus.completed.value,
    scored_spans: list[dict] | None = None,
    error: str | None = None,
) -> dict:
    return {
        "segment_id": str(segment_id),
        "sequence_index": sequence_index,
        "status": status,
        "scored_spans": scored_spans or [],
        "error": error,
    }


# ---------------------------------------------------------------------------
# BUG #1 (found + fixed): transitive chain of >2 boundary duplicates
# ---------------------------------------------------------------------------


class TestTransitiveDuplicateChains:
    """A single real-world moment can, in principle, sit close enough to
    BOTH neighbors of a middle segment to be independently detected (and
    marked touches_boundary=True) by all three of segments 0, 1, and 2. All
    three should collapse to ONE surviving span, regardless of which one
    happens to have the highest raw_score."""

    def _three_way_chain(self, winner_position: int):
        seg0, seg1, seg2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        scores = [0.3, 0.3, 0.3]
        scores[winner_position] = 0.9
        span_a = _scored_span(
            seg0, start_ts=95.0, end_ts=105.0, raw_score=scores[0],
            touches_boundary=True, excerpt="alpha",
        )
        span_b = _scored_span(
            seg1, start_ts=96.0, end_ts=104.0, raw_score=scores[1],
            touches_boundary=True, excerpt="bravo",
        )
        span_c = _scored_span(
            seg2, start_ts=97.0, end_ts=103.0, raw_score=scores[2],
            touches_boundary=True, excerpt="charlie",
        )
        seq_map = {str(seg0): 0, str(seg1): 1, str(seg2): 2}
        return [span_a, span_b, span_c], seq_map

    def test_chain_collapses_when_outer_span_wins(self):
        # This is the exact scenario the original pairwise-accumulator
        # implementation got wrong: A (segment 0, an OUTER member of the
        # chain) wins the A-vs-B duel, so the pre-fix "current" identity
        # became A's segment_id (sequence_index 0) — not adjacent to C
        # (sequence_index 2) — silently leaving C undeduped.
        spans, seq_map = self._three_way_chain(winner_position=0)
        surviving, dropped = dedupe_boundary_spans(spans, seq_map)
        assert dropped == 2
        assert len(surviving) == 1
        assert surviving[0]["raw_score"] == 0.9

    def test_chain_collapses_when_middle_span_wins(self):
        spans, seq_map = self._three_way_chain(winner_position=1)
        surviving, dropped = dedupe_boundary_spans(spans, seq_map)
        assert dropped == 2
        assert len(surviving) == 1
        assert surviving[0]["raw_score"] == 0.9

    def test_chain_collapses_when_other_outer_span_wins(self):
        spans, seq_map = self._three_way_chain(winner_position=2)
        surviving, dropped = dedupe_boundary_spans(spans, seq_map)
        assert dropped == 2
        assert len(surviving) == 1
        assert surviving[0]["raw_score"] == 0.9

    def test_chain_merges_transcript_from_all_members_not_just_two(self):
        spans, seq_map = self._three_way_chain(winner_position=0)
        surviving, _ = dedupe_boundary_spans(spans, seq_map)
        excerpt = surviving[0]["transcript_excerpt"]
        assert "alpha" in excerpt
        assert "bravo" in excerpt
        assert "charlie" in excerpt

    def test_chain_collapses_end_to_end_through_reduce_segment_results(self):
        seg0, seg1, seg2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(
                seg0, 0,
                scored_spans=[_scored_span(seg0, start_ts=95.0, end_ts=105.0, raw_score=0.9, touches_boundary=True)],
            ),
            _segment_result(
                seg1, 1,
                scored_spans=[_scored_span(seg1, start_ts=96.0, end_ts=104.0, raw_score=0.3, touches_boundary=True)],
            ),
            _segment_result(
                seg2, 2,
                scored_spans=[_scored_span(seg2, start_ts=97.0, end_ts=103.0, raw_score=0.3, touches_boundary=True)],
            ),
        ]

        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        assert len(output["spans"]) == 1
        assert output["dropped_duplicate_count"] == 2

    def test_four_way_chain_across_four_segments_collapses_to_one(self):
        """Stress the transitive-closure grouping beyond 3 members."""
        segs = [uuid.uuid4() for _ in range(4)]
        seq_map = {str(s): i for i, s in enumerate(segs)}
        spans = [
            _scored_span(segs[i], start_ts=100.0 + i, end_ts=110.0 + i, raw_score=0.1 * (i + 1), touches_boundary=True)
            for i in range(4)
        ]
        # Each adjacent pair [i, i+1] overlaps heavily: windows are
        # [100,110],[101,111],[102,112],[103,113] -> every adjacent pair has
        # intersection=9, union=10 -> IoU=0.9 > 0.5.
        surviving, dropped = dedupe_boundary_spans(spans, seq_map)
        assert dropped == 3
        assert len(surviving) == 1
        # Highest raw_score (segs[3], score 0.4) wins.
        assert surviving[0]["raw_score"] == pytest.approx(0.4)

    def test_two_separate_duplicate_pairs_are_not_conflated_with_each_other(self):
        """A wider adversarial check for dropped_duplicate_count/grouping
        correctness: two independent duplicate PAIRS (not one big chain)
        among 4 segments must produce two independent collapses, not get
        merged into a single group or miscounted."""
        segs = [uuid.uuid4() for _ in range(4)]
        seq_map = {str(s): i for i, s in enumerate(segs)}
        # Pair 1: segments 0-1 overlap heavily.
        pair1_a = _scored_span(segs[0], start_ts=0.0, end_ts=10.0, raw_score=0.9, touches_boundary=True, excerpt="p1a")
        pair1_b = _scored_span(segs[1], start_ts=1.0, end_ts=9.0, raw_score=0.2, touches_boundary=True, excerpt="p1b")
        # Segment 1-2 do NOT overlap (distinct moments) -> no bridge between
        # the two pairs.
        # Pair 2: segments 2-3 overlap heavily.
        pair2_a = _scored_span(segs[2], start_ts=500.0, end_ts=510.0, raw_score=0.7, touches_boundary=True, excerpt="p2a")
        pair2_b = _scored_span(segs[3], start_ts=501.0, end_ts=509.0, raw_score=0.1, touches_boundary=True, excerpt="p2b")

        surviving, dropped = dedupe_boundary_spans(
            [pair1_a, pair1_b, pair2_a, pair2_b], seq_map
        )

        assert dropped == 2
        assert len(surviving) == 2
        raw_scores = sorted(s["raw_score"] for s in surviving)
        assert raw_scores == [0.7, 0.9]


# ---------------------------------------------------------------------------
# BUG #2 (found + fixed): malformed raw_score/start_ts/end_ts values
# ---------------------------------------------------------------------------


class TestMalformedSpanValuesDoNotCrashOrPoisonTheJob:
    def test_none_raw_score_span_is_skipped_not_a_hard_failure(self):
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(
                seg_a, 0,
                scored_spans=[_scored_span(seg_a, start_ts=0.0, end_ts=10.0, raw_score=None)],
            ),
            _segment_result(
                seg_b, 1,
                scored_spans=[_scored_span(seg_b, start_ts=100.0, end_ts=110.0, raw_score=0.5)],
            ),
        ]

        # Must not raise (pre-fix: float(None) -> TypeError inside
        # normalize_scores, propagated as an unhandled exception from
        # reduce_segment_results).
        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        assert len(output["spans"]) == 1
        assert output["spans"][0]["raw_score"] == 0.5

    def test_nan_raw_score_does_not_poison_every_spans_normalized_score(self):
        """Before the fix: a single NaN raw_score silently made
        normalize_scores emit NaN for EVERY span in the job (not just the
        bad one), because NaN poisons the shared population mean/std. The
        malformed span must instead be excluded before normalization ever
        runs."""
        seg_a, seg_b, seg_c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(
                seg_a, 0,
                scored_spans=[_scored_span(seg_a, start_ts=0.0, end_ts=10.0, raw_score=float("nan"))],
            ),
            _segment_result(
                seg_b, 1,
                scored_spans=[_scored_span(seg_b, start_ts=100.0, end_ts=110.0, raw_score=0.4)],
            ),
            _segment_result(
                seg_c, 2,
                scored_spans=[_scored_span(seg_c, start_ts=200.0, end_ts=210.0, raw_score=0.8)],
            ),
        ]

        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        assert len(output["spans"]) == 2
        for span in output["spans"]:
            assert math.isfinite(span["normalized_score"])
            assert not math.isnan(span["normalized_score"])

    def test_infinite_raw_score_span_is_skipped(self):
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(
                seg_a, 0,
                scored_spans=[_scored_span(seg_a, start_ts=0.0, end_ts=10.0, raw_score=float("inf"))],
            ),
            _segment_result(
                seg_b, 1,
                scored_spans=[_scored_span(seg_b, start_ts=100.0, end_ts=110.0, raw_score=0.6)],
            ),
        ]

        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        assert len(output["spans"]) == 1
        assert math.isfinite(output["spans"][0]["normalized_score"])

    def test_non_numeric_string_raw_score_is_skipped_not_fatal(self):
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(
                seg_a, 0,
                scored_spans=[_scored_span(seg_a, start_ts=0.0, end_ts=10.0, raw_score="not-a-number")],
            ),
            _segment_result(
                seg_b, 1,
                scored_spans=[_scored_span(seg_b, start_ts=100.0, end_ts=110.0, raw_score=0.6)],
            ),
        ]

        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        assert len(output["spans"]) == 1

    def test_none_start_ts_span_is_skipped_not_fatal(self):
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(
                seg_a, 0,
                scored_spans=[_scored_span(seg_a, start_ts=None, end_ts=10.0, raw_score=0.5)],
            ),
            _segment_result(
                seg_b, 1,
                scored_spans=[_scored_span(seg_b, start_ts=100.0, end_ts=110.0, raw_score=0.6)],
            ),
        ]

        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        assert len(output["spans"]) == 1

    def test_segment_with_only_malformed_spans_is_still_usable_not_degraded(self):
        """status=='completed' is the ONLY thing that makes a segment
        usable — a completed segment whose every span turns out to be
        unreducible garbage still is not "degraded" (that label is reserved
        for status != completed). It simply contributes zero spans."""
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(
                seg_a, 0,
                scored_spans=[_scored_span(seg_a, start_ts=0.0, end_ts=10.0, raw_score=float("nan"))],
            ),
            _segment_result(
                seg_b, 1,
                scored_spans=[_scored_span(seg_b, start_ts=100.0, end_ts=110.0, raw_score=0.5)],
            ),
        ]

        usable, degraded_ids, degraded_labels = classify_segments(results, min_required_fraction=0.6)

        assert len(usable) == 2
        assert degraded_ids == []
        assert degraded_labels == []


# ---------------------------------------------------------------------------
# Numerical soundness: normalized_score must never be NaN/inf
# ---------------------------------------------------------------------------


class TestNormalizedScoreNeverNanOrInf:
    def test_extreme_magnitude_finite_raw_scores_do_not_overflow_to_nan(self):
        """Every individual raw_score below is a genuine finite float, but
        their sum/sum-of-squares (mean/variance) can overflow float range,
        which (pre-fix) yielded `inf - inf -> nan` in the z-score formula."""
        spans = [
            _scored_span(uuid.uuid4(), start_ts=0.0, end_ts=1.0, raw_score=1e308),
            _scored_span(uuid.uuid4(), start_ts=1.0, end_ts=2.0, raw_score=1e308),
            _scored_span(uuid.uuid4(), start_ts=2.0, end_ts=3.0, raw_score=-1e308),
        ]

        normalized = normalize_scores(spans)

        for span in normalized:
            assert math.isfinite(span["normalized_score"])

    def test_normal_case_still_produces_real_nonzero_normalized_scores(self):
        """Guard against over-correcting: the finite-fallback must not
        accidentally zero out legitimate, well-behaved normalized scores."""
        spans = [
            _scored_span(uuid.uuid4(), raw_score=0.0),
            _scored_span(uuid.uuid4(), raw_score=1.0),
            _scored_span(uuid.uuid4(), raw_score=2.0),
        ]

        normalized = normalize_scores(spans)
        scores = [s["normalized_score"] for s in normalized]

        assert scores[0] < 0
        assert scores[1] == pytest.approx(0.0, abs=1e-9)
        assert scores[2] > 0


# ---------------------------------------------------------------------------
# classify_segments: exactly one segment, and unrecognized chord shapes
# ---------------------------------------------------------------------------


class TestSingleSegmentAndMalformedChordEntries:
    def test_exactly_one_completed_segment_is_usable_and_sufficient(self):
        seg = uuid.uuid4()
        results = [_segment_result(seg, 0, scored_spans=[_scored_span(seg, start_ts=0.0, end_ts=10.0, raw_score=0.5)])]

        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        assert len(output["spans"]) == 1
        assert output["dropped_duplicate_count"] == 0
        assert output["degraded_segment_ids"] == []

    def test_exactly_one_non_completed_segment_raises(self):
        seg = uuid.uuid4()
        results = [_segment_result(seg, 0, status=SegmentStatus.failed.value, scored_spans=[], error="boom")]

        with pytest.raises(InsufficientSegmentsError):
            classify_segments(results, min_required_fraction=0.6)

    def test_non_dict_non_none_chord_entry_is_treated_as_degraded_not_crashed(self):
        """A chord entry that is neither a dict nor None nor iterable into a
        dict (e.g. a bare int/str from some serialization mishap) must be
        defensively absorbed, not raise from inside classify_segments."""
        seg = uuid.uuid4()
        results: list = [
            _segment_result(seg, 0, scored_spans=[_scored_span(seg, start_ts=0.0, end_ts=10.0)]),
            12345,  # nonsensical chord entry shape
            "also nonsensical",
        ]

        usable, degraded_ids, degraded_labels = classify_segments(results, min_required_fraction=0.3)

        assert len(usable) == 1
        assert len(degraded_labels) == 2
        assert any("unrecognized chord result shape" in label for label in degraded_labels)

    def test_malformed_segment_id_never_crashes_degraded_id_recovery(self):
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        results = [
            _segment_result(seg_a, 0, scored_spans=[_scored_span(seg_a, start_ts=0.0, end_ts=10.0)]),
            _segment_result(seg_b, 1, scored_spans=[_scored_span(seg_b, start_ts=100.0, end_ts=110.0)]),
            {
                "segment_id": "not-a-valid-uuid",
                "sequence_index": 2,
                "status": SegmentStatus.failed.value,
                "scored_spans": [],
                "error": "boom",
            },
        ]

        usable, degraded_ids, degraded_labels = classify_segments(results, min_required_fraction=0.5)

        assert len(usable) == 2
        # The malformed id is never recoverable into degraded_segment_ids
        # (that field is typed as a list of real UUIDs downstream), but it
        # must still show up in the human-readable labels for logging.
        assert degraded_ids == []
        assert any("sequence_index=2" in label for label in degraded_labels)


# ---------------------------------------------------------------------------
# IoU float-precision sensitivity right at the dedup threshold
# ---------------------------------------------------------------------------


class TestIouFloatPrecisionAtTheThreshold:
    def test_just_above_half_iou_is_merged(self):
        """Complements the existing exactly-0.5 (not merged) boundary test:
        confirms the strict `>` comparison actually DOES merge once IoU
        clears 0.5, even by a tiny margin, so the boundary is a true
        two-sided cutoff and not accidentally off in one direction."""
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        # [0, 20] and [0, 10.0001]: intersection=10.0001, union=20 -> IoU
        # just over 0.5.
        span_a = _scored_span(seg_a, start_ts=0.0, end_ts=20.0, touches_boundary=True)
        span_b = _scored_span(seg_b, start_ts=0.0, end_ts=10.0001, touches_boundary=True)

        surviving, dropped = dedupe_boundary_spans(
            [span_a, span_b], {str(seg_a): 0, str(seg_b): 1}
        )

        assert dropped == 1
        assert len(surviving) == 1

    def test_float_rounding_noise_near_the_boundary_is_handled_deterministically(self):
        """Windows engineered so that plain decimal arithmetic would land
        exactly on 0.5 IoU, but IEEE-754 float subtraction actually produces
        a value fractionally below 0.5 (0.1 is not exactly representable).
        This isn't "fixable" (it's inherent float representation, not a
        reducer bug) but it must resolve deterministically and never raise
        — pinning the actual behavior here catches any future regression
        that makes this flaky."""
        seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
        span_a = _scored_span(seg_a, start_ts=0.0, end_ts=0.3, touches_boundary=True)
        span_b = _scored_span(seg_b, start_ts=0.1, end_ts=0.4, touches_boundary=True)

        # Documented actual float behavior (run deterministically, not
        # asserted as "correct" either way — see docstring above).
        intersection = min(0.3, 0.4) - max(0.0, 0.1)
        union = max(0.3, 0.4) - min(0.0, 0.1)
        expected_iou = intersection / union

        surviving, dropped = dedupe_boundary_spans(
            [span_a, span_b], {str(seg_a): 0, str(seg_b): 1}
        )

        if expected_iou > 0.5:
            assert dropped == 1 and len(surviving) == 1
        else:
            assert dropped == 0 and len(surviving) == 2


# ---------------------------------------------------------------------------
# Multi-duplicate + multi-degraded combined accounting correctness
# ---------------------------------------------------------------------------


class TestCombinedDegradedAndDuplicateAccounting:
    def test_counts_are_exact_not_just_present(self):
        """6 segments: 2 degraded (failed/timeout), 4 completed containing
        two SEPARATE duplicate boundary pairs among otherwise-unique spans.
        Asserts exact values, not just "something got dropped/degraded"."""
        segs = [uuid.uuid4() for _ in range(6)]

        results = [
            _segment_result(segs[0], 0, scored_spans=[
                _scored_span(segs[0], start_ts=0.0, end_ts=10.0, raw_score=0.9, touches_boundary=True, excerpt="dup1-a"),
            ]),
            _segment_result(segs[1], 1, scored_spans=[
                _scored_span(segs[1], start_ts=1.0, end_ts=9.0, raw_score=0.2, touches_boundary=True, excerpt="dup1-b"),
                _scored_span(segs[1], start_ts=500.0, end_ts=510.0, raw_score=0.4, touches_boundary=False, excerpt="unique1"),
            ]),
            _segment_result(segs[2], 2, status=SegmentStatus.failed.value, scored_spans=[], error="scorer_timeout"),
            _segment_result(segs[3], 3, scored_spans=[
                _scored_span(segs[3], start_ts=1000.0, end_ts=1010.0, raw_score=0.7, touches_boundary=True, excerpt="dup2-a"),
            ]),
            _segment_result(segs[4], 4, scored_spans=[
                _scored_span(segs[4], start_ts=1001.0, end_ts=1009.0, raw_score=0.3, touches_boundary=True, excerpt="dup2-b"),
            ]),
            _segment_result(segs[5], 5, status=SegmentStatus.timeout.value, scored_spans=[]),
        ]

        output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

        # 4/6 usable = 66.7%, clears the default 0.6 fraction.
        assert len(output["spans"]) == 3  # dup1 collapsed, unique1, dup2 collapsed
        assert output["dropped_duplicate_count"] == 2
        assert set(output["degraded_segment_ids"]) == {str(segs[2]), str(segs[5])}
        assert len(output["degraded_segment_ids"]) == 2

        excerpts = {span["transcript_excerpt"] for span in output["spans"]}
        assert any("dup1-a" in e and "dup1-b" in e for e in excerpts)
        assert any("dup2-a" in e and "dup2-b" in e for e in excerpts)
        assert "unique1" in excerpts
