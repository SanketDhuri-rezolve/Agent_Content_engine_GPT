"""Unit tests for orchestrator.splitter — pure logic, no DB, no Celery.

Segment/span timestamps are GLOBAL (relative to the whole film), so every
assertion here is expressed in those terms, never segment-local.
"""

import pytest

from orchestrator.splitter import DurationProbe, MockDurationProbe, compute_segment_plan


class TestComputeSegmentPlanEvenSplit:
    def test_even_split_produces_correct_count_and_sequence_indices(self):
        plan = compute_segment_plan(total_duration_seconds=400.0, segment_count=4, overlap_seconds=0.0)

        assert len(plan) == 4
        assert [item["sequence_index"] for item in plan] == [0, 1, 2, 3]

    def test_even_split_core_boundaries_tile_exactly(self):
        plan = compute_segment_plan(total_duration_seconds=400.0, segment_count=4, overlap_seconds=0.0)

        expected_boundaries = [0.0, 100.0, 200.0, 300.0, 400.0]
        assert plan[0]["start_ts"] == pytest.approx(expected_boundaries[0])
        for i, item in enumerate(plan):
            assert item["start_ts"] == pytest.approx(expected_boundaries[i])
            assert item["end_ts"] == pytest.approx(expected_boundaries[i + 1])

        # No gap and no overlap between adjacent segments' *core* windows.
        for prev_item, next_item in zip(plan, plan[1:]):
            assert prev_item["end_ts"] == pytest.approx(next_item["start_ts"])

    def test_first_segment_starts_at_zero_last_segment_ends_at_total(self):
        plan = compute_segment_plan(total_duration_seconds=400.0, segment_count=4, overlap_seconds=0.0)

        assert plan[0]["start_ts"] == pytest.approx(0.0)
        assert plan[-1]["end_ts"] == pytest.approx(400.0)


class TestComputeSegmentPlanUnevenSplit:
    def test_uneven_remainder_split_still_covers_full_duration_with_no_gaps(self):
        total_duration = 100.0
        segment_count = 3
        plan = compute_segment_plan(total_duration_seconds=total_duration, segment_count=segment_count, overlap_seconds=0.0)

        assert len(plan) == 3
        assert plan[0]["start_ts"] == pytest.approx(0.0)
        assert plan[-1]["end_ts"] == pytest.approx(total_duration)

        for prev_item, next_item in zip(plan, plan[1:]):
            assert prev_item["end_ts"] == pytest.approx(next_item["start_ts"])

        # Each core segment is total/segment_count long (33.33... here),
        # not floor-divided/truncated.
        for item in plan:
            assert (item["end_ts"] - item["start_ts"]) == pytest.approx(total_duration / segment_count)

    def test_uneven_split_with_odd_prime_segment_count(self):
        total_duration = 731.0  # deliberately not evenly divisible by 7
        segment_count = 7
        plan = compute_segment_plan(total_duration_seconds=total_duration, segment_count=segment_count, overlap_seconds=0.0)

        assert len(plan) == segment_count
        assert plan[0]["start_ts"] == pytest.approx(0.0)
        assert plan[-1]["end_ts"] == pytest.approx(total_duration)
        for prev_item, next_item in zip(plan, plan[1:]):
            assert prev_item["end_ts"] == pytest.approx(next_item["start_ts"])


class TestComputeSegmentPlanOverlap:
    def test_overlap_extends_symmetrically_across_each_internal_boundary(self):
        total_duration = 100.0
        overlap = 5.0
        plan = compute_segment_plan(total_duration_seconds=total_duration, segment_count=4, overlap_seconds=overlap)

        for prev_item, next_item in zip(plan, plan[1:]):
            # prev segment's overlap_end reaches `overlap` seconds into the
            # next segment's core window.
            assert prev_item["overlap_end"] - next_item["start_ts"] == pytest.approx(overlap)
            # next segment's overlap_start reaches `overlap` seconds back
            # into the prev segment's core window.
            assert prev_item["end_ts"] - next_item["overlap_start"] == pytest.approx(overlap)

    def test_first_segment_overlap_start_clamped_to_zero(self):
        plan = compute_segment_plan(total_duration_seconds=100.0, segment_count=4, overlap_seconds=5.0)

        assert plan[0]["overlap_start"] == pytest.approx(0.0)

    def test_last_segment_overlap_end_clamped_to_total_duration(self):
        total_duration = 100.0
        plan = compute_segment_plan(total_duration_seconds=total_duration, segment_count=4, overlap_seconds=5.0)

        assert plan[-1]["overlap_end"] == pytest.approx(total_duration)

    def test_zero_overlap_means_overlap_bounds_equal_core_bounds(self):
        plan = compute_segment_plan(total_duration_seconds=100.0, segment_count=4, overlap_seconds=0.0)

        for item in plan:
            assert item["overlap_start"] == pytest.approx(item["start_ts"])
            assert item["overlap_end"] == pytest.approx(item["end_ts"])

    def test_large_overlap_still_clamps_within_film_bounds(self):
        # Overlap larger than a single segment's core length should still
        # never push overlap_start below 0 or overlap_end above total.
        total_duration = 40.0
        plan = compute_segment_plan(total_duration_seconds=total_duration, segment_count=4, overlap_seconds=50.0)

        for item in plan:
            assert item["overlap_start"] >= 0.0
            assert item["overlap_end"] <= total_duration


class TestComputeSegmentPlanSingleSegment:
    def test_segment_count_one_covers_the_whole_film_with_no_overlap_needed(self):
        total_duration = 250.0
        plan = compute_segment_plan(total_duration_seconds=total_duration, segment_count=1, overlap_seconds=5.0)

        assert len(plan) == 1
        segment = plan[0]
        assert segment["sequence_index"] == 0
        assert segment["start_ts"] == pytest.approx(0.0)
        assert segment["end_ts"] == pytest.approx(total_duration)
        # Nothing to overlap with outside the film — both clamped.
        assert segment["overlap_start"] == pytest.approx(0.0)
        assert segment["overlap_end"] == pytest.approx(total_duration)


class TestComputeSegmentPlanValidation:
    def test_rejects_zero_segment_count(self):
        with pytest.raises(ValueError):
            compute_segment_plan(total_duration_seconds=100.0, segment_count=0, overlap_seconds=0.0)

    def test_rejects_negative_segment_count(self):
        with pytest.raises(ValueError):
            compute_segment_plan(total_duration_seconds=100.0, segment_count=-2, overlap_seconds=0.0)

    def test_rejects_zero_total_duration(self):
        with pytest.raises(ValueError):
            compute_segment_plan(total_duration_seconds=0.0, segment_count=4, overlap_seconds=0.0)

    def test_rejects_negative_overlap(self):
        with pytest.raises(ValueError):
            compute_segment_plan(total_duration_seconds=100.0, segment_count=4, overlap_seconds=-1.0)


class TestMockDurationProbe:
    def test_is_a_duration_probe(self):
        probe = MockDurationProbe(fixed_duration_seconds=1200.0)
        assert isinstance(probe, DurationProbe)

    def test_returns_fixed_duration_regardless_of_url(self):
        probe = MockDurationProbe(fixed_duration_seconds=1200.0)

        assert probe.probe("https://example.com/movie_a.mp4") == 1200.0
        assert probe.probe("file:///tmp/movie_b.mkv") == 1200.0

    def test_default_fixed_duration_is_positive(self):
        probe = MockDurationProbe()
        assert probe.probe("anything") > 0
