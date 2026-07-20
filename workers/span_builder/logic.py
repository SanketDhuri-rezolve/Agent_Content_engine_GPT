"""Stage 3 (span_builder) pure logic.

No Celery, no DB, no I/O — a pure function that merges Stage 2
(segment_worker) shot boundaries + transcript segments into candidate
highlight spans (Stage 4/scorer input). Keeping this free of side effects
means it can be unit-tested without a broker, a worker process, or Postgres.

All timestamps in and out are GLOBAL (relative to the whole source film),
never segment-local — see models/orm.py's note on Segment/CandidateSpan.
Stage 5 (reducer) dedupes candidates across segments using these global
values, so segment-local timestamps here would silently corrupt that logic.
"""

from __future__ import annotations

from typing import Any


def _shot_ts(shot: Any) -> float:
    return shot["ts"] if isinstance(shot, dict) else shot.ts


def _shot_keyframe(shot: Any) -> str | None:
    return shot.get("keyframe_ref") if isinstance(shot, dict) else shot.keyframe_ref


def _transcript_field(segment: Any, field: str) -> Any:
    return segment[field] if isinstance(segment, dict) else getattr(segment, field)


def _touches_boundary(
    start_ts: float,
    end_ts: float,
    overlap_start: float | None,
    overlap_end: float | None,
) -> bool:
    """True if either edge of the span falls within the segment's
    overlap_start/overlap_end window (models.schemas.SegmentTaskPayload).
    Unknown window (both None) conservatively means "no", never a guess."""
    if overlap_start is None or overlap_end is None:
        return False
    lo, hi = (overlap_start, overlap_end) if overlap_start <= overlap_end else (overlap_end, overlap_start)
    return (lo <= start_ts <= hi) or (lo <= end_ts <= hi)


DEFAULT_MERGE_GAP_SECONDS = 3.0
DEFAULT_MAX_MOMENT_DURATION_SECONDS = 60.0


def build_candidate_spans(
    segment_worker_output: dict,
    overlap_start: float | None = None,
    overlap_end: float | None = None,
    source_video_url: str | None = None,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    max_moment_duration_seconds: float = DEFAULT_MAX_MOMENT_DURATION_SECONDS,
) -> list[dict]:
    """Merge Stage 2 shot boundaries + transcript into Stage 3 candidate
    spans. Each returned dict matches models.schemas.CandidateSpanPayload
    (JSON-safe: segment_id as str, plain floats/dicts/lists/None).

    models.schemas.SegmentWorkerOutput does not carry overlap_start/
    overlap_end (those live on SegmentTaskPayload, Stage 2's *input*, not its
    output) so the caller — workers.span_builder.tasks.build_spans — must
    thread them through explicitly as extra arguments. As a fallback, if an
    upstream producer chooses to stuff them directly into
    segment_worker_output anyway, those values are used when the explicit
    arguments are omitted.

    merge_gap_seconds/max_moment_duration_seconds: confirmed necessary on
    real GPU hardware in Whisper+Gemma4-only mode (config.Settings.
    enable_secondary_adapters = False) — with shot_boundaries always empty
    in that mode, spans used to merge ONLY on exact overlap/adjacency, so
    every natural pause between Whisper transcript lines (even 0.1s) started
    a brand new span. A 13-minute clip fragmented into 17-21 tiny spans per
    segment instead of the intended handful of scene-level "moments" —
    multiplying real Gemma4 API calls by ~5-10x (measured: one segment's 17
    spans took 400s of scoring alone). merge_gap_seconds bridges natural
    conversational pauses into one moment; max_moment_duration_seconds stops
    a long continuous conversation from merging into one unbounded span.
    """
    segment_id = str(segment_worker_output["segment_id"])

    if overlap_start is None:
        overlap_start = segment_worker_output.get("overlap_start")
    if overlap_end is None:
        overlap_end = segment_worker_output.get("overlap_end")

    shot_boundaries = sorted(segment_worker_output.get("shot_boundaries") or [], key=_shot_ts)
    transcript = list(segment_worker_output.get("transcript") or [])
    motion_features = dict(segment_worker_output.get("motion_features") or {})
    audio_features = dict(segment_worker_output.get("audio_features") or {})
    visual_embeddings = dict(segment_worker_output.get("visual_embeddings") or {})

    if not shot_boundaries and not transcript:
        return []

    # Raw pieces: shot-derived windows (the interval between two consecutive
    # cuts) and transcript-derived windows (one per dialogue segment). Each
    # piece carries whatever local evidence justifies its own interval; a
    # single shot boundary with no partner cannot form a window and is
    # dropped (there is nothing to merge it into).
    pieces: list[dict] = []

    for shot_a, shot_b in zip(shot_boundaries, shot_boundaries[1:]):
        start_ts, end_ts = _shot_ts(shot_a), _shot_ts(shot_b)
        if end_ts <= start_ts:
            continue
        pieces.append(
            {"start_ts": start_ts, "end_ts": end_ts, "text": None, "keyframe_ref": _shot_keyframe(shot_a)}
        )

    for segment in transcript:
        start_ts = _transcript_field(segment, "start_ts")
        end_ts = _transcript_field(segment, "end_ts")
        if end_ts <= start_ts:
            continue
        pieces.append(
            {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "text": _transcript_field(segment, "text"),
                "keyframe_ref": None,
            }
        )

    if not pieces:
        return []

    pieces.sort(key=lambda piece: (piece["start_ts"], piece["end_ts"]))

    # Sweep-merge overlapping/adjacent (within merge_gap_seconds) pieces into
    # contiguous groups, regardless of whether they came from shots or
    # transcript — but never grow a single group past
    # max_moment_duration_seconds, so one long continuous scene doesn't
    # collapse into one unbounded moment.
    groups: list[dict] = []
    for piece in pieces:
        fits_gap = groups and piece["start_ts"] <= groups[-1]["end_ts"] + merge_gap_seconds
        fits_duration = (
            groups
            and max(groups[-1]["end_ts"], piece["end_ts"]) - groups[-1]["start_ts"]
            <= max_moment_duration_seconds
        )
        if fits_gap and fits_duration:
            group = groups[-1]
            group["end_ts"] = max(group["end_ts"], piece["end_ts"])
            if piece["text"]:
                group["texts"].append(piece["text"])
            if piece["keyframe_ref"]:
                group["keyframe_refs"].append(piece["keyframe_ref"])
                group["shot_count"] += 1
        else:
            groups.append(
                {
                    "start_ts": piece["start_ts"],
                    "end_ts": piece["end_ts"],
                    "texts": [piece["text"]] if piece["text"] else [],
                    "keyframe_refs": [piece["keyframe_ref"]] if piece["keyframe_ref"] else [],
                    "shot_count": 1 if piece["keyframe_ref"] else 0,
                }
            )

    spans: list[dict] = []
    for group in groups:
        feature_vector: dict = dict(motion_features)
        feature_vector.update(audio_features)
        feature_vector["shot_count"] = group["shot_count"]

        if group["keyframe_refs"]:
            feature_vector["keyframe_refs"] = group["keyframe_refs"]
            relevant_embeddings = [
                visual_embeddings[ref] for ref in group["keyframe_refs"] if ref in visual_embeddings
            ]
            if relevant_embeddings:
                dim = len(relevant_embeddings[0])
                feature_vector["visual_embedding_mean"] = [
                    sum(vec[i] for vec in relevant_embeddings) / len(relevant_embeddings) for i in range(dim)
                ]

        spans.append(
            {
                "segment_id": segment_id,
                "start_ts": group["start_ts"],
                "end_ts": group["end_ts"],
                "transcript_excerpt": " ".join(group["texts"]) if group["texts"] else None,
                "feature_vector": feature_vector,
                "touches_boundary": _touches_boundary(
                    group["start_ts"], group["end_ts"], overlap_start, overlap_end
                ),
                "source_video_url": source_video_url,
            }
        )

    return spans
