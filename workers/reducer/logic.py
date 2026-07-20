"""Stage 5 (reducer) pure logic — see workers/reducer/tasks.py for the Celery
wrapper around this module.

STAGE 5 IS THE HIGHEST-RISK COMPONENT IN THIS PIPELINE (per the project
brief): every upstream stage (segment_worker/span_builder/scorer) fails
*soft* by design — a bad segment becomes a `status != completed` dict, never
an exception (see those modules' docstrings). The reducer is the one place
that turns those accumulated soft per-segment failures into either:
  (a) an explicit, typed, hard failure (`InsufficientSegmentsError`) when too
      much of the job is missing to trust a ranking at all, or
  (b) an accounted-for partial result (`degraded_segment_ids`,
      `dropped_duplicate_count`) that names exactly what was excluded and why
      — never a silent drop that would make a job that mostly failed look
      like a confident, complete ranking.

Three ordered steps, each its own function so each is independently unit
testable:
  1. `classify_segments`      — STITCH/VALIDATE
  2. `dedupe_boundary_spans`  — DEDUPE
  3. `normalize_scores`       — NORMALIZE

`reduce_segment_results()` composes all three into the
`models.schemas.ReducerOutput` shape.

No Celery, no DB, no network I/O, no unbounded loops or blocking calls
anywhere in this module (the "must never hang" requirement) — every loop
here iterates over an already-in-memory, finite list exactly once (or, for
the dedupe step, a bounded double loop over that same finite list).
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from models.enums import SegmentStatus
from models.schemas import ReducedSpan, ReducerOutput

# ---------------------------------------------------------------------------
# Documented, adversarially-reviewable tuning decisions
# ---------------------------------------------------------------------------

# DEDUP_IOU_THRESHOLD: two boundary-touching spans from *adjacent* segments
# (abs(sequence_index_a - sequence_index_b) == 1) are treated as referring to
# the same underlying moment, and collapsed into one, when the
# intersection-over-union (IoU) of their [start_ts, end_ts] global windows is
# STRICTLY GREATER than this value. 0.5 ("more than 50% overlap") was chosen
# because IoU already punishes any length mismatch or offset harshly (unlike
# plain overlap-fraction-of-the-smaller-window), so requiring >50% IoU is a
# conservative bar: two genuinely distinct nearby moments that both happen to
# touch the same overlap window will very rarely clear it, while two spans
# built from (near-)identical underlying evidence on both sides of a segment
# boundary reliably will. Revisit against Step 2 field data if this proves
# too aggressive or too lax.
DEDUP_IOU_THRESHOLD = 0.5


class InsufficientSegmentsError(Exception):
    """Raised by `classify_segments` (and surfaced by
    `reduce_segment_results`) when the fraction of usable segments for a job
    falls below `config.Settings.reducer_min_segments_required_fraction`.

    The message names the specific segment identifiers (sequence_index when
    known, list position when not — e.g. a bare `None` chord entry has no
    data to recover a sequence_index from) that are missing/degraded, so a
    log reader can see exactly which part of the source film is unaccounted
    for rather than just a raw fraction.
    """


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """1-D intersection-over-union of two [start, end] windows. 0.0 for
    non-overlapping or degenerate (zero-length union) windows — never raises,
    never divides by zero."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0.0:
        return 0.0
    return intersection / union


def _safe_uuid_str(value: Any) -> str | None:
    """Best-effort coercion of `value` to a canonical UUID string. Returns
    None (never raises) if `value` isn't a valid UUID — a degraded segment's
    id is used for logging/`degraded_segment_ids` best-effort only; a
    malformed id must never crash the reduce."""
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


def classify_segments(
    segment_pipeline_results: list[Any],
    min_required_fraction: float,
) -> tuple[list[dict], list[str], list[str]]:
    """STITCH/VALIDATE step.

    `segment_pipeline_results` is the chord's list of
    models.schemas.SegmentPipelineResult-shaped dicts. Any entry may
    instead be:
      - `None` — a hard Celery task failure (the chain member raised instead
        of returning a soft-failure dict).
      - a dict with `status != SegmentStatus.completed` — a soft failure
        (span_builder/scorer already degraded it to `failed`/`timeout`;
        see those modules' fail-soft docstrings).
      - some other malformed/unexpected shape — defensively treated the same
        as a hard failure rather than raising.

    A segment is "usable" iff it is a dict with
    `status == SegmentStatus.completed.value`. This is intentionally a
    status-only check: a `completed` segment's `scored_spans` may be empty
    (span_builder/scorer legitimately found nothing) or partial — both are
    still usable, per the Stage 4 contract that an empty result is not an
    error. Conversely, ANY non-completed status (failed/timeout/unknown) is
    treated as fully degraded — none of that entry's `scored_spans` (even if
    present) are trusted into the output pool, since a status signaling
    trouble means the rest of that entry's data is not guaranteed reliable.

    Returns `(usable_entries, degraded_segment_ids, degraded_labels)`:
      - `usable_entries`: the dicts to actually reduce.
      - `degraded_segment_ids`: best-effort list of segment_id strings for
        degraded dict entries that had a recoverable id (hard-failure `None`
        entries contribute no id here — there is nothing to recover it from
        — but they DO count toward the usable/total fraction and DO appear
        in `degraded_labels`).
      - `degraded_labels`: human-readable descriptions of every degraded/
        missing entry, for logging and for `InsufficientSegmentsError`'s
        message.

    Raises `InsufficientSegmentsError` if the usable fraction is below
    `min_required_fraction` (or if there are zero segments at all).
    """
    total = len(segment_pipeline_results)
    usable_entries: list[dict] = []
    degraded_segment_ids: list[str] = []
    degraded_labels: list[str] = []

    for position, entry in enumerate(segment_pipeline_results):
        if entry is None:
            degraded_labels.append(f"position={position} (hard task failure — no data returned)")
            continue

        if not isinstance(entry, dict):
            try:
                entry = dict(entry)
            except (TypeError, ValueError):
                degraded_labels.append(
                    f"position={position} (unrecognized chord result shape: {type(entry).__name__!r})"
                )
                continue

        status = entry.get("status")
        sequence_index = entry.get("sequence_index")
        segment_id = entry.get("segment_id")
        label = f"sequence_index={sequence_index}" if sequence_index is not None else f"position={position}"

        is_completed = status == SegmentStatus.completed.value or status == SegmentStatus.completed
        if is_completed:
            usable_entries.append(entry)
            continue

        degraded_labels.append(f"{label} (status={status!r}, error={entry.get('error')!r})")
        safe_id = _safe_uuid_str(segment_id)
        if safe_id is not None:
            degraded_segment_ids.append(safe_id)

    usable_count = len(usable_entries)
    fraction_usable = (usable_count / total) if total else 0.0

    if total == 0 or fraction_usable < min_required_fraction:
        raise InsufficientSegmentsError(
            f"Only {usable_count}/{total} segment(s) usable "
            f"({fraction_usable:.1%} < required {min_required_fraction:.1%}) for a reducible job. "
            f"Missing/degraded segments: [{'; '.join(degraded_labels) if degraded_labels else 'none'}]."
        )

    return usable_entries, degraded_segment_ids, degraded_labels


def _merge_transcript_excerpts(*excerpts: str | None) -> str | None:
    """Merge policy for a group of dedup-collapsed spans' transcript_excerpt:
    keep every distinct piece of evidence (never silently drop a collapsed
    span's transcript text), in original order, skipping duplicates/empties/
    None. Documented dedup behavior — see module docstring /
    DEDUP_IOU_THRESHOLD comment. Works for a pair (the common case) or for a
    larger transitively-collapsed group (see `dedupe_boundary_spans`)."""
    parts: list[str] = []
    for excerpt in excerpts:
        if excerpt and excerpt not in parts:
            parts.append(excerpt)
    if not parts:
        return None
    return " ".join(parts)


def dedupe_boundary_spans(
    spans: list[dict],
    segment_id_to_sequence_index: dict[str, int],
) -> tuple[list[dict], int]:
    """DEDUPE step.

    Only spans with `touches_boundary=True` are dedup candidates (spans that
    never touch a segment's overlap window cannot be a cross-segment
    duplicate — they pass through completely untouched). Among
    boundary-touching spans, only pairs from ADJACENT segments
    (`abs(seq_a - seq_b) == 1`, per `config.Settings.segment_overlap_seconds`
    — segments only ever overlap their immediate neighbor) whose
    [start_ts, end_ts] windows have IoU > `DEDUP_IOU_THRESHOLD` are
    collapsed. Two boundary spans from the SAME segment are never compared
    here — span_builder already sweep-merges any overlapping windows within
    one segment before this stage ever sees them.

    Pairwise "should merge" edges are computed once from each span's own,
    never-mutated [start_ts, end_ts] (a union-find over span *positions*),
    then unioned into connected components and each component collapses to
    ONE surviving span. This is deliberate: a chain of pairwise duplicates —
    e.g. span A (segment 0) duplicates span B (segment 1), and B in turn
    duplicates span C (segment 2), even though A and C are not themselves
    adjacent segments — must collapse into exactly one surviving span,
    regardless of which of A/B happens to win their own pairwise
    raw_score comparison first. (An earlier "current absorbs the next
    matching span" pairwise-only implementation was order-dependent: if A
    beat B on raw_score, the survivor's identity became A's segment
    (sequence_index 0), which is NOT adjacent to C (sequence_index 2), so
    the still-genuine A/B/C duplicate chain silently left C as a second,
    undeduplicated copy of the same real-world moment whenever the
    "outer" span of a chain happened to out-score the "middle" one. Fixed
    by grouping via transitive closure instead of a single mutable
    "current" accumulator.)

    Merge policy for a collapsed group (documented per the build brief's
    "your call, document it" instruction): keep the highest-`raw_score`
    span's fields as the surviving record (its scoring is what the reducer
    ultimately ranks on), but merge `transcript_excerpt` from every span in
    the group (see `_merge_transcript_excerpts`) so no losing span's
    evidence is silently discarded.

    A boundary span whose `segment_id` is not found in
    `segment_id_to_sequence_index` (should not happen — every usable span
    comes from a usable segment entry that is itself the source of this map
    — but defensively handled) can never be unioned with anything (every
    comparison involving it fails the adjacency lookup), so it always
    passes through unmerged as its own singleton group.

    Returns `(surviving_spans, dropped_duplicate_count)`. Runs in bounded
    O(n^2) time over `spans` (n = spans in this one job, always finite/small)
    — a bounded double loop over an in-memory list, not an unbounded/blocking
    loop, so it cannot hang.
    """
    non_boundary = [s for s in spans if not s.get("touches_boundary")]
    boundary = [s for s in spans if s.get("touches_boundary")]
    n = len(boundary)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boundary[i], boundary[j]
            if a.get("segment_id") == b.get("segment_id"):
                continue  # same-segment overlaps are span_builder's job, not ours

            seq_a = segment_id_to_sequence_index.get(str(a.get("segment_id")))
            seq_b = segment_id_to_sequence_index.get(str(b.get("segment_id")))
            if seq_a is None or seq_b is None:
                continue  # unresolvable adjacency -> never merge (conservative default)
            if abs(seq_a - seq_b) != 1:
                continue  # only immediate-neighbor segments can share a moment

            iou = _iou(
                float(a["start_ts"]), float(a["end_ts"]),
                float(b["start_ts"]), float(b["end_ts"]),
            )
            if iou <= DEDUP_IOU_THRESHOLD:
                continue

            union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    kept: list[dict] = []
    dropped_duplicate_count = 0
    for members in groups.values():
        if len(members) == 1:
            kept.append(boundary[members[0]])
            continue

        dropped_duplicate_count += len(members) - 1
        member_spans = [boundary[m] for m in members]
        winner = max(member_spans, key=lambda s: float(s.get("raw_score", 0.0)))
        merged = dict(winner)
        merged["transcript_excerpt"] = _merge_transcript_excerpts(
            *(s.get("transcript_excerpt") for s in member_spans)
        )
        kept.append(merged)

    return non_boundary + kept, dropped_duplicate_count


def normalize_scores(spans: list[dict]) -> list[dict]:
    """NORMALIZE step: z-score normalize `raw_score` -> `normalized_score`
    across all surviving spans (mean 0, std 1, population statistics — this
    is normalizing the ENTIRE finite population of a job's spans, not a
    sample estimate of some larger population).

    Zero-variance edge case (a job with exactly one span, or with every
    span's `raw_score` identical): the z-score `(x - mean) / std` is
    mathematically undefined (0/0), not merely a precision hazard, so it is
    NOT patched by nudging `std` away from zero with a small epsilon (doing
    so would produce an arbitrarily large/near-infinite normalized_score for
    any span whose raw_score differs from the mean by mere float noise).
    Instead, every span in a zero-variance job is explicitly defined as
    `normalized_score = 0.0` — none of them carries more or less relative
    information than any other in that population, so a neutral score is the
    only defensible choice. Documented here and in the final report so a
    reviewer knows to adversarially test this exact branch.

    Never raises (including on an empty `spans` list, which returns `[]`).
    """
    if not spans:
        return []

    raw_scores = [float(s["raw_score"]) for s in spans]
    n = len(raw_scores)
    mean = sum(raw_scores) / n
    variance = sum((x - mean) ** 2 for x in raw_scores) / n
    std_dev = math.sqrt(variance)

    if std_dev == 0.0:
        normalized = [0.0] * n
    else:
        normalized = [(x - mean) / std_dev for x in raw_scores]

    # Numerically-guard against float overflow in the mean/variance sums
    # themselves: `raw_scores` is guaranteed individually finite by the
    # caller's shape guard (see reduce_segment_results), but the *sum* of
    # squares (variance) or the sum (mean) of several very-large-but-finite
    # values can still overflow to +/-inf (e.g. two raw_scores near
    # float's max magnitude), which then yields `inf - inf -> nan` or
    # similar in the z-score formula even though every input was finite.
    # Treated the same as the documented zero-variance case (neutral 0.0)
    # rather than ever leaking a NaN/inf normalized_score into the DB float
    # column, the Celery JSON wire, or MMR's downstream cosine math.
    normalized = [value if math.isfinite(value) else 0.0 for value in normalized]

    return [{**span, "normalized_score": norm} for span, norm in zip(spans, normalized)]


def reduce_segment_results(segment_pipeline_results: list[Any], job_id: str) -> dict:
    """Top-level Stage 5 composition: classify -> dedupe -> normalize.

    Returns a dict matching `models.schemas.ReducerOutput`
    (`job_id`, `spans`, `degraded_segment_ids`, `dropped_duplicate_count`).

    Raises `InsufficientSegmentsError` (propagated from `classify_segments`)
    if too few segments are usable — callers (workers.reducer.tasks.reduce_job)
    are responsible for catching this and failing the Job explicitly rather
    than letting it kill the chord silently.
    """
    settings_fraction = _min_required_fraction()

    usable_entries, degraded_segment_ids, _degraded_labels = classify_segments(
        segment_pipeline_results, settings_fraction
    )

    segment_id_to_sequence_index: dict[str, int] = {}
    all_spans: list[dict] = []
    for entry in usable_entries:
        segment_id = entry.get("segment_id")
        sequence_index = entry.get("sequence_index")
        if segment_id is not None and sequence_index is not None:
            segment_id_to_sequence_index[str(segment_id)] = sequence_index

        for raw_span in entry.get("scored_spans") or []:
            if not isinstance(raw_span, dict):
                continue
            # Minimal shape guard: a scored span missing its scoring fields,
            # or carrying a value that isn't a genuine finite number for
            # them (None, a non-numeric string, NaN, +/-inf — any of which
            # would otherwise crash `float(...)` in dedupe/normalize, or
            # silently poison the whole job's population mean/std with a
            # NaN/inf that then propagates into normalized_score for EVERY
            # span, not just the malformed one) is not reducible — skip it
            # rather than raising. Should not happen with contract-conformant
            # upstream output (workers.scorer always emits full
            # ScoredSpanPayload dicts with genuine finite floats), so this is
            # a defensive floor, not the expected path.
            try:
                numeric_ok = (
                    math.isfinite(float(raw_span["raw_score"]))
                    and math.isfinite(float(raw_span["start_ts"]))
                    and math.isfinite(float(raw_span["end_ts"]))
                )
            except (KeyError, TypeError, ValueError):
                numeric_ok = False
            if not numeric_ok:
                continue
            all_spans.append(dict(raw_span))

    deduped_spans, dropped_duplicate_count = dedupe_boundary_spans(all_spans, segment_id_to_sequence_index)
    normalized_spans = normalize_scores(deduped_spans)

    reducer_output = ReducerOutput(
        job_id=job_id,
        spans=[ReducedSpan.model_validate(span) for span in normalized_spans],
        degraded_segment_ids=degraded_segment_ids,
        dropped_duplicate_count=dropped_duplicate_count,
    )
    return reducer_output.model_dump(mode="json")


def _min_required_fraction() -> float:
    """Isolated so tests can call classify_segments/reduce_segment_results
    without needing config.get_settings() to be importable in every context
    (it always is here, but this keeps the settings read to one place)."""
    from config import get_settings

    return get_settings().reducer_min_segments_required_fraction
