"""Stage 6 (ranker) — pure MMR (Maximal Marginal Relevance) diversity re-rank.

No Celery, no DB, no I/O here on purpose: `mmr_rerank` is a pure function so
it can be unit tested in isolation (tests/test_ranker.py) and reused by
`workers.ranker.tasks.rank_and_persist` without dragging in infrastructure.

Input spans are dict-shaped like models.schemas.ReducedSpan (must carry at
least `normalized_score` and `feature_vector`; any other keys are passed
through untouched on the returned copies).
"""

import math
from typing import Any


def _scalar_items(vector: dict[str, Any]) -> dict[str, float]:
    """workers.span_builder.logic.build_candidate_spans emits feature_vector
    entries that are NOT plain scalars (e.g. `keyframe_refs`: list[str],
    `visual_embedding_mean`: list[float]) alongside genuinely scalar ones
    (motion/audio features, `shot_count`). Only scalar numeric entries are
    comparable dimensions for this cosine similarity — non-scalar values are
    dropped rather than crashing `float(...)` on a list."""
    return {k: v for k, v in vector.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _cosine_similarity(vector_a: dict[str, Any], vector_b: dict[str, Any]) -> float:
    """Cosine similarity treating each feature_vector dict's scalar entries as
    a sparse vector keyed by feature name. Keys absent from a vector are
    implicitly 0, so the dot product only needs to sum over keys present in
    *both* dicts, while each vector's own norm is computed over its own keys.

    If the two dicts share no comparable scalar keys at all, similarity is
    defined as 0.0 (no comparable dimensions -> treated as maximally
    dissimilar for MMR purposes), per the Stage 6 contract.
    """
    scalar_a = _scalar_items(vector_a)
    scalar_b = _scalar_items(vector_b)

    common_keys = scalar_a.keys() & scalar_b.keys()
    if not common_keys:
        return 0.0

    dot = sum(scalar_a[k] * scalar_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v**2 for v in scalar_a.values()))
    norm_b = math.sqrt(sum(v**2 for v in scalar_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rerank(
    spans: list[dict[str, Any]],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[dict[str, Any]]:
    """Greedy MMR selection over `spans`.

    At each step, picks the span maximizing:
        lambda_param * normalized_score - (1 - lambda_param) * max_similarity_to_already_picked

    where max_similarity_to_already_picked is the highest cosine similarity
    (over feature_vector) between the candidate and any span already
    selected (0.0 for the first pick, since nothing is selected yet).

    Returns the picked spans, in rank order, as shallow copies of the input
    dicts with a `final_score` key added (the MMR score at the moment the
    span was selected). Rank number and RankedHighlight assembly are the
    caller's (workers.ranker.tasks) responsibility, not this function's.

    Never mutates the input dicts or the input list.
    """
    if not spans or top_k <= 0:
        return []

    remaining = list(spans)
    selected: list[dict[str, Any]] = []
    selected_feature_vectors: list[dict[str, float]] = []

    picks = min(top_k, len(remaining))
    for _ in range(picks):
        best_idx = -1
        best_score = float("-inf")

        for idx, candidate in enumerate(remaining):
            relevance = float(candidate.get("normalized_score", 0.0))
            candidate_fv = candidate.get("feature_vector") or {}

            if selected_feature_vectors:
                max_similarity = max(
                    _cosine_similarity(candidate_fv, picked_fv)
                    for picked_fv in selected_feature_vectors
                )
            else:
                max_similarity = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        chosen_with_score = dict(chosen)
        chosen_with_score["final_score"] = best_score
        selected.append(chosen_with_score)
        selected_feature_vectors.append(chosen.get("feature_vector") or {})

    return selected
