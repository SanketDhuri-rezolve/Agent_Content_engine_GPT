"""Stage 3.5 (new, opt-in via config.Settings.use_global_memory_pipeline):
global movie-level moment selection.

Runs ONCE per job, as the new chord callback (before reduce_job), after
every segment's local_memory + candidate_spans are available (see
workers.global_selector.tasks.select_and_score_job). Unlike
workers.scorer.adapters.Gemma4Scorer (genuinely multimodal, per span), this
stage is deliberately TEXT-ONLY: sending every candidate span's images+audio
into one call would blow the endpoint's context budget for a 2-hour movie's
worth of candidates (50-100+), re-introducing the exact per-span multimodal
cost this stage exists to avoid. Transcript + global memory is enough to
judge which moments matter in light of the whole film — including moments
that only matter because of something set up in an earlier chunk or paid off
in a later one, which a chunk-blind per-span scorer can never see. Multimodal
detail is deferred to Gemma4Scorer, which still runs afterward but ONLY on
the selected top_n winners.
"""

import json
from abc import ABC, abstractmethod

import httpx

from config import get_settings

_SELECTION_INSTRUCTION = """You are a film analyst selecting the most important highlight-worthy moments from an entire movie/show, given a compressed memory of the whole story and a list of candidate moments with their timestamps and transcript excerpts.

Use the global memory (characters, events, objects, and open/resolved story threads across every chunk of the film) to judge which candidates matter most in light of the WHOLE story — including moments that only matter because of something set up earlier or paid off later, not just moments that look exciting in isolation.

Respond with ONLY a JSON array (no other text, no markdown fences) of the selected candidates' "index" values, ordered from most to least important. Select at most {top_n} candidates."""


class GlobalSelectorError(Exception):
    """Base type for a failure to select moments."""


class GlobalSelectorAdapter(ABC):
    @abstractmethod
    def select(self, global_memory: dict, candidates: list[dict], top_n: int) -> list[int]:
        """Returns the indices (into `candidates`) of the selected moments,
        ordered most to least important, length <= top_n. Must raise
        GlobalSelectorError on failure rather than returning a sentinel."""


class Gemma4GlobalSelector(GlobalSelectorAdapter):
    MODEL_VERSION = "gemma-4-selector"

    def __init__(self, endpoint_url=None, api_key=None, timeout_seconds=None, model=None):
        settings = get_settings()
        self._endpoint_url = endpoint_url or settings.gemma4_endpoint_url
        self._api_key = api_key or settings.gemma4_api_key
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.global_selector_timeout_seconds
        )
        self._model = model or settings.gemma4_model_name
        if not self._endpoint_url:
            raise GlobalSelectorError("Gemma4GlobalSelector requires gemma4_endpoint_url to be configured")

    def select(self, global_memory: dict, candidates: list[dict], top_n: int) -> list[int]:
        if not candidates:
            return []

        # Transcript excerpts are truncated (not the full span) — this call
        # only needs enough text to judge relevance, not the full multimodal
        # detail Gemma4Scorer will produce later for the actual winners.
        candidate_briefs = [
            {
                "index": i,
                "segment_id": c.get("segment_id"),
                "start_ts": c.get("start_ts"),
                "end_ts": c.get("end_ts"),
                "transcript_excerpt": (c.get("transcript_excerpt") or "")[:400],
            }
            for i, c in enumerate(candidates)
        ]

        prompt = (
            _SELECTION_INSTRUCTION.format(top_n=top_n)
            + "\n\nGlobal memory:\n"
            + json.dumps(global_memory, ensure_ascii=False)
            + "\n\nCandidates:\n"
            + json.dumps(candidate_briefs, ensure_ascii=False)
        )

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request_body = {"model": self._model, "messages": [{"role": "user", "content": prompt}]}

        try:
            response = httpx.post(
                self._endpoint_url, json=request_body, headers=headers, timeout=self._timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise GlobalSelectorError(f"Global selector timed out after {self._timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise GlobalSelectorError(f"Global selector request failed: {exc}") from exc

        if not (200 <= response.status_code < 300):
            raise GlobalSelectorError(
                f"Global selector returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            body = response.json()
            message_content = body["choices"][0]["message"]["content"]
            json_start = message_content.index("[")
            json_end = message_content.rindex("]") + 1
            indices = json.loads(message_content[json_start:json_end])
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise GlobalSelectorError(f"Global selector returned an unexpected response shape: {exc}") from exc

        seen: set[int] = set()
        deduped: list[int] = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                deduped.append(i)
        return deduped[:top_n]


class MockGlobalSelector(GlobalSelectorAdapter):
    """Deterministic stand-in — zero network. Picks the top_n candidates by
    transcript_excerpt length (a crude but stable proxy for "has substance"),
    breaking ties by start_ts for reproducibility."""

    def select(self, global_memory: dict, candidates: list[dict], top_n: int) -> list[int]:
        ranked = sorted(
            range(len(candidates)),
            key=lambda i: (-len(candidates[i].get("transcript_excerpt") or ""), candidates[i].get("start_ts", 0.0)),
        )
        return ranked[:top_n]


def get_global_selector() -> GlobalSelectorAdapter:
    """Single call site controlling which GlobalSelectorAdapter is active —
    same config-driven pattern as workers.scorer.adapters.get_scorer."""
    settings = get_settings()
    if settings.gemma4_endpoint_url:
        return Gemma4GlobalSelector()
    return MockGlobalSelector()
