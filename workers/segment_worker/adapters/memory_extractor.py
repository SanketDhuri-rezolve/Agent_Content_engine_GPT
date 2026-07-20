"""Local memory extraction — Stage 2 addition, opt-in via config.Settings.
use_global_memory_pipeline.

Unlike every other Stage 2 adapter (shot_detector, diarizer, visual_embedder,
etc. — all skipped entirely in Whisper+Gemma4-only mode, see
workers/segment_worker/tasks.py), this one is a lightweight TEXT-ONLY Gemma4
call over the segment's own transcript, producing a compact structured
summary: characters, events, objects, locations, and open/resolved story
threads. This is Stage 1 of the global-memory architecture — see
workers/global_selector/adapters.py for how these per-chunk memories get
merged and used for job-wide moment selection.

Deliberately text-only (no keyframes/audio, unlike workers.scorer.adapters.
Gemma4Scorer): this call runs once per segment regardless of how many
candidate spans that segment produces, so keeping it cheap matters — the
transcript alone is enough to extract characters/events/threads.
"""

import json
from abc import ABC, abstractmethod

import httpx

from config import get_settings
from models.schemas import TranscriptSegment

_MEMORY_INSTRUCTION = """You are building a compact memory of one chunk of a longer movie/show, from its transcript, to be merged later with memories from other chunks and used to judge which moments matter across the WHOLE story.

Respond with ONLY a single JSON object (no other text, no markdown fences) matching EXACTLY this shape:

{
  "characters": ["<character names or short descriptions that appear/are mentioned in this chunk>"],
  "events": ["<short descriptions of what happens, in order>"],
  "objects": ["<notable physical objects/props mentioned or implied>"],
  "locations": ["<places this chunk's scenes take place, if inferable from dialogue>"],
  "open_story_threads": ["<questions/setups raised in this chunk that are not resolved yet>"],
  "resolved_story_threads": ["<earlier setups this chunk pays off or resolves, if any are evident from this transcript alone>"]
}

Base every field only on what this transcript actually shows — leave a list empty rather than guessing."""


class MemoryExtractorError(Exception):
    """Base type for a failure to extract local memory for a segment."""


class MemoryExtractor(ABC):
    @abstractmethod
    def extract(self, segment_id: str, transcript: list[TranscriptSegment]) -> dict:
        """Returns a JSON-safe local-memory dict for this segment (see
        _MEMORY_INSTRUCTION for the shape). Must raise MemoryExtractorError
        on failure rather than returning a sentinel — callers treat this
        fail-soft (a segment without local_memory just contributes nothing
        to the global memory, it does not fail the segment)."""


class Gemma4MemoryExtractor(MemoryExtractor):
    MODEL_VERSION = "gemma-4-memory"

    def __init__(self, endpoint_url=None, api_key=None, timeout_seconds=None, model=None):
        settings = get_settings()
        self._endpoint_url = endpoint_url or settings.gemma4_endpoint_url
        self._api_key = api_key or settings.gemma4_api_key
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.gemma4_timeout_seconds
        self._model = model or settings.gemma4_model_name
        if not self._endpoint_url:
            raise MemoryExtractorError("Gemma4MemoryExtractor requires gemma4_endpoint_url to be configured")

    def extract(self, segment_id: str, transcript: list[TranscriptSegment]) -> dict:
        transcript_text = "\n".join(f"[{line.start_ts:.1f}-{line.end_ts:.1f}] {line.text}" for line in transcript)
        if not transcript_text:
            transcript_text = "(empty transcript for this chunk)"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request_body = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": f"{_MEMORY_INSTRUCTION}\n\nTranscript:\n{transcript_text}"}
            ],
        }

        try:
            response = httpx.post(
                self._endpoint_url, json=request_body, headers=headers, timeout=self._timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise MemoryExtractorError(f"Memory extractor timed out after {self._timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise MemoryExtractorError(f"Memory extractor request failed: {exc}") from exc

        if not (200 <= response.status_code < 300):
            raise MemoryExtractorError(
                f"Memory extractor returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            body = response.json()
            message_content = body["choices"][0]["message"]["content"]
            json_start = message_content.index("{")
            json_end = message_content.rindex("}") + 1
            memory = json.loads(message_content[json_start:json_end])
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise MemoryExtractorError(f"Memory extractor returned an unexpected response shape: {exc}") from exc

        memory["segment_id"] = segment_id
        return memory


class MockMemoryExtractor(MemoryExtractor):
    """Deterministic stand-in — zero network. Derives a trivial memory
    straight from the transcript text so tests get stable, non-empty output
    without calling any real model."""

    def extract(self, segment_id: str, transcript: list[TranscriptSegment]) -> dict:
        events = [line.text for line in transcript if line.text]
        return {
            "segment_id": segment_id,
            "characters": [],
            "events": events[:5],
            "objects": [],
            "locations": [],
            "open_story_threads": [],
            "resolved_story_threads": [],
        }


def get_memory_extractor() -> MemoryExtractor:
    settings = get_settings()
    if settings.gemma4_endpoint_url:
        return Gemma4MemoryExtractor()
    return MockMemoryExtractor()
