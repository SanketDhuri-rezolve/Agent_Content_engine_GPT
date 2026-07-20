"""Stage 4 scorer adapters — LLM-as-judge scoring of candidate spans.

`ScorerAdapter` is the interface every implementation honors:
    score(span: dict) -> tuple[raw_score: float, justification: str,
                                llm_model_version: str, clip_url: str | None,
                                rich_data: dict]

Two implementations exist:
  - `MockScorer`  — deterministic, zero-network, zero-GPU. This is what
    `workers.scorer.tasks` actually calls in Step 1. Always returns
    clip_url=None (no real video to crop); `rich_data` is a deterministic
    fake matching the same shape Gemma4Scorer produces.
  - `Gemma4Scorer` — real, genuinely MULTIMODAL call to the deployed Gemma 4
    endpoint (config.get_settings().gemma4_endpoint_url / gemma4_api_key /
    gemma4_timeout_seconds / gemma4_model_name), a real OpenAI-compatible
    chat completions API (vLLM), CONFIRMED working end-to-end for text,
    image, and audio content blocks — see CLAUDE.md for the verification.
    For each span (given `source_video_url` is present — see
    models.schemas.CandidateSpanPayload) it:
      1. Crops the span's actual [start_ts, end_ts) video+audio into a real
         clip via workers.segment_worker.adapters._media.extracted_av_clip.
      2. Saves that clip durably via storage.object_storage (so it's also a
         real, deliverable artifact for the product — not just scorer input)
         and keeps the resulting clip_url to return alongside the score.
      3. Extracts several representative keyframe images (see _NUM_KEYFRAMES)
         + a capped-duration audio excerpt (see _MAX_AUDIO_SECONDS — full-span
         audio risks exceeding the endpoint's max_model_len for long spans,
         since audio tokenizes at a real per-second cost, confirmed via a
         real request: ~100 prompt tokens for a 5s clip) from that SAME
         cropped clip, and base64-encodes them.
      4. Sends transcript + images + audio to Gemma4 as one multimodal chat
         message, asking for the FULL rich moment-analysis JSON shape (see
         _SCORING_INSTRUCTION) — not just a bare score.
    Falls back to text-only scoring if `source_video_url` is missing from
    the span (e.g. older test fixtures) rather than raising.

`get_scorer()` at the bottom of this file is the single, obvious line the
task layer calls — swapping Step 1 -> Step 2 is changing that one line
(`return MockScorer()` -> `return Gemma4Scorer()`), not touching any caller.
"""

import base64
import hashlib
import json
from abc import ABC, abstractmethod

import httpx

from config import get_settings

# Number of representative keyframes attached per span — evenly spaced across
# [start_ts, end_ts). Confirmed on real GPU hardware that 3 was too sparse
# for "give (near-)complete video coverage" — raised to 6. Not raised further
# without also raising the endpoint's --max-model-len: each image + the full
# rich-schema instruction + transcript + audio all share one token budget
# (see _MAX_AUDIO_SECONDS below for the same concern on the audio side).
_NUM_KEYFRAMES = 6

# Caps how much of a span's audio actually gets sent, regardless of the
# span's own duration. Real spans in this project ran up to ~195s+; sending
# a whole span's audio risks exceeding the endpoint's max_model_len (8192
# tokens, deliberately capped when deploying — see CLAUDE.md) since audio
# tokenizes at a real, non-negligible per-second cost (confirmed via a real
# request: ~100 prompt tokens for just 5s of audio — a 195s span extrapolated
# linearly would be ~4000 tokens on audio ALONE, before images/text). Take a
# fixed-length excerpt from the middle of the span instead of the whole
# thing — the middle is more likely to contain the span's core content than
# its edges (which often overlap into a neighboring cut/dialogue turn).
_MAX_AUDIO_SECONDS = 20.0


class ScorerError(Exception):
    """Base type for any failure to produce a score for a span."""


class ScorerTimeoutError(ScorerError):
    """The scoring backend did not respond within the configured timeout."""


class ScorerResponseError(ScorerError):
    """The scoring backend responded, but with a non-2xx status or an
    unexpected response shape."""


class ScorerAdapter(ABC):
    """Common interface for Stage 4 (LLM-as-judge) scoring implementations."""

    @abstractmethod
    def score(self, span: dict) -> tuple[float, str, str, str | None, dict]:
        """Score one candidate span.

        `span` is a CandidateSpanPayload-shaped dict (segment_id, start_ts,
        end_ts, transcript_excerpt, feature_vector, touches_boundary,
        source_video_url).

        Returns (raw_score, justification, llm_model_version, clip_url,
        rich_data). raw_score mirrors rich_data's own "moment_score" (kept
        as a separate scalar for the reducer's existing z-score
        normalization/ranking logic — see workers/reducer/logic.py — which
        predates rich_data and only knows how to rank a flat float).
        clip_url is the saved highlight clip's URL if one was
        extracted/saved for this span, else None. rich_data is the full
        structured moment analysis (see _SCORING_INSTRUCTION for the exact
        shape) as a JSON-safe dict. Implementations must raise a
        `ScorerError` subclass on failure rather than returning a sentinel
        value — callers rely on the exception to decide fail-soft behavior.
        """


def _format_timestamp_range(start_ts: float, end_ts: float) -> str:
    """"MM:SS - MM:SS", matching the target schema's time_stamp format
    exactly. Computed here, not asked of the LLM — these are already known
    exactly from the span itself; asking the model to also produce them
    risks it hallucinating a slightly different (wrong) value."""

    def _mmss(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        return f"{total // 60:02d}:{total % 60:02d}"

    return f"{_mmss(start_ts)} - {_mmss(end_ts)}"


def _make_moment_id(span: dict) -> str:
    """Best-effort human-readable reference id, NOT a database primary key
    (CandidateSpanPayload has no stable id yet at scorer time — that's
    assigned later, in the ranker — see workers/ranker/tasks.py). Good
    enough for a human reading rich_data to distinguish moments within a
    segment; not meant to be globally unique/stable across reruns."""
    segment_id = str(span.get("segment_id", "unknown"))[:8]
    start_ts = span.get("start_ts", 0.0)
    return f"MOM_{segment_id}_{int(start_ts):06d}"


class Gemma4Scorer(ScorerAdapter):
    """Real, multimodal implementation — calls the already-deployed Gemma 4
    endpoint. See module docstring for the full design and its caveats.
    """

    MODEL_VERSION = "gemma-4-judge"

    # Asks for the FULL rich moment-analysis schema in one completion,
    # rather than just a bare score — see CLAUDE.md for the discussion of
    # why this is one big prompt rather than several focused calls (the
    # trade-off chosen here: fewer LLM calls/lower cost per span, at some
    # risk the model spreads its attention thin across this many fields).
    # moment_id/time_stamp are deliberately EXCLUDED from what the LLM must
    # produce — those are computed from the span itself (see
    # _make_moment_id/_format_timestamp_range) and injected into rich_data
    # after parsing, not trusted from the model's own output.
    _SCORING_INSTRUCTION = """You are a film analyst evaluating a short moment from a movie/show as a candidate for a trailer highlight, advertisement placement, or shoppable/commerce moment. You are given the moment's real transcript, several representative frame images sampled across the moment, and an excerpt of its real audio.

Respond with ONLY a single JSON object (no other text, no markdown code fences) matching EXACTLY this shape:

{
  "moment_title": "<short evocative title, 3-6 words>",
  "moment_description": "<one sentence describing what happens>",
  "moment_score": <float, -1.0 to 1.0, how good a highlight/trailer moment this is>,
  "moment_location": "<where this appears to take place, inferred from the visuals>",
  "action_took": "<one sentence describing the physical action/behavior in the moment>",
  "dialogue_by_character": [
    {
      "name": "<character name if identifiable from dialogue/context, else a short visual description like 'Man in blue shirt'>",
      "emotion": "<the character's apparent emotional state>",
      "line": "<the actual line of dialogue they say, from the transcript>",
      "subtext": "<what they really mean/feel beneath the words>",
      "actions_done_by_character": "<their physical actions/expressions while speaking>",
      "tags": ["<short lowercase tags describing tone/delivery, e.g. 'whisper', 'shock'>"]
    }
  ],
  "props_involved": ["<notable physical objects/props visible or referenced, empty list if none>"],
  "impact_of_action": "<one sentence on what this moment means for the story going forward>",
  "characters_present": [
    {"name": "<character name or visual description>", "screen_percentage": "<rough percent of frame time/area this character occupies, as a string like '40%'>"}
  ],
  "sound_design": {
    "sfx": "<notable sound effects heard, or 'None'>",
    "bg_sound": {
      "score": "<one short phrase describing the background score/music, or 'None' if none audible>",
      "notes": "<brief description of its character>",
      "instruments": ["<instruments you can identify, if any>"],
      "reason": "<why this music/sound choice fits the moment>"
    }
  },
  "cinematography": {
    "shot_type": "<e.g. 'Extreme close-up', 'Wide shot', 'Medium shot'>",
    "camera_movement": "<e.g. 'Static', 'Quick zoom cuts', 'Handheld'>",
    "lighting": "<e.g. 'High contrast dramatic lighting', 'Soft natural light'>"
  },
  "ad_placement": {
    "is_suitable_ad_break": <true or false, whether this moment is a natural pause point for inserting an ad>,
    "ad_break_timestamp": "<MM:SS within this moment where an ad break would fit best, or null if not suitable>",
    "suitability_score": <float, 0.0 to 1.0>,
    "reason": "<one sentence explaining the ad-break suitability judgment>"
  },
  "risk_hints": {
    "flags": ["<content flags if any, e.g. 'violence', 'profanity' — or 'none' if clean>"],
    "risk_level": "<'safe', 'moderate', or 'high'>",
    "description": "<one short sentence explaining the risk assessment>"
  },
  "inversion_score": {
    "scores": {
      "narrative_necessity": <int, 0-100>,
      "character_significance": <int, 0-100>,
      "emotional_impact": <int, 0-100>,
      "visual_memorability": <int, 0-100>,
      "rewatchability_retention": <int, 0-100>,
      "pacing_structural_placement": <int, 0-100>,
      "thematic_relevance": <int, 0-100>,
      "audio_visual_execution": <int, 0-100>,
      "safety_reputation": <int, 0-100>
    },
    "composite_score": <float, 0.0-1.0, your own weighted overall judgment of this moment's highlight value>,
    "impact_tier": "<'Low', 'Medium', or 'High'>"
  }
}

Do NOT invent a moment_id or a time_stamp field yourself — those are supplied separately, outside this JSON. Base every field on what you can actually observe/hear in the provided frames, audio, and transcript — do not fabricate details you cannot support from the material given."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self._endpoint_url = endpoint_url or settings.gemma4_endpoint_url
        self._api_key = api_key or settings.gemma4_api_key
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.gemma4_timeout_seconds
        )
        self._model = model or settings.gemma4_model_name
        if not self._endpoint_url:
            raise ScorerError("Gemma4Scorer requires gemma4_endpoint_url to be configured")

    def _clip_storage_key(self, span: dict) -> str:
        segment_id = span.get("segment_id", "unknown")
        start_ts = span.get("start_ts", 0.0)
        end_ts = span.get("end_ts", 0.0)
        return f"highlight_clips/{segment_id}/{start_ts:.3f}_{end_ts:.3f}.mp4"

    def _build_multimodal_content(self, span: dict, local_clip_path: str) -> list[dict]:
        """Builds the OpenAI-style `content` blocks (text + image_url +
        input_audio) for one span, sourced from its already-cropped local
        clip (not the original remote source_video_url — cheaper, and the
        clip's own [0, duration) timeline is simpler to sample evenly)."""
        from workers.segment_worker.adapters._media import extracted_audio_wav, extracted_frame

        start_ts = float(span.get("start_ts", 0.0))
        end_ts = float(span.get("end_ts", 0.0))
        duration = max(end_ts - start_ts, 0.0)
        transcript_excerpt = span.get("transcript_excerpt") or "(no transcript)"

        content: list[dict] = [
            {"type": "text", "text": f"{self._SCORING_INSTRUCTION}\n\nTranscript: {transcript_excerpt}"}
        ]

        if duration > 0:
            # Evenly spaced sample points across the clip's own local
            # timeline (e.g. 6 frames -> 1/12, 3/12, ..., 11/12 of the way
            # through) — avoids sampling exactly at the cut boundaries,
            # which are more likely to be transitional/blurry frames.
            for i in range(_NUM_KEYFRAMES):
                frame_ts = duration * (2 * i + 1) / (2 * _NUM_KEYFRAMES)
                with extracted_frame(local_clip_path, frame_ts) as frame_path:
                    with open(frame_path, "rb") as f:
                        frame_b64 = base64.b64encode(f.read()).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    }
                )

            # See _MAX_AUDIO_SECONDS: cap audio duration regardless of the
            # span's own length, taking a fixed-length excerpt from the
            # middle of the span (more likely to hold its core content than
            # the edges, which often overlap a neighboring cut).
            audio_duration = min(duration, _MAX_AUDIO_SECONDS)
            audio_start = max(0.0, (duration - audio_duration) / 2.0)
            with extracted_audio_wav(local_clip_path, audio_start, audio_start + audio_duration) as audio_path:
                with open(audio_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("ascii")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            )

        return content

    def score(self, span: dict) -> tuple[float, str, str, str | None, dict]:
        source_video_url = span.get("source_video_url")
        start_ts = span.get("start_ts")
        end_ts = span.get("end_ts")

        clip_url: str | None = None
        content: list[dict]

        if source_video_url and start_ts is not None and end_ts is not None and end_ts > start_ts:
            from workers.segment_worker.adapters._media import extracted_av_clip

            with extracted_av_clip(source_video_url, float(start_ts), float(end_ts)) as local_clip_path:
                # Save the actual highlight clip as a real deliverable
                # artifact BEFORE building the multimodal request, so a
                # downstream LLM-call failure still leaves a usable saved
                # clip rather than losing the work entirely.
                from storage.object_storage import get_object_storage

                storage = get_object_storage()
                clip_key = self._clip_storage_key(span)
                clip_url = storage.put(clip_key, local_clip_path)

                content = self._build_multimodal_content(span, local_clip_path)
        else:
            # No real video available for this span (e.g. an older test
            # fixture, or span_builder ran without source_video_url threaded
            # through) — degrade to text-only rather than fail the span.
            transcript_excerpt = span.get("transcript_excerpt") or "(no transcript)"
            content = [
                {"type": "text", "text": f"{self._SCORING_INSTRUCTION}\n\nTranscript: {transcript_excerpt}"}
            ]

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request_body = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
        }

        try:
            response = httpx.post(
                self._endpoint_url,
                json=request_body,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ScorerTimeoutError(
                f"Gemma4 scorer timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ScorerError(f"Gemma4 scorer request failed: {exc}") from exc

        if not (200 <= response.status_code < 300):
            raise ScorerResponseError(
                f"Gemma4 scorer returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            body = response.json()
            message_content = body["choices"][0]["message"]["content"]
            # The model is instructed to respond with ONLY a JSON object, but
            # some servers/models still wrap it in prose or markdown fences —
            # defensively extract the first {...} block rather than assuming
            # message_content is pure JSON.
            json_start = message_content.index("{")
            json_end = message_content.rindex("}") + 1
            rich_data = json.loads(message_content[json_start:json_end])
            raw_score = float(rich_data["moment_score"])
            justification = str(rich_data.get("moment_description", ""))
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ScorerResponseError(
                f"Gemma4 scorer returned an unexpected response shape: {exc}"
            ) from exc

        # Inject the code-computed reference fields — never trust the LLM's
        # own guess for these (see _SCORING_INSTRUCTION's explicit
        # instruction not to invent them; this is belt-and-suspenders in
        # case it does anyway).
        rich_data["moment_id"] = _make_moment_id(span)
        rich_data["time_stamp"] = _format_timestamp_range(float(start_ts or 0.0), float(end_ts or 0.0))

        return raw_score, justification, self.MODEL_VERSION, clip_url, rich_data


class MockScorer(ScorerAdapter):
    """Deterministic stand-in for Step 1 — zero network/GPU dependency.

    Derives a reproducible pseudo-score from a hash of the span's start_ts +
    transcript_excerpt plus the excerpt's length, so tests get stable,
    comparable output without calling any real model. Always returns
    clip_url=None — no real video is ever touched here. rich_data is a
    deterministic fake matching Gemma4Scorer's own shape (same top-level
    keys), so downstream code (workers.ranker, api) can treat both
    implementations' output identically without a None-shape special case.
    """

    MODEL_VERSION = "mock-scorer-v1"

    def score(self, span: dict) -> tuple[float, str, str, str | None, dict]:
        start_ts = float(span.get("start_ts", 0.0))
        end_ts = float(span.get("end_ts", start_ts))
        excerpt = span.get("transcript_excerpt") or ""

        digest = hashlib.sha256(f"{start_ts:.3f}:{excerpt}".encode("utf-8")).hexdigest()
        digest_component = int(digest[:8], 16) / 0xFFFFFFFF  # stable float in [0, 1]
        length_component = min(len(excerpt), 200) / 200.0

        raw_score = round(0.7 * digest_component + 0.3 * length_component, 6)
        justification = (
            f"mock score derived from start_ts={start_ts:.2f} "
            f"and transcript_excerpt length={len(excerpt)}"
        )
        rich_data = {
            "moment_id": _make_moment_id(span),
            "moment_title": "Mock moment",
            "moment_description": justification,
            "moment_score": raw_score,
            "moment_location": None,
            "action_took": None,
            "time_stamp": _format_timestamp_range(start_ts, end_ts),
            "dialogue_by_character": [],
            "props_involved": [],
            "impact_of_action": None,
            "characters_present": [],
            "sound_design": None,
            "cinematography": None,
            "ad_placement": None,
            "risk_hints": None,
            "inversion_score": None,
        }
        return raw_score, justification, self.MODEL_VERSION, None, rich_data


def get_scorer() -> ScorerAdapter:
    """Single call site controlling which ScorerAdapter implementation is
    active. Config-driven rather than a manual code flip: Gemma4Scorer
    requires gemma4_endpoint_url, so its mere presence in config IS the
    Step 1 -> Step 2 switch — no caller (workers.scorer.tasks) needs to
    change either way."""
    settings = get_settings()
    if settings.gemma4_endpoint_url:
        return Gemma4Scorer()
    return MockScorer()
