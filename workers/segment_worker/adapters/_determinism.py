"""Deterministic pseudo-random helpers shared by the Step 1 mock adapters.

Every mock adapter in this package derives its fake output purely from its
inputs (a segment's global start_ts/end_ts, a keyframe ref, etc.) via a
stable hash — never from `random`/`numpy.random` — so the exact same payload
always produces the exact same output. That determinism is required by
tests/test_segment_worker.py and matters downstream too: the reducer dedups
spans across segments using their (global) timestamps, and a flaky/random
segment_worker output would make that dedup logic untestable and the
pipeline non-reproducible.
"""

import hashlib

_HASH_HEX_DIGITS = 12  # first 12 hex chars of the sha256 digest -> 48 bits of entropy, plenty for fake fixtures
_HASH_HEX_MAX = 16**_HASH_HEX_DIGITS


def stable_unit_interval(*parts: object, salt: int = 0) -> float:
    """Deterministic value in [0, 1) derived from `parts` (+ salt to
    decorrelate multiple values drawn from the same parts, e.g. successive
    components of a fake embedding vector)."""
    key = "|".join(str(p) for p in parts) + f"|salt={salt}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:_HASH_HEX_DIGITS], 16) / float(_HASH_HEX_MAX)


def stable_signed_unit(*parts: object, salt: int = 0) -> float:
    """Deterministic value in [-1, 1) — a convenient range for fake
    embedding components."""
    return stable_unit_interval(*parts, salt=salt) * 2.0 - 1.0


def stable_vector(dim: int, *parts: object) -> list[float]:
    """Deterministic fixed-length fake vector, one independent component per
    salt index, derived from `parts`."""
    return [round(stable_signed_unit(*parts, salt=i), 6) for i in range(dim)]
