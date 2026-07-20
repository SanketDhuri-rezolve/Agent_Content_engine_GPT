"""ShotDetector adapter interface.

Step 2 backs this with TransNetV2 running against the actual source video
frames. Step 1 only needs the interface plus a deterministic mock so callers
(workers.segment_worker.tasks) depend only on the `ShotDetector` interface,
never on a concrete implementation.
"""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from models.schemas import ShotBoundary

DEFAULT_SHOT_INTERVAL_SECONDS = 5.0

# --- TransNetV2ShotDetector defaults ----------------------------------------
# 0.5 is the paper's/original repo's own default binarization threshold for
# the per-frame shot-boundary probability — confirmed directly against
# soCzech/TransNetV2's inference/transnetv2.py (`predictions_to_scenes(...,
# threshold: float = 0.5)`), and the transnetv2-pytorch PyPI package mirrors
# the same default on its own predictions_to_scenes/detect_scenes methods.
DEFAULT_TRANSNETV2_THRESHOLD = 0.5
# TransNetV2 was designed/evaluated to run at a source video's native frame
# rate; that rate isn't threaded through the ShotDetector interface (only
# start_ts/end_ts are), so this is a deliberate, DOCUMENTED SIMPLIFICATION
# rather than a fact about the model itself: near-native fps (25) was the
# initial stand-in, but measured on real GPU hardware to make frame
# decode+preprocess volume (near-native-fps frame count over a whole
# segment, not just at cuts) the single largest per-segment cost in this
# pipeline (~48% of segment time — see CLAUDE.md's adapter timing table).
# Lowered to 8fps: still far finer than any real cinematic shot length (even
# fast-cut editing rarely goes below ~0.5-1s per shot; 8fps resolves cut
# timing to ~0.125s, more than adequate for "which few seconds is this
# highlight" purposes), for a ~3x reduction in frames decoded/processed per
# segment. Pass a different sample_fps to the constructor to tune this
# per-deployment if finer boundary precision is ever needed.
DEFAULT_TRANSNETV2_SAMPLE_FPS = 8.0
# Optional env var fallback for TransNetV2ShotDetector(model_path=...),
# mirroring PyannoteDiarizer's *_AUTH_TOKEN env-var convention (see
# diarizer.py).
_TRANSNETV2_MODEL_PATH_ENV_VAR = "TRANSNETV2_MODEL_PATH"

# Thread-pool size for CPU-bound frame decode/resize concurrency (see
# detect() below). See visual_embedder.py's ClipVisualEmbedder for the full
# explanation: `nproc` reports the HOST machine's core count, not what this
# container is actually entitled to — the enforced cgroup CPU quota capped
# it at ~7.65 cores (matching RunPod's own ~9 vCPU console figure for this
# pod tier). Kept intentionally below the full quota since two
# segment_worker processes can run concurrently, each spinning up a pool of
# this size.
_MAX_PREPROCESS_WORKERS = 8


class ShotDetector(ABC):
    @abstractmethod
    def detect(self, source_video_url: str, start_ts: float, end_ts: float) -> list[ShotBoundary]:
        """Return shot boundaries (with keyframe references) inside
        [start_ts, end_ts). start_ts/end_ts are GLOBAL timestamps relative to
        the whole film, not segment-local — implementations must return
        boundaries in that same global timeline."""


class MockShotDetector(ShotDetector):
    """Step 1 stand-in for TransNetV2. Emits one shot boundary every
    `shot_interval_seconds`, deterministically derived from start_ts/end_ts —
    no video is actually read or decoded."""

    def __init__(self, shot_interval_seconds: float = DEFAULT_SHOT_INTERVAL_SECONDS):
        self._interval = shot_interval_seconds

    def detect(self, source_video_url: str, start_ts: float, end_ts: float) -> list[ShotBoundary]:
        if end_ts <= start_ts or self._interval <= 0:
            # Degenerate/zero-duration segment — nothing to detect, not an error.
            return []

        boundaries: list[ShotBoundary] = []
        index = 0
        ts = start_ts
        while ts < end_ts:
            boundaries.append(
                ShotBoundary(ts=round(ts, 4), keyframe_ref=f"kf_{start_ts:.3f}_{index:04d}")
            )
            index += 1
            ts = start_ts + index * self._interval
        return boundaries


class TransNetV2ShotDetector(ShotDetector):
    """Step 2 real implementation backed by TransNetV2 (Souček & Lokoč,
    "TransNet V2: An effective deep network architecture for fast shot
    transition detection", https://arxiv.org/abs/2008.04838) — a GPU-native
    CNN for shot-boundary detection.

    Packaging (researched July 2026 — verified, not guessed): TransNetV2
    itself has no official PyPI package. The original repo
    (github.com/soCzech/TransNetV2) ships a TensorFlow model plus a
    hand-rolled `inference-pytorch/transnetv2_pytorch.py` port with NO
    packaged pretrained-weights file — its own README says to run its
    `convert_weights.py` against the TF weights yourself. However, a
    maintained third-party package DOES exist and was verified directly
    (PyPI's JSON API + unzipping the actual wheel — not just a web-search
    summary): `transnetv2-pytorch` (PyPI project
    https://pypi.org/project/transnetv2-pytorch/, homepage
    https://github.com/allenday/transnetv2_pytorch), currently at 1.0.5
    (released 2025-06-01, requires Python >=3.10). Its wheel bundles the
    ~30MB pretrained weights file
    (`transnetv2_pytorch/transnetv2-pytorch-weights.pth`) inside the package
    itself, auto-loaded the instant `transnetv2_pytorch.TransNetV2(...)` is
    constructed — so, unusually for this codebase's other Step 2 adapters,
    no separate weights download/URL is required to get the stock model
    running. Caveat: this is a single-maintainer, unofficial port (not the
    paper authors' own package) with modest download counts — moderate, not
    absolute, confidence it stays maintained; pin the version and re-check
    before depending on it in production.

    New pip dependency needed (NOT added to pyproject.toml by this change —
    see task/PR notes): `transnetv2-pytorch>=1.0.5`. Pulls in transitively:
    `torch>=1.9.0`, `numpy`, `pillow`, `pandas`, `ffmpeg-python`, `tqdm`.

    Model contract (confirmed from the installed package's own source,
    `transnetv2_pytorch/transnetv2_pytorch.py`):
      - native input: RGB frames resized to 48x27 (W x H), uint8, shape
        [batch, num_frames, 27, 48, 3].
      - windowing/batching is delegated to the package's own
        `model.predict_frames()`, which implements the same sliding-window
        scheme as the original paper/repo — window=100 frames, stride=50,
        25-frame edge padding (repeats of the first/last frame), keeping
        only the middle 50 predictions per window — rather than
        reimplementing that scheme independently here, which would only
        risk silently diverging from the tested upstream implementation.
      - per-frame shot-boundary probability is sigmoid of the model's
        "single_frame" head (`model.predict_frames` already applies the
        sigmoid), thresholded at DEFAULT_TRANSNETV2_THRESHOLD (0.5 — the
        paper's/package's own default, see above).
      - sampling fps: see DEFAULT_TRANSNETV2_SAMPLE_FPS above — a documented
        simplification, not a rediscovered model fact, since this interface
        doesn't thread the source video's real native fps through to us.

    All heavy imports (torch, transnetv2_pytorch, PIL, numpy) are deferred to
    `_get_model`/`detect`, never at module import time — this module is
    imported unconditionally by workers.segment_worker.tasks and a local
    test suite that runs with none of those packages installed (mirrors
    FasterWhisperTranscriber/PyannoteDiarizer's lazy-import pattern
    elsewhere in this package).
    """

    # Class-level cache shared by every instance in this process, keyed by
    # (model_path, device) — loading pretrained weights is expensive and
    # should happen once per process, not once per detect() call (mirrors
    # FasterWhisperTranscriber._MODEL_CACHE in transcriber.py).
    _MODEL_CACHE: dict[tuple[str | None, str], object] = {}

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        threshold: float = DEFAULT_TRANSNETV2_THRESHOLD,
        sample_fps: float = DEFAULT_TRANSNETV2_SAMPLE_FPS,
    ):
        # No torch/transnetv2_pytorch import happens here — only on first
        # detect() call (see _get_model) — so constructing this class is
        # always cheap/safe even without those packages installed.
        #
        # model_path: optional path to an alternate/fine-tuned .pth
        # state_dict to load OVER transnetv2-pytorch's own bundled weights
        # (see _get_model). Left as None — the default, and the expected
        # case — uses the stock pretrained model that ships inside the
        # transnetv2-pytorch wheel; no download or user-supplied file is
        # required for that. Also overridable via the TRANSNETV2_MODEL_PATH
        # env var so ops can swap the checkpoint without a code change (same
        # convention as PyannoteDiarizer's *_AUTH_TOKEN env-var fallback in
        # diarizer.py).
        self._model_path = model_path or os.environ.get(_TRANSNETV2_MODEL_PATH_ENV_VAR)
        self._device = device
        self._threshold = threshold
        self._sample_fps = sample_fps

    def _get_model(self):
        cache_key = (self._model_path, self._device)
        cached = TransNetV2ShotDetector._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        import torch
        from transnetv2_pytorch import TransNetV2 as _TransNetV2Model  # deferred: heavy/optional dependency

        model = _TransNetV2Model(device=self._device)
        if self._model_path:
            # Override the package's bundled weights with a caller-supplied
            # checkpoint. torch.load raises on a missing/bad path — allowed
            # to propagate: workers.segment_worker.tasks already fail-softs
            # every adapter call, so this detector must not add its own
            # try/except (see task notes / module docstring convention used
            # throughout this package's other Step 2 adapters).
            state_dict = torch.load(self._model_path, map_location=model.device)
            model.load_state_dict(state_dict)
        model.eval()
        TransNetV2ShotDetector._MODEL_CACHE[cache_key] = model
        return model

    def detect(self, source_video_url: str, start_ts: float, end_ts: float) -> list[ShotBoundary]:
        if end_ts <= start_ts:
            # Degenerate/zero-duration segment — nothing to detect, not an
            # error (consistent with MockShotDetector and
            # extracted_frames_uniform's own handling).
            return []

        import numpy as np
        import torch
        from PIL import Image

        from workers.segment_worker.adapters._media import extracted_frames_uniform, make_keyframe_ref

        model = self._get_model()

        with extracted_frames_uniform(source_video_url, start_ts, end_ts, self._sample_fps) as frame_paths:
            if not frame_paths:
                return []

            # Resize to TransNetV2's native 48x27 (W x H) RGB input. At the
            # default 25fps sample rate a single segment can produce
            # thousands of frame files (near-native fps for most movie
            # footage) — decoding/resizing them was a sequential Python loop,
            # measured as a significant share of detect()'s ~50-80s cost on
            # real GPU hardware. PIL's decode/resize release the GIL for
            # their actual C-level work, so (like ClipVisualEmbedder's frame
            # loading — see that adapter for why this differs from the
            # earlier failed attempt to parallelize GPU model calls
            # themselves) this is genuine CPU-bound work that benefits from
            # real thread concurrency.
            def _load_and_resize(path: str):
                return np.asarray(Image.open(path).convert("RGB").resize((48, 27)), dtype=np.uint8)

            with ThreadPoolExecutor(max_workers=min(len(frame_paths), _MAX_PREPROCESS_WORKERS)) as pool:
                frame_arrays = list(pool.map(_load_and_resize, frame_paths))
            frames = torch.from_numpy(np.stack(frame_arrays)).to(model.device)

            # model.predict_frames handles the window=100/stride=50/pad=25
            # sliding-window batching itself (see class docstring) and
            # already applies sigmoid — single_frame_pred is the per-frame
            # shot-boundary probability, one entry per input frame, in the
            # same order as frame_paths/frame_arrays.
            single_frame_pred, _all_frame_pred = model.predict_frames(frames, quiet=True)
            probs = single_frame_pred.detach().cpu().numpy()

        boundaries: list[ShotBoundary] = []
        for frame_index, prob in enumerate(probs):
            if prob > self._threshold:
                # frame_index is LOCAL to this sampled sequence — recover
                # the GLOBAL timestamp per the ShotDetector contract.
                ts = start_ts + frame_index / self._sample_fps
                boundaries.append(ShotBoundary(ts=round(ts, 4), keyframe_ref=make_keyframe_ref(ts)))
        return boundaries
