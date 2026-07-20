"""VisualEmbedder adapter interface.

Step 2 backs this with InternVideo2, run only against the keyframes
ShotDetector emits for a segment (never against raw/all frames — embedding
every frame would be prohibitively expensive). Step 1 only needs the
interface plus a deterministic mock.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from workers.segment_worker.adapters._determinism import stable_vector
from workers.segment_worker.adapters._media import extracted_frame, parse_keyframe_timestamp

DEFAULT_EMBEDDING_DIM = 16

# --- InternVideo2VisualEmbedder defaults ------------------------------------
# Researched July 2026 — verified directly against the model's own HF model
# card + raw config.json + raw modeling_internvideo2.py (not just a web-search
# summary). See InternVideo2VisualEmbedder's docstring for full citations.
DEFAULT_INTERNVIDEO2_MODEL = "OpenGVLab/InternVideo2-Stage2_6B"
# config.json: "image_res"/"img_size" = 224, "patch_size" = 14.
_INTERNVIDEO2_FRAME_SIZE = 224
# config.json: "num_frames" = 4 — the temporal window this checkpoint's
# position embeddings were fit to. Read from the loaded model's own
# `model.config.num_frames` at runtime when available, falling back to this.
_INTERNVIDEO2_DEFAULT_NUM_FRAMES = 4
# Standard ImageNet normalization stats — confirmed against the reference
# preprocessing in OpenGVLab/InternVideo's own
# InternVideo2/multi_modality/demo/utils.py (`frames2tensor`), which backs
# this same model family's official demo.
_INTERNVIDEO2_MEAN = (0.485, 0.456, 0.406)
_INTERNVIDEO2_STD = (0.229, 0.224, 0.225)

# --- ClipVisualEmbedder defaults --------------------------------------------
# openai/clip-vit-base-patch32: ~151M params (vs. InternVideo2-Stage2_6B's 6B)
# — measured ~47.6s/keyframe-batch on a real A40 for InternVideo2 vs. this,
# a single-image encoder that doesn't need the "tile a static frame into a
# fake video clip" workaround InternVideo2 requires. See ClipVisualEmbedder's
# docstring for why this replaced InternVideo2 as the default real adapter.
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Thread-pool size for CPU-bound frame-loading concurrency (see embed()
# below). Confirmed on real GPU hardware: `nproc`/`/proc/cpuinfo` report the
# HOST machine's full core count (96 on our test pod), but that is NOT what
# the container is actually entitled to — the enforced cgroup CPU quota
# (cpu.cfs_quota_us / cpu.cfs_period_us) capped it at ~7.65 cores, matching
# the ~9 vCPUs shown in RunPod's own console for this pod tier. Sizing the
# pool to 96 (or any number well above the real quota) doesn't add genuine
# throughput past the quota ceiling — the kernel throttles back down to it
# regardless of thread count — it just adds scheduling overhead. Also note
# TWO segment_worker processes can run concurrently (concurrency=2), each
# spinning up its own pool of this size, so this is intentionally
# conservative relative to the full quota, not equal to it.
_MAX_PREPROCESS_WORKERS = 8


class VisualEmbedder(ABC):
    @abstractmethod
    def embed(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[float]]:
        """Return {keyframe_ref: embedding_vector}. Only ever called with
        keyframe_refs produced by a ShotDetector for the same segment."""

    def embed_faces(
        self, source_video_url: str, face_detections: dict[str, list[dict]]
    ) -> dict[str, list[float]]:
        """Return {face_id: embedding_vector} for detected faces (see
        face_detector.py's FaceBox shape) — foundation for character
        identification (cross-segment clustering/naming happens later, in
        the reducer). Concrete (not abstract): InternVideo2VisualEmbedder
        (legacy, no longer the default — see ClipVisualEmbedder) never
        implemented this, and forcing it to would serve no purpose now that
        nothing constructs it by default. Only MockVisualEmbedder and
        ClipVisualEmbedder override this."""
        raise NotImplementedError(f"{type(self).__name__} does not support embed_faces")


class MockVisualEmbedder(VisualEmbedder):
    """Step 1 stand-in for InternVideo2. Emits a fixed-length deterministic
    fake vector per keyframe ref (hash of the ref) — identical keyframe refs
    always produce identical embeddings, and no frame is actually decoded or
    run through a model."""

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        self._dim = embedding_dim

    def embed(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[float]]:
        return {ref: stable_vector(self._dim, "visual_embedding", ref) for ref in keyframe_refs}

    def embed_faces(
        self, source_video_url: str, face_detections: dict[str, list[dict]]
    ) -> dict[str, list[float]]:
        face_ids = [face["face_id"] for faces in face_detections.values() for face in faces]
        return {face_id: stable_vector(self._dim, "face_embedding", face_id) for face_id in face_ids}


class InternVideo2VisualEmbedder(VisualEmbedder):
    """Step 2 real implementation backed by OpenGVLab's InternVideo2 video
    foundation model (Wang et al., "InternVideo2: Scaling Video Foundation
    Models for Multimodal Video Understanding", https://arxiv.org/abs/2403.15377).

    Model/loading (researched July 2026, verified directly — not guessed):
    the checkpoint used here is
    "OpenGVLab/InternVideo2-Stage2_6B" (https://huggingface.co/OpenGVLab/InternVideo2-Stage2_6B),
    a public (non-gated, MIT-licensed) HF repo whose OWN model card documents
    loading it via plain `transformers.AutoModel`:

        from transformers import AutoModel
        model = AutoModel.from_pretrained("OpenGVLab/InternVideo2-Stage2_6B",
                                           trust_remote_code=True).eval()

    Other InternVideo2 variants (e.g. InternVideo2-CLIP-1B/6B-224p-f8) were
    considered but rejected for this adapter: their own repos require the
    full github.com/OpenGVLab/InternVideo checkout plus a hand-rolled
    `setup_internvideo2()`/manual `torch.load()` checkpoint flow rather than
    a documented `AutoModel.from_pretrained(...)` path, so picking one of
    those here would have meant guessing an unconfirmed loading procedure.

    CAVEATS the user should verify before relying on this in production:
      - This checkpoint's remote modeling code (modeling_internvideo2.py,
        confirmed via its raw source) has a hard top-level
        `from flash_attn...` import chain — the `flash-attn` pip package
        MUST be installed (GPU-only, CUDA-toolchain/compiler-sensitive, slow
        to build) or `AutoModel.from_pretrained(...)` will fail at import
        time before any of our code runs.
      - A HF discussion on this exact model
        (huggingface.co/OpenGVLab/InternVideo2-Stage2_6B/discussions/4,
        reported 2025-04-17, no visible fix in that thread as of this
        research) reports the bundled remote code loads config files
        (e.g. "configs/config_bert_large.json") via paths that are only
        valid when the current working directory is the repo root — i.e.
        `trust_remote_code=True` loading may raise `FileNotFoundError`
        depending on the process's cwd when this adapter is instantiated.
        Not worked around here (unverifiable without a GPU/flash-attn
        environment) — test `AutoModel.from_pretrained(...)` directly in the
        actual deployment environment first.
      - This is a 6B-parameter multimodal (video+text) model — heavyweight
        for a per-keyframe embedder. Only its vision tower + projection head
        are exercised here (get_vid_feat), but the full model (including its
        BERT-large text tower) is loaded into memory regardless, since
        AutoModel loads the whole `InternVideo2_Stage2` module.
      - InternVideo2 is a *video*, not single-image, encoder: per config.json
        it expects `num_frames` (4 for this checkpoint) frames per clip, with
        position embeddings fit to that count. Since VisualEmbedder is only
        ever given single keyframe timestamps, each extracted frame is tiled
        into a static num_frames-long "clip" (every frame identical) before
        being run through the model — the standard "image-as-video" trick
        for adapting video transformers to single-frame input. This is a
        deliberate, documented simplification, not a confirmed fact about
        how InternVideo2 "should" be fed a single frame.
      - Embedding dimension: 512, per this checkpoint's own config.json
        top-level `"embed_dim"` field — the shared vision/text projection
        space `get_vid_feat`/`get_txt_feat` return (used for cosine-similarity
        video-text retrieval in the model's own demo). Confirmed from
        config.json directly; NOT independently confirmed against
        get_vid_feat's literal forward-pass source, so double-check
        `len(embedding)` at runtime.

    All heavy imports (torch, transformers, PIL, numpy) are deferred to
    `_get_model`/`embed`, never at module import time — this module is
    imported unconditionally by workers.segment_worker.tasks and by a large
    local test suite that runs with none of those packages installed
    (mirrors FasterWhisperTranscriber/PyannoteDiarizer/TransNetV2ShotDetector's
    lazy-import pattern elsewhere in this package).

    Batching: keyframes are embedded one at a time (a simple loop), not
    batched together into a single forward pass — not confident enough in
    get_vid_feat's batching semantics (batch dim vs. temporal dim, given the
    "code not portable"/undocumented-shape caveats above) to risk silently
    mixing them up; a loop is correct even if slower.
    """

    # Class-level cache shared by every instance in this process, keyed by
    # (model_name_or_path, device) — loading a 6B-parameter checkpoint is
    # expensive and should happen once per process, not once per embed()
    # call (mirrors TransNetV2ShotDetector._MODEL_CACHE / FasterWhisperTranscriber._MODEL_CACHE).
    _MODEL_CACHE: dict[tuple[str, str], object] = {}

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_INTERNVIDEO2_MODEL,
        device: str = "auto",
    ):
        # No torch/transformers import happens here — only on first embed()
        # call (see _get_model) — so constructing this class is always cheap
        # and safe even without those packages installed.
        self._model_name_or_path = model_name_or_path
        self._device = device

    def _resolve_device(self, torch_module) -> str:
        if self._device != "auto":
            return self._device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _get_model(self):
        import torch
        from transformers import AutoModel  # deferred: heavy/optional dependency

        device = self._resolve_device(torch)
        cache_key = (self._model_name_or_path, device)
        cached = InternVideo2VisualEmbedder._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # Compat shim, confirmed necessary on real GPU hardware (transformers
        # 5.13.1 AND 4.57.6 both hit this): this checkpoint's remote code
        # (trust_remote_code=True) does `from transformers.modeling_utils
        # import (PreTrainedModel, apply_chunking_to_forward,
        # find_pruneable_heads_and_indices, prune_linear_layer)` — confirmed
        # via the actual cached remote source, line ~1096. Current
        # transformers releases relocated the three non-PreTrainedModel
        # helpers to transformers.pytorch_utils and dropped the
        # modeling_utils re-export. Patch all three back rather than pinning
        # to a fragile old transformers version that may not stay installable.
        import transformers.modeling_utils as _modeling_utils
        import transformers.pytorch_utils as _pytorch_utils

        for _symbol in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(_modeling_utils, _symbol) and hasattr(_pytorch_utils, _symbol):
                setattr(_modeling_utils, _symbol, getattr(_pytorch_utils, _symbol))

        # Two more real bugs confirmed on GPU hardware, both in the
        # checkpoint's own remote code, both fixed by pre-staging files it
        # expects to already exist rather than fetch itself:
        #
        # 1. It loads its text encoder via
        #    `BertTokenizer.from_pretrained("bert-large-uncased",
        #    local_files_only=True, ...)` — hardcoded local_files_only=True,
        #    so on a fresh machine (nothing cached yet) this fails. Pre-fetch
        #    it once here so it's already in the HF cache by the time the
        #    remote code asks for it "locally only".
        from transformers import BertTokenizer

        BertTokenizer.from_pretrained("bert-large-uncased")

        # 2. It loads its BERT config via a bare RELATIVE path,
        #    `BertConfig.from_json_file("configs/config_bert_large.json")` —
        #    only works if the current process's cwd happens to already
        #    contain that exact relative path (an assumption from the
        #    original GitHub repo's layout, not valid for an arbitrary
        #    AutoModel.from_pretrained() call from any other directory). The
        #    file IS a real repo file on the HF Hub (confirmed), just not
        #    fetched by transformers' normal dynamic-module download (which
        #    only pulls the .py files) — so fetch it explicitly via
        #    huggingface_hub and place it at that same relative path under
        #    the CURRENT working directory before construction.
        import os

        from huggingface_hub import hf_hub_download

        _bert_config_relpath = "configs/config_bert_large.json"
        if not os.path.isfile(_bert_config_relpath):
            _cached_path = hf_hub_download(
                repo_id=self._model_name_or_path, filename=_bert_config_relpath
            )
            os.makedirs(os.path.dirname(_bert_config_relpath), exist_ok=True)
            import shutil

            shutil.copyfile(_cached_path, _bert_config_relpath)

        model = AutoModel.from_pretrained(self._model_name_or_path, trust_remote_code=True)
        model = model.to(device).eval()
        InternVideo2VisualEmbedder._MODEL_CACHE[cache_key] = model
        return model

    def embed(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[float]]:
        # Resolve each ref's GLOBAL timestamp up front, skipping any ref that
        # doesn't match the real-adapter kf_t<timestamp> contract (e.g. a
        # Step-1 Mock-style ref, which only happens in test fixtures) instead
        # of crashing the whole call.
        timestamps_by_ref: dict[str, float] = {}
        for ref in keyframe_refs:
            try:
                timestamps_by_ref[ref] = parse_keyframe_timestamp(ref)
            except ValueError:
                continue

        if not timestamps_by_ref:
            return {}

        import numpy as np
        import torch
        from PIL import Image

        model = self._get_model()
        device = self._resolve_device(torch)
        model_dtype = next(model.parameters()).dtype
        num_frames = getattr(model.config, "num_frames", _INTERNVIDEO2_DEFAULT_NUM_FRAMES)

        mean = np.array(_INTERNVIDEO2_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(_INTERNVIDEO2_STD, dtype=np.float32).reshape(1, 1, 3)

        embeddings: dict[str, list[float]] = {}
        with torch.no_grad():
            for ref, timestamp in timestamps_by_ref.items():
                with extracted_frame(source_video_url, timestamp) as frame_path:
                    image = Image.open(frame_path).convert("RGB").resize(
                        (_INTERNVIDEO2_FRAME_SIZE, _INTERNVIDEO2_FRAME_SIZE)
                    )
                    array = np.asarray(image, dtype=np.float32) / 255.0
                    array = (array - mean) / std
                    # HWC -> CHW, then tile the single frame into a static
                    # num_frames-long clip (see class docstring) and add the
                    # batch dim: [1, T, C, H, W].
                    frame_tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
                    clip = frame_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)
                    batch = clip.unsqueeze(0).to(device=device, dtype=model_dtype)

                    vid_feat = model.get_vid_feat(batch)
                    embeddings[ref] = vid_feat.reshape(-1).float().cpu().tolist()

        return embeddings


class ClipVisualEmbedder(VisualEmbedder):
    """Step 2 real implementation backed by OpenAI CLIP (ViT-B/32), replacing
    InternVideo2VisualEmbedder as the default real VisualEmbedder.

    Why: measured on real GPU hardware (RunPod A40), InternVideo2-Stage2_6B
    took ~47.6s to embed one segment's keyframes and required ~24-37GB of
    VRAM per worker process — a 6B-parameter video+text foundation model
    (loading a full BERT-large text tower it never uses here) applied to a
    task that only ever gets single static keyframes, not real video clips
    (VisualEmbedder.embed's contract is one image per keyframe_ref; the
    InternVideo2 adapter had to tile each frame into a fake static "clip" to
    even feed it in — see InternVideo2VisualEmbedder's docstring). That
    mismatch made it both the single largest per-segment latency cost AND
    the reason a single GPU could only run one segment_worker process at a
    time (two InternVideo2 copies alone caused a CUDA OOM on a 44GB A40 —
    see CLAUDE.md).

    CLIP ViT-B/32 (~151M params) is a plain single-image encoder — no fake
    "video" framing needed, no trust_remote_code, no flash-attn dependency,
    no BERT-tokenizer/config-file workarounds InternVideo2 required. It also
    embeds into a joint image/text space (useful for Qdrant's semantic
    search over highlight moments, unlike InternVideo2's video-text space
    which this adapter never used the text side of anyway).

    Trade-off: CLIP has no temporal/video-specific understanding — it embeds
    each keyframe as a plain image, same as InternVideo2VisualEmbedder
    effectively did in practice (via the frame-tiling workaround). This is
    not believed to be a quality regression for keyframe-level embeddings
    specifically; it has not been evaluated for retrieval/scoring quality
    against InternVideo2's output on this project's real data.

    Batching: unlike InternVideo2VisualEmbedder (looped one keyframe at a
    time — see that class's docstring for why), CLIPProcessor's batching
    semantics are simple and well-documented (a plain list-of-images ->
    batched pixel_values), so all of a segment's keyframes are embedded in
    one forward pass here.
    """

    _MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str = "auto"):
        # No torch/transformers import happens here — only on first embed()
        # call (see _get_model_and_processor) — mirrors every other Real*
        # adapter's lazy-import pattern in this package.
        self._model_name = model_name
        self._device = device

    def _resolve_device(self, torch_module) -> str:
        if self._device != "auto":
            return self._device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _get_model_and_processor(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor  # deferred heavy import

        device = self._resolve_device(torch)
        cache_key = (self._model_name, device)
        cached = ClipVisualEmbedder._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        model = CLIPModel.from_pretrained(self._model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(self._model_name)
        ClipVisualEmbedder._MODEL_CACHE[cache_key] = (model, processor)
        return model, processor

    def _load_frame(self, source_video_url: str, timestamp: float):
        from PIL import Image

        with extracted_frame(source_video_url, timestamp) as frame_path:
            # .copy() (via .load() below) forces the pixel data to be read
            # into memory before the temp file is deleted on context exit —
            # required since this runs in a worker thread whose extracted_frame
            # context closes as soon as this function returns.
            image = Image.open(frame_path).convert("RGB")
            image.load()
            return image

    def embed(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[float]]:
        timestamps_by_ref: dict[str, float] = {}
        for ref in keyframe_refs:
            try:
                timestamps_by_ref[ref] = parse_keyframe_timestamp(ref)
            except ValueError:
                continue

        if not timestamps_by_ref:
            return {}

        import torch

        model, processor = self._get_model_and_processor()
        device = self._resolve_device(torch)

        refs = list(timestamps_by_ref.keys())
        # Frame extraction is one `ffmpeg` subprocess per keyframe — CPU/disk
        # I/O work, not GPU work, so (unlike the earlier attempt to
        # parallelize the six GPU adapters, which made things WORSE — see
        # build_segment_output's docstring) this genuinely benefits from
        # concurrency: each ffmpeg child process runs on its own CPU core
        # (measured segment_worker: shot_detector alone found 30-50+ shots
        # per segment, i.e. 30-50+ sequential ffmpeg calls previously — see
        # CLAUDE.md's per-adapter timing table showing visual_embedder
        # scaling directly with shot count).
        with ThreadPoolExecutor(max_workers=min(len(refs), _MAX_PREPROCESS_WORKERS)) as pool:
            images = list(pool.map(lambda ref: self._load_frame(source_video_url, timestamps_by_ref[ref]), refs))

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        return {ref: vec for ref, vec in zip(refs, image_features.float().cpu().tolist())}

    def embed_faces(
        self, source_video_url: str, face_detections: dict[str, list[dict]]
    ) -> dict[str, list[float]]:
        """Embeds each detected face CROP (see face_detector.py's FaceBox
        shape: {face_id, bbox: [x0,y0,x1,y1] normalized, confidence}) using
        this SAME already-loaded CLIP model/processor — no second visual
        model for character identification. Returns {face_id: embedding}.

        Character identification (the actual clustering/naming) happens
        later, cross-segment, in the reducer — this only produces the raw
        per-face embeddings for that later step to consume.
        """
        # (keyframe_ref, timestamp, face_id, bbox) tuples — one keyframe can
        # have zero, one, or several faces.
        face_entries: list[tuple[str, float, str, list[float]]] = []
        for ref, faces in face_detections.items():
            try:
                timestamp = parse_keyframe_timestamp(ref)
            except ValueError:
                continue
            for face in faces:
                face_entries.append((ref, timestamp, face["face_id"], face["bbox"]))

        if not face_entries:
            return {}

        import torch

        model, processor = self._get_model_and_processor()
        device = self._resolve_device(torch)

        # Group by keyframe so each frame is only decoded once even if it
        # has multiple faces.
        refs_needed = {ref: ts for ref, ts, _, _ in face_entries}
        with ThreadPoolExecutor(max_workers=min(len(refs_needed), _MAX_PREPROCESS_WORKERS)) as pool:
            ref_list = list(refs_needed.keys())
            frames = list(pool.map(lambda ref: self._load_frame(source_video_url, refs_needed[ref]), ref_list))
        frames_by_ref = dict(zip(ref_list, frames))

        face_ids: list[str] = []
        crops = []
        for ref, _ts, face_id, bbox in face_entries:
            frame = frames_by_ref[ref]
            width, height = frame.size
            x0, y0, x1, y1 = bbox
            crop_box = (
                max(0, int(x0 * width)),
                max(0, int(y0 * height)),
                min(width, int(x1 * width)),
                min(height, int(y1 * height)),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                continue  # degenerate/zero-area box — skip rather than crash
            face_ids.append(face_id)
            crops.append(frame.crop(crop_box))

        if not crops:
            return {}

        inputs = processor(images=crops, return_tensors="pt").to(device)
        with torch.no_grad():
            face_features = model.get_image_features(**inputs)

        return {fid: vec for fid, vec in zip(face_ids, face_features.float().cpu().tolist())}
