"""FaceDetector adapter interface.

Step towards character identification (see CLAUDE.md): before any face can be
embedded/clustered/named, we need bounding boxes for where faces actually are
within each keyframe ShotDetector already emits — this adapter only answers
"where are the faces", never "who is this". Downstream, VisualEmbedder (CLIP)
is reused to embed each detected face CROP (not the whole frame) into the
same embedding space it already produces for keyframes — no second visual
model is introduced by this feature.

Face boxes are returned in NORMALIZED [0, 1] coordinates (fraction of frame
width/height), not pixel coordinates — keeps this adapter's output
resolution-independent of whatever frame size a given source video happens
to have, consistent with how downstream consumers (embedding crop
extraction) don't need to know the original frame's pixel dimensions ahead
of time.
"""

from abc import ABC, abstractmethod

from workers.segment_worker.adapters._determinism import stable_unit_interval
from workers.segment_worker.adapters._media import extracted_frame, parse_keyframe_timestamp

# --- MTCNNFaceDetector defaults ---------------------------------------------
# MTCNN (Zhang et al., "Joint Face Detection and Alignment using Multi-task
# Cascaded Convolutional Networks", 2016) via the `facenet-pytorch` package
# (https://github.com/timesler/facenet-pytorch) — a small (~2M param), fast,
# widely-used, actively-maintained pretrained face detector with a plain
# `MTCNN().detect(image) -> (boxes, probs)` API. Chosen over a heavier/newer
# detector specifically because this adapter only needs "where is the face",
# not state-of-the-art accuracy — MTCNN is a well-established, low-risk
# choice for that narrower job.
DEFAULT_MIN_FACE_CONFIDENCE = 0.90


class FaceBox:
    """Not a pydantic model — this adapter returns plain dicts (matching
    every other Step 2 adapter's SegmentWorkerOutput-bound dict convention)
    so this class exists only to document the shape:
        {"face_id": str, "bbox": [x0, y0, x1, y1], "confidence": float}
    bbox is normalized [0, 1] (fraction of frame width/height), corners
    (x0, y0) top-left / (x1, y1) bottom-right — NOT (x, y, w, h) — matching
    facenet-pytorch's own MTCNN.detect() box convention directly, so no
    coordinate-convention translation happens inside this adapter."""


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[dict]]:
        """Return {keyframe_ref: [FaceBox, ...]} — only ever called with
        keyframe_refs produced by a ShotDetector for the same segment (same
        contract VisualEmbedder.embed follows)."""


class MockFaceDetector(FaceDetector):
    """Step 1 stand-in. Emits a deterministic 0 or 1 fake face per keyframe
    ref (hash of the ref decides presence), no frame is actually decoded or
    run through a model."""

    def detect(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for ref in keyframe_refs:
            has_face = stable_unit_interval(ref, "mock_face_presence") > 0.5
            if not has_face:
                result[ref] = []
                continue
            cx = stable_unit_interval(ref, "mock_face_cx")
            cy = stable_unit_interval(ref, "mock_face_cy")
            half = 0.1
            result[ref] = [
                {
                    "face_id": f"{ref}_face0",
                    "bbox": [
                        max(0.0, cx - half),
                        max(0.0, cy - half),
                        min(1.0, cx + half),
                        min(1.0, cy + half),
                    ],
                    "confidence": round(0.9 + stable_unit_interval(ref, "mock_face_conf") * 0.1, 4),
                }
            ]
        return result


class MTCNNFaceDetector(FaceDetector):
    """Step 2 real implementation backed by facenet-pytorch's MTCNN.

    All heavy imports (torch, facenet_pytorch, PIL) are deferred to
    `_get_model`/`detect`, never at module import time — mirrors every other
    Real* adapter's lazy-import pattern in this package (see
    TransNetV2ShotDetector/ClipVisualEmbedder for the same convention).
    """

    _MODEL_CACHE: dict[str, object] = {}

    def __init__(self, device: str = "auto", min_confidence: float = DEFAULT_MIN_FACE_CONFIDENCE):
        self._device = device
        self._min_confidence = min_confidence

    def _resolve_device(self, torch_module) -> str:
        if self._device != "auto":
            return self._device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _get_model(self):
        import torch
        from facenet_pytorch import MTCNN  # deferred: heavy/optional dependency

        device = self._resolve_device(torch)
        cached = MTCNNFaceDetector._MODEL_CACHE.get(device)
        if cached is not None:
            return cached

        # keep_all=True: return every detected face above threshold, not
        # just the single most-confident one — a keyframe can (and often
        # does, per the target output schema's characters_present list)
        # contain multiple characters at once.
        model = MTCNN(keep_all=True, device=device)
        MTCNNFaceDetector._MODEL_CACHE[device] = model
        return model

    def detect(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[dict]]:
        timestamps_by_ref: dict[str, float] = {}
        for ref in keyframe_refs:
            try:
                timestamps_by_ref[ref] = parse_keyframe_timestamp(ref)
            except ValueError:
                continue

        if not timestamps_by_ref:
            return {}

        from PIL import Image

        model = self._get_model()
        result: dict[str, list[dict]] = {}

        for ref, timestamp in timestamps_by_ref.items():
            with extracted_frame(source_video_url, timestamp) as frame_path:
                image = Image.open(frame_path).convert("RGB")
                width, height = image.size
                boxes, probs = model.detect(image)

            faces: list[dict] = []
            if boxes is not None:
                for index, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob is None or prob < self._min_confidence:
                        continue
                    x0, y0, x1, y1 = box
                    faces.append(
                        {
                            "face_id": f"{ref}_face{index}",
                            "bbox": [
                                max(0.0, x0 / width),
                                max(0.0, y0 / height),
                                min(1.0, x1 / width),
                                min(1.0, y1 / height),
                            ],
                            "confidence": round(float(prob), 4),
                        }
                    )
            result[ref] = faces

        return result
