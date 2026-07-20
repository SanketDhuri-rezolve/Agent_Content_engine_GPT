"""ObjectDetector adapter interface.

First step towards the object/commerce roadmap discussed in CLAUDE.md: find
CLEAR SHOPPABLE ITEMS in a highlight's keyframes — no vector embedding, no
catalog matching, no significance/ranking scoring yet. Just "is there a
recognizable, purchasable-looking object here, and where." Those later
stages (product matching, ranking, brand catalogue) all depend on this
existing first, so this is deliberately scoped to detection only.

Backed by Grounding DINO (Liu et al., "Grounding DINO: Marrying DINO with
Grounded Pre-Training for Open-Set Object Detection", 2023) — an
open-vocabulary detector queried with a fixed TEXT PROMPT of shoppable
category phrases, rather than a closed-set detector needing per-category
training. This lets us ask specifically for shoppable things instead of
detecting everything in frame and filtering after.
"""

from abc import ABC, abstractmethod

from workers.segment_worker.adapters._determinism import stable_unit_interval
from workers.segment_worker.adapters._media import extracted_frame, parse_keyframe_timestamp

# --- GroundingDinoObjectDetector defaults -----------------------------------
# "grounding-dino-tiny" — the smallest/fastest official checkpoint, available
# directly via transformers' AutoModelForZeroShotObjectDetection (no custom
# remote code, unlike InternVideo2's trust_remote_code headaches earlier in
# this project) — https://huggingface.co/IDEA-Research/grounding-dino-tiny.
DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Grounding DINO's own documented prompt convention: lowercase category
# phrases separated by ". " (a period + space) — confirmed against the
# model's own HF model card usage example. Deliberately a fixed, broad-but-
# focused list of common purchasable product categories, not "detect
# everything" — the whole point of using an open-vocabulary detector here is
# querying specifically for shoppable things, not post-filtering a
# general-purpose detector's full output.
DEFAULT_SHOPPABLE_CATEGORIES = [
    "bottle", "bag", "backpack", "handbag", "wallet", "shoe", "sneaker",
    "watch", "sunglasses", "jewelry", "necklace", "ring", "earring",
    "phone", "laptop", "headphones", "earphones", "camera", "mug", "cup",
    "jacket", "shirt", "dress", "hat", "cap", "perfume", "bottle of perfume",
    "book", "bicycle", "car", "chair", "lamp", "toy", "guitar",
]

DEFAULT_MIN_OBJECT_CONFIDENCE = 0.35
# Grounding DINO also scores how well a detected box matches the specific
# text phrase (as opposed to just "is this a box"); both thresholds matter —
# a box can be confidently a box but a poor match for any given phrase.
DEFAULT_MIN_TEXT_MATCH_CONFIDENCE = 0.25


def _prompt_text(categories: list[str]) -> str:
    return ". ".join(categories) + "."


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[dict]]:
        """Return {keyframe_ref: [DetectedObject, ...]} where DetectedObject
        is {"label": str, "bbox": [x0,y0,x1,y1] normalized [0,1] (top-left/
        bottom-right corners), "confidence": float}. Only ever called with
        keyframe_refs produced by a ShotDetector for the same segment (same
        contract VisualEmbedder.embed / FaceDetector.detect follow)."""


class MockObjectDetector(ObjectDetector):
    """Step 1 stand-in. Emits a deterministic 0 or 1 fake shoppable object
    per keyframe ref (hash of the ref decides presence/category), no frame
    is actually decoded or run through a model."""

    def __init__(self, categories: list[str] | None = None):
        self._categories = categories or DEFAULT_SHOPPABLE_CATEGORIES

    def detect(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for ref in keyframe_refs:
            has_object = stable_unit_interval(ref, "mock_object_presence") > 0.5
            if not has_object:
                result[ref] = []
                continue
            category_index = int(stable_unit_interval(ref, "mock_object_category") * len(self._categories))
            category_index = min(category_index, len(self._categories) - 1)
            cx = stable_unit_interval(ref, "mock_object_cx")
            cy = stable_unit_interval(ref, "mock_object_cy")
            half = 0.15
            result[ref] = [
                {
                    "label": self._categories[category_index],
                    "bbox": [
                        max(0.0, cx - half),
                        max(0.0, cy - half),
                        min(1.0, cx + half),
                        min(1.0, cy + half),
                    ],
                    "confidence": round(0.5 + stable_unit_interval(ref, "mock_object_conf") * 0.5, 4),
                }
            ]
        return result


class GroundingDinoObjectDetector(ObjectDetector):
    """Step 2 real implementation backed by Grounding DINO, loaded via plain
    `transformers.AutoModelForZeroShotObjectDetection` (no trust_remote_code,
    no custom loading — unlike InternVideo2 earlier in this project).

    All heavy imports (torch, transformers, PIL) are deferred to
    `_get_model_and_processor`/`detect`, never at module import time —
    mirrors every other Real* adapter's lazy-import pattern in this package.
    """

    _MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_GROUNDING_DINO_MODEL,
        device: str = "auto",
        categories: list[str] | None = None,
        min_object_confidence: float = DEFAULT_MIN_OBJECT_CONFIDENCE,
        min_text_match_confidence: float = DEFAULT_MIN_TEXT_MATCH_CONFIDENCE,
    ):
        self._model_name = model_name
        self._device = device
        self._categories = categories or DEFAULT_SHOPPABLE_CATEGORIES
        self._min_object_confidence = min_object_confidence
        self._min_text_match_confidence = min_text_match_confidence

    def _resolve_device(self, torch_module) -> str:
        if self._device != "auto":
            return self._device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _get_model_and_processor(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = self._resolve_device(torch)
        cache_key = (self._model_name, device)
        cached = GroundingDinoObjectDetector._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        processor = AutoProcessor.from_pretrained(self._model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self._model_name).to(device).eval()
        GroundingDinoObjectDetector._MODEL_CACHE[cache_key] = (model, processor)
        return model, processor

    def _load_frame(self, source_video_url: str, timestamp: float):
        from PIL import Image

        with extracted_frame(source_video_url, timestamp) as frame_path:
            image = Image.open(frame_path).convert("RGB")
            image.load()
            return image

    def detect(self, source_video_url: str, keyframe_refs: list[str]) -> dict[str, list[dict]]:
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
        prompt = _prompt_text(self._categories)

        refs = list(timestamps_by_ref.keys())
        images = [self._load_frame(source_video_url, timestamps_by_ref[ref]) for ref in refs]

        # Grounding DINO's processor accepts a batch of images against one
        # shared text query — one forward pass for the whole segment's
        # keyframes, mirroring ClipVisualEmbedder's batching approach rather
        # than looping per-frame.
        inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1] for image in images])
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self._min_object_confidence,
            text_threshold=self._min_text_match_confidence,
            target_sizes=target_sizes,
        )

        detections: dict[str, list[dict]] = {}
        for ref, image, result in zip(refs, images, results):
            width, height = image.size
            objects: list[dict] = []
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                x0, y0, x1, y1 = box.tolist()
                objects.append(
                    {
                        "label": label,
                        "bbox": [
                            max(0.0, x0 / width),
                            max(0.0, y0 / height),
                            min(1.0, x1 / width),
                            min(1.0, y1 / height),
                        ],
                        "confidence": round(float(score), 4),
                    }
                )
            detections[ref] = objects

        return detections
