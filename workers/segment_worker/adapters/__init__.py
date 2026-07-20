"""Stage 2 (segment_worker) model adapters.

Each adapter is an ABC interface + a Step 1 deterministic Mock
implementation, one pair per module, so Step 2 can drop in a real
implementation (TransNetV2, faster-whisper, pyannote.audio, InternVideo2,
OpenCV/RAFT, OpenSMILE/torchaudio) behind the same interface without any
caller needing to change:

  shot_detector.py     -> ShotDetector / MockShotDetector
  transcriber.py        -> Transcriber / MockTranscriber
  diarizer.py            -> Diarizer / MockDiarizer
  visual_embedder.py    -> VisualEmbedder / MockVisualEmbedder
  motion_features.py    -> MotionFeatureExtractor / MockMotionFeatureExtractor
  audio_features.py     -> AudioFeatureExtractor / MockAudioFeatureExtractor
"""

from workers.segment_worker.adapters.audio_features import (
    AudioFeatureExtractor,
    MockAudioFeatureExtractor,
)
from workers.segment_worker.adapters.diarizer import Diarizer, MockDiarizer, SpeakerTurn
from workers.segment_worker.adapters.motion_features import (
    MockMotionFeatureExtractor,
    MotionFeatureExtractor,
)
from workers.segment_worker.adapters.shot_detector import MockShotDetector, ShotDetector
from workers.segment_worker.adapters.transcriber import MockTranscriber, Transcriber
from workers.segment_worker.adapters.visual_embedder import (
    MockVisualEmbedder,
    VisualEmbedder,
)

__all__ = [
    "AudioFeatureExtractor",
    "MockAudioFeatureExtractor",
    "Diarizer",
    "MockDiarizer",
    "SpeakerTurn",
    "MotionFeatureExtractor",
    "MockMotionFeatureExtractor",
    "ShotDetector",
    "MockShotDetector",
    "Transcriber",
    "MockTranscriber",
    "VisualEmbedder",
    "MockVisualEmbedder",
]
