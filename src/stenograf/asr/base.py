"""ASR backend interface.

Backends wrap one model + runtime combination. Planned implementations:

- ``parakeet_mlx`` — Parakeet-TDT-0.6B-v3 via parakeet-mlx (default for both
  finalize and live pass on macOS; Canary-1B-v2 was dropped — no Apple Silicon
  runtime with word timestamps, see PLAN.md)
- ``voxtral_mlx`` — Voxtral Small 24B via mlx-voxtral (opt-in max accuracy)
- ONNX/CTranslate2 equivalents for Linux/Windows

Word-level timestamps are mandatory: speaker assignment intersects them with
diarization turns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from stenograf.config import Language


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    words: tuple[Word, ...] = field(default=())


class ASRBackend(ABC):
    """Transcribes mono int16/float32 PCM at 16 kHz into timestamped segments."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights (downloads to the local cache on first use)."""

    @abstractmethod
    def transcribe(self, samples: np.ndarray, language: Language) -> list[Segment]:
        """Transcribe a complete buffer; ``language`` is always known here —
        detection happens upstream, backends never auto-detect per chunk."""

    @abstractmethod
    def unload(self) -> None:
        """Release model memory."""
