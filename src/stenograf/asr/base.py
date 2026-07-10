"""ASR backend interface.

Backends wrap one model + runtime combination. Implementations:

- ``parakeet`` — Parakeet-TDT-0.6B-v3 via parakeet-mlx (default for both
  finalize and live pass on macOS; Canary-1B-v2 was dropped — no Apple Silicon
  runtime with word timestamps, see PLAN.md)
- ``parakeet-onnx`` — the same model, fp32 ONNX via onnx-asr on CPU (the
  cross-platform default off macOS; Phase 5)
- planned: ``voxtral_mlx`` — Voxtral Small 24B via mlx-voxtral (opt-in max
  accuracy)

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
    def transcribe(self, samples: np.ndarray, language: Language | None) -> list[Segment]:
        """Transcribe a complete buffer.

        ``language`` is the resolved meeting language, or ``None`` when it is
        not (yet) known: the default backend (Parakeet) is multilingual and
        ignores it, and Phase-1 language detection runs over the finalized text
        (see ``stenograf.lid``). A language-*requiring* backend must handle
        ``None`` itself (detect once, then lock) rather than assume a value."""

    @abstractmethod
    def unload(self) -> None:
        """Release model memory."""
