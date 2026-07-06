"""Diarization interface (who-spoke-when within one channel).

Diarization runs per channel: never across mic + system together, so it only
has to separate voices *within* a channel and can be given an exact speaker
count — the biggest single accuracy lever.

Planned implementations: speakrs (pyannote community-1 pipeline, CoreML) on
macOS; sherpa-onnx elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    """Cluster label local to this run, e.g. ``"S0"``; mapping to display
    labels (``Local-1`` / ``Remote-2`` / profile names) happens in the core."""
    start: float
    end: float


class Diarizer(ABC):
    @abstractmethod
    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        """Segment mono 16 kHz PCM by speaker. ``num_speakers=None`` lets the
        pipeline estimate the count (less accurate than providing it)."""
