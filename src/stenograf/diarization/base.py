"""Diarization interface (who-spoke-when within one channel).

Diarization runs per channel: never across mic + system together, so it only
has to separate voices *within* a channel.
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


@dataclass(frozen=True)
class DiarizationResult:
    """Turns plus, optionally, a mean voice embedding per cluster.

    The embedding lets the core match a run's clusters against saved speaker
    profiles for cross-meeting re-ID. Each value is an
    L2-normalized mean embedding keyed by the same cluster label used in
    ``turns`` (e.g. ``"S0"``); a cluster with too little clean audio to embed is
    simply absent from the mapping. Backends that cannot produce embeddings
    return an empty mapping, so callers must treat it as best-effort."""

    turns: list[SpeakerTurn]
    embeddings: dict[str, np.ndarray]


class Diarizer(ABC):
    @abstractmethod
    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        """Segment mono 16 kHz PCM by speaker. ``num_speakers=None`` leaves
        the count to the diarizer; a stated one caps it."""

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        """Diarize and also return a per-cluster voice embedding for re-ID.

        The default runs :meth:`diarize` and returns no embeddings, which is
        what a test double that only needs turns wants; the real diarizer
        overrides it with :class:`~stenograf.diarization.embedding.VoiceEmbedder`."""
        return DiarizationResult(self.diarize(samples, num_speakers), {})

    def channel_embedding(
        self, samples: np.ndarray, turns: list[SpeakerTurn]
    ) -> np.ndarray | None:
        """One voice embedding over a single-speaker channel's speech ``turns``.

        A solo channel never diarizes — one stated speaker has nothing to
        separate — but a meeting with speaker identity on still needs its
        voice, for re-ID naming and for post-meeting enrollment. No
        segmentation runs: the turns are the transcript's own speech spans.
        ``None`` when the backend cannot embed (the default) or the audio
        yields no embeddable span."""
        return None
