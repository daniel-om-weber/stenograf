"""The diarizer base class for the owned loop: turns plus re-ID voices.

The voice embedding and the shared worker pool are
:class:`~stenograf.diarization.embedding.VoiceEmbedder`'s; this class only
pairs a subclass's ``diarize`` with them. The diarization loop itself is
:class:`~stenograf.diarization.loop.OwnDiarizer`; sherpa's
``OfflineSpeakerDiarization`` monolith is not called anywhere (one path — its
reference semantics live on in ``eval/diarization-loop-spec.md`` and the parity
gate ``eval/loop_parity.py`` constructs it directly from the installed
package)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from stenograf import assets
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.diarization.embedding import VoiceEmbedder


class SherpaOnnxDiarizer(Diarizer):
    """Per-cluster embeddings and re-ID plumbing over a
    :class:`VoiceEmbedder`; ``diarize`` is the subclass's (abstract here —
    construct :class:`~stenograf.diarization.loop.OwnDiarizer`, not this)."""

    def __init__(
        self,
        segmentation_model: Path | None = None,
        embedding_model: Path | None = None,
        *,
        clustering_threshold: float = 0.5,
        workers: int | None = None,
        progress: assets.ProgressHook | None = None,
    ) -> None:
        self._segmentation_model = segmentation_model
        self._threshold = clustering_threshold
        self._progress = progress
        self.embedder = VoiceEmbedder(embedding_model, workers=workers, progress=progress)
        self._init_lock = threading.Lock()  # guards the subclass's lazy sessions

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        turns = self.diarize(samples, num_speakers)
        return DiarizationResult(turns, self.embedder.cluster_embeddings(turns, samples))

    def channel_embedding(
        self, samples: np.ndarray, turns: list[SpeakerTurn]
    ) -> np.ndarray | None:
        return self.embedder.channel_embedding(samples, turns)

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        return self.embedder.embed(audio)

    def _executor(self) -> ThreadPoolExecutor:
        return self.embedder.pool()
