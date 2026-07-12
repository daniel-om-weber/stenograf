"""Speaker diarization via sherpa-onnx (pyannote segmentation-3.0 + CAM++
embeddings, ONNX/CPU).

This is the cross-platform baseline diarizer. PLAN.md's accuracy target is
the pyannote community-1 pipeline; the macOS-native port of that (speakrs /
FluidAudio — both libraries, so a thin wrapper binary is needed) replaces
this on Mac in a later step, behind the same ``Diarizer`` interface.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stenograf import models
from stenograf.audio import SAMPLE_RATE, to_float32
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn

MIN_EMBED_SECONDS = 0.5
"""Turns shorter than this are too brief for a reliable voice embedding; they are
skipped when a cluster has any longer turn, and used only as a last resort."""

_MAX_THREADS = 8
"""sherpa defaults every model to a single ORT intra-op thread, which makes
diarization the finalize bottleneck (measured 2026-07-12 on a 12-core box,
2.3-min clip: 48.7s at 1 thread vs 17.9s at 8, identical turns). Scaling
plateaus around 8 threads, so cap there and leave the rest of the machine to
the ASR and the UI."""


def _num_threads() -> int:
    return min(_MAX_THREADS, os.cpu_count() or 1)


class SherpaOnnxDiarizer(Diarizer):
    def __init__(
        self,
        segmentation_model: Path | None = None,
        embedding_model: Path | None = None,
        *,
        clustering_threshold: float = 0.5,
        progress: models.ProgressHook | None = None,
    ) -> None:
        self._segmentation_model = segmentation_model
        self._embedding_model = embedding_model
        self._threshold = clustering_threshold
        self._progress = progress
        self._pipeline = None
        self._num_clusters = -1
        self._extractor = None  # lazy SpeakerEmbeddingExtractor for re-ID

    def _build(self, num_clusters: int) -> None:
        import sherpa_onnx

        segmentation = self._segmentation_model or models.fetch(
            models.PYANNOTE_SEGMENTATION, self._progress
        )
        embedding = self._embedding_model or models.fetch(models.SPEAKER_EMBEDDING, self._progress)
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation)
                ),
                num_threads=_num_threads(),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding), num_threads=_num_threads()
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters, threshold=self._threshold
            ),
        )
        if self._pipeline is None:
            self._pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
        else:
            # Reuse the loaded ONNX models; only clustering changes per run.
            self._pipeline.set_config(config)
        self._num_clusters = num_clusters

    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        num_clusters = num_speakers if num_speakers is not None else -1
        if self._pipeline is None or self._num_clusters != num_clusters:
            self._build(num_clusters)
        pipeline = self._pipeline
        assert pipeline is not None  # _build() sets it or raises

        result = pipeline.process(to_float32(samples))
        return [
            SpeakerTurn(speaker=f"S{seg.speaker}", start=seg.start, end=seg.end)
            for seg in result.sort_by_start_time()
        ]

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        """Diarize, then a duration-weighted mean voice embedding per cluster.

        sherpa's ``OfflineSpeakerDiarization`` result carries no embeddings
        (verified against the installed package), so a separate
        ``SpeakerEmbeddingExtractor`` — the same ``models.SPEAKER_EMBEDDING`` file
        the clustering uses — embeds each cluster's turn slices via
        :func:`cluster_embeddings`."""
        turns = self.diarize(samples, num_speakers)
        return DiarizationResult(
            turns=turns, embeddings=cluster_embeddings(turns, samples, self.embed)
        )

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """L2-normalized voice embedding of a mono 16 kHz float32 slice, or None
        when the slice is empty or the extractor cannot form an embedding."""
        if len(audio) == 0:
            return None
        extractor = self._embedder()
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        return _l2_normalize(np.asarray(extractor.compute(stream), dtype=np.float32))

    def _embedder(self):
        if self._extractor is None:
            import sherpa_onnx

            embedding = self._embedding_model or models.fetch(
                models.SPEAKER_EMBEDDING, self._progress
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(embedding), num_threads=_num_threads()
                )
            )
        return self._extractor


def cluster_embeddings(
    turns: list[SpeakerTurn],
    samples: np.ndarray,
    embed: Callable[[np.ndarray], np.ndarray | None],
) -> dict[str, np.ndarray]:
    """A duration-weighted mean voice embedding per cluster of ``turns``.

    Shared by every diarization backend that pairs its turns with sherpa's
    ``SpeakerEmbeddingExtractor`` (the ``embed`` callable) — re-ID voiceprints
    must come from one embedding model regardless of which backend produced
    the turns. Slices shorter than :data:`MIN_EMBED_SECONDS` are skipped
    unless they are all a cluster has; each embedding is L2-normalized,
    duration-weighted, and averaged, and the mean re-normalized. A cluster
    with no embeddable audio is omitted."""
    audio = to_float32(samples)
    by_cluster: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        by_cluster.setdefault(turn.speaker, []).append(turn)

    embeddings: dict[str, np.ndarray] = {}
    for speaker, cluster_turns in by_cluster.items():
        long = [t for t in cluster_turns if t.end - t.start >= MIN_EMBED_SECONDS]
        selected = long or cluster_turns  # fall back to short turns if that's all there is
        vectors, weights = [], []
        for turn in selected:
            slice_ = audio[int(turn.start * SAMPLE_RATE) : int(turn.end * SAMPLE_RATE)]
            vector = embed(slice_)
            if vector is not None:
                vectors.append(vector)
                weights.append(turn.end - turn.start)
        if vectors:
            mean = np.average(vectors, axis=0, weights=weights)
            embeddings[speaker] = _l2_normalize(mean)
    return embeddings


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector
