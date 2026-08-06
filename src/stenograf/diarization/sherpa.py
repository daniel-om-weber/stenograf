"""Speaker diarization via sherpa-onnx (pyannote segmentation-3.0 + ERes2Net
embeddings, ONNX/CPU; see assets.SPEAKER_EMBEDDING for why not CAM++).

This is the cross-platform baseline diarizer. The accuracy target is
the pyannote community-1 pipeline; the macOS-native port of that (speakrs /
FluidAudio — both libraries, so a thin wrapper binary is needed) replaces
this on Mac in a later step, behind the same ``Diarizer`` interface.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stenograf import assets
from stenograf.audio import SAMPLE_RATE, l2_normalize, to_float32
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn

MIN_EMBED_SECONDS = 0.5
"""Speech spans shorter than this are too brief for a reliable voice embedding;
they are skipped when a cluster has any longer span, and used only as a last
resort (see :func:`cluster_embeddings`)."""

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
        progress: assets.ProgressHook | None = None,
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

        segmentation = self._segmentation_model or assets.fetch(
            assets.PYANNOTE_SEGMENTATION, self._progress
        )
        embedding = self._embedding_model or assets.fetch(assets.SPEAKER_EMBEDDING, self._progress)
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
        ``SpeakerEmbeddingExtractor`` — the same ``assets.SPEAKER_EMBEDDING`` file
        the clustering uses — embeds each cluster's turn slices via
        :func:`cluster_embeddings`."""
        turns = self.diarize(samples, num_speakers)
        return DiarizationResult(
            turns=turns, embeddings=cluster_embeddings(turns, samples, self.embed)
        )

    def channel_embedding(
        self, samples: np.ndarray, turns: list[SpeakerTurn]
    ) -> np.ndarray | None:
        if not turns:
            return None
        return cluster_embeddings(turns, samples, self.embed).get(turns[0].speaker)

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
        return l2_normalize(np.asarray(extractor.compute(stream), dtype=np.float32))

    def _embedder(self):
        if self._extractor is None:
            import sherpa_onnx

            embedding = self._embedding_model or assets.fetch(
                assets.SPEAKER_EMBEDDING, self._progress
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
    the turns. Spans where a *different* cluster is active too are excluded
    before embedding: overlapped audio measurably worsens the embedding
    (12.84 → 14.11 % EER with overlap included, and frame-level identity in
    overlap is near-chance — the research record in
    ``eval/diarization-sota-2026.md``). Spans shorter than
    :data:`MIN_EMBED_SECONDS` are skipped unless they are all a cluster has;
    a cluster whose speech is entirely overlapped falls back to its raw turns
    (a contaminated embedding still serves naming and the collapse/fold rules,
    an absent one blocks them). Each embedding is L2-normalized,
    duration-weighted, and averaged, and the mean re-normalized. A cluster
    with no embeddable audio is omitted."""
    audio = to_float32(samples)
    by_cluster: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        by_cluster.setdefault(turn.speaker, []).append(turn)
    overlap = _overlap_spans(by_cluster)

    embeddings: dict[str, np.ndarray] = {}
    for speaker, cluster_turns in by_cluster.items():
        raw = [(t.start, t.end) for t in cluster_turns]
        spans = _subtract_spans(raw, overlap) or raw
        long = [s for s in spans if s[1] - s[0] >= MIN_EMBED_SECONDS]
        vectors, weights = [], []
        for start, end in long or spans:
            vector = embed(audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)])
            if vector is not None:
                vectors.append(vector)
                weights.append(end - start)
        if vectors:
            mean = np.average(vectors, axis=0, weights=weights)
            embeddings[speaker] = l2_normalize(mean)
    return embeddings


def _overlap_spans(by_cluster: dict[str, list[SpeakerTurn]]) -> list[tuple[float, float]]:
    """Spans where at least two clusters speak at once, sorted and disjoint.

    Each cluster's turns are unioned first, so a cluster overlapping itself
    (back-to-back or re-emitted turns) never counts as overlapped speech."""
    events: list[tuple[float, int]] = []
    for cluster_turns in by_cluster.values():
        merged: list[tuple[float, float]] = []
        for t in sorted(cluster_turns, key=lambda t: t.start):
            if merged and t.start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t.end))
            else:
                merged.append((t.start, t.end))
        for start, end in merged:
            events += [(start, 1), (end, -1)]

    events.sort()  # ends before starts at equal times: touching spans never overlap
    spans: list[tuple[float, float]] = []
    depth = 0
    overlap_start = 0.0
    for time_, delta in events:
        was = depth
        depth += delta
        if was < 2 <= depth:
            overlap_start = time_
        elif was >= 2 > depth:
            spans.append((overlap_start, time_))
    return spans


def _subtract_spans(
    spans: list[tuple[float, float]], cuts: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``spans`` minus ``cuts`` (both lists of (start, end); cuts sorted, disjoint)."""
    result = []
    for start, end in spans:
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor:
                continue
            if cut_start >= end:
                break
            if cut_start > cursor:
                result.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result
