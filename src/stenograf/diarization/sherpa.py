"""Speaker diarization via sherpa-onnx (pyannote segmentation-3.0 + CAM++
embeddings, ONNX/CPU).

This is the cross-platform baseline diarizer. PLAN.md's accuracy target is
the pyannote community-1 pipeline; the macOS-native port of that (speakrs /
FluidAudio — both libraries, so a thin wrapper binary is needed) replaces
this on Mac in a later step, behind the same ``Diarizer`` interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stenograf import models
from stenograf.audio import to_float32
from stenograf.diarization.base import Diarizer, SpeakerTurn


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

    def _build(self, num_clusters: int) -> None:
        import sherpa_onnx

        segmentation = self._segmentation_model or models.fetch(
            models.PYANNOTE_SEGMENTATION, self._progress
        )
        embedding = self._embedding_model or models.fetch(
            models.SPEAKER_EMBEDDING, self._progress
        )
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation)
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding)),
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

        result = self._pipeline.process(to_float32(samples))
        return [
            SpeakerTurn(speaker=f"S{seg.speaker}", start=seg.start, end=seg.end)
            for seg in result.sort_by_start_time()
        ]
