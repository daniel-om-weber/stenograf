"""Real-backend SherpaOnnxDiarizer test (Phase 3, Task 0c).

Every other diarizer test uses ``FakeDiarizer``; this one drives the actual
sherpa-onnx ONNX pipeline (pyannote segmentation-3.0 + eres2net embeddings) —
the surface speaker re-ID (Stage 1) extends. Precedent for why it matters: the
MLX thread-stream bug showed a real backend can break what every mocked test
passes green.

It is **gated**: it runs only when sherpa-onnx is installed, both diarization
models are already cached, and a real eval clip is present (the audio is
gitignored private meeting content, the models are a multi-hundred-MB download).
A fresh checkout / CI therefore skips it; run it on a dev machine that has done
Phase 0/1 setup. Assertions are structural (turns are well-formed, sorted, in
bounds; the known count caps clusters; the estimation and count-change-rebuild
paths run) rather than accuracy numbers — DER scoring is Task 0d.
"""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stenograf import models
from stenograf.diarization.base import SpeakerTurn

CLIP = Path(__file__).resolve().parents[1] / "eval" / "audio" / "de-1.wav"
CLIP_SECONDS = 40  # long enough that a second speaker appears in de-1


def _sherpa_available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        return False
    return True


def _models_cached() -> bool:
    return (
        models.cached_path(models.PYANNOTE_SEGMENTATION) is not None
        and models.cached_path(models.SPEAKER_EMBEDDING) is not None
    )


pytestmark = pytest.mark.skipif(
    not (_sherpa_available() and _models_cached() and CLIP.exists()),
    reason="real sherpa test needs sherpa-onnx, cached diarization models, and eval audio",
)


def _load_clip() -> np.ndarray:
    with wave.open(str(CLIP), "rb") as wv:
        rate = wv.getframerate()
        frames = wv.readframes(min(wv.getnframes(), CLIP_SECONDS * rate))
    return np.frombuffer(frames, dtype=np.int16)


@pytest.fixture(scope="module")
def diarized() -> SimpleNamespace:
    """Run the real pipeline once (inference is the slow part) and stash results.

    Reuses a single diarizer across the count changes, which is exactly what
    exercises the ``set_config`` rebuild path (the ONNX models load once; only
    the clustering config changes)."""
    from stenograf.diarization.sherpa import SherpaOnnxDiarizer

    pcm = _load_clip()
    diarizer = SherpaOnnxDiarizer()

    known2 = diarizer.diarize(pcm, num_speakers=2)
    pipeline_after_2, nclu_after_2 = diarizer._pipeline, diarizer._num_clusters

    known3 = diarizer.diarize(pcm, num_speakers=3)
    pipeline_after_3, nclu_after_3 = diarizer._pipeline, diarizer._num_clusters

    estimated = diarizer.diarize(pcm, num_speakers=None)
    nclu_estimate = diarizer._num_clusters

    return SimpleNamespace(
        duration=len(pcm) / 16000,
        known2=known2,
        known3=known3,
        estimated=estimated,
        pipeline_after_2=pipeline_after_2,
        pipeline_after_3=pipeline_after_3,
        nclu_after_2=nclu_after_2,
        nclu_after_3=nclu_after_3,
        nclu_estimate=nclu_estimate,
    )


def _assert_well_formed(turns: list[SpeakerTurn], duration: float) -> None:
    assert turns, "real speech should yield at least one turn"
    assert all(isinstance(t, SpeakerTurn) for t in turns)
    assert turns == sorted(turns, key=lambda t: t.start)  # diarize returns sorted
    for t in turns:
        assert 0.0 <= t.start < t.end <= duration + 0.5  # within the clip, non-empty


def test_known_count_returns_well_formed_capped_turns(diarized):
    _assert_well_formed(diarized.known2, diarized.duration)
    # FastClustering with num_clusters=2 can never emit more than 2 speakers.
    assert len({t.speaker for t in diarized.known2}) <= 2


def test_estimation_runs_unconstrained(diarized):
    # num_speakers=None → num_clusters=-1 (let the pipeline estimate). It must
    # run without raising and produce well-formed turns.
    assert diarized.nclu_estimate == -1
    _assert_well_formed(diarized.estimated, diarized.duration)


def test_count_change_reuses_pipeline_via_set_config(diarized):
    # A new count rebuilds the config but reuses the loaded ONNX models
    # (set_config), so the pipeline object is the *same* instance across counts —
    # never reconstructed. _num_clusters tracks the requested count.
    assert diarized.pipeline_after_2 is diarized.pipeline_after_3
    assert diarized.nclu_after_2 == 2
    assert diarized.nclu_after_3 == 3
