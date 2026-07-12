"""Unit coverage for SherpaOnnxDiarizer's pure-python aggregation (Phase 3→4).

The real-backend test (``test_diarization_sherpa.py``) is gated on cached ONNX
models + private audio, so on a fresh checkout / Linux CI ``sherpa.py`` executes
zero test lines — yet Phase 4 ships this "cross-platform baseline diarizer" to
Linux. These drive ``diarize_with_embeddings`` and ``l2_normalize`` through a
fake ``SpeakerEmbeddingExtractor`` + fake pipeline, so the aggregation logic
(per-cluster unit-norm mean, duration weighting, empty-cluster omission,
short-turn fallback, zero-vector guard) is verified without any model.

The precedent for why this matters is the MLX thread-stream bug: a real backend
broke what every mocked test passed green.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stenograf.audio import SAMPLE_RATE, l2_normalize
from stenograf.diarization.sherpa import SherpaOnnxDiarizer


class _FakeStream:
    def accept_waveform(self, rate: int, audio: np.ndarray) -> None:
        self.audio = audio

    def input_finished(self) -> None:
        pass


class _SeqExtractor:
    """Returns preset vectors in call order; ``_embed`` L2-normalizes the result."""

    def __init__(self, vectors: list[np.ndarray]) -> None:
        self._vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        self._i = 0

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def is_ready(self, stream: _FakeStream) -> bool:
        return True

    def compute(self, stream: _FakeStream) -> np.ndarray:
        vector = self._vectors[self._i]
        self._i += 1
        return vector


class _FakeResult:
    def __init__(self, segs: list[SimpleNamespace]) -> None:
        self._segs = segs

    def sort_by_start_time(self) -> list[SimpleNamespace]:
        return sorted(self._segs, key=lambda s: s.start)


class _FakePipeline:
    def __init__(self, segs: list[SimpleNamespace]) -> None:
        self._segs = segs

    def process(self, audio: np.ndarray) -> _FakeResult:
        return _FakeResult(self._segs)


def _seg(speaker: int, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(speaker=speaker, start=start, end=end)


def _wired(segs: list[SimpleNamespace], vectors: list[np.ndarray]) -> SherpaOnnxDiarizer:
    """A diarizer with the ONNX pieces replaced by fakes (num_speakers=None path)."""
    d = SherpaOnnxDiarizer()
    d._pipeline = _FakePipeline(segs)
    d._num_clusters = -1  # matches num_speakers=None, so diarize() won't rebuild
    d._extractor = _SeqExtractor(vectors)
    return d


_AUDIO = np.ones(3 * SAMPLE_RATE, dtype=np.int16)


def test_embeddings_are_per_cluster_and_unit_norm():
    d = _wired([_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0)], [[3.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    result = d.diarize_with_embeddings(_AUDIO)

    assert [t.speaker for t in result.turns] == ["S0", "S1"]  # turns pass through
    assert set(result.embeddings) == {"S0", "S1"}
    for vector in result.embeddings.values():
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
    assert float(result.embeddings["S0"] @ result.embeddings["S1"]) == pytest.approx(0.0, abs=1e-6)


def test_mean_is_duration_weighted():
    # One cluster, two long turns of 1 s and 2 s pointing along x and y. The longer
    # turn dominates: mean = normalize(x*1 + y*2) = [1, 2, 0]/sqrt(5).
    d = _wired([_seg(0, 0.0, 1.0), _seg(0, 1.0, 3.0)], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    embedding = d.diarize_with_embeddings(_AUDIO).embeddings["S0"]
    expected = np.array([1.0, 2.0, 0.0]) / np.sqrt(5.0)
    assert embedding == pytest.approx(expected, abs=1e-6)


def test_cluster_with_no_embeddable_audio_is_omitted():
    # S1's only turn is zero-length → empty slice → no embedding → cluster dropped
    # (its turns still appear in `turns`, just not in `embeddings`).
    d = _wired([_seg(0, 0.0, 1.0), _seg(1, 2.0, 2.0)], [[1.0, 0.0, 0.0]])
    result = d.diarize_with_embeddings(_AUDIO)
    assert set(result.embeddings) == {"S0"}
    assert "S1" in {t.speaker for t in result.turns}


def test_short_only_cluster_falls_back_to_its_short_turns():
    # A cluster whose only turn is below MIN_EMBED_SECONDS still gets embedded
    # (the `selected = long or cluster_turns` fallback), rather than being dropped.
    d = _wired([_seg(0, 0.0, 0.2)], [[0.0, 5.0, 0.0]])
    result = d.diarize_with_embeddings(_AUDIO)
    assert set(result.embeddings) == {"S0"}
    assert np.linalg.norm(result.embeddings["S0"]) == pytest.approx(1.0, abs=1e-6)


def test_l2_normalize_guards_the_zero_vector():
    zero = l2_normalize(np.zeros(3, dtype=np.float32))
    assert np.all(zero == 0.0) and not np.any(np.isnan(zero))  # no div-by-zero / NaN
    assert l2_normalize(np.array([3.0, 4.0, 0.0])) == pytest.approx([0.6, 0.8, 0.0])
