"""Model-free tests for the onnx-asr ASR backend (Phase 5).

The real model is a ~2.4 GB download, so ``transcribe`` is exercised against
a fake onnx-asr adapter returning the token/timestamp lists the real one
produces; MLX↔ONNX agreement on real audio is the eval harness's job
(eval/parity.py, PLAN.md Phase 5 verification).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx_asr")

from stenograf.asr import create_backend  # noqa: E402
from stenograf.asr.base import Word  # noqa: E402
from stenograf.asr.parakeet_onnx import (  # noqa: E402
    ParakeetOnnxBackend,
    _approximate_ends,
    _split_sentences,
)


class _FakeResult:
    def __init__(self, tokens, timestamps):
        self.tokens = tokens
        self.timestamps = timestamps


class _FakeModel:
    def __init__(self, result):
        self._result = result
        self.calls: list[tuple] = []

    def recognize(self, samples, sample_rate):
        self.calls.append((samples, sample_rate))
        return self._result


def test_transcribe_builds_words_and_sentences():
    result = _FakeResult(
        tokens=[" Hal", "lo", " Welt", ".", " Gut", "."],
        timestamps=[0.16, 0.24, 0.56, 0.88, 2.0, 2.16],
    )
    backend = ParakeetOnnxBackend()
    backend._model = _FakeModel(result)

    segments = backend.transcribe(np.zeros(16000 * 3, dtype=np.int16), None)

    assert [s.text for s in segments] == ["Hallo Welt.", "Gut."]
    first = segments[0]
    assert [w.text for w in first.words] == ["Hallo", "Welt."]
    hallo, welt = first.words
    assert hallo.start == pytest.approx(0.16)
    assert hallo.end == pytest.approx(0.56)  # runs to the next token's start
    assert welt.start == pytest.approx(0.56)
    # the "." token starts at 0.88; the next token is 1.12 s away, so the
    # TDT duration ceiling caps the word end instead
    assert welt.end == pytest.approx(0.88 + 0.32)


def test_transcribe_feeds_float32_at_16k():
    backend = ParakeetOnnxBackend()
    model = _FakeModel(_FakeResult([], []))
    backend._model = model

    assert backend.transcribe(np.zeros(160, dtype=np.int16), None) == []
    ((samples, rate),) = model.calls
    assert rate == 16000
    assert samples.dtype == np.float32


def test_approximate_ends_caps_at_the_duration_ceiling():
    tokens = _approximate_ends([" a", " b"], [1.0, 10.0])
    assert tokens[0].end == pytest.approx(1.32)  # capped, not stretched to 10.0
    assert tokens[1].end == pytest.approx(10.32)  # last token: start + ceiling


def test_split_sentences_on_final_punctuation():
    words = [
        Word("Ja.", 0.0, 0.2),
        Word("Das", 0.5, 0.7),
        Word("stimmt!", 0.8, 1.1),
        Word("Und", 1.5, 1.6),  # trailing run without punctuation survives
    ]
    segments = _split_sentences(words)
    assert [s.text for s in segments] == ["Ja.", "Das stimmt!", "Und"]
    assert segments[1].start == 0.5
    assert segments[1].end == 1.1
    assert segments[1].words == (words[1], words[2])


def test_factory_constructs_without_loading_the_model():
    backend = create_backend("parakeet-onnx")
    assert isinstance(backend, ParakeetOnnxBackend)
    assert backend.name == "parakeet-onnx"
    assert backend._model is None  # nothing downloaded or loaded yet
