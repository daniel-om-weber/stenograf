"""Shared test doubles and helpers.

Import them explicitly — ``from conftest import FakeASR`` — pytest loads this
file once as the ``conftest`` module, so these are ordinary importable names.
Specialized doubles (``GermanASR``, ``WordlessASR``, ``TwoSpeakerASR``, …)
live in the test files that need them, as thin subclasses of these.
"""

import wave

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.capture.base import SAMPLE_RATE
from stenograf.diarization.base import Diarizer, SpeakerTurn


class FakeASR(ASRBackend):
    """One "wort" per transcribed window; records each call's sample count."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.calls.append(len(samples))
        return [Segment(text="wort", start=0.1, end=0.5, words=(Word("wort", 0.1, 0.5),))]

    def unload(self) -> None:
        pass


class FakeDiarizer(Diarizer):
    """Preset turns; records the speaker count of the last ``diarize`` call.

    ``seen_num_speakers`` starts as ``"unset"`` so a never-called diarizer is
    distinguishable from one called with ``num_speakers=None``."""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self.turns = turns
        self.seen_num_speakers: object = "unset"

    def diarize(self, samples, num_speakers=None):
        self.seen_num_speakers = num_speakers
        return self.turns


class RaisingDiarizer(Diarizer):
    """Fails on every call — stands in for a backend that throws on odd input."""

    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, samples, num_speakers=None):
        self.calls += 1
        raise RuntimeError("diarizer exploded")


def write_wav(path, samples: np.ndarray | None = None, *, seconds: float = 1.0,
              rate: int = SAMPLE_RATE, channels: int = 1) -> None:
    """Write an int16 WAV; ``samples=None`` writes ``seconds`` of silence."""
    if samples is None:
        samples = np.zeros(int(rate * seconds), dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
