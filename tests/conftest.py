"""Shared test doubles and helpers.

Import them explicitly — ``from conftest import FakeASR`` — pytest loads this
file once as the ``conftest`` module, so these are ordinary importable names.
Specialized doubles (``WordlessASR``, ``TwoSpeakerASR``, …) live in the test
files that need them, as thin subclasses of these; a double lives here once a
second file needs it.
"""

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.view import LiveView

# Shell-level overrides a developer's environment may carry; any one of them
# would give that machine a different run than CI, so isolation clears them.
# STENOGRAF_CACHE deliberately survives: the live byte-parity tests skip
# without a cached model, and redirecting the cache would skip them on the
# one machine that has it.
_ENV_OVERRIDES = (
    "STENOGRAF_ASR_BACKEND",
    "STENOGRAF_ASR_PROVIDER",
    "STENOGRAF_NOTES_BACKEND",
    "STENOGRAF_NOTES_MODEL",
    "STENOGRAF_OUTPUT_HOME",
    "STENOGRAF_MEETING_DIR",
    "STENOGRAF_CAPTURE_HELPER",
    "STENOGRAF_DIAR_HELPER",
    "STENOGRAF_APP_BUNDLE",
)


@pytest.fixture(autouse=True)
def _isolated_machine(tmp_path, monkeypatch):
    """No test reads or writes the real machine.

    Settings and voiceprints land in a per-test ``$STENOGRAF_DATA``; a run
    without ``--out`` creates its meeting folder in a per-test output home
    instead of the real ~/Documents/Meetings; and the developer's own
    STENOGRAF_* overrides are cleared. Autouse and function-scoped, so it runs
    before every other function-scope fixture — fixture-time code (the GUI
    shell build) sees only the isolated paths too.
    """
    from stenograf import output

    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "steno-data"))
    monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings-home")
    for name in _ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def write_settings(body: str) -> Path:
    """Drop a settings.toml into the test's isolated ``$STENOGRAF_DATA`` dir."""
    data = Path(os.environ["STENOGRAF_DATA"])
    data.mkdir(parents=True, exist_ok=True)
    path = data / "settings.toml"
    path.write_text(body, encoding="utf-8")
    return path


class CallbackView(LiveView):
    """A LiveView over plain test callbacks (the orchestrator is view-only).

    ``update``/``status`` forward to their callback; ``language``/``error``
    fold onto ``on_status`` as text so a status-collecting test sees them too.
    """

    def __init__(self, on_update=None, on_status=None) -> None:
        self._on_update = on_update
        self._on_status = on_status

    def update(self, channel, update) -> None:
        if self._on_update is not None:
            self._on_update(channel, update)

    def status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)

    def language(self, language) -> None:
        if self._on_status is not None:
            self._on_status(f"detected language: {language.value}")

    def error(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)


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


class GermanASR(FakeASR):
    """Transcribes to fixed German text, for language-detection paths."""

    name = "german"

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.calls.append(len(samples))
        text = "und das ist wirklich eine gute idee für uns"
        return [Segment(text=text, start=0.1, end=1.0, words=(Word(text, 0.1, 1.0),))]


class CliASR(FakeASR):
    """Fixed German text *with word timestamps* — the whole CLI path (LID
    included) runs offline, and word-level output (subtitles, captions)
    has something to show."""

    model_id = "fake/model"

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.calls.append(len(samples))
        return [
            Segment(
                text="und das ist wirklich eine gute idee für uns alle",
                start=0.1,
                end=1.0,
                words=(Word("und", 0.1, 0.3), Word("das", 0.3, 0.6)),
            )
        ]


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


class EmbeddingDiarizer(Diarizer):
    """A diarizer that also carries per-cluster embeddings (the re-ID surface)."""

    def __init__(self, turns: list[SpeakerTurn], embeddings: dict[str, np.ndarray]):
        self.turns = turns
        self.embeddings = embeddings
        self.diarize_calls = 0
        self.embed_calls = 0

    def diarize(self, samples, num_speakers=None):
        self.diarize_calls += 1
        return self.turns

    def diarize_with_embeddings(self, samples, num_speakers=None):
        self.embed_calls += 1
        return DiarizationResult(turns=self.turns, embeddings=self.embeddings)


class EnrollmentDiarizer(Diarizer):
    """Fixed clusters with fixed unit embeddings — no ONNX model, no real audio.

    Lets the re-ID/enrollment paths run offline: enrollment reads
    ``diarize_with_embeddings`` and the finalize pass matches against the same
    vectors, so a profile enrolled from this diarizer self-matches (cosine 1.0).
    """

    def __init__(self, embeddings, turns=None):
        self._embeddings = {k: np.asarray(v, dtype=np.float32) for k, v in embeddings.items()}
        # One long turn per cluster by default (covers every word's midpoint).
        self._turns = turns or [SpeakerTurn(s, 0.0, 1e9) for s in embeddings]

    def diarize(self, samples, num_speakers=None):
        return list(self._turns)

    def diarize_with_embeddings(self, samples, num_speakers=None):
        return DiarizationResult(turns=list(self._turns), embeddings=dict(self._embeddings))


class RaisingDiarizer(Diarizer):
    """Fails on every call — stands in for a backend that throws on odd input."""

    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, samples, num_speakers=None):
        self.calls += 1
        raise RuntimeError("diarizer exploded")


class ListProvider(CaptureProvider):
    """Yields a preset list of frames — an in-process stand-in for a device.

    Stops yielding once ``stop()`` lands; ``yielded`` counts the frames
    actually consumed, so a test can see an early stop (``max_seconds``)."""

    def __init__(self, frames: list[AudioFrame]):
        self._frames = frames
        self.started_channels: set[Channel] | None = None
        self.stopped = False
        self.yielded = 0

    def start(self, channels: set[Channel]) -> None:
        self.started_channels = channels

    def frames(self):
        for f in self._frames:
            if self.stopped:
                return
            self.yielded += 1
            yield f

    def stop(self) -> None:
        self.stopped = True


def fake_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None, announce=None, **_):
    """The loaders seam, offline: no VAD (whole buffer is one window), no diarizer."""
    return CliASR(), None, None


@pytest.fixture
def stub_backends(monkeypatch):
    """Route the loaders seam to the offline fakes — no weights, no downloads.

    Tests that need a *custom* fake (a recording wrapper, a specific diarizer)
    still patch ``loaders.load_backends`` themselves."""
    from stenograf import loaders

    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)


@pytest.fixture
def backend_calls(monkeypatch):
    """``stub_backends`` plus a record of the call's keyword arguments.

    Assert on ``backend_calls["need_diarizer"]`` (or any other kwarg the
    caller passed) after the run."""
    from stenograf import loaders

    calls: dict[str, object] = {}

    def recording(*, need_diarizer, **kwargs):
        calls["need_diarizer"] = need_diarizer
        calls.update(kwargs)
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording)
    return calls


def write_wav(
    path,
    samples: np.ndarray | None = None,
    *,
    seconds: float = 1.0,
    rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> None:
    """Write an int16 WAV; ``samples=None`` writes ``seconds`` of silence."""
    if samples is None:
        samples = np.zeros(int(rate * seconds), dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
