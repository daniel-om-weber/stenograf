"""Unit coverage for the stenodiar-backed diarizer (no Rust binary needed).

A fake helper script stands in for stenodiar (the ``command`` seam), so these
verify the Python half of the contract: raw-PCM piping, JSON parsing, speakrs'
``SPEAKER_NN`` → ``S<N>`` label normalization, the known-count route to sherpa,
error surfacing, and binary discovery. The real binary is exercised by the
eval harness, not the test suite.
"""

from __future__ import annotations

import json
import sys
import textwrap

import numpy as np
import pytest

from stenograf.audio import SAMPLE_RATE
from stenograf.diarization.base import DiarizationResult, SpeakerTurn
from stenograf.diarization.speakrs import (
    DiarizerHelperNotFoundError,
    SpeakrsCliDiarizer,
    find_stenodiar,
)

_TURNS_JSON = json.dumps(
    {
        "turns": [
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 7.5},
            {"speaker": "SPEAKER_00", "start": 0.5, "end": 4.25},
        ]
    }
)


def _fake_helper(tmp_path, *, stdout=_TURNS_JSON, exit_code=0, stderr=""):
    """A stand-in helper: records its stdin byte count, prints canned output."""
    script = tmp_path / "fake_stenodiar.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            data = sys.stdin.buffer.read()
            open({str(tmp_path / "stdin_bytes")!r}, "w").write(str(len(data)))
            sys.stderr.write({stderr!r})
            sys.stdout.write({stdout!r})
            sys.exit({exit_code})
            """
        )
    )
    return [sys.executable, str(script)]


class _StubSherpa:
    """Records routing; ``embed`` returns a fixed unit vector."""

    def __init__(self):
        self.calls = []

    def diarize(self, samples, num_speakers=None):
        self.calls.append(num_speakers)
        return [SpeakerTurn(speaker="S0", start=0.0, end=1.0)]

    def diarize_with_embeddings(self, samples, num_speakers=None):
        return DiarizationResult(self.diarize(samples, num_speakers), {})

    def embed(self, audio):
        return np.array([1.0, 0.0], dtype=np.float32) if len(audio) else None


_AUDIO = np.ones(8 * SAMPLE_RATE, dtype=np.int16)


def test_estimate_mode_runs_helper_and_normalizes_labels(tmp_path):
    d = SpeakrsCliDiarizer(_StubSherpa(), command=_fake_helper(tmp_path))
    turns = d.diarize(_AUDIO)  # num_speakers=None → the helper path

    assert [(t.speaker, t.start, t.end) for t in turns] == [
        ("S0", 0.5, 4.25),  # sorted by start, SPEAKER_00 → S0
        ("S1", 5.0, 7.5),
    ]
    # the full PCM went down the pipe as s16le
    assert int((tmp_path / "stdin_bytes").read_text()) == _AUDIO.size * 2


def test_explicit_count_routes_to_sherpa(tmp_path):
    sherpa = _StubSherpa()
    d = SpeakrsCliDiarizer(sherpa, command=_fake_helper(tmp_path, exit_code=1))
    turns = d.diarize(_AUDIO, num_speakers=3)  # helper would fail; must not run

    assert sherpa.calls == [3]
    assert turns[0].speaker == "S0"
    assert not (tmp_path / "stdin_bytes").exists()


def test_helper_failure_raises_with_stderr_detail(tmp_path):
    command = _fake_helper(tmp_path, exit_code=1, stderr="stenodiar: kaboom\n")
    d = SpeakrsCliDiarizer(_StubSherpa(), command=command)
    with pytest.raises(RuntimeError, match="kaboom"):
        d.diarize(_AUDIO)


def test_unparseable_output_raises(tmp_path):
    d = SpeakrsCliDiarizer(_StubSherpa(), command=_fake_helper(tmp_path, stdout="not json"))
    with pytest.raises(RuntimeError, match="unparseable"):
        d.diarize(_AUDIO)


def test_embeddings_come_from_sherpa_extractor(tmp_path):
    d = SpeakrsCliDiarizer(_StubSherpa(), command=_fake_helper(tmp_path))
    result = d.diarize_with_embeddings(_AUDIO)

    assert {t.speaker for t in result.turns} == {"S0", "S1"}
    assert set(result.embeddings) == {"S0", "S1"}  # embedded via _StubSherpa.embed
    for vector in result.embeddings.values():
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_find_stenodiar_env_override_and_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("STENOGRAF_DIAR_HELPER", str(tmp_path / "stenodiar"))
    assert find_stenodiar() == tmp_path / "stenodiar"

    monkeypatch.delenv("STENOGRAF_DIAR_HELPER")
    monkeypatch.setattr(
        "stenograf.diarization.speakrs.resources.files",
        lambda _: tmp_path / "nowhere",
    )
    monkeypatch.setattr(
        "stenograf.diarization.speakrs.Path.resolve", lambda self: tmp_path / "elsewhere"
    )
    with pytest.raises(DiarizerHelperNotFoundError, match="native/stenodiar"):
        find_stenodiar()


def test_find_stenodiar_dev_build_uses_platform_suffix(tmp_path, monkeypatch):
    """The dev fallback finds the raw ``cargo build`` output under target/release,
    with the ``.exe`` suffix on Windows."""
    from stenograf.diarization import speakrs as mod

    monkeypatch.delenv("STENOGRAF_DIAR_HELPER", raising=False)
    monkeypatch.setattr(mod.resources, "files", lambda _: tmp_path / "nowhere")
    # parents[3] of the faked resolve() path is tmp_path (the repo root stand-in).
    fake_module = tmp_path / "a" / "b" / "c" / "speakrs.py"
    monkeypatch.setattr(mod.Path, "resolve", lambda self: fake_module)

    built = tmp_path / "native" / "stenodiar" / "target" / "release" / mod._HELPER_FILENAME
    built.parent.mkdir(parents=True)
    built.write_bytes(b"")
    assert find_stenodiar() == built
