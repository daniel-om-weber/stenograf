import wave

import numpy as np
from click.testing import CliRunner

from stenograf import cli
from stenograf.asr.base import ASRBackend, Segment, Word


class FakeASR(ASRBackend):
    """Returns fixed German text so the whole CLI path (incl. LID) runs offline."""

    name = "fake"
    model_id = "fake/model"

    def load(self) -> None:
        pass

    def transcribe(self, samples, language) -> list[Segment]:
        return [
            Segment(
                text="und das ist wirklich eine gute idee für uns alle",
                start=0.1,
                end=1.0,
                words=(Word("und", 0.1, 0.3), Word("das", 0.3, 0.6)),
            )
        ]

    def unload(self) -> None:
        pass


def fake_load_backends(*, need_diarizer):
    # No VAD (whole buffer is one window) and no diarizer (single speaker).
    return FakeASR(), None, None


def write_wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(np.zeros(int(16_000 * seconds), dtype=np.int16).tobytes())


def test_transcribe_writes_outputs_and_detects_language(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "meeting.transcript.md").exists()
    assert (tmp_path / "meeting.transcript.json").exists()
    assert "language: detected de" in result.output  # LID ran over the German text


def test_start_replay_writes_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        ["start", "--local", "1", "--remote", "0", "--replay", str(mic), "--out", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    transcripts = list(tmp_path.glob("meeting-*.transcript.md"))
    assert len(transcripts) == 1
    assert "detected language: de" in result.output


def test_doctor_runs_and_prints_checks():
    result = CliRunner().invoke(cli.main, ["doctor"])
    # Exit code is environment-dependent (0 all-ok, 1 if e.g. models uncached);
    # what matters is it ran and printed the check table without crashing.
    assert result.exit_code in (0, 1)
    assert "Python" in result.output
    assert "ASR backend" in result.output
