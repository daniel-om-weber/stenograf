"""`steno transcribe`: formats, language/count provenance, diarization
resolution, glossary, and the split-channel paths."""

import json
import wave

import conftest
import numpy as np
from click.testing import CliRunner
from conftest import (
    write_settings,
    write_wav,
)

from stenograf import cli, loaders
from stenograf.asr.base import Segment, Word


def test_transcribe_writes_outputs_and_detects_language(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.md").exists()
    assert (tmp_path / "transcript.json").exists()
    assert (tmp_path / "transcript.txt").exists()
    assert "language: detected de" in result.output  # LID ran over the German text


def test_transcribe_records_parameter_provenance_in_json(tmp_path, stub_backends):
    # No --lang and no --speakers: both are auto, so the JSON must record them as
    # detected (language via LID, count via the finalize), not as user-set (3b).
    # --diarization opts in so the count is estimated rather than pinned to 1.
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--diarization"]
    )

    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "transcript.json").read_text())["parameters"]
    assert params["language"] == {"value": "de", "provenance": "detected"}
    assert params["speakers"]["audio"]["provenance"] == "detected"


def test_transcribe_explicit_language_is_recorded_as_explicit(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main,
        ["transcribe", str(audio), "--out", str(tmp_path), "--lang", "de", "--speakers", "1"],
    )

    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "transcript.json").read_text())["parameters"]
    assert params["language"] == {"value": "de", "provenance": "explicit"}
    assert params["speakers"]["audio"] == {"value": 1, "provenance": "explicit"}


def test_transcribe_format_writes_requested_subtitle_files(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--format", "srt,vtt"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.srt").exists()
    assert (tmp_path / "transcript.vtt").exists()
    # Only the requested formats — the defaults are not written when --format overrides them.
    assert not (tmp_path / "transcript.md").exists()
    assert not (tmp_path / "transcript.json").exists()
    assert not (tmp_path / "transcript.txt").exists()
    assert (tmp_path / "transcript.vtt").read_text().startswith("WEBVTT")
    assert "transcript.srt" in result.output


def test_transcribe_no_diarization_skips_the_diarizer(tmp_path, backend_calls):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--no-diarization"]
    )

    assert result.exit_code == 0, result.output
    assert backend_calls["need_diarizer"] is False  # the diarizer model is never requested
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Speaker 1"}


def test_transcribe_no_diarization_conflicts_with_a_speaker_count(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main,
        ["transcribe", str(audio), "--out", str(tmp_path), "--no-diarization", "--speakers", "2"],
    )

    assert result.exit_code != 0
    assert "--no-diarization conflicts" in result.output


def test_apply_no_diarization_preserves_a_disabled_channel():
    # --local 0 (listen-only) stays off; unknown counts collapse to 1 (no estimate).
    assert cli.run._apply_no_diarization(True, 0, None) == (0, 1)
    assert cli.run._apply_no_diarization(False, None, 3) == (None, 3)


def test_resolve_diarization_precedence():
    # flag > explicit count > settings.toml > off.
    resolve = cli.run._resolve_diarization
    assert resolve(None, None, None) is False  # everything unset → off
    assert resolve(None, False, None) is False  # explicit off in the file too
    assert resolve(None, None, 3) is True  # an explicit count asks to diarize
    assert resolve(None, False, 3) is True  # …and beats an explicit file off too
    assert resolve(True, False, None) is True  # --diarization beats the file
    assert resolve(False, None, 3) is False  # --no-diarization wins (UsageError later)
    assert resolve(None, True, None, 1) is True  # explicit on in the file


def test_transcribe_diarization_off_by_default(tmp_path, backend_calls):
    # No settings file, no flag, no count: diarization must not run — the
    # built-in default is off, and the run says so.
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert backend_calls["need_diarizer"] is False
    assert "diarization: off" in result.output
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Speaker 1"}


def test_transcribe_settings_diarization_off_skips_the_diarizer(tmp_path, backend_calls):
    write_settings("[speakers]\ndiarization = false\n")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert backend_calls["need_diarizer"] is False
    assert "diarization: off" in result.output  # the file's default is announced
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Speaker 1"}


def test_transcribe_diarization_flag_beats_settings_off(tmp_path, backend_calls):
    write_settings("[speakers]\ndiarization = false\n")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--diarization"]
    )

    assert result.exit_code == 0, result.output
    assert backend_calls["need_diarizer"] is True
    assert "diarization: off" not in result.output


def test_transcribe_explicit_count_beats_settings_off(tmp_path, backend_calls):
    # A per-run --speakers above 1 is itself a request to diarize; the file's
    # default must not force it off (or error like the explicit flag does).
    write_settings("[speakers]\ndiarization = false\n")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--speakers", "2"]
    )

    assert result.exit_code == 0, result.output
    assert backend_calls["need_diarizer"] is True


def test_transcribe_rejects_unknown_format(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--format", "docx"]
    )

    assert result.exit_code != 0
    assert "unknown format" in result.output


def test_transcribe_glossary_corrects_the_transcript(tmp_path, stub_backends):
    # CliASR emits "...eine gute idee für uns alle"; the glossary snaps "idee"
    # to its canonical spelling in the written transcript.
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--glossary", "Idee"]
    )

    assert result.exit_code == 0, result.output
    assert "glossary: 1 term(s), 0 name(s)" in result.output
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "gute Idee für" in md


class ChannelASR(conftest.FakeASR):
    """Decodes each buffer's peak amplitude into a channel-specific word stem.

    A split run's mic (amplitude 1000) decodes to ``foxtrot…`` and its system
    channel (amplitude 3000) to ``quebec…`` — letter-disjoint stems, so the
    tests can see exactly which channel every transcript line came from (and
    the echo backstop can never mistake one channel's text for the other's).
    """

    name = "channel"
    model_id = "fake/channel"

    def transcribe(self, samples, language) -> list[Segment]:
        pcm = np.asarray(samples)
        if pcm.dtype == np.int16:
            pcm = pcm.astype(np.float32) / 32768.0
        peak = float(np.abs(pcm).max()) * 32768
        if peak == 0:
            return []
        stem = "foxtrot" if peak < 2000 else "quebec"
        words = tuple(Word(f"{stem}{i}", 0.4 * i + 0.1, 0.4 * i + 0.4) for i in range(4))
        return [Segment(" ".join(w.text for w in words), words[0].start, words[-1].end, words)]


def fake_channel_backends(
    *, need_diarizer, asr_backend=None, asr_provider=None, announce=None, **_
):
    return ChannelASR(), None, None


def write_stereo_wav(path, left: np.ndarray, right: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(np.column_stack([left, right]).ravel().astype(np.int16).tobytes())


def _voice_channel_pcms(seconds: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Turn-taking voice channels: local speaks the first half, remote the second."""
    left = np.zeros(seconds * 16_000, dtype=np.int16)
    right = np.zeros(seconds * 16_000, dtype=np.int16)
    left[: seconds * 8_000] = 1000
    right[seconds * 8_000 :] = 3000
    return left, right


def test_transcribe_auto_splits_independent_voice_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
    audio = tmp_path / "meeting.wav"
    write_stereo_wav(audio, *_voice_channel_pcms())

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--diarization"]
    )

    assert result.exit_code == 0, result.output
    assert "2 voice channels" in result.output
    assert "left → Local, right → Remote" in result.output
    assert "local (detected)" in result.output  # meeting-style per-channel counts
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    by_speaker = {e["speaker"] for e in entries}
    assert by_speaker == {"Local-1", "Remote-1"}
    # No cross-channel bleed: each channel decoded its own audio only.
    for entry in entries:
        stem = "foxtrot" if entry["speaker"] == "Local-1" else "quebec"
        assert stem in entry["text"]


def test_transcribe_channels_mix_forces_the_downmix(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
    audio = tmp_path / "meeting.wav"
    write_stereo_wav(audio, *_voice_channel_pcms())

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--channels", "mix", "--out", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "voice channels" not in result.output
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Speaker 1"}  # classic single stream


def test_transcribe_auto_downmixes_a_stereo_image(tmp_path, monkeypatch):
    # The same programme on both channels (panned): every voice would be
    # transcribed twice if split, so auto must keep the classic downmix.
    monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
    left, _ = _voice_channel_pcms()
    audio = tmp_path / "meeting.wav"
    write_stereo_wav(audio, left, (left * 0.5).astype(np.int16))

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "stereo image" in result.output
    assert "--channels split" in result.output  # the override is advertised
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Speaker 1"}


def test_transcribe_split_needs_two_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--channels", "split", "--out", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "needs 2-channel audio" in result.output


def test_transcribe_split_conflicts_with_speakers(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
    audio = tmp_path / "meeting.wav"
    write_stereo_wav(audio, *_voice_channel_pcms())

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--speakers", "3", "--out", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "--local/--remote" in result.output


def test_transcribe_local_remote_require_split_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--local", "1", "--out", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "split voice channels" in result.output


def test_transcribe_split_matches_start_replay(tmp_path, monkeypatch):
    # The unification promise: a split-channel transcribe IS the meeting
    # pipeline, so it must produce the same transcript as replaying the two
    # channels through `steno start` (batch mode; --no-aec because a recording
    # is past capture-time cancellation).
    monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
    left, right = _voice_channel_pcms()
    stereo = tmp_path / "stereo.wav"
    write_stereo_wav(stereo, left, right)
    mic, system = tmp_path / "mic.wav", tmp_path / "system.wav"
    for path, pcm in ((mic, left), (system, right)):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16_000)
            w.writeframes(pcm.tobytes())

    split_out, replay_out = tmp_path / "split", tmp_path / "replay"
    split = CliRunner().invoke(
        cli.main, ["transcribe", str(stereo), "--channels", "split", "--out", str(split_out)]
    )
    replay = CliRunner().invoke(
        cli.main,
        [
            "start",
            "--replay",
            f"{mic},{system}",
            "--no-live",
            "--no-aec",
            "--out",
            str(replay_out),
        ],
    )

    assert split.exit_code == 0, split.output
    assert replay.exit_code == 0, replay.output
    split_entries = json.loads((split_out / "transcript.json").read_text())["entries"]
    replay_entries = json.loads((replay_out / "transcript.json").read_text())["entries"]
    assert split_entries == replay_entries


def test_transcribe_surfaces_estimated_count_as_editable(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--diarization"]
    )

    assert result.exit_code == 0, result.output
    assert "speakers: 1 detected" in result.output
    assert "re-run with --speakers 1" in result.output
