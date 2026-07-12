import json
import sys
import wave
from pathlib import Path

import conftest
import numpy as np
import pytest
from click.testing import CliRunner
from conftest import write_wav

from stenograf import cli, loaders, output
from stenograf.asr.base import Segment, Word
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.view import LiveView


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Point the data dir (profile store, settings) and the meetings output home
    at throwaway locations.

    A run without ``--out`` creates its meeting folder in the output home
    (~/Documents/Meetings by default); patching :func:`default_output_home`
    keeps every CLI test out of the real one, and ``$STENOGRAF_DATA`` keeps
    settings/profiles off the real data dir."""
    from stenograf import output

    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "steno-data"))
    monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings-home")


class FakeASR(conftest.FakeASR):
    """Returns fixed German text so the whole CLI path (incl. LID) runs offline."""

    model_id = "fake/model"

    def transcribe(self, samples, language) -> list[Segment]:
        return [
            Segment(
                text="und das ist wirklich eine gute idee für uns alle",
                start=0.1,
                end=1.0,
                words=(Word("und", 0.1, 0.3), Word("das", 0.3, 0.6)),
            )
        ]


class FakeDiarizer(Diarizer):
    """Fixed clusters with fixed unit embeddings — no ONNX model, no real audio.

    Lets the CLI re-ID/enrollment paths run offline: enrollment reads
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


def fake_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
    # No VAD (whole buffer is one window) and no diarizer (single speaker).
    return FakeASR(), None, None


@pytest.fixture
def stub_backends(monkeypatch):
    """Route the loaders seam to the offline fakes — no weights, no downloads.

    Tests that need a *custom* fake (a recording wrapper, a specific diarizer)
    still patch ``loaders.load_backends`` themselves."""
    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)


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
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

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


def test_transcribe_no_diarization_skips_the_diarizer(tmp_path, monkeypatch):
    calls = {}

    def recording_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
        calls["need_diarizer"] = need_diarizer
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--no-diarization"]
    )

    assert result.exit_code == 0, result.output
    assert calls["need_diarizer"] is False  # the diarizer model is never requested
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
    # flag > explicit count > settings.toml > on.
    resolve = cli.run._resolve_diarization
    assert resolve(None, None, None) is True  # everything unset → on
    assert resolve(None, False, None) is False  # settings turn it off
    assert resolve(True, False, None) is True  # --diarization beats the file
    assert resolve(None, False, 3) is True  # an explicit count asks to diarize
    assert resolve(False, None, 3) is False  # --no-diarization wins (UsageError later)
    assert resolve(None, True, None, 1) is True  # explicit on in the file


def _write_settings(monkeypatch_env_dir: Path, body: str) -> None:
    """Drop a settings.toml into the test's $STENOGRAF_DATA dir."""
    data = monkeypatch_env_dir / "steno-data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "settings.toml").write_text(body, encoding="utf-8")


def test_transcribe_settings_diarization_off_skips_the_diarizer(tmp_path, monkeypatch):
    calls = {}

    def recording_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
        calls["need_diarizer"] = need_diarizer
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording_load_backends)
    _write_settings(tmp_path, "[speakers]\ndiarization = false\n")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls["need_diarizer"] is False
    assert "diarization: off" in result.output  # the file's default is announced
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Speaker 1"}


def test_transcribe_diarization_flag_beats_settings_off(tmp_path, monkeypatch):
    calls = {}

    def recording_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
        calls["need_diarizer"] = need_diarizer
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording_load_backends)
    _write_settings(tmp_path, "[speakers]\ndiarization = false\n")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--diarization"]
    )

    assert result.exit_code == 0, result.output
    assert calls["need_diarizer"] is True
    assert "diarization: off" not in result.output


def test_transcribe_explicit_count_beats_settings_off(tmp_path, monkeypatch):
    # A per-run --speakers above 1 is itself a request to diarize; the file's
    # default must not force it off (or error like the explicit flag does).
    calls = {}

    def recording_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
        calls["need_diarizer"] = need_diarizer
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording_load_backends)
    _write_settings(tmp_path, "[speakers]\ndiarization = false\n")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--speakers", "2"]
    )

    assert result.exit_code == 0, result.output
    assert calls["need_diarizer"] is True


def test_cleanup_checkpoints_removes_every_checkpoint_format(tmp_path):
    for fmt in ("md", "json", "txt"):
        (tmp_path / f"transcript.partial.{fmt}").write_text("x", encoding="utf-8")
    output.cleanup_checkpoints(tmp_path, "transcript")
    assert not list(tmp_path.glob("transcript.partial.*"))


def test_transcribe_rejects_unknown_format(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--format", "docx"]
    )

    assert result.exit_code != 0
    assert "unknown format" in result.output


def test_transcribe_glossary_corrects_the_transcript(tmp_path, stub_backends):
    # FakeASR emits "...eine gute idee für uns alle"; the glossary snaps "idee"
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


def fake_channel_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
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

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

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


def test_start_replay_streams_live_captions_by_default(tmp_path, stub_backends):
    # Default is live: a non-TTY runner gets the plain caption stream, then the
    # on-stop finalize swap. The whole live path runs through the real orchestrator.
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        ["start", "--local", "1", "--remote", "0", "--replay", str(mic), "--out", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.md").exists()  # --out is this meeting's managed dir
    assert "You:" in result.output  # a live caption streamed
    assert "language: de" in result.output  # structured language event, plain-rendered
    assert "finalized:" in result.output  # the on-stop finalize swap was announced


def test_start_no_diarization_skips_the_diarizer(tmp_path, monkeypatch):
    calls = {}

    def recording_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
        calls["need_diarizer"] = need_diarizer
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        [
            "start",
            "--no-diarization",
            "--remote",
            "0",
            "--replay",
            str(mic),
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["need_diarizer"] is False  # counts collapsed to 1 → no diarizer load
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Local-1"}


def test_start_settings_diarization_off_skips_the_diarizer(tmp_path, monkeypatch):
    calls = {}

    def recording_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
        calls["need_diarizer"] = need_diarizer
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording_load_backends)
    _write_settings(tmp_path, "[speakers]\ndiarization = false\n")
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        ["start", "--remote", "0", "--replay", str(mic), "--out", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls["need_diarizer"] is False
    assert "diarization: off" in result.output
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Local-1"}


def test_start_no_live_uses_the_batch_path(tmp_path, stub_backends):
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        [
            "start",
            "--no-live",
            "--local",
            "1",
            "--remote",
            "0",
            "--replay",
            str(mic),
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.md").exists()
    assert "detected language: de" in result.output  # legacy status-string wording
    assert "You:" not in result.output  # no live captions in batch mode


def test_start_surfaces_estimated_local_count_as_editable(tmp_path, stub_backends):
    # Omitting --local estimates the mic count (Stage 3a); the summary shows the
    # detected count and the exact flag to lock or correct it by re-running.
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        ["start", "--remote", "0", "--replay", str(mic), "--no-live", "--out", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "local (detected)" in result.output
    assert "re-run with --local 1" in result.output  # the correction hint


def test_start_reports_given_counts_without_a_correction_hint(tmp_path, stub_backends):
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        [
            "start",
            "--local",
            "1",
            "--remote",
            "0",
            "--replay",
            str(mic),
            "--no-live",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 local (given)" in result.output
    assert "re-run with" not in result.output  # nothing was estimated


def test_transcribe_surfaces_estimated_count_as_editable(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "speakers: 1 detected" in result.output
    assert "re-run with --speakers 1" in result.output


def test_flush_interval_and_checkpoint_interval_are_aliases(tmp_path, stub_backends):
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    for flag in ("--flush-interval", "--checkpoint-interval"):
        result = CliRunner().invoke(
            cli.main,
            [
                "start",
                "--no-live",
                "--local",
                "1",
                "--remote",
                "0",
                "--replay",
                str(mic),
                flag,
                "0",
                "--out",
                str(tmp_path / flag.lstrip("-")),
            ],
        )
        assert result.exit_code == 0, result.output


def test_resolve_flush_interval_defaults_are_mode_aware():
    # The live checkpoint is zero-inference file I/O → tight default; the batch
    # checkpoint runs VAD+ASR over the tail → sparse default. Explicit values
    # (including 0 = disabled) win in both modes.
    assert cli.start._resolve_flush_interval(None, live=True) == 15.0
    assert cli.start._resolve_flush_interval(None, live=False) == 180.0
    assert cli.start._resolve_flush_interval(45.0, live=True) == 45.0
    assert cli.start._resolve_flush_interval(0.0, live=True) == 0.0


def test_persist_once_writes_once_and_replays_paths():
    sentinel = object()
    calls = []
    persist = cli.start._PersistOnce(lambda t: calls.append(t) or [Path("t.md")])
    assert persist(sentinel) == [Path("t.md")]
    assert persist(sentinel) == [Path("t.md")]  # second call replays, no rewrite
    assert calls == [sentinel]


def test_persist_once_retries_after_a_failed_write():
    attempts = []

    def flaky(transcript):
        attempts.append(transcript)
        if len(attempts) == 1:
            raise OSError("disk full")
        return [Path("t.md")]

    persist = cli.start._PersistOnce(flaky)
    with pytest.raises(OSError):
        persist(object())  # the event-time write fails...
    assert persist.paths is None  # ...and is not marked done
    assert persist(object()) == [Path("t.md")]  # the exit-path call retries


def test_plain_forces_the_stream_even_on_a_tty(tmp_path, monkeypatch):
    served = []

    class FakeTUI(LiveView):  # records if the TUI was chosen; never opens a terminal
        def __init__(self, profile, *, language=None, stop=None, persist=None):
            self.stop = stop
            self.persist = persist

        def serve(self, meeting):
            served.append(self)
            return meeting()

    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
    monkeypatch.setattr(cli.start, "_stdout_is_tty", lambda: True)  # pretend we're on a terminal
    monkeypatch.setattr("stenograf.ui.meeting.TextualLiveView", FakeTUI)
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    tui_run = CliRunner().invoke(
        cli.main,
        [
            "start",
            "--local",
            "1",
            "--remote",
            "0",
            "--replay",
            str(mic),
            "--out",
            str(tmp_path / "tui"),
        ],
    )
    assert tui_run.exit_code == 0, tui_run.output
    assert served, "on a TTY the default live run should pick the Textual view"
    assert served[0].persist is not None, "the TUI should get the write-at-finalize hook"

    served.clear()
    plain_run = CliRunner().invoke(
        cli.main,
        [
            "start",
            "--plain",
            "--local",
            "1",
            "--remote",
            "0",
            "--replay",
            str(mic),
            "--out",
            str(tmp_path / "plain"),
        ],
    )
    assert plain_run.exit_code == 0, plain_run.output
    assert not served, "--plain must bypass the TUI even on a TTY"
    assert "You:" in plain_run.output


def test_doctor_runs_and_prints_checks():
    result = CliRunner().invoke(cli.main, ["doctor"])
    # Exit code is environment-dependent (0 all-ok, 1 if e.g. models uncached);
    # what matters is it ran and printed the check table without crashing.
    assert result.exit_code in (0, 1)
    assert "Python" in result.output
    assert "ASR backend" in result.output


# ---- speaker profiles (Task 1c) -------------------------------------------


def _isolate_store(tmp_path, monkeypatch):
    """Point the profile store at a throwaway dir so tests never touch the real one."""
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))


def _patch_diarizer(monkeypatch, diarizer):
    monkeypatch.setattr(loaders, "load_diarizer", lambda: diarizer)


def test_profiles_list_empty(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    result = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert result.exit_code == 0, result.output
    assert "no speaker profiles yet" in result.output


def test_profiles_enroll_then_list(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    _patch_diarizer(monkeypatch, FakeDiarizer({"S0": [1.0, 0, 0]}))
    audio = tmp_path / "daniel.wav"
    write_wav(audio)

    enroll = CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])
    assert enroll.exit_code == 0, enroll.output
    assert "enrolled 'Daniel'" in enroll.output

    listing = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert "Daniel" in listing.output
    assert "(1 sample)" in listing.output


def test_profiles_enroll_duplicate_then_reinforce(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    _patch_diarizer(monkeypatch, FakeDiarizer({"S0": [1.0, 0, 0]}))
    audio = tmp_path / "a.wav"
    write_wav(audio)
    CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])

    dup = CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])
    assert dup.exit_code != 0
    assert "--reinforce" in dup.output  # points the user at the right flag

    again = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Daniel", str(audio), "--reinforce"]
    )
    assert again.exit_code == 0, again.output
    assert "2 samples" in again.output


def test_profiles_enroll_multispeaker_needs_speaker_choice(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    diar = FakeDiarizer(
        {"S0": [1.0, 0, 0], "S1": [0, 1.0, 0]},
        turns=[SpeakerTurn("S0", 0.0, 2.0), SpeakerTurn("S1", 2.0, 3.0)],
    )
    _patch_diarizer(monkeypatch, diar)
    audio = tmp_path / "m.wav"
    write_wav(audio)

    ambiguous = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Anna", str(audio), "--speakers", "2"]
    )
    assert ambiguous.exit_code != 0
    assert "S0" in ambiguous.output and "S1" in ambiguous.output  # lists the choices

    chosen = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Anna", str(audio), "--speakers", "2", "--speaker", "S1"]
    )
    assert chosen.exit_code == 0, chosen.output


def test_profiles_rename_and_remove(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    _patch_diarizer(monkeypatch, FakeDiarizer({"S0": [1.0, 0, 0]}))
    audio = tmp_path / "a.wav"
    write_wav(audio)
    CliRunner().invoke(cli.main, ["profiles", "enroll", "Speaker 1", str(audio)])

    renamed = CliRunner().invoke(cli.main, ["profiles", "rename", "Speaker 1", "Daniel"])
    assert renamed.exit_code == 0, renamed.output
    after = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert "Daniel" in after.output and "Speaker 1" not in after.output

    removed = CliRunner().invoke(cli.main, ["profiles", "remove", "Daniel", "--yes"])
    assert removed.exit_code == 0, removed.output
    assert "no speaker profiles yet" in CliRunner().invoke(cli.main, ["profiles", "list"]).output


def test_transcribe_reid_relabels_enrolled_speaker(tmp_path, monkeypatch):
    # End-to-end: enroll Daniel, then a diarized transcribe relabels his cluster
    # to "Daniel" instead of the generic "Speaker 1"; --no-reid restores it.
    _isolate_store(tmp_path, monkeypatch)
    diar = FakeDiarizer({"S0": [1.0, 0, 0]})
    _patch_diarizer(monkeypatch, diar)
    audio = tmp_path / "m.wav"
    write_wav(audio)
    CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])

    monkeypatch.setattr(
        loaders,
        "load_backends",
        lambda *, need_diarizer, asr_backend=None, asr_provider=None: (FakeASR(), None, diar),
    )
    reid = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--speakers", "2", "--out", str(tmp_path)]
    )
    assert reid.exit_code == 0, reid.output
    assert "re-ID: 1 profile(s) active" in reid.output
    assert "Daniel" in (tmp_path / "transcript.md").read_text()

    # The --no-reid re-run replaces the transcript in place — the --force flow.
    no_reid = CliRunner().invoke(
        cli.main,
        [
            "transcribe",
            str(audio),
            "--speakers",
            "2",
            "--no-reid",
            "--out",
            str(tmp_path),
            "--force",
        ],
    )
    assert no_reid.exit_code == 0, no_reid.output
    assert "re-ID:" not in no_reid.output
    md = (tmp_path / "transcript.md").read_text()
    assert "Daniel" not in md and "Speaker 1" in md


class TestSpeakerCountHints:
    """The 'lock the detected count' hint must stay actionable (Phase 3→4 audit).

    An unconstrained diarizer can detect more (or, on silence, zero) speakers than
    the user can set, so the hint is clamped to the settable range and suppressed
    when there is nothing to lock — a form-driven web UI inherits these paths.
    """

    def test_lock_hint_clamps_and_guards(self):
        assert cli.format._lock_hint(0, 8) is None  # no speech found → nothing to lock
        assert cli.format._lock_hint(1, 8) == (1, False)
        assert cli.format._lock_hint(3, 8) == (3, False)
        assert cli.format._lock_hint(13, 8) == (8, True)  # over-cluster → clamp to the max

    def test_silent_channel_gives_no_bogus_zero_hint(self, capsys):
        from stenograf.capture.base import Channel
        from stenograf.session import SpeakerCount

        cli.format._report_speaker_counts([SpeakerCount(Channel.MIC, None, 0)])
        out = capsys.readouterr().out
        assert "0 local (detected)" in out
        assert "re-run with" not in out  # never suggests the nonsensical `--local 0`

    def test_over_range_estimate_is_clamped_in_the_hint(self, capsys):
        from stenograf.capture.base import Channel
        from stenograf.session import SpeakerCount

        cli.format._report_speaker_counts([SpeakerCount(Channel.MIC, None, 13)])
        out = capsys.readouterr().out
        assert "13 local (detected)" in out  # the raw estimate is still shown
        assert "re-run with --local 8" in out  # clamped to the settable max
        assert "exceeded the 8-speaker max" in out


def test_start_with_no_speakers_errors_cleanly(tmp_path, monkeypatch):
    # --local 0 --remote 0 violates MeetingProfile; the CLI must report it as a
    # clean error, not leak the ValueError traceback (a web UI feeds form values in).
    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)
    result = CliRunner().invoke(
        cli.main,
        ["start", "--local", "0", "--remote", "0", "--replay", str(mic), "--out", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "at least one speaker" in result.output
    assert not isinstance(result.exception, ValueError)  # handled as a ClickException


# ---- output home (Stage C1/C2) ---------------------------------------------


def _start_batch(tmp_path, monkeypatch, *extra):
    """Run a minimal, deterministic ``steno start`` (batch replay) and return the result."""
    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)
    return CliRunner().invoke(
        cli.main,
        ["start", "--local", "1", "--remote", "0", "--replay", str(mic), "--no-live", *extra],
    )


def test_start_writes_a_dated_folder_into_the_output_home(tmp_path, monkeypatch):
    # No --out: the meeting gets its own meeting-YYYYMMDD-HHMMSS/ folder in the
    # visible output home, holding plainly named transcript files.
    from stenograf.transcript import Transcript

    result = _start_batch(tmp_path, monkeypatch, "--title", "Weekly sync")
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert meeting_dir.name.startswith("meeting-")
    assert (meeting_dir / "transcript.json").exists()
    assert str(meeting_dir) in result.output  # the CLI says where the files landed
    transcript = Transcript.from_json((meeting_dir / "transcript.json").read_text())
    assert transcript.profile.title == "Weekly sync"


def test_start_out_is_the_meetings_own_folder(tmp_path, monkeypatch):
    out = tmp_path / "custom"
    result = _start_batch(tmp_path, monkeypatch, "--out", str(out))
    assert result.exit_code == 0, result.output
    assert (out / "transcript.json").exists()  # files land directly in --out
    assert not (tmp_path / "meetings-home").exists()  # the home is untouched


def test_out_refuses_an_existing_transcript_unless_forced(tmp_path, monkeypatch):
    # Fixed file names mean a reused --out would silently replace the previous
    # meeting — refuse, and let --force say overwriting is the point.
    out = tmp_path / "custom"
    assert _start_batch(tmp_path, monkeypatch, "--out", str(out)).exit_code == 0
    first = (out / "transcript.md").read_text(encoding="utf-8")

    refused = _start_batch(tmp_path, monkeypatch, "--out", str(out), "--title", "Second")
    assert refused.exit_code != 0
    assert "--force" in refused.output
    assert (out / "transcript.md").read_text(encoding="utf-8") == first  # untouched

    forced = _start_batch(tmp_path, monkeypatch, "--out", str(out), "--force")
    assert forced.exit_code == 0, forced.output


def test_out_overwrite_guard_ignores_partial_checkpoints(tmp_path, monkeypatch):
    # A crashed run leaves only .partial files; recovering into the same folder
    # must not demand --force.
    out = tmp_path / "custom"
    out.mkdir()
    (out / "transcript.partial.md").write_text("crashed", encoding="utf-8")
    result = _start_batch(tmp_path, monkeypatch, "--out", str(out))
    assert result.exit_code == 0, result.output


def test_transcribe_out_refusal_happens_before_any_transcription(tmp_path, monkeypatch):
    def explode(*, need_diarizer, asr_backend=None, asr_provider=None):
        raise AssertionError("backends must not load when --out is refused")

    monkeypatch.setattr(loaders, "load_backends", explode)
    out = tmp_path / "custom"
    out.mkdir()
    (out / "transcript.md").write_text("previous meeting", encoding="utf-8")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(out)])
    assert result.exit_code != 0
    assert "--force" in result.output


def test_transcribe_writes_into_the_output_home_by_default(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio)])
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert (meeting_dir / "transcript.md").exists()
    assert str(meeting_dir) in result.output


def test_no_index_is_ever_written(tmp_path, monkeypatch):
    # Stage C2: the filesystem is the index. A run leaves exactly the meeting
    # folder — no index.json in the home or the data dir, ever.
    result = _start_batch(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert not list((tmp_path / "meetings-home").rglob("index.json"))
    assert not list((tmp_path / "steno-data").rglob("index.json"))


def test_record_audio_lands_in_the_meeting_folder(tmp_path, monkeypatch):
    result = _start_batch(tmp_path, monkeypatch, "--record-audio")
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert (meeting_dir / "audio.wav").exists()


def _helper_wrapper(tmp_path, *forced_args):
    """An executable stand-in for stenocap; forced_args replace the real argv."""
    fake = Path(__file__).parent / "fake_stenocap.py"
    args = " ".join(forced_args) if forced_args else '"$@"'
    script = tmp_path / "stenocap"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" {args}\n')
    script.chmod(0o755)
    return script


@pytest.mark.skipif(sys.platform != "darwin", reason="steno setup is macOS-only")
def test_setup_grants_permissions_then_prefetches(tmp_path, monkeypatch):
    monkeypatch.setenv("STENOGRAF_CAPTURE_HELPER", str(_helper_wrapper(tmp_path)))
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup"])
    assert result.exit_code == 0, result.output
    assert "granted" in result.output
    assert fetched  # downloads run after the permission step
    assert "setup complete" in result.output


@pytest.mark.skipif(sys.platform != "darwin", reason="steno setup is macOS-only")
def test_setup_fails_when_helper_dies_without_mic_frames(tmp_path, monkeypatch):
    # A denied permission means the helper exits before its first mic frame;
    # emitting only system frames then exiting reproduces that shape.
    monkeypatch.setenv("STENOGRAF_CAPTURE_HELPER", str(_helper_wrapper(tmp_path, "--system")))
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup"])
    assert result.exit_code != 0
    assert "denied" in result.output
    assert not fetched  # no downloads on a failed permission grant


def test_setup_models_only_skips_the_permission_step(monkeypatch):
    # No STENOGRAF_CAPTURE_HELPER and no fake helper: reaching the permission
    # code would fail loudly, so success proves it was skipped. Runs on any OS.
    monkeypatch.delenv("STENOGRAF_CAPTURE_HELPER", raising=False)
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup", "--models-only"])
    assert result.exit_code == 0, result.output
    assert fetched
    assert "granted" not in result.output


def test_prefetch_models_downloads_missing_and_loads_asr(monkeypatch, tmp_path):
    from stenograf import models
    from stenograf.asr.base import ASRBackend

    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    # One asset pre-cached, the rest missing: only the missing ones are fetched.
    (tmp_path / models.SILERO_VAD.name).write_bytes(b"\x00")
    fetched = []
    monkeypatch.setattr(models, "fetch", lambda asset, progress=None: fetched.append(asset.name))

    class PrefetchASR(ASRBackend):
        name = "fake"
        model_id = "fake/model"
        calls = []

        def load(self):
            self.calls.append("load")

        def transcribe(self, samples, language):
            return []

        def unload(self):
            self.calls.append("unload")

    import stenograf.asr as asr
    from stenograf import doctor

    monkeypatch.setattr(doctor, "installed", lambda module: True)  # deps "present" (any OS)
    monkeypatch.setattr(asr, "create_backend", lambda name=None, **kw: PrefetchASR())
    loaders.prefetch_models()
    assert set(fetched) == {models.PYANNOTE_SEGMENTATION.name, models.SPEAKER_EMBEDDING.name}
    assert PrefetchASR.calls == ["load", "unload"]  # weights pulled and released


def test_prefetch_models_skips_asr_when_backend_deps_absent(monkeypatch, tmp_path, capsys):
    from stenograf import doctor, models

    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    monkeypatch.setattr(models, "fetch", lambda asset, progress=None: None)
    monkeypatch.setattr(doctor, "installed", lambda module: False)  # the Linux shape
    import stenograf.asr as asr

    def boom(name=None, **kw):
        raise AssertionError("create_backend must not run without its deps")

    monkeypatch.setattr(asr, "create_backend", boom)
    loaders.prefetch_models()  # must not raise
    assert "skipping its weights" in capsys.readouterr().out


def test_load_backends_refuses_uninstalled_backend(monkeypatch):
    """A selected backend whose runtime is absent must be a CLI error, not an
    import traceback from deep inside ``asr.load()``."""
    import click

    from stenograf import doctor

    monkeypatch.setattr(doctor, "installed", lambda module: False)
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "parakeet")
    with pytest.raises(click.ClickException, match="parakeet-mlx is not installed"):
        loaders.load_backends(need_diarizer=False)


def test_load_backends_refuses_unknown_backend(monkeypatch):
    import click

    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "no-such-backend")
    with pytest.raises(click.ClickException, match="unknown ASR backend"):
        loaders.load_backends(need_diarizer=False)


# ---------------------------------------------------------------------------
# settings.toml wiring — one test per *mechanism* (the resolution helpers are
# shared, so file-beats-default / flag-beats-file / tri-state / merge each need
# proving once, not per field).


def _write_settings(tmp_path, text):
    """Write settings.toml into the isolated $STENOGRAF_DATA dir."""
    data_dir = tmp_path / "steno-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.toml").write_text(text, encoding="utf-8")


def test_settings_formats_are_the_default_but_format_flag_wins(tmp_path, stub_backends):
    _write_settings(tmp_path, '[transcript]\nformats = ["srt"]\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    out1 = tmp_path / "one"
    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(out1)])
    assert result.exit_code == 0, result.output
    assert (out1 / "transcript.srt").exists()
    assert not (out1 / "transcript.md").exists()

    out2 = tmp_path / "two"
    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(out2), "--format", "md"]
    )
    assert result.exit_code == 0, result.output
    assert (out2 / "transcript.md").exists()
    assert not (out2 / "transcript.srt").exists()


def test_settings_output_dir_replaces_the_home_and_out_flag_wins(tmp_path, stub_backends):
    home = tmp_path / "configured-home"
    # as_posix(): a raw Windows path in a TOML basic string is invalid (\U…).
    _write_settings(tmp_path, f'[output]\ndir = "{home.as_posix()}"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    # File-beats-default: no flags → the meeting folder is created in [output] dir.
    result = CliRunner().invoke(cli.main, ["transcribe", str(audio)])
    assert result.exit_code == 0, result.output
    (meeting_dir,) = home.iterdir()
    assert (meeting_dir / "transcript.md").exists()

    # Flag-beats-file: --out bypasses the configured home for this run.
    out = tmp_path / "explicit"
    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "transcript.md").exists()
    assert len(list(home.iterdir())) == 1  # nothing new in the home


def test_settings_vocab_merges_with_flags(tmp_path, stub_backends):
    glossary_file = tmp_path / "glossary.txt"
    glossary_file.write_text("Idee\n", encoding="utf-8")
    _write_settings(
        tmp_path, f'[vocab]\nglossary_file = "{glossary_file.as_posix()}"\nattendees = ["Ada"]\n'
    )
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--glossary", "Wirklich"]
    )

    assert result.exit_code == 0, result.output
    # Configured file + inline flag merge (2 terms), attendees ride along (1 name).
    assert "glossary: 2 term(s), 1 name(s)" in result.output
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "gute Idee für" in md  # the settings-file term corrected the transcript


def test_settings_missing_glossary_file_is_a_clean_error(tmp_path, stub_backends):
    _write_settings(tmp_path, '[vocab]\nglossary_file = "/nonexistent/glossary.txt"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code != 0
    assert "cannot read glossary file" in result.output
    assert "[vocab] glossary_file" in result.output  # says where the bad path came from


def test_broken_settings_fail_fast_with_a_clean_error(tmp_path, stub_backends):
    _write_settings(tmp_path, '[vocab]\nglossry_file = "x"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code != 0
    assert "invalid settings" in result.output
    assert "glossry_file" in result.output
    assert "Traceback" not in result.output


def test_settings_asr_backend_reaches_the_loader(tmp_path, monkeypatch):
    calls = {}

    def recording(*, need_diarizer, asr_backend=None, asr_provider=None):
        calls["asr_backend"] = asr_backend
        calls["asr_provider"] = asr_provider
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording)
    _write_settings(tmp_path, '[asr]\nbackend = "parakeet"\nprovider = "dml"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls["asr_backend"] == "parakeet"
    assert calls["asr_provider"] == "dml"


def test_settings_profile_store_stays_off_the_transcript(tmp_path, stub_backends):
    # The configured store must feed re-ID loading only: MeetingProfile serializes
    # into every transcript, and keeping machine-local paths out of shared files
    # is the settings file's founding rule.
    _write_settings(
        tmp_path, f'[speakers]\nprofile_store = "{tmp_path.as_posix()}/profiles.json"\n'
    )
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    profile = json.loads((tmp_path / "transcript.json").read_text())["profile"]
    assert profile.get("speaker_profile_store") is None


def test_settings_show_reports_values_and_sources(tmp_path, monkeypatch):
    _write_settings(tmp_path, '[transcript]\nformats = ["srt"]\n')

    result = CliRunner().invoke(cli.main, ["settings", "show"])

    assert result.exit_code == 0, result.output
    assert 'formats = ["srt"]  (settings.toml)' in result.output
    assert "glossary_threshold = 0.82  (default)" in result.output
    assert "[notes.export]" in result.output

    # An env override wins over the file and is attributed to the variable.
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "parakeet")
    result = CliRunner().invoke(cli.main, ["settings", "show"])
    assert "backend  = parakeet  ($STENOGRAF_ASR_BACKEND)" in result.output
    assert "provider = cpu  (default)" in result.output


def test_settings_show_names_a_missing_file(tmp_path):
    result = CliRunner().invoke(cli.main, ["settings", "show"])
    assert result.exit_code == 0, result.output
    assert "not present — all defaults" in result.output


def test_settings_show_broken_file_points_at_edit(tmp_path):
    _write_settings(tmp_path, "[vocab]\nbad_key = 1\n")
    result = CliRunner().invoke(cli.main, ["settings", "show"])
    assert result.exit_code != 0
    assert "bad_key" in result.output
    assert "steno settings edit" in result.output


def test_settings_edit_creates_the_template_and_validates(tmp_path, monkeypatch):
    opened = {}
    monkeypatch.setattr(cli.click, "edit", lambda filename=None: opened.update(path=filename))

    result = CliRunner().invoke(cli.main, ["settings", "edit"])

    assert result.exit_code == 0, result.output
    path = tmp_path / "steno-data" / "settings.toml"
    assert opened["path"] == str(path)
    assert "created" in result.output
    assert "OK" in result.output
    assert path.read_text(encoding="utf-8").startswith("# stenograf settings")


def test_settings_edit_keeps_and_reports_a_bad_save(tmp_path, monkeypatch):
    def fake_edit(filename=None):
        Path(filename).write_text('[vocab]\nglossry_file = "x"\n', encoding="utf-8")

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = CliRunner().invoke(cli.main, ["settings", "edit"])

    assert result.exit_code != 0
    assert "glossry_file" in result.output
    assert "your edits are saved" in result.output
    # The bad content was not reverted — the user's work survives the failure.
    path = tmp_path / "steno-data" / "settings.toml"
    assert "glossry_file" in path.read_text(encoding="utf-8")


# -- bare invocation (Phase 7: the launcher entry) ---------------------------


def test_bare_invocation_without_a_tty_prints_help():
    # A pipe/script hitting bare `steno` wants usage text, not a Textual app
    # (which needs a real TTY anyway). CliRunner streams are never TTYs.
    result = CliRunner().invoke(cli.main, [])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "transcribe" in result.output  # the subcommands are listed


def test_bare_invocation_on_a_tty_opens_the_launcher(monkeypatch):
    import stenograf.ui

    calls = []
    monkeypatch.setattr(cli, "_interactive_terminal", lambda: True)
    monkeypatch.setattr(stenograf.ui, "run_launcher", lambda: calls.append(1))

    result = CliRunner().invoke(cli.main, [])

    assert result.exit_code == 0
    assert calls == [1]
    assert "Usage:" not in result.output


def test_subcommands_never_open_the_launcher(monkeypatch):
    import stenograf.ui

    monkeypatch.setattr(cli, "_interactive_terminal", lambda: True)
    monkeypatch.setattr(
        stenograf.ui, "run_launcher", lambda: (_ for _ in ()).throw(AssertionError)
    )

    result = CliRunner().invoke(cli.main, ["profiles", "list"])

    assert result.exit_code == 0
