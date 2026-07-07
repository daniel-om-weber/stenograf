import json
import wave

import numpy as np
from click.testing import CliRunner

from stenograf import cli
from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.view import LiveView


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


def test_transcribe_records_parameter_provenance_in_json(tmp_path, monkeypatch):
    # No --lang and no --speakers: both are auto, so the JSON must record them as
    # detected (language via LID, count via the finalize), not as user-set (3b).
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "meeting.transcript.json").read_text())["parameters"]
    assert params["language"] == {"value": "de", "provenance": "detected"}
    assert params["speakers"]["audio"]["provenance"] == "detected"


def test_transcribe_explicit_language_is_recorded_as_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main,
        ["transcribe", str(audio), "--out", str(tmp_path), "--lang", "de", "--speakers", "1"],
    )

    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "meeting.transcript.json").read_text())["parameters"]
    assert params["language"] == {"value": "de", "provenance": "explicit"}
    assert params["speakers"]["audio"] == {"value": 1, "provenance": "explicit"}


def test_transcribe_format_writes_requested_subtitle_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--format", "srt,vtt"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "meeting.transcript.srt").exists()
    assert (tmp_path / "meeting.transcript.vtt").exists()
    # Only the requested formats — md/json are not written when --format overrides them.
    assert not (tmp_path / "meeting.transcript.md").exists()
    assert not (tmp_path / "meeting.transcript.json").exists()
    assert (tmp_path / "meeting.transcript.vtt").read_text().startswith("WEBVTT")
    assert "meeting.transcript.srt" in result.output


def test_transcribe_rejects_unknown_format(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--format", "docx"]
    )

    assert result.exit_code != 0
    assert "unknown format" in result.output


def test_transcribe_glossary_corrects_the_transcript(tmp_path, monkeypatch):
    # FakeASR emits "...eine gute idee für uns alle"; the glossary snaps "idee"
    # to its canonical spelling in the written transcript.
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--glossary", "Idee"]
    )

    assert result.exit_code == 0, result.output
    assert "glossary: 1 term(s), 0 name(s)" in result.output
    md = (tmp_path / "meeting.transcript.md").read_text()
    assert "gute Idee für" in md


def test_start_replay_streams_live_captions_by_default(tmp_path, monkeypatch):
    # Default is live: a non-TTY runner gets the plain caption stream, then the
    # on-stop finalize swap. The whole live path runs through the real orchestrator.
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
    assert "You:" in result.output  # a live caption streamed
    assert "language: de" in result.output  # structured language event, plain-rendered
    assert "finalized:" in result.output  # the on-stop finalize swap was announced


def test_start_no_live_uses_the_batch_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
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
    assert len(list(tmp_path.glob("meeting-*.transcript.md"))) == 1
    assert "detected language: de" in result.output  # legacy status-string wording
    assert "You:" not in result.output  # no live captions in batch mode


def test_start_surfaces_estimated_local_count_as_editable(tmp_path, monkeypatch):
    # Omitting --local estimates the mic count (Stage 3a); the summary shows the
    # detected count and the exact flag to lock or correct it by re-running.
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        ["start", "--remote", "0", "--replay", str(mic), "--no-live", "--out", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "local (detected)" in result.output
    assert "re-run with --local 1" in result.output  # the correction hint


def test_start_reports_given_counts_without_a_correction_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
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


def test_transcribe_surfaces_estimated_count_as_editable(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "speakers: 1 detected" in result.output
    assert "re-run with --speakers 1" in result.output


def test_flush_interval_and_checkpoint_interval_are_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
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
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output


def test_plain_forces_the_stream_even_on_a_tty(tmp_path, monkeypatch):
    served = []

    class FakeTUI(LiveView):  # records if the TUI was chosen; never opens a terminal
        def __init__(self, profile, *, language=None, stop=None):
            self.stop = stop

        def serve(self, meeting):
            served.append(self)
            return meeting()

    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    monkeypatch.setattr(cli, "_stdout_is_tty", lambda: True)  # pretend we're on a terminal
    monkeypatch.setattr("stenograf.tui.TextualLiveView", FakeTUI)
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    tui_run = CliRunner().invoke(
        cli.main,
        ["start", "--local", "1", "--remote", "0", "--replay", str(mic), "--out", str(tmp_path)],
    )
    assert tui_run.exit_code == 0, tui_run.output
    assert served, "on a TTY the default live run should pick the Textual view"

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
            str(tmp_path),
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
    monkeypatch.setattr(cli, "_load_diarizer", lambda *, need=True: diarizer)


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

    monkeypatch.setattr(cli, "_load_backends", lambda *, need_diarizer: (FakeASR(), None, diar))
    reid = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--speakers", "2", "--out", str(tmp_path)]
    )
    assert reid.exit_code == 0, reid.output
    assert "re-ID: 1 profile(s) active" in reid.output
    assert "Daniel" in (tmp_path / "m.transcript.md").read_text()

    no_reid = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--speakers", "2", "--no-reid", "--out", str(tmp_path)]
    )
    assert no_reid.exit_code == 0, no_reid.output
    assert "re-ID:" not in no_reid.output
    md = (tmp_path / "m.transcript.md").read_text()
    assert "Daniel" not in md and "Speaker 1" in md


class TestSpeakerCountHints:
    """The 'lock the detected count' hint must stay actionable (Phase 3→4 audit).

    An unconstrained diarizer can detect more (or, on silence, zero) speakers than
    the user can set, so the hint is clamped to the settable range and suppressed
    when there is nothing to lock — a form-driven web UI inherits these paths.
    """

    def test_lock_hint_clamps_and_guards(self):
        assert cli._lock_hint(0, 8) is None  # no speech found → nothing to lock
        assert cli._lock_hint(1, 8) == (1, False)
        assert cli._lock_hint(3, 8) == (3, False)
        assert cli._lock_hint(13, 8) == (8, True)  # over-cluster → clamp to the max

    def test_silent_channel_gives_no_bogus_zero_hint(self, capsys):
        from stenograf.capture.base import Channel
        from stenograf.session import SpeakerCount

        cli._report_speaker_counts([SpeakerCount(Channel.MIC, None, 0)])
        out = capsys.readouterr().out
        assert "0 local (detected)" in out
        assert "re-run with" not in out  # never suggests the nonsensical `--local 0`

    def test_over_range_estimate_is_clamped_in_the_hint(self, capsys):
        from stenograf.capture.base import Channel
        from stenograf.session import SpeakerCount

        cli._report_speaker_counts([SpeakerCount(Channel.MIC, None, 13)])
        out = capsys.readouterr().out
        assert "13 local (detected)" in out  # the raw estimate is still shown
        assert "re-run with --local 8" in out  # clamped to the settable max
        assert "exceeded the 8-speaker max" in out


def test_start_with_no_speakers_errors_cleanly(tmp_path, monkeypatch):
    # --local 0 --remote 0 violates MeetingProfile; the CLI must report it as a
    # clean error, not leak the ValueError traceback (a web UI feeds form values in).
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)
    result = CliRunner().invoke(
        cli.main,
        ["start", "--local", "0", "--remote", "0", "--replay", str(mic), "--out", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "at least one speaker" in result.output
    assert not isinstance(result.exception, ValueError)  # handled as a ClickException
