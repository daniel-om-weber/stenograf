"""`steno start`: the live/batch meeting command — captions, diarization
switches, checkpoints, the notes tail, and speaker-count hints."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import (
    fake_load_backends,
    write_settings,
    write_wav,
)

from stenograf import cli, loaders
from stenograf.view import LiveView


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


def test_start_no_diarization_skips_the_diarizer(tmp_path, backend_calls):
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
    assert backend_calls["need_diarizer"] is False  # counts collapsed to 1 → no diarizer load
    entries = json.loads((tmp_path / "transcript.json").read_text())["entries"]
    assert {e["speaker"] for e in entries} == {"Local-1"}


def test_start_settings_diarization_off_skips_the_diarizer(tmp_path, backend_calls):
    write_settings("[speakers]\ndiarization = false\n")
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        ["start", "--remote", "0", "--replay", str(mic), "--out", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert backend_calls["need_diarizer"] is False
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
    # With --diarization, omitting --local estimates the mic count (Stage 3a);
    # the summary shows the detected count and the exact flag to lock or
    # correct it by re-running.
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    result = CliRunner().invoke(
        cli.main,
        [
            "start",
            "--diarization",
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


def test_resolve_flush_interval_defaults():
    # The live checkpoint is zero-inference file I/O → tight default. Explicit
    # values (including 0 = disabled) win.
    from stenograf.flow import RunOptions

    assert RunOptions(live=True).resolved_flush_interval() == 15.0
    assert RunOptions(live=True, flush_interval=45.0).resolved_flush_interval() == 45.0
    assert RunOptions(live=True, flush_interval=0.0).resolved_flush_interval() == 0.0


def test_persist_once_writes_once_and_replays_paths():
    from stenograf.output import PersistOnce

    sentinel = object()
    calls = []
    persist = PersistOnce(lambda t: calls.append(t) or [Path("t.md")])
    assert persist(sentinel) == [Path("t.md")]
    assert persist(sentinel) == [Path("t.md")]  # second call replays, no rewrite
    assert calls == [sentinel]


def test_persist_once_retries_after_a_failed_write():
    from stenograf.output import PersistOnce

    attempts = []

    def flaky(transcript):
        attempts.append(transcript)
        if len(attempts) == 1:
            raise OSError("disk full")
        return [Path("t.md")]

    persist = PersistOnce(flaky)
    with pytest.raises(OSError):
        persist(object())  # the event-time write fails...
    assert persist.paths is None  # ...and is not marked done
    assert persist(object()) == [Path("t.md")]  # the exit-path call retries


def test_capture_log_buffers_chatter_and_surfaces_problems(capsys):
    # The GUI's stderr sink (flow.py wires it into the Qt meeting run): routine
    # transport chatter (formats, stopped) is buffered and never reaches the
    # terminal; problem lines flash in the view the moment they arrive.
    class RecordingView(LiveView):
        def __init__(self):
            self.errors = []

        def error(self, message):
            self.errors.append(message)

    log = loaders.CaptureLog()
    log("stenocap: mic format: 48000.0 Hz, 1 ch")  # arrives before the view exists
    log.view = RecordingView()
    log("stenocap: WARNING channel 0 drifted 40 ms from wall clock")
    log("stenocap: stopped")
    assert log.view.errors == ["stenocap: WARNING channel 0 drifted 40 ms from wall clock"]
    assert capsys.readouterr().err == ""  # nothing touches the terminal while buffering
    assert len(log.lines) == 3  # the full chatter stays available for debugging


def test_plain_flag_stays_an_accepted_no_op(tmp_path, monkeypatch):
    # --plain used to force the line stream instead of the full-screen Textual
    # TUI. The TUI is retired and the stream is the only live terminal mode,
    # but external scripts still pass the flag — it must stay accepted (and
    # change nothing) rather than turn into a usage error.
    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    for name, flags in (("bare", []), ("plain", ["--plain"])):
        result = CliRunner().invoke(
            cli.main,
            [
                "start",
                *flags,
                "--local",
                "1",
                "--remote",
                "0",
                "--replay",
                str(mic),
                "--out",
                str(tmp_path / name),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "You:" in result.output  # the plain caption stream ran

    help_text = CliRunner().invoke(cli.main, ["start", "--help"]).output
    assert "--plain" not in help_text  # accepted, but no longer advertised


def test_start_generates_notes_through_the_finish_tail(tmp_path, monkeypatch):
    # There is exactly one notes path for `steno start`: MeetingRun's shared
    # run_notes tail, after the transcript is safely written — the same tail
    # the Qt app and `steno transcribe` use.
    notes_calls: list = []

    def fake_generate(transcript, out_dir, basename, **kwargs):
        from stenograf.notes import MeetingNotes

        notes_calls.append(basename)
        return [tmp_path / "notes.md"], MeetingNotes(title="T", body="S")

    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
    monkeypatch.setattr("stenograf.notes.run.generate_and_write_notes", fake_generate)
    mic = tmp_path / "mic.wav"
    write_wav(mic)

    run = CliRunner().invoke(
        cli.main,
        ["start", "--notes", "--local", "1", "--remote", "0", "--replay", str(mic)],
    )
    assert run.exit_code == 0, run.output
    assert notes_calls == ["transcript"]
    assert "notes: wrote" in run.output
    assert run.output.index("wrote transcript") < run.output.index("notes: wrote"), (
        "the transcript must be persisted before notes generation starts"
    )


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
