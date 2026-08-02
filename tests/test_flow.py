"""The UI-shared layer: meeting-request resolution and caption segmentation.

Both are library layer shared by every front-end (today: the Qt app), so
they are tested here once, directly, rather than through a screen. The
screens' own tests then only have to prove they call this correctly.
"""

import pytest

from stenograf.asr.base import Word
from stenograf.captions import (
    IDLE_FLUSH_S,
    INTERIM_TAIL_CHARS,
    LINE_FLUSH_CHARS,
    CaptionStream,
)
from stenograf.capture.base import Channel
from stenograf.flow import MeetingRequestError, resolve_meeting_request, settings_report


def _words(text, start=0.0, step=0.3):
    return [Word(word, start + i * step, start + i * step + 0.25) for i, word in enumerate(text)]


class TestMeetingRequest:
    """The switches → profile mapping both setup forms rely on."""

    def test_counts_only_mean_something_while_diarizing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        request = resolve_meeting_request(
            mic=True, system=True, diarize=True, local_speakers=3, remote_speakers=None
        )
        assert request.profile.local_speakers == 3
        assert request.profile.remote_speakers is None  # estimate it

        # Without diarization every live source is exactly one speaker — which
        # is what keeps the diarizer model unloaded.
        plain = resolve_meeting_request(
            mic=True, system=True, diarize=False, local_speakers=3, remote_speakers=None
        )
        assert (plain.profile.local_speakers, plain.profile.remote_speakers) == (1, 1)

    def test_a_source_switched_off_is_zero_speakers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        request = resolve_meeting_request(mic=True, system=False, diarize=True)
        assert request.profile.remote_speakers == 0
        assert request.profile.local_speakers is None  # auto by default

    def test_both_sources_off_is_a_reported_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        with pytest.raises(MeetingRequestError):
            resolve_meeting_request(mic=False, system=False, diarize=False)

    def test_a_broken_settings_file_is_reported_not_raised_raw(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text("not toml [", encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        with pytest.raises(MeetingRequestError) as excinfo:
            resolve_meeting_request(mic=True, system=True, diarize=False)
        assert "settings.toml" in str(excinfo.value)

    def test_a_preset_resolves_identically_to_the_cli_overlay(self, tmp_path, monkeypatch):
        # The parity the preset layer exists for: the UI path and the CLI must
        # produce the same effective configuration for the same preset.
        from stenograf.config import Language
        from stenograf.settings import apply_meeting_preset, load_settings

        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text(
            """
[notes]
backend = "command"
model = "claude-opus-4-8"
command = ["claude", "-p"]

[vocab]
attendees = ["Standing Name"]

[meetings.controlling]
title = "Controlling-Runde"
language = "de"

[meetings.controlling.notes]
backend = "mlx"

[meetings.controlling.vocab]
attendees = ["Preset Name"]
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        request = resolve_meeting_request(
            mic=True, system=False, diarize=False, preset="controlling"
        )
        expected, _preset = apply_meeting_preset(load_settings(), "controlling")
        assert request.settings.notes == expected.notes  # incl. model=None (pair rule)
        assert request.profile.title == "Controlling-Runde"
        assert request.profile.language == Language.GERMAN
        # Preset vocab merges — it never replaces the standing baseline.
        assert set(request.profile.attendee_names) >= {"Standing Name", "Preset Name"}

    def test_a_typed_title_still_beats_the_preset(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text(
            '[meetings.x]\ntitle = "Preset-Titel"\n', encoding="utf-8"
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        request = resolve_meeting_request(
            mic=True, system=False, diarize=False, title="Getippter Titel", preset="x"
        )
        assert request.profile.title == "Getippter Titel"

    def test_an_unknown_preset_is_a_reported_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        with pytest.raises(MeetingRequestError, match="unknown meeting preset"):
            resolve_meeting_request(mic=True, system=False, diarize=False, preset="nope")

    def test_the_settings_report_names_the_file_and_every_table(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        lines, ok = settings_report()
        assert ok is True
        assert lines[0].startswith("settings: ")
        assert "[asr]" in lines


class TestMeetingRunAbort:
    """The pre-capture cancel: between Start and ``set_stop`` nothing is
    installed to stop, so the run itself checks the abort flag around provider
    construction (gui/meeting.py routes Stop/Escape/quit here)."""

    def test_a_cancelled_run_never_builds_capture(self, tmp_path, monkeypatch):
        import threading

        from stenograf import loaders, output
        from stenograf.flow import MeetingRun
        from stenograf.view import LiveView

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")

        def no_provider(*args, **kwargs):
            raise AssertionError("a cancelled run must not open capture devices")

        monkeypatch.setattr(loaders, "make_provider", no_provider)
        abort = threading.Event()
        abort.set()  # the cancel landed before the worker got here
        request = resolve_meeting_request(mic=True, system=False, diarize=False)
        run = MeetingRun(request, abort=abort)

        assert run.run(LiveView()) is None
        assert not (tmp_path / "meetings").exists()  # nothing was ever written

    def test_a_cancel_during_construction_releases_the_provider(self, tmp_path, monkeypatch):
        import threading

        from stenograf import loaders, output
        from stenograf.flow import MeetingRun
        from stenograf.view import LiveView

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        abort = threading.Event()
        stopped = []

        class HalfBuiltProvider:
            def stop(self):
                stopped.append(True)

        def make_provider_then_cancel(*args, **kwargs):
            abort.set()  # the user pressed Cancel while this was running
            return HalfBuiltProvider()

        monkeypatch.setattr(loaders, "make_provider", make_provider_then_cancel)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("models must not load")),
        )
        request = resolve_meeting_request(mic=True, system=False, diarize=False)
        run = MeetingRun(request, abort=abort)

        assert run.run(LiveView()) is None
        assert stopped == [True]  # the devices were released on the way out


class TestBiasingReachesTheDecoder:
    """Both GUI entry points hand ``load_backends`` the same decode-time biasing
    the CLI passes. The regression this guards: the Qt paths once omitted
    ``glossary``/``attendee_names``/``boost``, so a clicked Start decoded
    unbiased while a flagless ``steno start`` boosted — invisible drift, because
    the post-hoc glossary layer still corrected the transcript text."""

    def _configure(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        glossary = tmp_path / "glossary.txt"
        glossary.write_text("Kubernetes\n", encoding="utf-8")
        (data / "settings.toml").write_text(
            f'[vocab]\nglossary_file = "{glossary.as_posix()}"\n'
            'attendees = ["Ada Lovelace"]\n\n[asr]\nboost = 2.5\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))

    def _recording_load_backends(self, monkeypatch):
        from stenograf import loaders

        seen = {}

        def record(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("recorded the load call — no models in tests")

        monkeypatch.setattr(loaders, "load_backends", record)
        return seen

    def _assert_biased(self, seen):
        assert "Kubernetes" in seen["glossary"]
        assert "Ada Lovelace" in seen["attendee_names"]
        assert seen["boost"] == 2.5

    def test_the_meeting_run_loads_backends_with_the_settings_biasing(
        self, tmp_path, monkeypatch
    ):
        from stenograf import loaders, output
        from stenograf.flow import MeetingRun
        from stenograf.view import LiveView

        self._configure(tmp_path, monkeypatch)
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")

        class Provider:
            def start(self, channels):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(loaders, "make_provider", lambda *args, **kwargs: Provider())
        seen = self._recording_load_backends(monkeypatch)
        request = resolve_meeting_request(mic=True, system=False, diarize=False)
        with pytest.raises(RuntimeError):
            MeetingRun(request).run(LiveView())
        self._assert_biased(seen)

    def test_transcribe_recording_loads_backends_with_the_settings_biasing(
        self, tmp_path, monkeypatch
    ):
        from conftest import write_wav

        from stenograf import output
        from stenograf.flow import transcribe_recording

        self._configure(tmp_path, monkeypatch)
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        wav = tmp_path / "meeting.wav"
        write_wav(wav)  # mono → the mixed-stream branch
        seen = self._recording_load_backends(monkeypatch)
        with pytest.raises(RuntimeError):
            transcribe_recording(wav, on_status=lambda message: None)
        self._assert_biased(seen)


class TestCaptionStream:
    """When a caption line continues, breaks, flushes — and what stays in flight."""

    def test_a_run_continues_until_the_channel_changes(self):
        lines = []
        stream = CaptionStream(lambda channel, text: lines.append((channel, text)))
        stream.commit(Channel.MIC, _words(["guten", "Morgen"]))
        stream.commit(Channel.MIC, _words(["zusammen"], start=0.6))
        assert lines == []  # still open: the run may continue
        stream.commit(Channel.SYSTEM, _words(["hallo"], start=2.0))
        assert lines == [(Channel.MIC, "guten Morgen zusammen")]

    def test_a_pause_breaks_the_line(self):
        lines = []
        stream = CaptionStream(lambda channel, text: lines.append(text))
        stream.commit(Channel.MIC, _words(["eins"]))
        stream.commit(Channel.MIC, _words(["zwei"], start=9.0))  # a long gap
        assert lines == ["eins"]

    def test_a_window_sized_batch_flushes_at_once(self):
        # The window pass commits ~30 s of speech per batch. Past the size cap it
        # must land in the scrollback immediately, not accumulate in the
        # (bottom-clipping) interim area — the "UI frozen while remote talks" bug.
        lines = []
        stream = CaptionStream(lambda channel, text: lines.append(text))
        stream.commit(Channel.SYSTEM, _words([f"wort{i}" for i in range(60)]))
        assert len(lines) == 1
        assert len(lines[0]) >= LINE_FLUSH_CHARS
        assert stream.open_words == []  # the next window starts a fresh line

    def test_an_idle_line_flushes_on_the_tick(self):
        lines = []
        stream = CaptionStream(lambda channel, text: lines.append(text))
        stream.commit(Channel.SYSTEM, _words(["hallo"]))
        assert stream.flush_if_idle() is False  # the commit is fresh
        stream._last_commit_at -= IDLE_FLUSH_S + 1
        assert stream.flush_if_idle() is True
        assert lines == ["hallo"]

    def test_tails_carry_the_open_line_and_the_provisional_text(self):
        stream = CaptionStream(lambda channel, text: None)
        stream.commit(Channel.MIC, _words(["guten", "Morgen"]))
        stream.interim(Channel.MIC, "zusa")
        stream.interim(Channel.SYSTEM, "hello")
        assert stream.tails() == [
            (Channel.MIC, "guten Morgen", "zusa"),
            (Channel.SYSTEM, "", "hello"),
        ]
        stream.interim(Channel.SYSTEM, "")  # an empty tail clears the row
        assert [row[0] for row in stream.tails()] == [Channel.MIC]

    def test_only_the_freshest_words_of_a_long_line_render_in_the_tail(self):
        stream = CaptionStream(lambda channel, text: None)
        # Just under the flush cap, so the line stays open and long.
        stream.commit(Channel.MIC, _words(["wort"] * 45))
        (_channel, open_text, _tail) = stream.tails()[0]
        assert open_text.startswith("…")
        assert len(open_text) == INTERIM_TAIL_CHARS + 1
        assert open_text.endswith("wort")

    def test_clear_drops_everything_in_flight(self):
        lines = []
        stream = CaptionStream(lambda channel, text: lines.append(text))
        stream.commit(Channel.MIC, _words(["hallo"]))
        stream.interim(Channel.MIC, "wel")
        stream.clear()
        assert stream.tails() == []
        assert lines == []  # the finalize swap replaces it, it is not logged
