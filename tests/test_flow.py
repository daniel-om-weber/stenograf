"""The UI-shared layer: meeting-request resolution and caption segmentation.

Both are library layer shared by every front-end (today: the Qt app), so
they are tested here once, directly, rather than through a screen. The
screens' own tests then only have to prove they call this correctly.
"""

import numpy as np
import pytest
from conftest import write_settings

from stenograf import flow
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

    def test_counts_only_mean_something_while_diarizing(self):
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

    def test_the_switch_survives_a_stated_1_to_1(self):
        # Counts 1/1 with the switch on: the counts alone read as "machinery
        # off", so the profile must carry the switch itself — a 1:1 call still
        # embeds each channel's voice for naming and assignment.
        request = resolve_meeting_request(
            mic=True, system=True, diarize=True, local_speakers=1, remote_speakers=1
        )
        assert request.profile.diarizes
        off = resolve_meeting_request(
            mic=True, system=True, diarize=False, local_speakers=None, remote_speakers=None
        )
        assert not off.profile.diarizes

    def test_a_source_switched_off_is_zero_speakers(self):
        request = resolve_meeting_request(mic=True, system=False, diarize=True)
        assert request.profile.remote_speakers == 0
        assert request.profile.local_speakers is None  # auto by default

    def test_both_sources_off_is_a_reported_error(self):
        with pytest.raises(MeetingRequestError):
            resolve_meeting_request(mic=False, system=False, diarize=False)

    def test_a_broken_settings_file_is_reported_not_raised_raw(self):
        write_settings("not toml [")
        with pytest.raises(MeetingRequestError) as excinfo:
            resolve_meeting_request(mic=True, system=True, diarize=False)
        assert "settings.toml" in str(excinfo.value)

    def test_a_preset_resolves_identically_to_the_cli_overlay(self):
        # The parity the preset layer exists for: the UI path and the CLI must
        # produce the same effective configuration for the same preset.
        from stenograf.config import Language
        from stenograf.settings import apply_meeting_preset, load_settings

        write_settings(
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
"""
        )
        request = resolve_meeting_request(
            mic=True, system=False, diarize=False, preset="controlling"
        )
        expected, _preset = apply_meeting_preset(load_settings(), "controlling")
        assert request.settings.notes == expected.notes  # incl. model=None (pair rule)
        assert request.profile.title == "Controlling-Runde"
        assert request.profile.language == Language.GERMAN
        # Preset vocab merges — it never replaces the standing baseline.
        assert set(request.profile.attendee_names) >= {"Standing Name", "Preset Name"}

    def test_a_typed_title_still_beats_the_preset(self):
        write_settings('[meetings.x]\ntitle = "Preset-Titel"\n')
        request = resolve_meeting_request(
            mic=True, system=False, diarize=False, title="Getippter Titel", preset="x"
        )
        assert request.profile.title == "Getippter Titel"

    def test_an_unknown_preset_is_a_reported_error(self):
        with pytest.raises(MeetingRequestError, match="unknown meeting preset"):
            resolve_meeting_request(mic=True, system=False, diarize=False, preset="nope")

    def test_the_settings_report_names_the_file_and_every_table(self):
        lines, ok = settings_report()
        assert ok is True
        assert lines[0].startswith("settings: ")
        assert "[asr]" in lines

    def test_the_report_attributes_a_presets_keys_to_it(self, tmp_path, monkeypatch):
        # The picker's whole point: which values does *this* meeting type
        # decide? A row the preset set must not read as plain settings.toml.
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text(
            """
[notes]
backend = "command"
instructions = "~/standing.md"

[notes.export]
dir = "~/vault"

[meetings.geheim]
title = "Vertraulich"

[meetings.geheim.notes]
backend = "mlx"

[meetings.geheim.notes.export]
dir = ""
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)

        plain, _ = settings_report()
        lines, ok = settings_report("geheim")
        assert ok is True
        assert any(line.startswith("preset:   [meetings.geheim]") for line in lines), lines

        def row(report, table, key):
            """One ``key`` row out of one ``[table]`` — ``dir`` exists twice."""
            rest = report[report.index(f"[{table}]") + 1 :]
            body = rest[: next(i for i, line in enumerate([*rest, ""]) if not line.strip())]
            return next(line for line in body if line.strip().split(" ")[0] == key)

        # Overlaid and attributed…
        assert "command" in row(plain, "notes", "backend")
        assert "mlx" in row(lines, "notes", "backend")
        assert row(lines, "notes", "backend").endswith("([meetings.geheim])")
        # …a key the preset leaves alone still reads as the file…
        assert row(lines, "notes", "instructions").endswith("(settings.toml)")
        # …and one it switched off with "" says so, rather than looking unset.
        assert row(lines, "notes.export", "dir").endswith("([meetings.geheim] switched it off)")

    def test_an_unknown_preset_fails_the_report_like_a_broken_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        lines, ok = settings_report("nope")
        assert ok is False
        assert any("unknown meeting preset" in line for line in lines), lines


class TestMeetingRunAbort:
    """The pre-capture cancel: between Start and ``set_stop`` nothing is
    installed to stop, so the run itself checks the abort flag around provider
    construction (gui/meeting.py routes Stop/Escape/quit here)."""

    def test_a_cancelled_run_never_builds_capture(self, tmp_path, monkeypatch):
        import threading

        from stenograf import loaders, output
        from stenograf.flow import MeetingRun
        from stenograf.view import LiveView

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

        from stenograf import loaders
        from stenograf.flow import MeetingRun
        from stenograf.view import LiveView

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

    def _configure(self, tmp_path):
        glossary = tmp_path / "glossary.txt"
        glossary.write_text("Kubernetes\n", encoding="utf-8")
        write_settings(
            f'[vocab]\nglossary_file = "{glossary.as_posix()}"\n'
            'attendees = ["Ada Lovelace"]\n\n[asr]\nboost = 2.5\n'
        )

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
        from stenograf import loaders
        from stenograf.flow import MeetingRun
        from stenograf.view import LiveView

        self._configure(tmp_path)

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

        from stenograf.flow import transcribe_recording

        self._configure(tmp_path)
        wav = tmp_path / "meeting.wav"
        write_wav(wav)  # mono → the mixed-stream branch
        seen = self._recording_load_backends(monkeypatch)
        with pytest.raises(RuntimeError):
            transcribe_recording(wav, on_status=lambda message: None)
        self._assert_biased(seen)


class TestTranscribeRecording:
    """``flow.transcribe_recording``: a file in, a written transcript out —
    both the mixed-stream and the split-channel branch, offline. This is the
    button UIs' whole transcribe path, so it gets end-to-end coverage here,
    not just through a screen."""

    def test_a_mono_file_lands_in_a_fresh_folder_with_progress(self, tmp_path, monkeypatch):
        from conftest import fake_load_backends, write_wav

        from stenograf import loaders
        from stenograf.flow import transcribe_recording

        monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
        wav = tmp_path / "meeting.wav"
        write_wav(wav, seconds=2.0)  # mono → the mixed-stream branch
        statuses: list[str] = []
        windows: list[tuple[int, int]] = []

        result = transcribe_recording(
            wav,
            on_status=statuses.append,
            on_windows=lambda done, total: windows.append((done, total)),
        )

        # A fresh dated folder under the (isolated) output home, default formats.
        assert result.out_dir.parent == tmp_path / "meetings-home"
        names = {p.name for p in result.paths}
        assert names == {"transcript.md", "transcript.json", "transcript.txt"}
        assert all(p.exists() for p in result.paths)
        assert result.duration == 2.0
        assert "wirklich eine gute" in (result.out_dir / "transcript.md").read_text(
            encoding="utf-8"
        )
        # Loader progress went to the callback (a worker thread has no stdio)...
        assert "loading models…" in statuses
        # ...and the ASR pass announced every window (done is 0-indexed, called
        # as each window starts) — the progress bar's whole data source.
        assert windows
        total = windows[-1][1]
        assert [done for done, _ in windows] == list(range(total))

    def test_a_split_recording_transcribes_per_channel(self, tmp_path, monkeypatch):
        import json

        from conftest import fake_channel_backends, voice_channel_pcms, write_stereo_wav

        from stenograf import loaders
        from stenograf.flow import transcribe_recording

        monkeypatch.setattr(loaders, "load_backends", fake_channel_backends)
        wav = tmp_path / "meeting.wav"
        write_stereo_wav(wav, *voice_channel_pcms())
        statuses: list[str] = []

        result = transcribe_recording(wav, on_status=statuses.append)

        assert any("2 voice channels" in s for s in statuses)
        entries = json.loads(
            (result.out_dir / "transcript.json").read_text(encoding="utf-8")
        )["entries"]
        # Diarization stays off (nothing enabled it): one speaker per channel,
        # and no cross-channel bleed — each channel decoded its own audio only.
        assert {e["speaker"] for e in entries} == {"Local-1", "Remote-1"}
        for entry in entries:
            stem = "foxtrot" if entry["speaker"] == "Local-1" else "quebec"
            assert stem in entry["text"]


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


class TestAssignSpeaker:
    """The rename-once loop: name a meeting speaker, enroll the voice."""

    MODEL = "eres2net-voxceleb-16k.onnx"

    def _meeting(self, tmp_path, speakers=None):
        from stenograf.config import MeetingProfile
        from stenograf.output import write_transcript
        from stenograf.transcript import Transcript, TranscriptEntry
        from stenograf.voiceprints import write_meeting_voiceprints

        mdir = tmp_path / "meeting-20260801-120000"
        transcript = Transcript(
            language=None,
            profile=MeetingProfile(),
            entries=[
                TranscriptEntry(speaker="Remote-1", text="hallo", start=0.0, end=1.0),
                TranscriptEntry(speaker="Local-1", text="hi", start=1.5, end=2.0),
            ],
        )
        write_transcript(transcript, mdir, "transcript")
        if speakers is None:
            speakers = {"Remote-1": np.array([1.0, 0.0], np.float32)}
        write_meeting_voiceprints(mdir, speakers, self.MODEL, date="2026-08-01")
        return mdir

    def test_assign_enrolls_and_rewrites_the_transcript(self, tmp_path):
        from stenograf.output import load_transcript
        from stenograf.voiceprints import ProfileStore, load_meeting_voiceprints

        mdir = self._meeting(tmp_path)
        store_path = tmp_path / "profiles.json"
        result = flow.assign_speaker(mdir, "Remote-1", "Anna", store_path=store_path)
        assert result.created and result.samples == 1 and result.name == "Anna"

        stored = ProfileStore.load(store_path).get("Anna", self.MODEL)
        assert stored is not None
        assert stored.embeddings[0].date == "2026-08-01"  # the meeting's date

        transcript, _, _ = load_transcript(mdir)
        assert {e.speaker for e in transcript.entries} == {"Anna", "Local-1"}
        assert "Anna" in (mdir / "transcript.md").read_text(encoding="utf-8")
        sidecar = load_meeting_voiceprints(mdir)
        assert set(sidecar.speakers) == {"Anna"}  # follows the transcript

    def test_assign_reinforces_an_existing_profile(self, tmp_path):
        from stenograf.voiceprints import ProfileStore

        mdir = self._meeting(tmp_path)
        store_path = tmp_path / "profiles.json"
        store = ProfileStore(store_path)
        store.enroll("Anna", np.array([0.0, 1.0], np.float32), self.MODEL, date="2026-07-01")
        store.save()
        result = flow.assign_speaker(mdir, "Remote-1", "Anna", store_path=store_path)
        assert not result.created and result.samples == 2

    def test_confirming_an_automatch_reinforces_without_rewrite(self, tmp_path):
        mdir = self._meeting(tmp_path, {"Anna": np.array([1.0, 0.0], np.float32)})
        store_path = tmp_path / "profiles.json"
        result = flow.assign_speaker(mdir, "Anna", "Anna", store_path=store_path)
        assert result.created and result.rewritten == []

    def test_unknown_label_names_the_available_ones(self, tmp_path):
        mdir = self._meeting(tmp_path)
        with pytest.raises(ValueError, match="Remote-1"):
            flow.assign_speaker(mdir, "Remote-9", "Anna", store_path=tmp_path / "p.json")

    def test_a_meeting_without_a_sidecar_says_so(self, tmp_path):
        from stenograf.voiceprints import MEETING_VOICEPRINTS_NAME

        mdir = self._meeting(tmp_path)
        (mdir / MEETING_VOICEPRINTS_NAME).unlink()
        with pytest.raises(ValueError, match="profiles enroll"):
            flow.assign_speaker(mdir, "Remote-1", "Anna", store_path=tmp_path / "p.json")


class TestMicDeviceResolution:
    """Which microphone a run records from: flag, then settings, then the OS."""

    @staticmethod
    def _settings(pinned: str | None):
        from stenograf.settings import CaptureSettings, Settings

        return Settings(capture=CaptureSettings(mic_device=pinned))

    def test_the_standing_pin_applies_when_nothing_is_chosen(self):
        from stenograf.flow import resolve_mic_device

        assert resolve_mic_device(None, self._settings("usb-1")) == "usb-1"
        assert resolve_mic_device("", self._settings("usb-1")) == "usb-1"

    def test_an_explicit_choice_beats_the_standing_pin(self):
        from stenograf.flow import resolve_mic_device

        assert resolve_mic_device("usb-2", self._settings("usb-1")) == "usb-2"

    def test_the_word_default_clears_a_pin_from_either_side(self):
        # settings.toml gets copied between machines, so both the flag and the
        # file must have a one-word way back to the system default.
        from stenograf.flow import resolve_mic_device

        assert resolve_mic_device("default", self._settings("usb-1")) is None
        assert resolve_mic_device(None, self._settings("default")) is None

    def test_nothing_configured_means_the_system_default(self):
        from stenograf.flow import resolve_mic_device

        assert resolve_mic_device(None, self._settings(None)) is None
