"""Phase 7: the launcher (shell, home menu, and one class per workflow screen).

Same harness as test_tui.py: each test wraps an async body driving Textual's
``run_test`` pilot in ``asyncio.run``. The load-bearing guarantees:

- the minimal-redraw budget covers the launcher (frame cap pinned via the
  shared ``ui._fps`` module, animations off at the app level);
- Home is the default screen, offers every workflow as a clickable button,
  and each button pushes its screen (escape returns Home);
- the menu is fully keyboard-drivable: focus starts on the first button and
  the arrow keys walk the buttons (they must NOT be swallowed as scroll keys
  by the menu container), Enter activates — even on a terminal too short to
  show the whole menu;
- quit works by button and by key;
- every workflow screen runs its pipeline off the event loop and mirrors what
  it shows on plain-text attributes (``status_text``/``lines``/``notices``).

Screens are asserted through those mirrors, and file selection is driven
through the ``on_directory_tree_*`` message handlers — walking DirectoryTree's
async node loading with the pilot is flaky, and the handler *is* the screen's
contract with the tree.
"""

import asyncio
from types import SimpleNamespace

import textual.constants as tconst

from stenograf.ui.app import StenografApp
from stenograf.ui.home import _MENU, HomeScreen


def _run(body) -> None:
    asyncio.run(body())


class TestMinimalRedraw:
    def test_frame_cap_and_animations_are_pinned(self):
        # Importing stenograf.ui.app pins TEXTUAL_FPS before textual imports
        # (the same shared module stenograf.ui.meeting uses).
        assert tconst.MAX_FPS == 15

        async def body():
            app = StenografApp()
            async with app.run_test():
                assert app.animation_level == "none"

        _run(body)


class TestHomeScreen:
    def test_home_is_the_default_screen_with_the_full_menu(self):
        async def body():
            app = StenografApp()
            async with app.run_test():
                assert isinstance(app.screen, HomeScreen)
                button_ids = [b.id for b in app.screen.query("Button").results()]
                assert button_ids == [entry_id for entry_id, _, _ in _MENU]

        _run(body)

    def test_every_workflow_button_pushes_its_screen(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        from stenograf import doctor, output
        from stenograf.ui.doctor import DoctorScreen
        from stenograf.ui.notes import NotesScreen
        from stenograf.ui.settings import SettingsScreen
        from stenograf.ui.transcribe import TranscribeScreen

        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        monkeypatch.setattr(  # the real checks probe helpers and spawn processes
            doctor, "run_checks", lambda: [doctor.Check(name="Python", ok=True, detail="ok")]
        )
        expected = {
            "transcribe": TranscribeScreen,
            "notes": NotesScreen,
            "settings": SettingsScreen,
            "doctor": DoctorScreen,
        }

        async def body():
            app = StenografApp()
            # Tall enough to show the whole menu — pilot.click cannot reach a
            # button scrolled out of view (real small terminals scroll #menu).
            async with app.run_test(size=(80, 40)) as pilot:
                for button_id, screen_type in expected.items():
                    await pilot.click(f"#{button_id}")
                    await pilot.pause()
                    assert isinstance(app.screen, screen_type)
                    await pilot.press("escape")
                    await pilot.pause()
                    assert isinstance(app.screen, HomeScreen)
                assert app.is_running

        _run(body)

    def test_quit_button_exits(self):
        async def body():
            app = StenografApp()
            async with app.run_test(size=(80, 40)) as pilot:  # quit is the last button
                await pilot.click("#quit")
                await pilot.pause()
                assert not app.is_running

        _run(body)

    def test_focus_starts_on_the_first_button(self):
        async def body():
            app = StenografApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.focused is not None
                assert app.focused.id == _MENU[0][0]  # not the scroll container

        _run(body)

    def test_arrow_keys_walk_the_buttons_and_enter_activates(self, monkeypatch):
        from stenograf import doctor
        from stenograf.ui.doctor import DoctorScreen

        monkeypatch.setattr(  # _MENU[-2] is the doctor button — keep its checks fast
            doctor, "run_checks", lambda: [doctor.Check(name="Python", ok=True, detail="ok")]
        )

        # Deliberately on a short terminal: the lower buttons start scrolled out
        # of view, and arrow-key traversal must still reach them (focus-follow
        # scrolling) — arrows may not be captured as scroll keys by #menu.
        async def body():
            app = StenografApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                for entry_id, _, _ in _MENU[1:]:
                    await pilot.press("down")
                    assert app.focused.id == entry_id
                await pilot.press("up")
                assert app.focused.id == _MENU[-2][0]
                await pilot.press("enter")
                await pilot.pause()
                assert isinstance(app.screen, DoctorScreen)

        _run(body)

    def test_q_key_exits(self):
        async def body():
            app = StenografApp()
            async with app.run_test() as pilot:
                await pilot.press("q")
                await pilot.pause()
                assert not app.is_running

        _run(body)


class TestMeetingSetupScreen:
    def test_submit_defaults_to_an_auto_profile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        from stenograf.ui.setup import MeetingSetupScreen

        async def body():
            app = StenografApp()
            results = []
            async with app.run_test(size=(80, 40)) as pilot:
                app.push_screen(MeetingSetupScreen(), results.append)
                await pilot.pause()
                await pilot.click("#go")
                await pilot.pause()
                assert len(results) == 1
                request = results[0]
                assert request.profile.local_speakers is None  # auto-detect
                assert request.profile.remote_speakers is None
                assert request.profile.language is None
                assert request.notes is False

        _run(body)

    def test_settings_diarization_off_defaults_the_counts_to_one(self, tmp_path, monkeypatch):
        # [speakers] diarization = false makes 1 (the diarizer-free path) the
        # form's starting point; the Selects stay editable, so Auto-detect or a
        # real count re-enables diarization for this one meeting.
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text("[speakers]\ndiarization = false\n", encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        from stenograf.ui.setup import MeetingSetupScreen

        async def body():
            app = StenografApp()
            results = []
            async with app.run_test(size=(80, 40)) as pilot:
                app.push_screen(MeetingSetupScreen(), results.append)
                await pilot.pause()
                await pilot.click("#go")
                await pilot.pause()
                assert len(results) == 1
                request = results[0]
                assert request.profile.local_speakers == 1  # not auto-estimated
                assert request.profile.remote_speakers == 1

        _run(body)

    def test_impossible_profile_keeps_the_form_open(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        from textual.widgets import Select

        from stenograf.ui.setup import MeetingSetupScreen

        async def body():
            app = StenografApp()
            results = []
            async with app.run_test(size=(80, 40)) as pilot:
                screen = MeetingSetupScreen()
                app.push_screen(screen, results.append)
                await pilot.pause()
                screen.query_one("#local", Select).value = 0
                screen.query_one("#remote", Select).value = 0
                await pilot.click("#go")
                await pilot.pause()
                assert results == []  # not dismissed — the form keeps the input
                assert app.screen is screen
                assert "at least one speaker" in screen.notices[-1]

        _run(body)

    def test_back_cancels(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        from stenograf.ui.setup import MeetingSetupScreen

        async def body():
            app = StenografApp()
            results = []
            async with app.run_test(size=(80, 40)) as pilot:
                app.push_screen(MeetingSetupScreen(), results.append)
                await pilot.pause()
                await pilot.click("#back")
                await pilot.pause()
                assert results == [None]
                assert isinstance(app.screen, HomeScreen)

        _run(body)


class TestMeetingFlow:
    def test_start_meeting_runs_end_to_end_from_home(self, tmp_path, monkeypatch):
        # The whole launcher path with offline fakes: Home → setup form →
        # meeting screen runs a (replayed, silent) meeting through the real
        # recorder → done → q → back Home with the transcript on disk.
        import conftest
        from textual.widgets import Select

        from stenograf import loaders, output
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider
        from stenograf.ui.meeting import MeetingScreen, Phase
        from stenograf.ui.setup import MeetingSetupScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda *, need_diarizer, asr_backend=None, asr_provider=None: (
                conftest.FakeASR(),
                None,
                None,
            ),
        )
        monkeypatch.setattr(
            loaders,
            "make_provider",
            lambda replay, plans, *, paced=False, aec=True, aec_dump=None: FileCaptureProvider(
                {Channel.MIC: mic}
            ),
        )

        async def body():
            app = StenografApp()
            async with app.run_test(size=(80, 40)) as pilot:
                await pilot.click("#start")
                await pilot.pause()
                assert isinstance(app.screen, MeetingSetupScreen)
                app.screen.query_one("#local", Select).value = 1
                app.screen.query_one("#remote", Select).value = 0
                await pilot.click("#go")
                for _ in range(100):  # the dismiss callback pushes the meeting screen
                    await pilot.pause(0.05)
                    if isinstance(app.screen, MeetingScreen):
                        break
                meeting_screen = app.screen
                assert isinstance(meeting_screen, MeetingScreen)
                for _ in range(400):  # fake pipeline: capture+finalize well within this
                    await pilot.pause(0.05)
                    if meeting_screen._phase is Phase.DONE and "saved" in meeting_screen._status:
                        break
                assert meeting_screen._phase is Phase.DONE
                assert meeting_screen.committed_lines  # the finalize swap rendered
                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, HomeScreen)  # dismissed, app still up
                assert app.is_running

        _run(body)
        transcripts = list(home_dir.glob("*/transcript.md"))
        assert len(transcripts) == 1, "the meeting folder should hold one transcript"
        assert "wort" in transcripts[0].read_text(encoding="utf-8")


class TestTranscribeScreen:
    def test_tree_filter_keeps_audio_files_and_directories(self, tmp_path):
        from stenograf.ui.transcribe import _shows_in_picker

        (tmp_path / "sub").mkdir()
        for name in ("rec.wav", "call.M4A", "notes.txt", ".hidden.wav"):
            (tmp_path / name).touch()

        kept = {p.name for p in tmp_path.iterdir() if _shows_in_picker(p)}
        assert kept == {"sub", "rec.wav", "call.M4A"}

    def test_pick_and_transcribe_end_to_end(self, tmp_path, monkeypatch):
        import conftest
        from textual.widgets import Button

        from stenograf import loaders, output
        from stenograf.ui.transcribe import TranscribeScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda *, need_diarizer, asr_backend=None, asr_provider=None: (
                conftest.FakeASR(),
                None,
                None,
            ),
        )
        audio = tmp_path / "rec.wav"
        conftest.write_wav(audio)

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = TranscribeScreen(root=tmp_path)
                app.push_screen(screen)
                await pilot.pause()
                assert screen.query_one("#go", Button).disabled  # nothing picked yet
                screen.on_directory_tree_file_selected(SimpleNamespace(path=audio))
                await pilot.pause()
                assert not screen.query_one("#go", Button).disabled
                await pilot.click("#go")
                for _ in range(400):  # fake pipeline: finishes well within this
                    await pilot.pause(0.05)
                    if screen.status_text.startswith(("wrote", "failed")):
                        break
                assert screen.status_text.startswith("wrote"), screen.status_text
                assert any("Files in" in n for n in screen.notices)
                await pilot.press("escape")
                await pilot.pause()
                assert isinstance(app.screen, HomeScreen)

        _run(body)
        transcripts = list(home_dir.glob("*/transcript.md"))
        assert len(transcripts) == 1
        assert "wort" in transcripts[0].read_text(encoding="utf-8")

    def test_settings_diarization_off_skips_the_diarizer(self, tmp_path, monkeypatch):
        # The launcher has no --diarization flag; [speakers] diarization = false
        # must keep the diarizer model unloaded and label one speaker.
        import conftest

        from stenograf import loaders, output
        from stenograf.ui.transcribe import TranscribeScreen

        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text("[speakers]\ndiarization = false\n", encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        calls = {}

        def recording_load_backends(*, need_diarizer, asr_backend=None, asr_provider=None):
            calls["need_diarizer"] = need_diarizer
            return conftest.FakeASR(), None, None

        monkeypatch.setattr(loaders, "load_backends", recording_load_backends)
        audio = tmp_path / "rec.wav"
        conftest.write_wav(audio)

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = TranscribeScreen(root=tmp_path)
                app.push_screen(screen)
                await pilot.pause()
                screen.on_directory_tree_file_selected(SimpleNamespace(path=audio))
                await pilot.pause()
                await pilot.click("#go")
                for _ in range(400):
                    await pilot.pause(0.05)
                    if screen.status_text.startswith(("wrote", "failed")):
                        break
                assert screen.status_text.startswith("wrote"), screen.status_text

        _run(body)
        assert calls["need_diarizer"] is False

    def test_a_failing_run_lands_on_the_status_line(self, tmp_path, monkeypatch):
        from stenograf import output
        from stenograf.ui.transcribe import TranscribeScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        bogus = tmp_path / "broken.wav"
        bogus.write_bytes(b"not audio at all")

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = TranscribeScreen(root=tmp_path)
                app.push_screen(screen)
                await pilot.pause()
                screen.on_directory_tree_file_selected(SimpleNamespace(path=bogus))
                await pilot.pause()
                await pilot.click("#go")
                for _ in range(400):
                    await pilot.pause(0.05)
                    if screen.status_text.startswith(("wrote", "failed")):
                        break
                assert screen.status_text.startswith("failed"), screen.status_text
                # The screen recovers: the run can be retried or abandoned.
                assert not screen._busy
                await pilot.press("escape")
                await pilot.pause()
                assert isinstance(app.screen, HomeScreen)

        _run(body)


class TestNotesScreen:
    def _fake_notes_backend(self, monkeypatch):
        from test_cli_notes import FakeBackend

        import stenograf.notes as notes_pkg

        backend = FakeBackend()
        monkeypatch.setattr(notes_pkg, "create_backend", lambda name, settings: backend)
        return backend

    def test_last_meeting_is_preselected_and_generates(self, tmp_path, monkeypatch):
        from test_cli_notes import write_transcript_json

        from stenograf import output
        from stenograf.ui.notes import NotesScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        home_dir = tmp_path / "meetings-home"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        older = home_dir / "meeting-20260701-100000"
        newest = home_dir / "meeting-20260710-143000"
        for meeting in (older, newest):
            meeting.mkdir(parents=True)
            write_transcript_json(meeting / "transcript.json")
        self._fake_notes_backend(monkeypatch)

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = NotesScreen()
                app.push_screen(screen)
                await pilot.pause()
                assert screen._target == newest  # the --last semantics
                await pilot.click("#go")
                for _ in range(400):
                    await pilot.pause(0.05)
                    if screen.status_text.startswith(("wrote", "failed")):
                        break
                assert screen.status_text.startswith("wrote"), screen.status_text

        _run(body)
        assert (newest / "transcript.notes.md").exists()
        assert not (older / "transcript.notes.md").exists()

    def test_picking_a_folder_overrides_the_default(self, tmp_path, monkeypatch):
        from test_cli_notes import write_transcript_json

        from stenograf import output
        from stenograf.ui.notes import NotesScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        home_dir = tmp_path / "meetings-home"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        older = home_dir / "meeting-20260701-100000"
        newest = home_dir / "meeting-20260710-143000"
        for meeting in (older, newest):
            meeting.mkdir(parents=True)
            write_transcript_json(meeting / "transcript.json")
        self._fake_notes_backend(monkeypatch)

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = NotesScreen()
                app.push_screen(screen)
                await pilot.pause()
                screen.on_directory_tree_directory_selected(SimpleNamespace(path=older))
                await pilot.pause()
                await pilot.click("#go")
                for _ in range(400):
                    await pilot.pause(0.05)
                    if screen.status_text.startswith(("wrote", "failed")):
                        break
                assert screen.status_text.startswith("wrote"), screen.status_text

        _run(body)
        assert (older / "transcript.notes.md").exists()

    def test_no_meetings_yet_disables_generate(self, tmp_path, monkeypatch):
        from textual.widgets import Button

        from stenograf import output
        from stenograf.ui.notes import NotesScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "nowhere")

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = NotesScreen()
                app.push_screen(screen)
                await pilot.pause()
                assert screen._target is None
                assert screen.query_one("#go", Button).disabled

        _run(body)


class TestSettingsScreen:
    def test_renders_every_table_with_value_provenance(self, tmp_path, monkeypatch):
        from stenograf.ui.settings import SettingsScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "settings.toml").write_text(
            '[transcript]\nformats = ["md"]\n', encoding="utf-8"
        )

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 50)) as pilot:
                screen = SettingsScreen()
                app.push_screen(screen)
                await pilot.pause()
                assert screen.lines[0].startswith("settings: ")
                for table in ("[transcript]", "[vocab]", "[output]", "[asr]", "[notes]"):
                    assert table in screen.lines
                formats_row = next(line for line in screen.lines if "formats" in line)
                assert "settings.toml" in formats_row  # the file's value, sourced
                assert any("(default)" in line for line in screen.lines)
                await pilot.press("escape")
                await pilot.pause()
                assert isinstance(app.screen, HomeScreen)

        _run(body)

    def test_a_broken_file_renders_its_error_instead(self, tmp_path, monkeypatch):
        from stenograf.ui.settings import SettingsScreen

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "settings.toml").write_text("not toml [", encoding="utf-8")

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 50)) as pilot:
                screen = SettingsScreen()
                app.push_screen(screen)
                await pilot.pause()
                assert any("settings.toml" in line for line in screen.lines[1:])
                assert screen.lines[-1] == "Press Edit to fix the file."

        _run(body)


class TestDoctorScreen:
    def test_report_renders_all_checks_and_a_summary(self, monkeypatch):
        from stenograf import doctor
        from stenograf.ui.doctor import DoctorScreen

        checks = [
            doctor.Check(name="Python", ok=True, detail="3.12.1"),
            doctor.Check(name="Notes backend", ok=False, detail="none installed", optional=True),
            doctor.Check(name="ffmpeg", ok=False, detail="not found"),
        ]
        monkeypatch.setattr(doctor, "run_checks", lambda: checks)

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = DoctorScreen()
                app.push_screen(screen)
                for _ in range(100):
                    await pilot.pause(0.05)
                    if len(screen.lines) == len(checks) + 1:  # report + summary
                        break
                assert screen.lines[:3] == [
                    "✓ Python: 3.12.1",
                    "○ Notes backend: none installed",
                    "✗ ffmpeg: not found",
                ]
                assert screen.lines[-1] == "1 problem(s) found — fix and reopen this screen."
                await pilot.press("escape")
                await pilot.pause()
                assert isinstance(app.screen, HomeScreen)

        _run(body)

    def test_a_healthy_machine_reads_as_such(self, monkeypatch):
        from stenograf import doctor
        from stenograf.ui.doctor import DoctorScreen

        monkeypatch.setattr(
            doctor,
            "run_checks",
            lambda: [doctor.Check(name="Python", ok=True, detail="3.12.1")],
        )

        async def body():
            app = StenografApp()
            async with app.run_test(size=(90, 40)) as pilot:
                screen = DoctorScreen()
                app.push_screen(screen)
                for _ in range(100):
                    await pilot.pause(0.05)
                    if len(screen.lines) == 2:
                        break
                assert screen.lines[-1] == "Everything looks good."

        _run(body)
