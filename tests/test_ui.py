"""Phase 7, Task 1: the launcher shell (StenografApp + HomeScreen).

Same harness as test_tui.py: each test wraps an async body driving Textual's
``run_test`` pilot in ``asyncio.run``. The load-bearing guarantees:

- the minimal-redraw budget covers the launcher (frame cap pinned via the
  shared ``ui._fps`` module, animations off at the app level);
- Home is the default screen and offers every workflow as a clickable button;
- stubbed buttons point at the CLI command that already does the job (and
  mirror the notice on ``notices`` — the plain-text-mirror rule);
- the menu is fully keyboard-drivable: focus starts on the first button and
  the arrow keys walk the buttons (they must NOT be swallowed as scroll keys
  by the menu container), Enter activates — even on a terminal too short to
  show the whole menu;
- quit works by button and by key.
"""

import asyncio

import textual.constants as tconst

from stenograf.ui.app import StenografApp
from stenograf.ui.home import _MENU, _STUB_HINT, HomeScreen


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

    def test_every_stub_button_names_its_cli_command(self):
        async def body():
            app = StenografApp()
            # Tall enough to show the whole menu — pilot.click cannot reach a
            # button scrolled out of view (real small terminals scroll #menu).
            async with app.run_test(size=(80, 40)) as pilot:
                home = app.screen
                for button_id, hint in _STUB_HINT.items():
                    await pilot.click(f"#{button_id}")
                    await pilot.pause()
                    assert hint in home.notices[-1]
                assert len(home.notices) == len(_STUB_HINT)
                assert app.is_running  # stubs never exit the app

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

    def test_arrow_keys_walk_the_buttons_and_enter_activates(self):
        # Deliberately on a short terminal: the lower buttons start scrolled out
        # of view, and arrow-key traversal must still reach them (focus-follow
        # scrolling) — arrows may not be captured as scroll keys by #menu.
        async def body():
            app = StenografApp()
            async with app.run_test(size=(80, 24)) as pilot:
                home = app.screen
                await pilot.pause()
                for entry_id, _, _ in _MENU[1:]:
                    await pilot.press("down")
                    assert app.focused.id == entry_id
                await pilot.press("up")
                assert app.focused.id == _MENU[-2][0]
                await pilot.press("enter")
                await pilot.pause()
                assert _STUB_HINT[_MENU[-2][0]] in home.notices[-1]

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
