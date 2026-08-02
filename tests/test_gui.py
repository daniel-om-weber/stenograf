"""Phase 8: the Qt desktop app (shell, QML tree, and one controller per screen).

Two kinds of test, because a GUI has two kinds of failure:

- **The QML actually loads.** Every page is built with its real controller and
  the Qt message handler is watched — a typo in a binding is a runtime warning,
  not an exception, so an unwatched app "works" while rendering nothing. This is
  the regression test that matters most and the one no amount of Python
  coverage replaces.
- **The controllers do what the screens promise**, driven exactly as QML drives
  them (``opened()``, ``start()``, ``stop()``) and asserted on ``state``,
  the screens' plain-text mirror.

Everything runs headless (``QT_QPA_PLATFORM=offscreen``); work started on a
worker thread is awaited by pumping the Qt event loop, which is also what
delivers the marshalled replies. Only the tray's close-to-hide test realizes a
window at all, and offscreen means nothing appears on a screen.
"""

import os
import sys
import threading
import time

import pytest

# PySide6 is a base dependency since the default flip — a broken Qt install
# must FAIL this suite, not skip it, so there is deliberately no importorskip.

# Must precede the first QGuiApplication: no display exists in CI, and none is
# needed — nothing here shows a window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlComponent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stenograf.gui.app import MENU, QML_DIR, build  # noqa: E402

PAGES = ("Home", "Setup", "Meeting", "Transcribe", "Notes", "Settings", "Doctor")


@pytest.fixture(scope="session")
def qt_app():
    """The one application object a process may have.

    A ``QApplication``, not the ``QGuiApplication`` a pure Qt Quick app would
    need, because that is what ``run()`` builds: ``QSystemTrayIcon``'s menu is a
    QtWidgets widget, and constructing one under a bare ``QGuiApplication``
    aborts the process on ``qFatal`` rather than raising."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(qt_app, monkeypatch):
    """A freshly built shell + engine — the real object graph, no window shown.

    The QML tree is *live*: asking the shell to navigate really instantiates the
    page, whose ``Component.onCompleted`` really calls ``opened()``. That is the
    point (it exercises the wiring end to end), but it also means a test that
    navigates to the meeting page starts a meeting — so capture is stubbed out
    by default and each test that wants a real run patches it back."""
    from stenograf import loaders

    def no_capture(*args, **kwargs):
        raise RuntimeError("capture is not available in tests")

    monkeypatch.setattr(loaders, "make_provider", no_capture)
    engine, shell = build(qt_app)
    yield shell, engine
    # Join any meeting the test left running *before* monkeypatch unwinds. The
    # worker is a daemon thread and `MeetingRun.__init__` does a first-time
    # import before it reaches capture, so a test that returns without joining
    # leaves a thread that calls `make_provider` after the stub above has been
    # restored — and on Windows, where capture needs no helper and no
    # permission, the real one then says yes and writes a meeting folder into
    # the developer's own ~/Documents/Meetings. pytest finalizes this fixture
    # before `monkeypatch` precisely because it depends on it.
    shell.screen("Meeting").shutdown()


def qt_app_instance():
    application = QApplication.instance()
    assert application is not None
    return application


def pump(until, timeout=30.0):
    """Run the event loop until ``until()`` holds — the queued-call delivery too."""
    app = QGuiApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout
    while not until():
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the GUI")
        time.sleep(0.005)
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


class _Warnings:
    """Collects Qt warnings (QML binding errors are warnings, not exceptions)."""

    def __enter__(self):
        self.seen: list[str] = []
        qInstallMessageHandler(self._handle)
        return self

    def __exit__(self, *exc):
        qInstallMessageHandler(None)

    # Font complaints are about the machine, not about our QML: the offscreen
    # platform has no "Sans Serif" and says so once, and on Windows PySide6
    # ships no fonts directory at all ("QFontDatabase: Cannot find font
    # directory …"). Neither can be fixed from a .qml file, and letting them
    # through means this test only ever passes on the developer's own desktop.
    _ENVIRONMENT_NOISE = ("font family", "font directory")

    def _handle(self, mode, context, message):
        loud = (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg)
        if mode in loud and not any(noise in message for noise in self._ENVIRONMENT_NOISE):
            self.seen.append(message)


class TestQmlTree:
    def test_the_window_and_every_page_build_without_warnings(self, qt_app, monkeypatch):
        from stenograf import doctor

        # Doctor's page runs its checks on open; the real ones probe helpers and
        # spawn processes.
        monkeypatch.setattr(doctor, "run_checks", lambda: [])
        with _Warnings() as log:
            engine, shell = build(qt_app)
            assert engine.rootObjects(), "Main.qml did not load"
            for page in PAGES:
                component = QQmlComponent(engine, str(QML_DIR / f"{page}.qml"))
                assert component.status() == QQmlComponent.Status.Ready, (
                    f"{page}.qml: {component.errorString()}"
                )
                # Exactly what Main.qml hands a pushed page (Home takes no
                # controller — it only reads the menu off the shell).
                properties = {"app": shell}
                if shell.screen(page) is not None:
                    properties["screen"] = shell.screen(page)
                item = component.createWithInitialProperties(properties)
                assert item is not None, f"{page}.qml did not instantiate"
            assert log.seen == []

    def test_the_app_has_an_icon_to_show(self, qt_app):
        # Ships in the wheel next to the .app bundle it was cut from. Without
        # it the Dock and the taskbar fall back to a generic Python tile —
        # which is also what a wheel that dropped the assets/ directory looks
        # like, so this doubles as a packaging check.
        from PySide6.QtGui import QIcon

        from stenograf import ASSETS

        icon = QIcon(str(ASSETS / "icon.png"))
        assert not icon.isNull(), f"{ASSETS / 'icon.png'} is missing or unreadable"

    def test_every_menu_entry_opens_a_real_page(self, gui):
        shell, _engine = gui
        for page, _label, _description in MENU:
            if page == "quit":
                continue
            assert (QML_DIR / f"{page}.qml").is_file(), f"no QML file for menu entry {page}"
            assert shell.screen(page) is not None, f"no controller for menu entry {page}"

    def test_every_controller_page_calls_opened_on_completion(self, gui):
        """A controller's ``opened()`` runs only if its page's QML calls it from
        ``Component.onCompleted`` — the controller tests call it by hand and so
        cannot notice a page that never wires it (Notes.qml shipped without the
        call, leaving the newest-meeting pre-selection dead in the real app).
        The base ``Screen.opened`` is a no-op, so the wiring is uniform: every
        page with a controller makes the call."""
        shell, _engine = gui
        checked = []
        for page in PAGES:
            if shell.screen(page) is None:  # Home takes no controller
                continue
            qml = (QML_DIR / f"{page}.qml").read_text(encoding="utf-8")
            assert "Component.onCompleted" in qml and "screen.opened()" in qml, page
            checked.append(page)
        assert "Notes" in checked  # the page this regression test exists for

    def test_navigation_is_one_signal_in_one_direction(self, gui):
        shell, _engine = gui
        seen = []
        shell.navigation.connect(lambda page, mode: seen.append((page, mode)))
        shell.open("Notes")
        shell.replace("Meeting")
        shell.back()
        assert seen == [("Notes", "push"), ("Meeting", "replace"), ("", "pop")]


class TestSetupScreen:
    def test_start_resolves_a_request_and_replaces_the_form(self, gui, tmp_path, monkeypatch):
        from stenograf import output

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        # This one really starts a meeting (recordAudio below), so it needs the
        # same output redirect as every other test that does — $STENOGRAF_DATA
        # does not cover it, the transcript home is resolved separately.
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        shell, _engine = gui
        setup = shell.screen("Setup")
        seen = []
        shell.navigation.connect(lambda page, mode: seen.append((page, mode)))

        setup.start(
            {
                "mic": True,
                "system": False,
                "diarize": True,
                "local": 2,
                "remote": -1,
                "language": "de",
                "title": "Weekly",
                "recordAudio": True,
                "notes": False,
            }
        )

        assert setup.state["error"] == ""
        # Replace, not push: Back from the meeting belongs on Home.
        assert seen == [("Meeting", "replace")]
        # The meeting page is live by then and has already consumed the request
        # (that is what starts the meeting), so what it says about it is the
        # observable proof it arrived: mic-only, two people in the room, German.
        meeting = shell.screen("Meeting")
        assert meeting.state["profile"] == "local 2"
        assert meeting.state["language"] == "de"

    def test_an_impossible_profile_keeps_the_form_open(self, gui, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        shell, _engine = gui
        seen = []
        shell.navigation.connect(lambda page, mode: seen.append((page, mode)))

        shell.screen("Setup").start(
            {
                "mic": False,  # both sources off — nothing to capture
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "language": "auto",
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )
        assert seen == [], "a form that cannot start a meeting must not navigate"
        error = shell.screen("Setup").state["error"]
        assert error and ("speaker" in error or "source" in error), error

    def test_standing_settings_preset_the_switches(self, gui, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text(
            "[speakers]\ndiarization = true\n\n[notes]\nauto = true\n", encoding="utf-8"
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()  # what the page does on every visit
        assert setup.state["diarize"] is True
        assert setup.state["notes"] is True
        assert setup.state["recordAudio"] is False


class TestMeetingScreen:
    def test_a_whole_meeting_runs_from_the_start_button(self, gui, tmp_path, monkeypatch):
        # The whole GUI path with offline fakes: the setup form resolves a
        # mic-only meeting, the meeting screen runs it through the real recorder
        # (replayed silence), the finalize swap lands on screen and the
        # transcript plus the teed audio are on disk.
        import conftest

        from stenograf import loaders, output
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic)
        announced = {}

        def fake_load_backends(
            *,
            need_diarizer,
            asr_backend=None,
            asr_provider=None,
            glossary=(),
            attendee_names=(),
            boost=None,
            announce=None,
        ):
            announced["load_backends"] = announce
            return conftest.FakeASR(), None, None

        def fake_make_provider(
            replay, plans, *, paced=False, aec=True, aec_dump=None, announce=None, on_log=None
        ):
            announced["make_provider"] = announce
            announced["on_log"] = on_log
            return FileCaptureProvider({Channel.MIC: mic})

        monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
        monkeypatch.setattr(loaders, "make_provider", fake_make_provider)

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        lines = []
        meeting.committed.connect(lambda _channel, who, line: lines.append(f"{who}  {line}"))
        restored = []
        meeting.restored.connect(restored.extend)

        shell.screen("Setup").start(
            {
                "mic": True,
                "system": False,  # mic-only meeting
                "diarize": False,
                "local": -1,
                "remote": -1,
                "language": "auto",
                "title": "",
                "recordAudio": True,  # keep audio.wav
                "notes": False,
            }
        )
        # No opened() call here: navigating instantiates Meeting.qml, whose
        # Component.onCompleted starts the run — the wiring under test.
        pump(lambda: meeting.state["phase"] == "done")
        meeting.join()
        assert restored, "the finalize swap must reach the screen"
        assert meeting.state["folder"].startswith(str(home_dir))

        transcripts = list(home_dir.glob("*/transcript.md"))
        assert len(transcripts) == 1
        assert "wort" in transcripts[0].read_text(encoding="utf-8")
        audio = transcripts[0].parent / "audio.wav"
        assert audio.exists() and audio.stat().st_size > 44
        # Loader progress must reach the view, never click: a GUI has no stdio,
        # and on Windows click.echo dies probing its proxy. `callable(...)` would
        # pass for click.echo too — prove the routing by firing the captured
        # callbacks and requiring the probe on the screen's status line.
        announced["load_backends"]("probe: backends announce")
        pump(lambda: meeting.state["status"] == "probe: backends announce")
        announced["make_provider"]("probe: provider announce")
        pump(lambda: meeting.state["status"] == "probe: provider announce")
        assert isinstance(announced["on_log"], loaders.CaptureLog)

    def test_a_failed_start_lands_on_the_status_line(self, gui, tmp_path, monkeypatch):
        from stenograf import loaders, output

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")

        def no_devices(*args, **kwargs):
            raise RuntimeError("no capture device")

        monkeypatch.setattr(loaders, "make_provider", no_devices)

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(
            {
                "mic": True,
                "system": True,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "language": "auto",
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )
        pump(lambda: meeting.state["phase"] == "failed")
        assert "no capture device" in meeting.state["status"]

    def test_stop_before_capture_cancels_the_run(self, gui, tmp_path, monkeypatch):
        # The gap between Start and set_stop has nothing installed to stop —
        # and provider construction is exactly where a wedged coreaudiod hangs
        # (the capture-conflict failure mode). Stop/Escape there cancels the
        # run: models never load, the provider is released, and the screen
        # pops back home with nothing written.
        import conftest

        from stenograf import loaders, output
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic)

        in_construction = threading.Event()
        release = threading.Event()
        released = []

        class TrackedProvider(FileCaptureProvider):
            def stop(self):
                released.append(True)
                super().stop()

        def slow_make_provider(replay, plans, **kwargs):
            in_construction.set()
            assert release.wait(10), "the test must release the construction"
            return TrackedProvider({Channel.MIC: mic})

        def no_models(**kwargs):
            raise AssertionError("a cancelled run must never load models")

        monkeypatch.setattr(loaders, "make_provider", slow_make_provider)
        monkeypatch.setattr(loaders, "load_backends", no_models)

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        moves = []
        shell.navigation.connect(lambda page, how: moves.append(how))
        shell.screen("Setup").start(
            {
                "mic": True,
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "language": "auto",
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )
        pump(lambda: in_construction.is_set())

        meeting.stop()  # what Escape and the footer button call in that gap
        assert meeting.state["status"] == "cancelling…"
        release.set()
        pump(lambda: "pop" in moves)  # the cancel lands back on Home
        meeting.join()
        assert released == [True]  # the half-built capture was released
        assert not (tmp_path / "meetings").exists()  # nothing was recorded

    def test_captions_become_lines_and_a_tail(self, gui):
        from stenograf.asr.base import Word
        from stenograf.capture.base import Channel

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        lines = []
        meeting.committed.connect(lambda channel, who, line: lines.append((channel, who, line)))

        meeting._commit(Channel.MIC, [Word("guten", 0.0, 0.4), Word("Morgen", 0.4, 0.8)])
        meeting._interim(Channel.MIC, "zusa")
        # Still open: the run may continue, so nothing is in the log yet — it is
        # in the tail, bright, with the provisional text behind it.
        assert lines == []
        assert meeting.state["tails"] == [
            {"channel": "mic", "speaker": "You", "open": "guten Morgen", "tail": "zusa"}
        ]
        # A different channel breaks the line and flushes it.
        meeting._commit(Channel.SYSTEM, [Word("hallo", 2.0, 2.4)])
        assert lines == [("mic", "You", "guten Morgen")]

    def test_stop_ends_capture_off_the_gui_thread(self, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        stopped = []
        meeting.set_stop(lambda: stopped.append(True))
        pump(lambda: meeting.state["canStop"] is True)  # marshalled from the worker

        meeting.stop()
        assert meeting.state["phase"] == "finalizing"  # immediate feedback
        pump(lambda: stopped == [True])

    def test_closing_the_window_mid_meeting_stops_capture_instead_of_hanging(self, gui):
        # The GUI's force-quit is the window's close button, and it can happen
        # mid-capture — where a plain join would wait forever on a meeting
        # nothing will ever stop.
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        stopped = []
        meeting.set_stop(lambda: stopped.append(True))
        pump(lambda: meeting.state["canStop"] is True)

        shell.join_meetings()  # no meeting thread yet: nothing to wait for
        assert stopped == []

        meeting._thread = threading.Thread(target=lambda: time.sleep(0.05))
        meeting._thread.start()
        shell.join_meetings()
        assert stopped == [True], "capture must be ended before waiting on the meeting"

    def test_stop_leaves_once_the_meeting_is_over(self, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        seen = []
        shell.navigation.connect(lambda page, mode: seen.append((page, mode)))
        meeting.set(phase="done")
        meeting.stop()
        assert seen == [("", "pop")]

    def _stub_offline_meeting(self, monkeypatch, tmp_path):
        """Offline meeting fakes: replayed silence in, FakeASR out."""
        import conftest

        from stenograf import loaders, output
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (conftest.FakeASR(), None, None),
        )
        monkeypatch.setattr(
            loaders,
            "make_provider",
            lambda replay, plans, *, paced=True, aec=True, announce=None, on_log=None: (
                FileCaptureProvider({Channel.MIC: mic})
            ),
        )

    @staticmethod
    def _mic_only_form(notes: bool) -> dict:
        return {
            "mic": True,
            "system": False,
            "diarize": False,
            "local": -1,
            "remote": -1,
            "language": "auto",
            "title": "",
            "recordAudio": False,
            "notes": notes,
        }

    def test_closing_the_window_does_not_wait_out_a_notes_run(self, gui, tmp_path, monkeypatch):
        # An agentic [notes] command backend legitimately runs for many
        # minutes, and by the time the notes tail starts the transcript is
        # already on disk — so the window's force-quit must not block on the
        # notes join for up to [notes] timeout_s.
        from stenograf.cli import notes as cli_notes

        self._stub_offline_meeting(monkeypatch, tmp_path)
        notes_started, release = threading.Event(), threading.Event()

        def slow_notes(view, transcript, out_dir, basename, *, created_at, notes_settings):
            notes_started.set()
            release.wait(10)
            return True

        monkeypatch.setattr(cli_notes, "_generate_notes", slow_notes)

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(self._mic_only_form(notes=True))
        pump(notes_started.is_set)

        began = time.monotonic()
        meeting.shutdown()
        assert time.monotonic() - began < 5, "shutdown must not wait out the notes run"
        assert meeting.running, "the notes thread is abandoned (daemon), not joined"
        release.set()
        meeting.join()

    def test_a_quit_before_the_notes_step_skips_it(self, gui, tmp_path, monkeypatch):
        # abandon_notes raised before the notes tail begins: the step is
        # skipped entirely (steno notes --last regenerates), never started
        # into a run whose window is already gone.
        import stenograf.flow as flow_mod
        from stenograf.cli import notes as cli_notes

        self._stub_offline_meeting(monkeypatch, tmp_path)
        calls = []
        monkeypatch.setattr(
            cli_notes, "_generate_notes", lambda *args, **kwargs: calls.append(args) or True
        )

        class AbandonedRun(flow_mod.MeetingRun):
            def __init__(self, request, **kwargs):
                super().__init__(request, **kwargs)
                self.abandon_notes.set()  # the quit arrived before the run

        monkeypatch.setattr(flow_mod, "MeetingRun", AbandonedRun)

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(self._mic_only_form(notes=True))
        pump(lambda: meeting.state["phase"] == "done" and not meeting.running)
        assert calls == [], "an abandoned run must not start its notes step"


class TestTranscribeScreen:
    def test_pick_and_transcribe_writes_a_transcript(self, gui, tmp_path, monkeypatch):
        import conftest

        from stenograf import loaders, output

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (conftest.FakeASR(), None, None),
        )
        audio = tmp_path / "recording.wav"
        conftest.write_wav(audio)

        shell, _engine = gui
        screen = shell.screen("Transcribe")
        screen.choose(audio.as_uri())
        assert screen.state["file"] == str(audio)

        screen.start()
        pump(lambda: not screen.state["busy"] and screen.state["status"])
        assert "wrote transcript.md" in screen.state["status"]
        assert "realtime" in screen.state["status"]
        assert list(home_dir.glob("*/transcript.md"))

    def test_a_failing_run_lands_on_the_status_line(self, gui, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        shell, _engine = gui
        screen = shell.screen("Transcribe")
        screen.choose((tmp_path / "not-audio.wav").as_uri())
        screen.start()  # the file does not exist
        assert screen.state["busy"] is False  # never started


class TestNotesScreen:
    def test_the_newest_meeting_is_preselected_and_generates(self, gui, tmp_path, monkeypatch):
        from test_cli_notes import FakeBackend, write_transcript_json

        import stenograf.notes as notes_pkg
        from stenograf import output

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        older = home_dir / "meeting-20260101-090000"
        newer = home_dir / "meeting-20260102-090000"
        for folder in (older, newer):
            folder.mkdir(parents=True)
            write_transcript_json(folder / "transcript.json")
        monkeypatch.setattr(notes_pkg, "create_backend", lambda name, settings: FakeBackend())

        shell, _engine = gui
        screen = shell.screen("Notes")
        screen.opened()
        assert screen.state["meeting"] == str(newer)

        screen.start()
        pump(lambda: not screen.state["busy"] and "wrote" in screen.state["status"])
        assert (newer / "transcript.notes.md").is_file()

    def test_no_meetings_yet_says_so(self, gui, tmp_path, monkeypatch):
        from stenograf import output

        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "nowhere")
        shell, _engine = gui
        screen = shell.screen("Notes")
        screen.opened()
        assert screen.state["meeting"] == ""
        assert "No finished meeting" in screen.state["status"]


class TestSettingsScreen:
    def test_renders_every_table_with_value_provenance(self, gui, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text('[transcript]\nformats = ["md"]\n', encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))

        shell, _engine = gui
        screen = shell.screen("Settings")
        screen.opened()
        assert screen.state["ok"] is True
        for table in ("[transcript]", "[vocab]", "[output]", "[asr]", "[notes]"):
            assert table in screen.lines
        assert any("settings.toml" in line and "formats" in line for line in screen.lines)

    def test_a_broken_file_renders_its_error(self, gui, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text("not toml [", encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))

        shell, _engine = gui
        screen = shell.screen("Settings")
        screen.opened()
        assert screen.state["ok"] is False
        assert any("settings.toml" in line for line in screen.lines[1:])


class TestDoctorScreen:
    def test_checks_render_with_their_state_and_a_summary(self, gui, monkeypatch):
        from stenograf import doctor
        from stenograf.doctor import Check

        monkeypatch.setattr(
            doctor,
            "run_checks",
            lambda: [
                Check(name="Python", detail="3.12", ok=True),
                Check(name="Notes backend", detail="not configured", ok=False, optional=True),
                Check(name="Capture helper", detail="missing", ok=False),
            ],
        )
        shell, _engine = gui
        screen = shell.screen("Doctor")
        screen.opened()
        pump(lambda: not screen.state["busy"])
        assert [check["state"] for check in screen.state["checks"]] == ["good", "optional", "bad"]
        assert "1 problem(s)" in screen.state["status"]


class TestTray:
    """Menu-bar / system-tray mode (Phase 8 step 6).

    The status item is built directly rather than through ``tray.install``,
    which correctly declines here: the offscreen platform hosts no tray, and
    that path is its own test below."""

    @pytest.fixture
    def tray(self, gui):
        from stenograf.gui.tray import Tray

        shell, _engine = gui
        return Tray(shell)

    def test_no_tray_host_means_no_status_item(self, gui):
        # Stock GNOME without the AppIndicator extension is the case that
        # matters; offscreen stands in for it. A None here is what keeps the
        # window quitting the app on close, so it must stay a supported result.
        from PySide6.QtWidgets import QSystemTrayIcon

        from stenograf.gui import tray as tray_module

        shell, _engine = gui
        assert not QSystemTrayIcon.isSystemTrayAvailable()
        assert tray_module.install(shell) is None

    def test_the_mark_renders_in_every_state(self, qt_app):
        # A missing or unrenderable tray.svg leaves an empty menu bar, which
        # looks exactly like an app that failed to start.
        from stenograf.gui.tray import MARK, _icon

        assert MARK.is_file(), f"{MARK} is missing from the wheel"
        for tint in (None, "#ff5f56"):
            icon = _icon(tint)
            assert not icon.isNull()
            assert not icon.pixmap(22, 22).toImage().allGray(), "the mark rendered blank"

    def test_the_status_icon_is_qt_s_wherever_there_is_no_real_shell(self, qt_app):
        # `_status_icon` gates on the platform *plugin*, not on sys.platform:
        # under offscreen (this suite, CI) the native Windows path would create
        # a real window and register a real tray icon for the test runner.
        # Windows CI runs this too, which is the point of the assertion.
        from PySide6.QtWidgets import QSystemTrayIcon

        from stenograf.gui.tray import _status_icon

        assert QGuiApplication.platformName() == "offscreen"
        assert type(_status_icon(qt_app)) is QSystemTrayIcon

    def test_a_status_icon_the_shell_refuses_falls_back_to_qt_s(self, gui, monkeypatch, capsys):
        # Only the Windows icon can report this: NIM_ADD is refused while
        # Explorer is still starting, which is precisely when a login item runs.
        # Ignoring it would leave `install` returning a Tray, `run` no longer
        # quitting on the last window, and close turning into hide — so in
        # --tray mode the user would have neither a window nor an icon.
        from PySide6.QtWidgets import QSystemTrayIcon

        from stenograf.gui import tray as tray_module

        class Refused(QSystemTrayIcon):
            def show(self) -> bool:
                return False

        shell, _engine = gui
        monkeypatch.setattr(tray_module, "_status_icon", Refused)
        tray = tray_module.Tray(shell)
        assert type(tray.icon) is QSystemTrayIcon, "the refusal was ignored"
        assert not tray.icon.icon().isNull(), "the replacement icon has no artwork"
        assert "falling back" in capsys.readouterr().err

    def test_the_icon_follows_the_meeting_but_not_its_clock(self, tray, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        assert tray.state() == "idle"  # `phase` alone says "rec" before any run

        meeting.set(active=True, phase="rec")
        assert tray.state() == "rec"
        assert "Recording" in tray.summary()

        # The notes tail runs on past phase="done"; the menu bar must still say
        # the meeting is being finished, not that nothing is happening.
        meeting.set(phase="done")
        assert tray.state() == "busy"
        meeting.set(active=False)
        assert tray.state() == "idle"

    def test_the_menu_labels_itself_only_when_it_opens(self, tray, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        meeting.set(active=True, phase="rec", elapsed="12:34")

        tray.menu.aboutToShow.emit()  # what a click on the icon does
        assert tray.status.text() == "Recording · 12:34"
        assert tray.stop.isEnabled()
        assert not tray.start.isEnabled(), "a second meeting cannot be started over this one"

        meeting.set(active=False, phase="done")
        assert tray.status.text() == "Recording · 12:34", "stale until the menu is opened again"
        tray.menu.aboutToShow.emit()
        assert tray.status.text() == "No meeting running"
        assert not tray.stop.isEnabled()
        assert tray.start.isEnabled()

    def test_the_window_can_always_be_opened_from_the_menu(self, tray, gui):
        # This entry used to grey itself out while the window was visible. Qt's
        # isVisible() means *mapped*, not looked at, so Plasma rendered it greyed
        # for a window merely buried behind the video call (measured 2026-07-25)
        # — the shape a meeting normally runs in, and exactly when the entry is
        # wanted. show_window() raises and focuses either way.
        shell, _engine = gui
        tray.menu.aboutToShow.emit()
        assert tray.open_window.isEnabled(), "hidden: the only way back"

        shell.window.show()
        tray.menu.aboutToShow.emit()
        assert tray.open_window.isEnabled(), "visible: possibly behind the call"

    def test_a_finished_meeting_is_announced_to_an_unfocused_window(self, tray, gui, monkeypatch):
        # Same measurement, other consequence: keyed off isVisible() the "Meeting
        # finished" notification would fire only in the rare case (no window at
        # all) and stay silent in the common one (window open, buried).
        from PySide6.QtWidgets import QSystemTrayIcon

        messages = []
        monkeypatch.setattr(QSystemTrayIcon, "supportsMessages", staticmethod(lambda: True))
        monkeypatch.setattr(
            QSystemTrayIcon, "showMessage", lambda self, *args: messages.append(args)
        )
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.window.show()  # visible, but an offscreen window never has focus
        meeting.set(folder="/meetings/meeting-1")

        meeting.set(active=True, phase="rec")
        tray._refresh()
        meeting.set(active=False, phase="done")
        tray._refresh()
        assert [message[0] for message in messages] == ["Meeting finished"]
        assert messages[0][1] == "/meetings/meeting-1", "the folder is the whole message"

    def test_stop_from_the_menu_bar_needs_no_window(self, tray, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        stopped = []
        meeting.set(active=True, phase="rec")
        meeting.set_stop(lambda: stopped.append(True))
        pump(lambda: meeting.state["canStop"] is True)

        tray._stop()
        assert meeting.state["phase"] == "finalizing"
        pump(lambda: stopped == [True])

    def test_start_from_the_menu_bar_unwinds_the_stack(self, tray, gui):
        # The menu bar has no idea what the window was left on, so it may not
        # push a page on top of a copy of itself.
        shell, _engine = gui
        seen = []
        shell.navigation.connect(lambda page, mode: seen.append((page, mode)))
        tray._start()
        assert seen == [("Setup", "root")]

    def test_re_opening_the_app_brings_a_hidden_window_back(self, tray, gui):
        # macOS delivers a double-click on Stenograf.app while it sits in the
        # menu bar as an activation, and AppKit has no window to order front —
        # so the gesture is ours to honour. Measured through the real bundle.
        from PySide6.QtCore import QEvent

        shell, _engine = gui
        assert shell.window is not None

        # The launch activation must NOT count, or --tray puts a window on
        # screen at startup, which is the one thing it exists to avoid.
        tray.eventFilter(qt_app_instance(), QEvent(QEvent.Type.ApplicationActivate))
        assert not shell.window.isVisible()

        tray.eventFilter(qt_app_instance(), QEvent(QEvent.Type.ApplicationActivate))
        assert shell.window.isVisible()
        shell.hide_window()

    def test_closing_the_window_hides_it_and_keeps_the_meeting(self, tray, gui):
        from PySide6.QtGui import QCloseEvent

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        stopped = []
        meeting.set(active=True, phase="rec")
        meeting.set_stop(lambda: stopped.append(True))
        pump(lambda: meeting.state["canStop"] is True)

        shell.show_window()
        assert shell.window is not None and shell.window.isVisible()
        QApplication.sendEvent(shell.window, QCloseEvent())

        assert not shell.window.isVisible()
        assert stopped == [], "closing to the tray must not end the meeting"
        assert meeting.state["phase"] == "rec"


class TestWindowsIdentity:
    """The AppUserModelID, without which the taskbar cannot find its own app.

    Windows' half of what ``setDesktopFileName`` does on Wayland: the shell
    matches a window to the shortcut that launched it by this string, so a
    missing one groups the window under ``pythonw.exe``, leaves a pinned
    shortcut unmatched, and signs our toasts as Python. Qt sets none."""

    def test_claiming_it_is_a_no_op_where_there_is_nothing_to_claim(self):
        from stenograf.gui.app import claim_windows_identity

        claim_windows_identity()  # every platform reaches this line in run()

    @pytest.mark.skipif(sys.platform != "win32", reason="no AppUserModelID exists elsewhere")
    def test_the_process_reports_the_id_the_shortcut_declares(self):
        import ctypes

        from stenograf.gui.app import claim_windows_identity
        from stenograf.shortcut import APP_USER_MODEL_ID

        claim_windows_identity()

        # Read back through the shell rather than trusting the setter: the call
        # returns an HRESULT nobody looks at, and the failure mode is silence.
        reported = ctypes.c_wchar_p()
        ctypes.oledll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(reported))
        try:
            # The same constant the .lnk carries — one string, or no match.
            assert reported.value == APP_USER_MODEL_ID
        finally:
            ctypes.windll.ole32.CoTaskMemFree(reported)


class TestSingleInstance:
    """One app per user, however often the launcher is clicked.

    macOS gets this from LaunchServices; Linux and Windows have nothing like it,
    and closing the window now only hides it — so clicking the launcher again,
    the natural way to ask for the window back, is exactly the gesture that
    would otherwise start a second app: two tray icons, two microphone claims,
    two meeting folders written side by side. ``QLocalServer`` needs no display,
    so the whole path is testable right here."""

    @pytest.fixture
    def name(self, monkeypatch):
        """A claim name of this test's own.

        The real one is per *user*, which is the point of it — and would collide
        with the app running on the developer's own desktop."""
        from PySide6.QtNetwork import QLocalServer

        from stenograf.gui import app as app_module

        unique = f"stenograf-test-{os.getpid()}"
        monkeypatch.setattr(app_module, "_instance_name", lambda: unique)
        yield unique
        QLocalServer.removeServer(unique)

    def test_the_second_launch_hands_over_its_click_and_leaves(self, qt_app, name):
        from stenograf.gui.app import claim_single_instance

        first = claim_single_instance()
        assert first is not None and first.isListening()
        try:
            assert claim_single_instance() is None, "the second launch must not build an app"
            # It said nothing beyond "a launch happened here" — the connection
            # *is* the message, and it has to reach the running app.
            pump(first.hasPendingConnections)
        finally:
            first.close()

    def test_the_running_app_answers_by_showing_its_window(self, qt_app, gui, name):
        from stenograf.gui.app import _relaunched, claim_single_instance

        shell, _engine = gui
        server = claim_single_instance()
        assert server is not None and shell.window is not None
        try:
            shell.hide_window()  # where closing the window leaves it, with a tray up
            assert claim_single_instance() is None
            pump(server.hasPendingConnections)

            _relaunched(server, shell)

            assert shell.window.isVisible()
            # Drained, not merely ignored: unaccepted connections count against
            # maxPendingConnections, and a server that stops accepting refuses
            # the *next* launch — which then starts the second app.
            assert not server.hasPendingConnections()
        finally:
            shell.hide_window()
            server.close()

    @pytest.mark.skipif(sys.platform == "win32", reason="a named pipe leaves no file behind")
    def test_a_crashed_instance_does_not_lock_the_next_one_out(self, qt_app, name):
        from pathlib import Path

        from PySide6.QtNetwork import QLocalServer

        from stenograf.gui.app import claim_single_instance

        scout = QLocalServer()  # only to learn where Qt puts the socket
        assert scout.listen(name)
        socket_path = Path(scout.fullServerName())
        scout.close()
        socket_path.write_bytes(b"")  # what a killed instance leaves behind

        server = claim_single_instance()

        try:
            assert server is not None and server.isListening(), server and server.errorString()
        finally:
            if server is not None:
                server.close()


class TestQuitting:
    def test_quitting_mid_meeting_finishes_it_before_leaving(self, gui):
        # The tray makes this the normal case rather than an accident: with no
        # window in front of it, Quit is how a meeting most often ends.
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        stopped = []
        meeting.set_stop(lambda: stopped.append(True))
        pump(lambda: meeting.state["canStop"] is True)
        meeting.set(active=True, phase="rec")
        meeting._thread = threading.Thread(target=lambda: time.sleep(0.05))
        meeting._thread.start()
        announced = []
        shell.quitting.connect(lambda: announced.append(True))

        shell.quit_app()
        assert shell.quitting_now
        assert announced == [True]  # so the menu bar can explain the wait
        pump(lambda: stopped == [True] and not meeting.running)

        # The meeting is finished, so the after-the-loop fallback has nothing
        # left to wait for.
        shell.join_meetings()
        assert stopped == [True]

    def test_a_second_quit_gives_up_on_the_finalize(self, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        meeting.set(active=True, phase="rec")
        meeting.set_stop(lambda: None)
        pump(lambda: meeting.state["canStop"] is True)
        never_ends = threading.Event()
        meeting._thread = threading.Thread(target=never_ends.wait, daemon=True)
        meeting._thread.start()

        shell.quit_app()
        shell.quit_app()  # impatient: a wait with no way out is the worse bug
        # Would otherwise block forever on the meeting thread.
        shell.join_meetings()
        never_ends.set()


class TestSignals:
    def test_a_sigterm_quits_the_event_loop_cleanly(self, qt_app):
        # Logout is a SIGTERM (the Linux session manager's; the app-bundle
        # stub forwards one on macOS), and the default disposition would kill
        # the process around run()'s finally — the call that stops capture
        # and lands the finalize. This drives the whole chain: signal →
        # wakeup byte → QSocketNotifier → quit, observed via the test's own
        # loop through the `quit` seam.
        import signal

        from PySide6.QtCore import QTimer

        from stenograf.gui.app import _hand_signals_to_qt

        before = {num: signal.getsignal(num) for num in (signal.SIGINT, signal.SIGTERM)}
        loop = QEventLoop()
        outcomes = []

        def quit_from_signal():
            outcomes.append("signal quit")
            loop.quit()

        restore = _hand_signals_to_qt(qt_app, quit=quit_from_signal)
        try:
            assert signal.getsignal(signal.SIGTERM) is not before[signal.SIGTERM]
            QTimer.singleShot(0, lambda: signal.raise_signal(signal.SIGTERM))
            QTimer.singleShot(5000, lambda: (outcomes.append("timed out"), loop.quit()))
            loop.exec()
        finally:
            restore()

        assert outcomes[0] == "signal quit"
        # join_meetings advertises "Ctrl-C abandons it" after exec(): the
        # dispositions must be back to what they were.
        for num, handler in before.items():
            assert signal.getsignal(num) is handler
