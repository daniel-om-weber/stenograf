"""Phase 8: the Qt desktop app (shell, QML tree, and one controller per screen).

Two kinds of test, because a GUI has two kinds of failure:

- **The QML actually loads.** Every page is built with its real controller and
  the Qt message handler is watched — a typo in a binding is a runtime warning,
  not an exception, so an unwatched app "works" while rendering nothing. This is
  the regression test that matters most and the one no amount of Python
  coverage replaces.
- **The controllers do what the screens promise**, driven exactly as QML drives
  them (``opened()``, ``start()``, ``stop()``) and asserted on ``state`` — the
  same plain-text-mirror rule the Textual screens follow.

Everything runs headless (``QT_QPA_PLATFORM=offscreen``) with no window ever
shown; work started on a worker thread is awaited by pumping the Qt event loop,
which is also what delivers the marshalled replies.
"""

import os
import threading
import time

import pytest

pytest.importorskip("PySide6", reason="the desktop app is the optional [gui] extra")

# Must precede the first QGuiApplication: no display exists in CI, and none is
# needed — nothing here shows a window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlComponent  # noqa: E402

from stenograf.gui.app import MENU, QML_DIR, build  # noqa: E402

PAGES = ("Home", "Setup", "Meeting", "Transcribe", "Notes", "Settings", "Doctor")


@pytest.fixture(scope="session")
def qt_app():
    """The one QGuiApplication a process may have."""
    return QGuiApplication.instance() or QGuiApplication([])


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

    def _handle(self, mode, context, message):
        loud = (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg)
        # The offscreen platform has no "Sans Serif" and says so once.
        if mode in loud and "font family" not in message:
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
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
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
            *, need_diarizer, asr_backend=None, asr_provider=None, announce=None
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
        meeting.committed.connect(lambda who, line: lines.append(f"{who}  {line}"))
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
        # and on Windows click.echo dies probing its proxy.
        assert callable(announced["load_backends"])
        assert callable(announced["make_provider"])
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

    def test_captions_become_lines_and_a_tail(self, gui):
        from stenograf.asr.base import Word
        from stenograf.capture.base import Channel

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        lines = []
        meeting.committed.connect(lambda who, line: lines.append((who, line)))

        meeting._commit(Channel.MIC, [Word("guten", 0.0, 0.4), Word("Morgen", 0.4, 0.8)])
        meeting._interim(Channel.MIC, "zusa")
        # Still open: the run may continue, so nothing is in the log yet — it is
        # in the tail, bright, with the provisional text behind it.
        assert lines == []
        assert meeting.state["tails"] == [
            {"speaker": "You", "open": "guten Morgen", "tail": "zusa"}
        ]
        # A different channel breaks the line and flushes it.
        meeting._commit(Channel.SYSTEM, [Word("hallo", 2.0, 2.4)])
        assert lines == [("You", "guten Morgen")]

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
        screen.choose(f"file://{audio}")
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
        screen.choose(f"file://{tmp_path / 'not-audio.wav'}")
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
