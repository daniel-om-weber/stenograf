"""The Qt desktop app (shell, QML tree, and one controller per screen).

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

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# PySide6 is a base dependency since the default flip — a broken Qt install
# must FAIL this suite, not skip it, so there is deliberately no importorskip.

# Must precede the first QGuiApplication: no display exists in CI, and none is
# needed — nothing here shows a window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import write_settings  # noqa: E402
from PySide6.QtCore import QEventLoop, QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlComponent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stenograf.capture.helper import InputDevice  # noqa: E402
from stenograf.gui.app import MENU, QML_DIR, build  # noqa: E402

PAGES = ("Home", "Setup", "Meeting", "Transcribe", "Notes", "Settings", "Doctor")

FAKE_INPUTS = [
    InputDevice(id="builtin", name="Built-in Microphone", is_default=True),
    InputDevice(id="usb-1", name="Desk Microphone", is_default=False),
]
"""What the stubbed helper reports to the setup form's picker."""


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
    from stenograf.capture import helper as capture_helper

    def no_capture(*args, **kwargs):
        raise RuntimeError("capture is not available in tests")

    monkeypatch.setattr(loaders, "make_provider", no_capture)
    # The setup form lists microphones on every visit, and nearly every test
    # here builds the shell — without this each of them would spawn the real
    # capture helper two or three times on a daemon thread. The screen imports
    # the function per call, so the seam is the helper module's attribute.
    monkeypatch.setattr(capture_helper, "list_input_devices", lambda: list(FAKE_INPUTS))
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


# The budget stays well clear of the per-test cap (30 s, pyproject.toml): the
# cap ends the whole session with os._exit(1), so a GUI wait that reached it
# first would cost the run every other test's result instead of raising the
# assertion below, which says what the loop was waiting for.
def pump(until, timeout=10.0):
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


def through_the_card(meeting):
    """Close the finish card the way its timeout does: the run waits for it
    after every meeting, so a test that waits for the run must answer it."""
    pump(lambda: meeting.state["namesOpen"])
    meeting.skip_names()


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

    def test_every_icon_a_page_names_is_drawn(self):
        # An unknown name draws nothing and warns nothing, so a typo would ship
        # a blank button. Every name an icon line can pick must be a known path.
        import re

        icon = (QML_DIR / "Icon.qml").read_text(encoding="utf-8")
        known = set(re.findall(r'^\s*"([a-z-]+)": "M', icon, re.MULTILINE))
        assert {"mic", "mic-off", "volume", "volume-off", "stop"} <= known
        used = set()
        for page in QML_DIR.glob("*.qml"):
            for line in page.read_text(encoding="utf-8").splitlines():
                if re.match(r"\s*(glyph|name):", line) and page.name != "Icon.qml":
                    line = re.sub(r'[!=]==\s*"[^"]*"', "", line)  # conditions, not names
                    used.update(word for word in re.findall(r'"([^"]*)"', line) if word)
        used.update(icon for _page, _label, _description, icon in MENU)  # named in Python
        assert used and used <= known, used - known

    def test_a_failed_load_carries_qts_reason(self, qt_app, monkeypatch, tmp_path):
        # Qt writes the one actionable line — a missing plugin DLL, a bad
        # import — to a stderr that an icon launch and `pythonw` do not have.
        # Users then report "the interface failed to load" and nothing else,
        # which names no cause and suggests no fix.
        broken = tmp_path / "qml"
        broken.mkdir()
        (broken / "Main.qml").write_text("import QtQuick\nNoSuchTypeAnywhere {}\n")
        monkeypatch.setattr("stenograf.gui.app.QML_DIR", broken)

        # The handler is here to swallow the failure Qt prints on the way out,
        # which is expected and would otherwise be read as a broken suite.
        with _Warnings(), pytest.raises(RuntimeError) as failure:
            build(qt_app)

        assert "NoSuchTypeAnywhere" in str(failure.value)

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
        for page, _label, _description, _icon in MENU:
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
        # observable proof it arrived: two people in the room, a language left
        # to detection, and system audio recorded but muted — the switch said
        # "not live at start".
        meeting = shell.screen("Meeting")
        assert meeting.state["profile"].startswith("local 2")
        assert meeting.state["language"] == ""
        assert meeting.state["channels"] == ["mic", "system"]
        assert (meeting.state["micMuted"], meeting.state["systemMuted"]) == (False, True)

    def test_the_microphone_picker_lists_devices_with_the_default_first(self, gui):
        shell, _engine = gui
        setup = shell.screen("Setup")
        pump(lambda: setup.state["micDevicesReady"])

        assert setup.state["micError"] == ""
        assert setup.state["micDevices"] == [
            {"label": "System default (Built-in Microphone)", "value": "default"},
            {"label": "Built-in Microphone (default)", "value": "builtin"},
            {"label": "Desk Microphone", "value": "usb-1"},
        ]

    def test_a_configured_microphone_is_the_first_entry(self, gui, tmp_path, monkeypatch):
        # Index 0 is what a combo starts on, so the standing choice must lead
        # the list — nothing writes currentIndex.
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text('[capture]\nmic_device = "usb-1"\n', encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))

        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()
        pump(lambda: setup.state["micDevicesReady"])

        assert setup.state["micDevices"][0] == {"label": "Desk Microphone", "value": "usb-1"}

    def test_a_configured_microphone_that_is_gone_says_so_first(self, gui, tmp_path, monkeypatch):
        # The form must never quietly show "System default" for a run that
        # would refuse to start.
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text(
            '[capture]\nmic_device = "studio-mic"\n', encoding="utf-8"
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))

        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()
        pump(lambda: setup.state["micDevicesReady"])

        assert setup.state["micDevices"][0] == {
            "label": "studio-mic — not connected",
            "value": "studio-mic",
        }

    def test_a_failed_listing_shows_a_hint_and_leaves_start_working(
        self, gui, tmp_path, monkeypatch
    ):
        from stenograf import output
        from stenograf.capture import helper as capture_helper
        from stenograf.capture.base import CaptureUnavailableError

        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")

        def boom():
            raise CaptureUnavailableError("the capture helper could not be run")

        monkeypatch.setattr(capture_helper, "list_input_devices", boom)
        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()
        pump(lambda: setup.state["micError"] != "")

        assert not setup.state["micDevicesReady"]  # the picker stays hidden
        assert "could not be run" in setup.state["micError"]

        setup.start(
            {
                "mic": True,
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )
        assert setup.state["error"] == ""  # a listing failure never blocks Start

    def test_picking_the_system_default_overrides_a_standing_choice(
        self, gui, tmp_path, monkeypatch
    ):
        # The form's own "System default" row must MEAN it. Sending the empty
        # string would fall through to [capture] mic_device and record the desk
        # microphone the user just chose against — silently.
        from stenograf import output

        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text('[capture]\nmic_device = "usb-1"\n', encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")

        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()
        pump(lambda: setup.state["micDevicesReady"])
        chosen = next(row for row in setup.state["micDevices"] if "System default" in row["label"])

        seen = {}
        monkeypatch.setattr(
            shell.screen("Meeting"), "begin", lambda request: seen.update(request=request)
        )
        setup.start({**self._form(), "micDevice": chosen["value"]})
        assert seen["request"].mic_device is None

    def test_a_form_with_no_picker_keeps_the_standing_choice(self, gui, tmp_path, monkeypatch):
        # The other half of the same rule: "" is what a form sends when the
        # listing failed and no picker was shown, and that must not silently
        # switch the meeting to the system default.
        from stenograf import output

        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text('[capture]\nmic_device = "usb-1"\n', encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")

        shell, _engine = gui
        seen = {}
        monkeypatch.setattr(
            shell.screen("Meeting"), "begin", lambda request: seen.update(request=request)
        )
        shell.screen("Setup").start({**self._form(), "micDevice": ""})
        assert seen["request"].mic_device == "usb-1"

    @staticmethod
    def _form() -> dict:
        return {
            "mic": True,
            "system": False,
            "diarize": False,
            "local": -1,
            "remote": -1,
            "title": "",
            "recordAudio": False,
            "notes": False,
        }

    def test_the_chosen_microphone_reaches_the_request(self, gui, tmp_path, monkeypatch):
        from stenograf import output

        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        shell, _engine = gui
        setup = shell.screen("Setup")
        seen = {}
        monkeypatch.setattr(
            shell.screen("Meeting"), "begin", lambda request: seen.update(request=request)
        )

        setup.start(
            {
                "micDevice": "usb-1",
                "mic": True,
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )

        assert seen["request"].mic_device == "usb-1"

    def test_the_meeting_type_picker_offers_the_presets_and_applies_one(
        self, gui, tmp_path, monkeypatch
    ):
        from stenograf import output

        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text(
            """
[meetings.controlling]
title = "Controlling-Runde"
language = "de"
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))
        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")

        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()
        # "" first (no preset), then the sections, each with the line `steno
        # presets` prints.
        assert setup.state["presets"] == [
            {"label": "None", "value": "", "hint": ""},
            {
                "label": "controlling",
                "value": "controlling",
                "hint": 'title "Controlling-Runde", de',
            },
        ]

        setup.start(
            {
                "preset": "controlling",
                "mic": True,
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "title": "",  # the preset's title fills the gap
                "recordAudio": False,
                "notes": False,
            }
        )

        # The form has no language field, so a German meeting on the live page
        # is the preset having reached the library through the picker.
        assert setup.state["error"] == ""
        assert shell.screen("Meeting").state["language"] == "de"

    def test_a_preset_the_file_no_longer_defines_keeps_the_form_open(
        self, gui, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "data"))
        shell, _engine = gui
        seen = []
        shell.navigation.connect(lambda page, mode: seen.append((page, mode)))

        shell.screen("Setup").start(
            {
                "preset": "gone",
                "mic": True,
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )
        assert seen == []
        assert "unknown meeting preset" in shell.screen("Setup").state["error"]

    def test_both_sources_off_starts_a_meeting_with_both_muted(
        self, gui, tmp_path, monkeypatch
    ):
        # "Off" means "not live at the start": both are still recorded, so the
        # meeting can switch either on — no longer a form that cannot start.
        from stenograf import output

        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        shell, _engine = gui
        seen = []
        shell.navigation.connect(lambda page, mode: seen.append((page, mode)))
        shell.screen("Setup").start(
            {
                "mic": False,
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )
        assert seen == [("Meeting", "replace")]
        meeting = shell.screen("Meeting")
        assert (meeting.state["micMuted"], meeting.state["systemMuted"]) == (True, True)

    def test_standing_settings_preset_the_switches(self, gui, tmp_path, monkeypatch):
        write_settings("[speakers]\ndiarization = true\n\n[notes]\nauto = true\n")
        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()  # what the page does on every visit
        assert setup.state["diarize"] is True
        assert setup.state["notes"] is True
        assert setup.state["recordAudio"] is False

    def test_speakers_are_separated_unless_the_settings_say_not(self, gui):
        shell, _engine = gui
        setup = shell.screen("Setup")
        setup.opened()
        assert setup.state["diarize"] is True, "on with no settings file"
        write_settings("[speakers]\ndiarization = false\n")
        setup.opened()
        assert setup.state["diarize"] is False


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

        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic)
        announced = {}

        def fake_load_backends(
            *,
            need_diarizer,
            asr_backend=None,
            asr_ep=None,
            glossary=(),
            attendee_names=(),
            boost=None,
            announce=None,
        ):
            announced["load_backends"] = announce
            return conftest.FakeASR(), conftest.WholeBufferVAD(), None

        def fake_make_provider(replay, plans, *, announce=None, on_log=None, **kwargs):
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
                "title": "",
                "recordAudio": True,  # keep audio.opus
                "notes": False,
            }
        )
        # No opened() call here: navigating instantiates Meeting.qml, whose
        # Component.onCompleted starts the run — the wiring under test.
        pump(lambda: meeting.state["phase"] == "done")
        through_the_card(meeting)
        meeting.join()
        assert restored, "the finalize swap must reach the screen"
        assert meeting.state["folder"].startswith(str(home_dir))

        transcripts = list(home_dir.glob("*/transcript.md"))
        assert len(transcripts) == 1
        assert "wort" in transcripts[0].read_text(encoding="utf-8")
        audio = transcripts[0].parent / "audio.opus"
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

    def test_muting_the_mic_reaches_the_transcript(self, gui, tmp_path, monkeypatch):
        # The header toggle → MeetingRun.mute → silence at the intake → the span
        # on the finished transcript, through the real recorder.
        import conftest

        from stenograf import loaders, output
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider

        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic, seconds=3.0)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (conftest.FakeASR(), conftest.WholeBufferVAD(), None),
        )
        monkeypatch.setattr(
            loaders,
            "make_provider",
            lambda replay, plans, **kwargs: FileCaptureProvider({Channel.MIC: mic}, paced=True),
        )
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(
            {
                "mic": True,
                "system": False,
                "diarize": False,
                "local": -1,
                "remote": -1,
                "title": "",
                "recordAudio": False,
                "notes": False,
            }
        )
        pump(lambda: meeting.state["canStop"])
        assert meeting.state["channels"] == ["mic", "system"]
        assert meeting.state["systemMuted"] is True  # switched off in the form
        meeting.mute("mic", True)
        assert meeting.state["micMuted"] is True
        time.sleep(0.3)
        meeting.mute("mic", False)
        assert meeting.state["micMuted"] is False
        meeting.stop()
        pump(lambda: meeting.state["phase"] == "done", timeout=20)
        through_the_card(meeting)
        meeting.join()

        folder = Path(meeting.state["folder"])
        saved = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
        spans = {span["channel"]: span for span in saved["muted"]}
        assert spans["mic"]["end"] - spans["mic"]["start"] >= 0.25
        assert spans["system"]["start"] == 0.0, "muted from the first sample"
        assert "*(microphone muted" in (folder / "transcript.md").read_text(encoding="utf-8")
        meeting.mute("mic", True)  # after the recording: nothing left to mute
        assert meeting.state["micMuted"] is False

    def test_a_failed_start_lands_on_the_status_line(self, gui, tmp_path, monkeypatch):
        from stenograf import loaders, output

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

        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (conftest.FakeASR(), conftest.WholeBufferVAD(), None),
        )
        monkeypatch.setattr(
            loaders,
            "make_provider",
            lambda replay, plans, **kwargs: FileCaptureProvider({Channel.MIC: mic}),
        )

    @staticmethod
    def _mic_only_form(notes: bool) -> dict:
        return {
            "mic": True,
            "system": False,
            "diarize": False,
            "local": -1,
            "remote": -1,
            "title": "",
            "recordAudio": False,
            "notes": notes,
        }

    def test_closing_the_window_does_not_wait_out_a_notes_run(self, gui, tmp_path, monkeypatch):
        # An agentic [notes] command backend legitimately runs for many
        # minutes, and by the time the notes tail starts the transcript is
        # already on disk — so the window's force-quit must not block on the
        # notes join for up to [notes] timeout_s.
        from stenograf.notes import run as notes_run

        self._stub_offline_meeting(monkeypatch, tmp_path)
        notes_started, release = threading.Event(), threading.Event()

        def slow_notes(view, transcript, out_dir, basename, **kwargs):
            notes_started.set()
            release.wait(10)
            return True

        monkeypatch.setattr(notes_run, "run_notes", slow_notes)

        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(self._mic_only_form(notes=True))
        through_the_card(meeting)
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
        from stenograf.notes import run as notes_run

        self._stub_offline_meeting(monkeypatch, tmp_path)
        calls = []
        monkeypatch.setattr(
            notes_run, "run_notes", lambda *args, **kwargs: calls.append(args) or True
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


class TestWhoSpoke:
    """The done screen's "who spoke?" panel, and the notes that wait for it."""

    @staticmethod
    def _diarized_meeting(monkeypatch, tmp_path, notes_calls):
        """A mic-only diarized meeting (one voice, Local-1) whose notes step
        records the transcript it was handed instead of running a model."""
        import conftest

        from stenograf import loaders, output
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider
        from stenograf.diarization.base import SpeakerTurn
        from stenograf.notes import run as notes_run

        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
        mic = tmp_path / "mic.wav"
        conftest.write_wav(mic)
        diarizer = conftest.EmbeddingDiarizer(
            [SpeakerTurn("S0", 0.0, 1e9)], {"S0": np.array([1.0, 0.0], np.float32)}
        )
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (
                conftest.FakeASR(),
                conftest.WholeBufferVAD(),
                diarizer if kwargs["need_diarizer"] else None,  # as the real loader
            ),
        )
        monkeypatch.setattr(
            loaders,
            "make_provider",
            lambda replay, plans, **kwargs: FileCaptureProvider({Channel.MIC: mic}),
        )

        def record_notes(view, transcript, out_dir, basename, **kwargs):
            notes_calls.append({e.speaker for e in transcript.entries})
            return True

        monkeypatch.setattr(notes_run, "run_notes", record_notes)

    @staticmethod
    def _done(meeting, names=None, *, title="", notes=True, preset="", context=""):
        """The finish card's Done button."""
        meeting.finish(names or {}, [], title, notes, preset, context)

    @staticmethod
    def _start(shell, notes=True):
        shell.screen("Setup").start(
            {
                "mic": True,
                "system": False,
                "diarize": True,
                "local": -1,
                "remote": -1,
                "title": "",
                "recordAudio": False,
                "notes": notes,
            }
        )

    def test_notes_wait_for_the_names_and_carry_them(self, gui, tmp_path, monkeypatch):
        from stenograf.voiceprints import ProfileStore

        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        with _Warnings() as log:
            self._start(shell)
            pump(lambda: meeting.state["namesOpen"])
            (row,) = meeting.state["speakers"]
            assert row["label"] == "Local-1" and not row["recognised"]
            assert row["quotes"] == [{"text": "“wort”", "line": 0}]
            assert row["clips"] == 1  # one line, one voice sample
            time.sleep(0.5)
            assert notes == [], "notes must wait for the names"
            assert "press Done" in meeting.state["status"]

            self._done(meeting, {"Local-1": "Anna"})
            # The card closes when the names job returns, which may be after
            # the notes it released have finished (Windows CI, 2026-11-20).
            pump(lambda: notes != [] and not meeting.running and not meeting.state["namesOpen"])
        assert log.seen == []
        assert notes == [{"Anna"}], "the notes see the name, not the placeholder"
        assert meeting.state["speakers"][0]["label"] == "Anna"
        assert ProfileStore.load().get("Anna", "redimnet2-b6-vb2-lm.onnx") is not None
        assert meeting.suggest("an") == ["Anna"], "a name saved here is offered next"

    def test_the_folder_preview_shows_when_the_card_opens(self, gui, tmp_path, monkeypatch):
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        expected = meeting.preview("")
        assert expected.startswith("meeting-")
        texts = [
            item.property("text")
            for item in _items(shell)
            if item.metaObject().className().startswith("QQuickText")
        ]
        assert expected in texts, "blank until the user types a title"

    def test_leaving_before_the_samples_arrive_keeps_none(self, gui, tmp_path, monkeypatch):
        # The page reads done at the finalized event; the run stores the
        # samples only after the recorder returns (the audio file closes in
        # between). A Back to menu in that gap must still drop them.
        from stenograf import flow

        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        release = threading.Event()

        class SlowReturn(flow.MeetingRecorder):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                release.wait(10)
                return result

        monkeypatch.setattr(flow, "MeetingRecorder", SlowReturn)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["phase"] == "done")
        meeting.stop()
        run = meeting._run
        release.set()
        pump(lambda: not meeting.running, timeout=10)
        assert run.result is not None
        assert run.voice_clips == {} and run.result.voice_clips == {}
        assert not meeting.state["namesOpen"]

    def test_a_copied_transcript_carries_names_given_after_the_finalize(
        self, gui, tmp_path, monkeypatch
    ):
        # The screen's captions are the finalize's; the names land on disk
        # later. The copy must be the named one, or a user pastes "Local-1".
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        assert meeting.state["transcriptReady"]
        self._done(meeting, {"Local-1": "Anna"})
        pump(lambda: notes != [] and not meeting.running)

        assert meeting.copy() is True
        copied = QGuiApplication.clipboard().text()
        assert "**Anna**" in copied and "Local-1" not in copied
        folder = Path(meeting.state["folder"])
        assert copied == (folder / "transcript.md").read_text(encoding="utf-8")

    def test_skip_starts_the_notes_with_the_labels(self, gui, tmp_path, monkeypatch):
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        meeting.skip_names()
        pump(lambda: notes != [] and not meeting.running)
        assert notes == [{"Local-1"}]
        assert not meeting.state["namesOpen"]
        meeting.open_names()  # names stay fixable after the notes
        assert meeting.state["namesOpen"]

    def test_leaving_the_screen_counts_as_skipping(self, gui, tmp_path, monkeypatch):
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        meeting.stop()  # "Back to menu"
        pump(lambda: notes != [] and not meeting.running)
        assert notes == [{"Local-1"}]

    def test_the_panel_closes_itself_after_the_timeout(self, gui, tmp_path, monkeypatch):
        import conftest

        conftest.write_settings("[speakers]\nnames_panel_timeout_s = 1\n")
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        assert meeting.state["namesCountdown"] == 1
        pump(lambda: notes != [] and not meeting.running)
        assert notes == [{"Local-1"}]
        assert not meeting.state["namesOpen"] and meeting.state["namesCountdown"] == 0

    def test_typing_stops_the_countdown(self, gui, tmp_path, monkeypatch):
        import conftest

        conftest.write_settings("[speakers]\nnames_panel_timeout_s = 1\n")
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        meeting.hold_names()  # what the name field calls on the first keystroke
        assert meeting.state["namesCountdown"] == 0
        deadline = time.monotonic() + 1.5
        pump(lambda: time.monotonic() > deadline)
        assert meeting.state["namesOpen"] and notes == [], "a held panel never closes itself"
        meeting.skip_names()
        pump(lambda: notes != [] and not meeting.running)

    def _enroll_the_voice(self):
        from stenograf.voiceprints import ProfileStore

        store = ProfileStore()
        store.enroll("Anna", np.array([1.0, 0.0], np.float32), "redimnet2-b6-vb2-lm.onnx")
        store.save()

    def test_the_panel_opens_even_when_everyone_was_recognised(
        self, gui, tmp_path, monkeypatch
    ):
        self._enroll_the_voice()
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        (row,) = meeting.state["speakers"]
        assert row["label"] == "Anna" and row["recognised"]
        time.sleep(0.5)
        assert notes == [], "a wrong name is worth a look before the notes use it"
        self._done(meeting, {"Anna": "Jonas"})  # the name button, corrected
        pump(lambda: notes != [] and not meeting.running)
        assert notes == [{"Jonas"}]

    def test_the_panel_can_skip_meetings_where_everyone_was_recognised(
        self, gui, tmp_path, monkeypatch
    ):
        import conftest

        conftest.write_settings("[speakers]\nnames_panel_skip_if_recognised = true\n")
        self._enroll_the_voice()
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: notes != [] and not meeting.running)
        assert notes == [{"Anna"}]
        assert not meeting.state["namesOpen"]
        assert meeting.state["speakers"], "the footer button can still open it"
        assert meeting.state["notesLocked"], "the notes went ahead; Edit must not offer them"

    def test_a_name_given_while_the_notes_run_reaches_the_notes(
        self, gui, tmp_path, monkeypatch
    ):
        import threading

        from stenograf.notes import run as notes_run

        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        writing, release = threading.Event(), threading.Event()

        def slow_notes(view, transcript, out_dir, basename, **kwargs):
            writing.set()
            release.wait(10)
            speaker = transcript.entries[0].speaker
            (out_dir / f"{basename}.notes.md").write_text(f"- {speaker} spoke\n", encoding="utf-8")
            return True

        monkeypatch.setattr(notes_run, "run_notes", slow_notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        self._done(meeting)
        pump(lambda: writing.is_set())
        meeting.open_names()
        self._done(meeting, {"Local-1": "Anna"})
        pump(lambda: not meeting.state["namesBusy"])
        assert "not renamed" not in meeting.state["status"], meeting.state["status"]
        release.set()
        pump(lambda: not meeting.running)
        notes_md = Path(meeting.state["folder"]) / "transcript.notes.md"
        assert notes_md.read_text(encoding="utf-8") == "- Anna spoke\n"

    def test_a_refused_voice_keeps_the_panel_and_the_notes_waiting(
        self, gui, tmp_path, monkeypatch
    ):
        from stenograf.voiceprints import ProfileStore

        store = ProfileStore()
        store.enroll("Anna", np.array([0.0, 1.0], np.float32), "redimnet2-b6-vb2-lm.onnx")
        store.save()
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        with _Warnings() as log:
            self._start(shell)
            pump(lambda: meeting.state["namesOpen"])
            folder = meeting.state["folder"]
            self._done(meeting, {"Local-1": "Anna"}, title="Kept back")
            pump(lambda: meeting.state["speakers"][0]["refused"] != "")
            row = meeting.state["speakers"][0]
            assert row["label"] == "Local-1" and row["name"] == "Anna"
            assert "doesn't sound like Anna" in row["refused"]
            assert meeting.state["namesOpen"] and notes == []
            assert meeting.state["folder"] == folder, "nothing after a refused name applies"

            meeting.save_names({"Local-1": "Anna"}, ["Local-1"])  # "Add anyway"
            pump(lambda: not meeting.state["namesBusy"])
            assert meeting.state["namesOpen"], "the card stays until Done"
            self._done(meeting)
            pump(lambda: notes != [] and not meeting.running)
        assert log.seen == []
        assert notes == [{"Anna"}]

    @staticmethod
    def _silent_player(monkeypatch):
        """Replace the speaker output: record what would play, play nothing."""
        from stenograf import playback

        played = []

        class SilentPlayback:
            def __init__(self, clip, on_done):
                self.clip, self.on_done = clip, on_done

            def start(self):
                played.append(self)
                return self

            def stop(self):
                self.on_done(None)

        monkeypatch.setattr(playback, "Playback", SilentPlayback)
        return played

    def test_a_voice_sample_plays_and_stops(self, gui, tmp_path, monkeypatch):
        import conftest

        conftest.write_settings("[speakers]\nnames_panel_timeout_s = 30\n")
        played = self._silent_player(monkeypatch)
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        with _Warnings() as log:
            self._start(shell)
            pump(lambda: meeting.state["namesOpen"])
            assert meeting.state["namesCountdown"] > 0

            meeting.play("Local-1")
            assert meeting.state["playing"] == "Local-1"
            assert meeting.state["namesCountdown"] == 0, "listening is answering"
            (sample,) = played
            assert sample.clip is meeting._run.voice_clips["Local-1"][0]

            meeting.play("Local-1")  # the same button again stops it
            pump(lambda: meeting.state["playing"] == "")

            played[0].on_done(None)  # a stale finish never clears a newer sample
            meeting.play("Local-1")
            assert len(played) == 2 and meeting.state["playing"] == "Local-1"
            played[0].on_done(None)
            pump(lambda: True)
            assert meeting.state["playing"] == "Local-1"
            meeting.skip_names()
            pump(lambda: notes != [] and not meeting.running)
        assert log.seen == []

    def test_samples_follow_a_name_and_leave_with_the_screen(self, gui, tmp_path, monkeypatch):
        played = self._silent_player(monkeypatch)
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        run = meeting._run
        self._done(meeting, {"Local-1": "Anna"})
        pump(lambda: notes != [] and not meeting.running)
        assert set(run.voice_clips) == {"Anna"}
        assert meeting.state["speakers"][0]["clips"] == 1

        meeting.play("Anna")
        assert played and meeting.state["playing"] == "Anna"
        meeting.stop()  # "Back to menu"
        assert meeting.state["playing"] == ""
        assert run.voice_clips == {}, "no voice outlives the screen"

    def test_a_meeting_without_diarization_asks_nothing(self, gui, tmp_path, monkeypatch):
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        form = {
            "mic": True,
            "system": False,
            "diarize": False,
            "local": -1,
            "remote": -1,
            "title": "",
            "recordAudio": False,
            "notes": True,
        }
        shell.screen("Setup").start(form)
        # Nothing to name, but the card still opens — the title and the notes
        # are on it — and the notes wait for it.
        pump(lambda: meeting.state["namesOpen"])
        assert meeting.state["speakers"] == []
        time.sleep(0.3)
        assert notes == []
        self._done(meeting)
        pump(lambda: notes != [] and not meeting.running and not meeting.state["namesOpen"])

    def test_done_renames_the_folder_and_titles_the_transcript(
        self, gui, tmp_path, monkeypatch
    ):
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        with _Warnings() as log:
            self._start(shell)
            pump(lambda: meeting.state["namesOpen"])
            before = Path(meeting.state["folder"])
            assert meeting.preview("Weekly sync") == before.name + "-weekly-sync"
            self._done(meeting, title="Weekly sync", context="Budget freigeben")
            # The folder line is set when the names job returns, which may be
            # after the notes it released have finished.
            pump(lambda: notes != [] and not meeting.running and not meeting.state["namesOpen"])
        assert log.seen == []
        after = Path(meeting.state["folder"])
        assert after.name == before.name + "-weekly-sync" and not before.exists()
        saved = json.loads((after / "transcript.json").read_text(encoding="utf-8"))
        assert saved["profile"]["title"] == "Weekly sync"
        assert saved["profile"]["context"] == "Budget freigeben"
        assert meeting.state["notesLocked"], "the notes choices are spent"

        meeting.open_names()  # the footer's Edit: the title can change again
        self._done(meeting, title="Weekly sync 2")
        pump(lambda: meeting.state["folder"].endswith("-weekly-sync-2"))
        assert len(notes) == 1, "a second Done never writes the notes again"

    def test_typing_a_name_offers_the_known_ones_without_insisting(
        self, gui, tmp_path, monkeypatch
    ):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from stenograf.voiceprints import ProfileStore

        store = ProfileStore()  # a voice unlike this meeting's, so the row stays unknown
        store.enroll("Annika Schulz", np.array([0.0, 1.0], np.float32), "redimnet2-b6-vb2-lm.onnx")
        store.save()
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        assert meeting.suggest("schu") == ["Annika Schulz"]

        def offered():
            return [
                i.property("text")
                for i in _items(shell)
                if "ItemDelegate" in i.metaObject().className() and i.isVisible()
            ]

        shell.show_window()
        try:
            field = next(
                i for i in _items(shell) if i.property("placeholderText") == "Who is this?"
            )
            field.forceActiveFocus()
            pump(lambda: field.hasActiveFocus())
            for key in "ann":
                QTest.keyClick(field.window(), key)
            pump(lambda: offered() == ["Annika Schulz"])  # the list lives in the overlay
            QTest.keyClick(field.window(), Qt.Key.Key_Down)
            QTest.keyClick(field.window(), Qt.Key.Key_Return)
            pump(lambda: True)
            assert field.property("text") == "Annika Schulz"
            assert meeting.state["namesOpen"], "Return took the offer, not the card"
            assert offered() == []

            field.setProperty("text", "")
            for key in "Ann":
                QTest.keyClick(field.window(), key)
            pump(lambda: offered() == ["Annika Schulz"])
            QTest.keyClick(field.window(), Qt.Key.Key_Down)
            QTest.keyClick(field.window(), Qt.Key.Key_Escape)  # the closing forgets the pick
            pump(lambda: offered() == [])
            QTest.keyClick(field.window(), Qt.Key.Key_Return)  # so the typed name is the answer
            pump(lambda: notes != [] and not meeting.running)
        finally:
            shell.hide_window()
        assert notes == [{"Ann"}]

    def test_keeping_the_recognised_name_closes_the_offers(self, gui, tmp_path, monkeypatch):
        from PySide6.QtCore import QMetaObject
        from PySide6.QtTest import QTest

        from stenograf.voiceprints import ProfileStore

        store = ProfileStore()
        store.enroll("Anna", np.array([1.0, 0.0], np.float32), "redimnet2-b6-vb2-lm.onnx")
        store.enroll("Annika", np.array([0.0, 1.0], np.float32), "redimnet2-b6-vb2-lm.onnx")
        store.save()
        self._diarized_meeting(monkeypatch, tmp_path, [])
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        shell.show_window()
        try:
            items = _items(shell)
            change = next(  # the recognised name, drawn as a button
                i
                for i in items
                if i.inherits("QQuickAbstractButton")
                and any(c.property("text") == "Anna" for c in _items_under(i))
            )
            keep = next(i for i in items if i.property("tip") == "Keep Anna")
            field = next(i for i in items if i.property("placeholderText") == "Who is this?")
            QMetaObject.invokeMethod(change, "clicked")
            pump(lambda: field.hasActiveFocus())
            for key in "ann":  # the first replaces the selected "Anna"
                QTest.keyClick(field.window(), key)

            def offered():
                return [
                    i.property("text")
                    for i in _items(shell)
                    if "ItemDelegate" in i.metaObject().className() and i.isVisible()
                ]

            pump(lambda: offered() == ["Anna", "Annika"])
            QMetaObject.invokeMethod(keep, "clicked")
            pump(lambda: offered() == [])
        finally:
            shell.hide_window()
        meeting.skip_names()
        meeting.join()

    def test_the_card_can_switch_notes_on_and_off(self, gui, tmp_path, monkeypatch):
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell, notes=False)  # the form said no notes
        pump(lambda: meeting.state["namesOpen"])
        assert meeting.state["notesOn"] is False
        self._done(meeting, notes=True)  # …the card says yes
        pump(lambda: notes != [] and not meeting.running)
        assert len(notes) == 1

    def test_discard_moves_the_meeting_to_the_trash(self, gui, tmp_path, monkeypatch):
        from PySide6.QtCore import QFile

        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        trashed = []
        monkeypatch.setattr(
            QFile, "moveToTrash", staticmethod(lambda path: trashed.append(path) or True)
        )
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        moves = []
        shell.navigation.connect(lambda page, mode: moves.append(mode))
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        folder = meeting.state["folder"]
        meeting.discard()
        assert trashed == [folder]
        assert "pop" in moves, "back to the menu"
        pump(lambda: not meeting.running)
        assert notes == [], "a discarded meeting writes no notes"

    def test_a_refused_trash_keeps_the_meeting_and_its_notes(self, gui, tmp_path, monkeypatch):
        from PySide6.QtCore import QFile

        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        monkeypatch.setattr(QFile, "moveToTrash", staticmethod(lambda path: False))
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        meeting.discard()
        assert "could not move" in meeting.state["status"]
        self._done(meeting)
        pump(lambda: notes != [] and not meeting.running)
        assert len(notes) == 1, "the card's Done still writes the notes"

    def test_leaving_drops_every_copy_of_the_voice_samples(self, gui, tmp_path, monkeypatch):
        notes = []
        self._diarized_meeting(monkeypatch, tmp_path, notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        self._start(shell)
        pump(lambda: meeting.state["namesOpen"])
        run = meeting._run
        assert run is not None and run.result is not None
        meeting.left()
        pump(lambda: not meeting.running)
        assert run.voice_clips == {} and run.result.voice_clips == {}


def _items(shell):
    """Every item in the window's tree, for tests that reach into a page."""

    def walk(item):
        yield item
        for child in item.childItems():
            yield from walk(child)

    assert shell.window is not None
    return list(walk(shell.window.contentItem()))


def _items_under(item):
    for child in item.childItems():
        yield child
        yield from _items_under(child)


def _switches(shell):
    return [i for i in _items(shell) if "Switch" in i.metaObject().className() and i.isVisible()]


def _toggle(switch):
    """Flip a switch the way a click does — not a write to ``checked``, which
    would drop the very binding these tests are about."""
    from PySide6.QtCore import QMetaObject

    QMetaObject.invokeMethod(switch, "toggle")


class TestOneMeetingAfterAnother:
    """The screen outlives its meetings; a run must not reach past its own."""

    _stub = TestMeetingScreen._stub_offline_meeting
    _form = staticmethod(TestMeetingScreen._mic_only_form)

    def test_a_finished_meetings_notes_leave_the_next_meeting_alone(
        self, gui, tmp_path, monkeypatch
    ):
        # Back to menu while the notes are still writing, then straight into
        # the next meeting: when the first run's notes return, the recording
        # meeting must stay recording, with its Stop.
        import conftest

        from stenograf import loaders
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider
        from stenograf.notes import run as notes_run

        self._stub(monkeypatch, tmp_path)
        started, release = threading.Event(), threading.Event()

        def slow_notes(view, transcript, out_dir, basename, **kwargs):
            started.set()
            release.wait(10)
            view.status("notes written")
            return True

        monkeypatch.setattr(notes_run, "run_notes", slow_notes)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(self._form(notes=True))
        through_the_card(meeting)
        assert started.wait(10)
        meeting.stop()  # Back to menu, notes still in flight
        first = meeting._thread

        second_mic = tmp_path / "second.wav"
        conftest.write_wav(second_mic, seconds=4.0)
        monkeypatch.setattr(
            loaders,
            "make_provider",
            lambda replay, plans, **kwargs: FileCaptureProvider(
                {Channel.MIC: second_mic}, paced=True
            ),
        )
        shell.screen("Setup").start(self._form(notes=False))
        pump(lambda: meeting.state["canStop"])

        release.set()
        assert first is not None
        first.join(10)
        pump(lambda: True)
        assert meeting.state["phase"] == "rec"
        # No audio is written by default, so the badge must not say "recording".
        assert meeting.state["phaseLabel"] == "LIVE"
        assert meeting.state["active"] is True
        assert meeting.state["canStop"] is True
        assert meeting.state["status"] != "notes written"

        meeting.stop()
        pump(lambda: meeting.state["phase"] == "done")
        through_the_card(meeting)
        meeting.join()

    def test_cancel_while_the_next_meeting_starts_cancels_it(self, gui, tmp_path, monkeypatch):
        # The first run's stop callback must not outlive it: the next
        # meeting's Cancel, pressed before its capture is up, cancels the run.
        from stenograf import loaders

        self._stub(monkeypatch, tmp_path)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(self._form(notes=False))
        through_the_card(meeting)
        meeting.join()
        pump(lambda: not meeting.state["active"])
        meeting.stop()

        gate = threading.Event()
        real = loaders.make_provider

        def slow_provider(*args, **kwargs):
            gate.wait(10)
            return real(*args, **kwargs)

        monkeypatch.setattr(loaders, "make_provider", slow_provider)
        shell.screen("Setup").start(self._form(notes=False))
        pump(lambda: meeting.state["status"] == "starting capture…")
        meeting.stop()
        assert meeting.state["phase"] == "rec"
        assert meeting.state["status"] == "cancelling…"
        gate.set()
        pump(lambda: not meeting.running and not meeting.state["active"])

    def test_the_form_refuses_a_second_meeting_while_one_records(self, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        never = threading.Event()
        meeting._thread = threading.Thread(target=never.wait, daemon=True)
        meeting._thread.start()
        meeting.set(active=True, phase="rec")
        try:
            setup = shell.screen("Setup")
            setup.start(self._form(notes=False))
            assert "still recording" in setup.state["error"]
        finally:
            never.set()
            meeting._thread.join()

    def test_a_menu_bar_stop_does_not_hold_the_notes_behind_the_card(
        self, gui, tmp_path, monkeypatch
    ):
        # Nobody is at the (hidden) window, so a card holding the notes would
        # hold them until a Quit skipped them. They run on the start form's
        # choices; the card waits, notes-free, for whenever the window opens.
        import conftest

        from stenograf import loaders
        from stenograf.capture.base import Channel
        from stenograf.capture.file import FileCaptureProvider
        from stenograf.notes import run as notes_run

        self._stub(monkeypatch, tmp_path)
        mic = tmp_path / "long.wav"
        conftest.write_wav(mic, seconds=3.0)
        monkeypatch.setattr(
            loaders,
            "make_provider",
            lambda replay, plans, **kwargs: FileCaptureProvider({Channel.MIC: mic}, paced=True),
        )
        calls = []
        monkeypatch.setattr(
            notes_run, "run_notes", lambda *args, **kwargs: calls.append(args) or True
        )
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(self._form(notes=True))
        pump(lambda: meeting.state["canStop"])
        assert shell.window is not None and not shell.window.isVisible()

        meeting.stop_from_menu_bar()
        # No Done: the run finishes on its own (the loop pumps, as the app's does).
        pump(lambda: not meeting.running and not meeting.state["active"])
        assert len(calls) == 1
        assert meeting.state["namesOpen"] is True
        assert meeting.state["notesLocked"] is True


class TestFinishCardInputs:
    """What the user types or flips on the card stays until Done."""

    _stub = TestMeetingScreen._stub_offline_meeting
    _form = staticmethod(TestMeetingScreen._mic_only_form)

    def _open_card(self, gui, tmp_path, monkeypatch):
        self._stub(monkeypatch, tmp_path)
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        shell.screen("Setup").start(self._form(notes=False))
        pump(lambda: meeting.state["namesOpen"])
        return shell, meeting

    def test_typed_and_flipped_answers_survive_a_status_line(self, gui, tmp_path, monkeypatch):
        # The state map is one property: every status line re-evaluates every
        # binding on it, and a bound field gets its old value written back.
        shell, meeting = self._open_card(gui, tmp_path, monkeypatch)
        title = next(
            i for i in _items(shell) if i.property("placeholderText") == "Untitled meeting"
        )
        title.setProperty("text", "Planning")
        (notes,) = _switches(shell)
        _toggle(notes)
        assert notes.property("checked") is True

        meeting.set(status="something else happened")
        meeting.hold_names()
        pump(lambda: True)
        assert title.property("text") == "Planning"
        assert notes.property("checked") is True
        meeting.skip_names()
        meeting.join()

    def test_escape_in_the_card_does_not_leave_the_meeting(self, gui, tmp_path, monkeypatch):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        shell, meeting = self._open_card(gui, tmp_path, monkeypatch)
        shell.show_window()
        try:
            title = next(
                i for i in _items(shell) if i.property("placeholderText") == "Untitled meeting"
            )
            title.forceActiveFocus()
            pump(lambda: title.hasActiveFocus())
            QTest.keyClick(title.window(), Qt.Key.Key_Escape)
            pump(lambda: True)
            assert meeting.state["namesOpen"] is True
            assert not meeting._run._names_settled.is_set()
        finally:
            shell.hide_window()
        meeting.skip_names()
        meeting.join()


class TestFinishCardLayout:
    _stub = TestMeetingScreen._stub_offline_meeting
    _form = staticmethod(TestMeetingScreen._mic_only_form)
    _open_card = TestFinishCardInputs._open_card

    def test_done_stays_in_sight_with_many_voices_in_the_smallest_window(
        self, gui, tmp_path, monkeypatch
    ):
        # The card gets only part of the window. A Done at the end of its
        # scrolled column fell below the edge from five voices up in the
        # default window, and nothing on screen said it was there.
        from PySide6.QtCore import QPointF

        shell, meeting = self._open_card(gui, tmp_path, monkeypatch)
        voices = [
            {
                "label": f"Remote-{n}",
                "recognised": False,
                "spoke": "3 min",
                "quotes": [{"text": "“something said”", "line": 0}],
                "clips": 1,
                "name": "",
                "refused": "",
            }
            for n in range(1, 10)
        ]
        window = shell.window
        assert window is not None
        window.setWidth(window.minimumWidth())
        window.setHeight(window.minimumHeight())
        shell.show_window()
        try:
            meeting.set(speakers=voices)
            pump(lambda: True)
            card = next(
                i for i in _items(shell) if i.property("opening") is not None and i.isVisible()
            )
            done = next(i for i in _items(shell) if i.property("text") == "Done")
            assert card.property("implicitHeight") > card.height(), "the card must overflow here"
            top = card.mapToScene(QPointF(0, 0)).y()
            bottom = card.mapToScene(QPointF(0, card.height())).y()
            assert top <= done.mapToScene(QPointF(0, 0)).y()
            assert done.mapToScene(QPointF(0, done.height())).y() <= bottom
        finally:
            shell.hide_window()
        meeting.skip_names()
        meeting.join()


class TestSmallWindow:
    """Every page in the smallest window the app allows — below a 1080p panel at
    200 %, the scaling a 7-inch handheld gets. A centred layout taller than the
    window used to run off both edges, so a Start, a Quit or a Back could be
    out of sight with nothing on screen saying it was there."""

    @staticmethod
    def _smallest(shell):
        window = shell.window
        assert window is not None
        window.setWidth(window.minimumWidth())
        window.setHeight(window.minimumHeight())
        shell.show_window()
        pump(lambda: True)
        return window

    @staticmethod
    def _scene_rect(item):
        from PySide6.QtCore import QPointF

        top_left = item.mapToScene(QPointF(0, 0))
        bottom_right = item.mapToScene(QPointF(item.width(), item.height()))
        return top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y()

    def _assert_in_window(self, window, item):
        """Inside the window and inside every clipping ancestor: a button
        under a scrolled body's edge is in the window and still not seen."""
        left, top, right, bottom = self._scene_rect(item)
        bounds = [(0.0, 0.0, float(window.width()), float(window.height()))]
        ancestor = item.parentItem()
        while ancestor is not None:
            if ancestor.clip():
                bounds.append(self._scene_rect(ancestor))
            ancestor = ancestor.parentItem()
        for b_left, b_top, b_right, b_bottom in bounds:
            assert b_left <= left and b_top <= top, item.property("text")
            assert right <= b_right and bottom <= b_bottom, item.property("text")

    @pytest.mark.parametrize(
        ("page", "button"),
        [
            ("Setup", "Start"),
            ("Transcribe", "Transcribe"),
            ("Notes", "Generate notes"),
            ("Settings", "Back"),
            ("Doctor", "Back"),
        ],
    )
    def test_a_forms_buttons_stay_in_the_window(self, gui, monkeypatch, page, button):
        from stenograf import doctor

        monkeypatch.setattr(doctor, "run_checks", lambda: [])
        shell, _engine = gui
        window = self._smallest(shell)
        try:
            shell.open(page)
            pump(lambda: True)
            if page == "Setup":
                # Every switch on: diarization adds the two speaker counts,
                # which is the form at its tallest.
                for switch in _switches(shell):
                    if not switch.property("checked"):
                        _toggle(switch)
                pump(lambda: True)
                flicks = [
                    i for i in _items(shell) if i.metaObject().className() == "QQuickFlickable"
                ]
                assert any(f.property("contentHeight") > f.height() for f in flicks), (
                    "the form must overflow here, or this test proves nothing"
                )
            target = next(
                i for i in _items(shell) if i.property("text") == button and i.isVisible()
            )
            self._assert_in_window(window, target)
        finally:
            shell.back()
            shell.hide_window()

    def test_the_menu_scrolls_to_quit_above_an_update_offer(self, gui):
        shell, _engine = gui
        shell.screen("Update").set(
            phase="available", version="9.9.9", text="Stenograf 9.9.9 is available", detail=""
        )
        window = self._smallest(shell)
        try:
            wordmark = next(i for i in _items(shell) if i.property("text") == "stenograf")
            self._assert_in_window(window, wordmark)
            update = next(
                i for i in _items(shell) if i.property("text") == "Update" and i.isVisible()
            )
            self._assert_in_window(window, update)
            flick = next(
                i
                for i in _items(shell)
                if i.metaObject().className() == "QQuickFlickable" and i.isVisible()
            )
            flick.setProperty(
                "contentY", max(0.0, flick.property("contentHeight") - flick.height())
            )
            pump(lambda: True)
            quit_label = next(i for i in _items(shell) if i.property("text") == "Quit")
            banner_top = self._scene_rect(update)[1]
            assert self._scene_rect(quit_label)[3] <= min(banner_top, window.height())
        finally:
            shell.screen("Update").set(phase="none")
            shell.hide_window()


class TestOpeningSize:
    class _Window:
        def __init__(self, width, height):
            from PySide6.QtCore import QSize

            size = QSize(width, height)
            self._screen = type("S", (), {"availableSize": lambda _self: size})()

        def screen(self):
            return self._screen

    @pytest.mark.parametrize(
        ("width", "height", "maximized"),
        [
            (1280, 720, True),  # a 1080p handheld at 150 %
            (960, 540, True),  # the same at 200 %
            (1512, 945, False),  # a 14-inch MacBook Pro at its default scaling
            (1920, 1040, False),
        ],
    )
    def test_a_small_desktop_opens_the_window_maximized(self, width, height, maximized):
        from stenograf.gui.app import opens_maximized

        assert opens_maximized(self._Window(width, height)) is maximized  # type: ignore[arg-type]

    def test_only_the_first_show_maximizes(self, gui):
        shell, _engine = gui
        window = shell.window
        assert window is not None
        shell.maximize_on_first_show = True
        try:
            shell.show_window()
            assert shell.maximize_on_first_show is False
        finally:
            shell.hide_window()


class TestSetupFormInputs:
    def test_a_flipped_switch_survives_the_microphone_listing(self, gui):
        # The listing lands after the form is up, through the same state map;
        # a switch bound to a default would flip back when it did.
        shell, _engine = gui
        shell.open("Setup")
        pump(lambda: bool(shell.screen("Setup").state["micDevicesReady"]))
        setup = shell.screen("Setup")
        switches = _switches(shell)
        notes = switches[-1]  # "Generate notes afterwards", the form's last switch
        before = notes.property("checked")
        _toggle(notes)
        setup.set(micDevicesReady=False)
        setup.set(micDevicesReady=True)
        pump(lambda: True)
        assert notes.property("checked") is (not before)
        shell.back()


class TestTranscribeScreen:
    def test_pick_and_transcribe_writes_a_transcript(self, gui, tmp_path, monkeypatch):
        import conftest

        from stenograf import loaders, output

        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (conftest.FakeASR(), conftest.WholeBufferVAD(), None),
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
        shell, _engine = gui
        screen = shell.screen("Transcribe")
        screen.choose((tmp_path / "not-audio.wav").as_uri())
        screen.start()  # the file does not exist
        assert screen.state["busy"] is False  # never started


class TestFolderActions:
    """The done screens' clickable folder and copy button (meeting and transcribe)."""

    @staticmethod
    def _folder(tmp_path) -> Path:
        from stenograf import output
        from stenograf.config import MeetingProfile
        from stenograf.transcript import Transcript, TranscriptEntry

        folder = tmp_path / "meeting-20261116-091500"
        transcript = Transcript(
            language=None,
            profile=MeetingProfile(),
            entries=[TranscriptEntry(speaker="Anna", text="Hallo.", start=0.0, end=1.0)],
        )
        output.write_transcript(transcript, folder, output.TRANSCRIPT_STEM)
        return folder

    @pytest.fixture
    def opened(self, monkeypatch):
        """Records what would have opened in the file manager — the real call
        would put a Finder window on the developer's screen."""
        from PySide6.QtGui import QDesktopServices

        urls = []
        monkeypatch.setattr(
            QDesktopServices, "openUrl", staticmethod(lambda url: urls.append(url) or True)
        )
        return urls

    @pytest.mark.parametrize("page", ["Meeting", "Transcribe"])
    def test_the_folder_opens_in_the_file_manager(self, gui, tmp_path, opened, page):
        folder = self._folder(tmp_path)
        shell, _engine = gui
        screen = shell.screen(page)
        screen.set(folder=str(folder), status="")
        screen.reveal()
        # toLocalFile() spells Windows paths with forward slashes; compare paths.
        assert [Path(url.toLocalFile()) for url in opened] == [folder]
        assert screen.state["status"] == ""

    def test_a_folder_not_written_yet_says_so(self, gui, tmp_path, opened):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        meeting.set(folder=str(tmp_path / "meeting-20261116-091500"))
        meeting.reveal()
        assert opened == []
        assert "does not exist yet" in meeting.state["status"]

    @pytest.mark.parametrize("page", ["Meeting", "Transcribe"])
    def test_copy_puts_the_transcript_on_the_clipboard(self, gui, tmp_path, page):
        folder = self._folder(tmp_path)
        shell, _engine = gui
        screen = shell.screen(page)
        screen.set(folder=str(folder))
        assert screen.copy() is True
        assert QGuiApplication.clipboard().text() == "**Anna** [0:00]: Hallo.\n"

    def test_copy_without_a_transcript_says_why(self, gui, tmp_path):
        QGuiApplication.clipboard().setText("left alone")
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        meeting.set(folder=str(tmp_path))  # a folder, but no transcript in it
        assert meeting.copy() is False
        assert meeting.state["status"].startswith("transcript not copied:")
        assert QGuiApplication.clipboard().text() == "left alone"

    def test_a_finished_transcription_offers_its_folder(self, gui, tmp_path, monkeypatch):
        import conftest

        from stenograf import loaders, output

        home_dir = tmp_path / "meetings"
        monkeypatch.setattr(output, "default_output_home", lambda: home_dir)
        monkeypatch.setattr(
            loaders,
            "load_backends",
            lambda **kwargs: (conftest.FakeASR(), conftest.WholeBufferVAD(), None),
        )
        audio = tmp_path / "recording.wav"
        conftest.write_wav(audio)
        shell, _engine = gui
        screen = shell.screen("Transcribe")
        screen.choose(audio.as_uri())
        screen.start()
        assert screen.state["folder"] == "", "no folder to offer while it runs"
        pump(lambda: not screen.state["busy"])
        (written,) = home_dir.glob("*/transcript.md")
        assert screen.state["folder"] == str(written.parent)
        screen.choose(audio.as_uri())  # a new pick forgets the last result
        assert screen.state["folder"] == ""


class TestNotesScreen:
    def test_the_newest_meeting_is_preselected_and_generates(self, gui, tmp_path, monkeypatch):
        from test_cli_notes import FakeBackend, write_transcript_json

        import stenograf.notes as notes_pkg
        from stenograf import output

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

        monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "nowhere")
        shell, _engine = gui
        screen = shell.screen("Notes")
        screen.opened()
        assert screen.state["meeting"] == ""
        assert "No finished meeting" in screen.state["status"]


class TestDroppedPaths:
    """A file or folder dropped onto the window arrives as a ``file:`` URL,
    and unlike a dialog's it can be the wrong kind."""

    def test_a_transcript_dropped_on_notes_stands_for_its_folder(self, gui, tmp_path):
        transcript = tmp_path / "meeting-20260101-090000" / "transcript.md"
        transcript.parent.mkdir()
        transcript.write_text("x", encoding="utf-8")
        shell, _engine = gui
        screen = shell.screen("Notes")
        screen.choose(transcript.as_uri())
        assert screen.state["meeting"] == str(transcript.parent)
        screen.choose(transcript.parent.as_uri())
        assert screen.state["meeting"] == str(transcript.parent)

    def test_a_link_that_is_not_a_file_is_refused(self, gui):
        shell, _engine = gui
        transcribe = shell.screen("Transcribe")
        transcribe.choose("https://example.com/talk.mp3")
        assert transcribe.state["file"] == ""
        assert "on this computer" in transcribe.state["status"]
        notes = shell.screen("Notes")
        before = notes.state["meeting"]
        notes.choose("https://example.com/")
        assert notes.state["meeting"] == before

    def test_a_folder_dropped_on_transcribe_is_refused(self, gui, tmp_path):
        shell, _engine = gui
        screen = shell.screen("Transcribe")
        screen.choose(tmp_path.as_uri())
        assert screen.state["file"] == ""
        assert "folder" in screen.state["status"]


class TestSettingsScreen:
    def test_renders_every_table_with_value_provenance(self, gui, tmp_path, monkeypatch):
        write_settings('[transcript]\nformats = ["md"]\n')

        shell, _engine = gui
        screen = shell.screen("Settings")
        screen.opened()
        assert screen.state["ok"] is True
        for table in ("[transcript]", "[vocab]", "[output]", "[asr]", "[notes]"):
            assert table in screen.lines
        assert any("settings.toml" in line and "formats" in line for line in screen.lines)

    def test_a_broken_file_renders_its_error(self, gui, tmp_path, monkeypatch):
        write_settings("not toml [")

        shell, _engine = gui
        screen = shell.screen("Settings")
        screen.opened()
        assert screen.state["ok"] is False
        assert any("settings.toml" in line for line in screen.lines[1:])

    def test_a_meeting_type_re_renders_the_report_under_its_overlay(
        self, gui, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)
        data = tmp_path / "data"
        data.mkdir()
        (data / "settings.toml").write_text(
            '[notes]\nbackend = "command"\ncommand = ["claude"]\n'
            '\n[meetings.controlling.notes]\nbackend = "mlx"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("STENOGRAF_DATA", str(data))

        shell, _engine = gui
        screen = shell.screen("Settings")
        screen.opened()
        assert [entry["value"] for entry in screen.state["presets"]] == ["", "controlling"]
        assert any("backend" in line and "command" in line for line in screen.lines)

        screen.show("controlling")
        assert screen.state["preset"] == "controlling"
        assert any("[meetings.controlling]" in line and "mlx" in line for line in screen.lines)

    def test_a_deleted_meeting_type_drops_off_the_picker_on_reload(
        self, gui, tmp_path, monkeypatch
    ):
        """The report must never render an error because the picker is still
        holding a name the file no longer defines."""
        data = tmp_path / "data"
        data.mkdir()
        settings = data / "settings.toml"
        settings.write_text('[meetings.controlling]\ntitle = "C"\n', encoding="utf-8")
        monkeypatch.setenv("STENOGRAF_DATA", str(data))

        shell, _engine = gui
        screen = shell.screen("Settings")
        screen.opened()
        screen.show("controlling")

        settings.write_text("", encoding="utf-8")
        screen.opened()  # the Reload button
        assert screen.state["preset"] == ""
        assert screen.state["ok"] is True


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
        # Problems first, whatever order the checks ran in.
        assert [check["state"] for check in screen.state["checks"]] == ["bad", "optional", "good"]
        assert "1 problem(s)" in screen.state["status"]


class TestTray:
    """Menu-bar / system-tray mode.

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

    def test_a_status_icon_the_shell_refuses_falls_back_to_qt_s(self, gui, monkeypatch, caplog):
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
        assert "falling back" in caplog.text

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

        # Finalizing is still this meeting; its notes and finish card are not.
        meeting.set(phase="finalizing")
        tray.menu.aboutToShow.emit()
        assert not tray.start.isEnabled()
        meeting.set(phase="done")
        tray.menu.aboutToShow.emit()
        assert tray.status.text() == "Finishing the meeting…"
        assert tray.start.isEnabled(), "the next meeting need not wait for the notes"

        meeting.set(active=True, phase="rec", elapsed="12:34")
        tray.menu.aboutToShow.emit()
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

    def test_a_muted_channel_shows_in_the_menu_bar(self, tray, gui, monkeypatch):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        meeting.set(
            active=True,
            phase="rec",
            elapsed="3:00",
            channels=["mic", "system"],
            canStop=True,
            micMuted=True,
            systemMuted=False,
        )
        assert tray.state() == "muted"
        tray._refresh()
        assert tray.icon.icon().cacheKey() == tray._icons["muted"].cacheKey()

        tray.menu.aboutToShow.emit()
        assert tray.status.text() == "Recording · 3:00 · mic muted"
        assert tray.mute_mic.isVisible() and tray.mute_mic.isChecked()
        assert tray.mute_system.isVisible() and not tray.mute_system.isChecked()

        pressed = []
        monkeypatch.setattr(meeting, "mute", lambda channel, on: pressed.append((channel, on)))
        tray.mute_mic.trigger()
        tray.mute_system.trigger()
        assert pressed == [("mic", False), ("system", True)]

        meeting.set(phase="done")  # the meeting is finishing: nothing to mute
        tray.menu.aboutToShow.emit()
        assert not tray.mute_mic.isVisible() and not tray.mute_system.isVisible()

    def test_a_mic_only_meeting_offers_no_system_mute(self, tray, gui):
        shell, _engine = gui
        meeting = shell.screen("Meeting")
        meeting.set(active=True, phase="rec", channels=["mic"], canStop=True)
        tray.menu.aboutToShow.emit()
        assert tray.mute_mic.isVisible() and not tray.mute_system.isVisible()

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
