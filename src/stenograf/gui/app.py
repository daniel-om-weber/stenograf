"""The desktop application shell: one window, a screen stack, one base class.

Phase 8. The Qt side of the launcher is deliberately thin — the
same rule the CLI and the Textual UI follow: a screen gathers inputs, shows
progress and calls a library entry point (:mod:`stenograf.flow`); anything it
needs that the library lacks belongs in the library, not here.

Three decisions carry most of the design:

**One state property per screen, not one per field.** :class:`Screen` exposes a
single ``state`` map to QML and one ``changed`` signal. ``screen.state.status``
in a binding costs nothing to add — no ``Property``/``Signal``/getter triplet
per field, which is where PySide UIs usually drown in boilerplate. Streams
(captions) still use real signals: appending to a list model must not
re-evaluate every binding on the screen.

**Slow work goes to a worker thread with a marshalled reply** (:meth:`Screen.work`
— the Qt equivalent of Textual's ``@work(thread=True)``). Blocking the GUI
thread freezes rendering, and a Qt object may only be touched from its own
thread; :meth:`Screen.post` is the single hop back, built on the fact that a
signal emitted off-thread to a slot on a GUI-thread object queues itself.

**The redraw budget carries over from the TUI** (``TEXTUAL_FPS`` +
``animation_level = "none"``): bind the view to model updates, never to a
clock. There is exactly one periodic timer in the whole app — the meeting
screen's 1 Hz elapsed clock — and no idle animations; the StackView's page
transitions are switched off in ``Main.qml`` for the same reason. Hover
feedback animates because it is event-driven and stops. A spinner, a pulsing
REC dot or a caption easing would hold the compositor awake for a whole
meeting, against a live pipeline tuned to ~0.6 W.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

if TYPE_CHECKING:
    from collections.abc import Callable

QML_DIR = Path(__file__).parent / "qml"
"""Where the ``.qml`` files live; they ship inside the wheel next to this module."""

MENU: tuple[tuple[str, str, str], ...] = (
    ("Setup", "Start meeting", "Capture this meeting with live captions."),
    ("Transcribe", "Transcribe a recording", "Turn an audio file into a transcript."),
    ("Notes", "Generate notes", "Summarize a finished meeting's transcript."),
    ("Settings", "Settings", "Show the active configuration."),
    ("Doctor", "Check setup", "Verify models, permissions, and audio devices."),
    ("quit", "Quit", "Leave the app."),
)
"""``(page, label, description)`` per home-menu entry, in display order. ``page``
names the QML file a click opens; ``quit`` is the one entry that is not a page."""

_T = TypeVar("_T")


class Screen(QObject):
    """Base for the object behind one QML page: reactive state, background work.

    Subclasses put display values in :meth:`set` and expose intents as ``@Slot``
    methods; QML reads ``screen.state.<field>`` and calls those slots. Every page
    calls :meth:`opened` from ``Component.onCompleted``, which is where a screen
    that must refresh itself on each visit (settings, doctor, the setup form's
    standing defaults) does its work — the objects are long-lived, the pages are
    not.
    """

    changed = Signal()
    """Emitted whenever :meth:`set` changes anything; re-evaluates QML bindings."""

    _posted = Signal(object)

    def __init__(self, app: StenografGui) -> None:
        # Parented to the shell (which is parented to the application), so a
        # screen outlives the QML that binds to it: an engine torn down first
        # destroys its items quietly, while the reverse order re-evaluates every
        # binding against a null object and floods stderr with TypeErrors.
        super().__init__(app)
        self.app = app
        self._state: dict[str, object] = {}
        # The thread hop. Both ends live on the GUI thread, so an emit from a
        # worker resolves to a queued connection and the call runs on the event
        # loop — Qt's supported equivalent of Textual's call_from_thread.
        self._posted.connect(self._run_posted)

    # -- state -------------------------------------------------------------

    @Property("QVariantMap", notify=changed)  # type: ignore[operator]  # PySide decorator
    def state(self) -> dict[str, object]:
        """Everything this screen displays, as one QML-readable map."""
        return self._state

    def set(self, **fields: object) -> None:
        """Update display fields and repaint the bindings that read them.

        Values must be QML-friendly (str/bool/number/list/dict) — a ``Path``
        reaches QML as an opaque object, so convert on the way in."""
        self._state.update(fields)
        self.changed.emit()

    # -- threading ---------------------------------------------------------

    def post(self, fn: Callable[..., object], *args: object, **kwargs: object) -> None:
        """Run ``fn(*args, **kwargs)`` on the GUI thread; safe to call from any thread."""
        self._posted.emit(lambda: fn(*args, **kwargs))

    @Slot(object)
    def _run_posted(self, call: Callable[[], object]) -> None:
        call()

    def work(
        self,
        job: Callable[[], _T],
        *,
        done: Callable[[_T], object] | None = None,
        failed: Callable[[str], object] | None = None,
        name: str = "gui-work",
    ) -> threading.Thread:
        """Run ``job`` off the GUI thread and deliver its outcome back on it.

        Exactly one of ``done``/``failed`` fires, both on the GUI thread. The
        thread is a daemon so a closed window never blocks process exit — the
        shell joins whatever is still meaningful (a finalize, a notes run) after
        the event loop stops."""

        def run() -> None:
            try:
                result = job()
            except Exception as exc:  # noqa: BLE001 — every failure is UI, not a crash
                if failed is not None:
                    self.post(failed, str(exc))
                return
            if done is not None:
                self.post(done, result)

        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread

    # -- lifecycle ---------------------------------------------------------

    @Slot()
    def opened(self) -> None:
        """The page for this screen just appeared (no-op by default)."""


class StenografGui(QObject):
    """The shell QML talks to: the menu, the screen objects, and navigation.

    Navigation is one signal in one direction — Python asks, ``Main.qml``'s
    StackView acts — so no screen needs a reference to the stack and pages stay
    plain QML files.
    """

    navigation = Signal(str, str)
    """``(page, mode)`` with mode ``push`` / ``replace`` / ``pop``."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        from stenograf.gui.meeting import MeetingScreen
        from stenograf.gui.screens import (
            DoctorScreen,
            NotesScreen,
            SettingsScreen,
            SetupScreen,
            TranscribeScreen,
        )

        self._screens: dict[str, Screen] = {
            "Setup": SetupScreen(self),
            "Meeting": MeetingScreen(self),
            "Transcribe": TranscribeScreen(self),
            "Notes": NotesScreen(self),
            "Settings": SettingsScreen(self),
            "Doctor": DoctorScreen(self),
        }

    @Property("QVariantList", constant=True)  # type: ignore[operator]  # PySide decorator
    def menu(self) -> list[dict[str, str]]:
        return [{"page": page, "label": label, "description": desc} for page, label, desc in MENU]

    @Slot(str, result=QObject)
    def screen(self, page: str) -> Screen | None:
        """The object behind a page, handed to it as a property when it opens."""
        return self._screens.get(page)

    # -- navigation --------------------------------------------------------

    @Slot(str)
    def open(self, page: str) -> None:
        self.navigation.emit(page, "push")

    @Slot(str)
    def replace(self, page: str) -> None:
        """Open ``page`` *instead of* the current one (setup → meeting: Back
        from the meeting must land on Home, not on the form that started it)."""
        self.navigation.emit(page, "replace")

    @Slot()
    def back(self) -> None:
        self.navigation.emit("", "pop")

    # -- shutdown ----------------------------------------------------------

    def join_meetings(self) -> None:
        """Finish meeting work the closed window would otherwise abandon.

        Meeting threads are daemons, so returning here kills whatever they are
        still doing. Closing the window mid-meeting therefore ends capture and
        waits for the finalize (and the notes tail) — see
        ``MeetingScreen.shutdown``. Say so on stderr first: if the app was
        started from a terminal, an unexplained pause reads as a hang."""
        from stenograf.gui.meeting import MeetingScreen

        meeting = self._screens["Meeting"]
        assert isinstance(meeting, MeetingScreen)
        if not meeting.running:
            return
        print(
            "the window closed with a meeting still running — stopping capture and "
            "finishing it (finalize/notes); Ctrl-C abandons it, and "
            "`steno notes --last` regenerates missing notes afterwards",
            file=sys.stderr,
        )
        meeting.shutdown()


def build(app: QGuiApplication) -> tuple[QQmlApplicationEngine, StenografGui]:
    """Create the shell and load the QML tree onto ``app`` (no window shown yet).

    Split out of :func:`run` so tests can build the real application object
    graph — the one thing worth checking automatically about QML — without
    entering an event loop or needing a display."""
    QQuickStyle.setStyle("Basic")  # full styling control; native styles resist it
    gui = StenografGui(app)  # owned by the application: it must outlive the engine
    engine = QQmlApplicationEngine()
    # Two roads not taken (measured in the Phase 8 spike, PySide6 6.11):
    # qmlRegisterSingletonInstance() makes the ApplicationWindow root reject
    # every QML-declared child, and setContextProperty() resolves late, so each
    # Component's first binding pass reads null and logs a TypeError. An initial
    # root property is set before component completion: every binding correct on
    # the first pass, zero warnings.
    engine.setInitialProperties({"app": gui})
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Main.qml")))
    if not engine.rootObjects():
        raise RuntimeError(f"the interface failed to load from {QML_DIR}")
    return engine, gui


def run() -> int:
    """Open the app and run it until the window closes; returns the exit code."""
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    assert isinstance(app, QGuiApplication)
    app.setApplicationName("Stenograf")
    app.setApplicationDisplayName("Stenograf")
    app.setOrganizationName("stenograf")

    engine, gui = build(app)
    # Shown from here, not by the QML: a headless test can build the whole tree
    # (catching every QML error) without a window ever being realized.
    engine.rootObjects()[0].setProperty("visible", True)
    code = app.exec()
    gui.join_meetings()
    return code


__all__ = ["MENU", "QML_DIR", "Screen", "StenografGui", "build", "run"]
