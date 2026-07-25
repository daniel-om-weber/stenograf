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
meeting, against a live pipeline tuned to ~0.6 W. The budget's real lever is
elsewhere, though: a window that is *visible at all* is redrawn at the display's
refresh rate, so the app is at its cheapest with no window — which is what the
menu bar buys (:mod:`stenograf.gui.tray`).

The application object is a ``QApplication`` rather than the ``QGuiApplication``
a pure Qt Quick app would need, for exactly one reason: ``QSystemTrayIcon``'s
menu is a QtWidgets widget, and building one under a bare ``QGuiApplication``
aborts the process on ``qFatal``.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from PySide6.QtCore import Property, QCoreApplication, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication

from stenograf import ASSETS
from stenograf.shortcut import DESKTOP_FILE_NAME

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtGui import QWindow

    from stenograf.gui.meeting import MeetingScreen

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

    def get(self, key: str, default: object = None) -> object:
        """One display field, for Python readers — ``state`` is a QML Property,
        and reaching through the descriptor from outside QML does not type."""
        return self._state.get(key, default)

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
    """``(page, mode)`` with mode ``push`` / ``replace`` / ``root`` / ``pop``."""

    quitting = Signal()
    """A quit is waiting on a meeting to finish (see :meth:`quit_app`)."""

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

        self.window: QWindow | None = None
        """The one window, once the QML has produced it (see :func:`build`)."""

        self._quitting = False
        self._abandoned = False
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

    @Slot(str)
    def reset_to(self, page: str) -> None:
        """Clear the stack and open ``page`` — for navigation from outside the window.

        The menu bar's *Start meeting* has no idea what the window was last left
        on, and pushing onto whatever that was would stack a second copy of a
        page the user is already looking at."""
        self.navigation.emit(page, "root")

    # -- the window --------------------------------------------------------

    @Slot()
    def show_window(self) -> None:
        """Bring the window up, from hidden or from behind."""
        from stenograf.gui.tray import set_dock_icon

        if self.window is None:
            return
        # Before the window, not after: an app in accessory mode cannot be
        # brought to the front, so a show() first would raise a window that
        # stays behind the frontmost app.
        set_dock_icon(True)
        self.window.show()
        self.window.raise_()
        self.window.requestActivate()

    @Slot()
    def hide_window(self) -> None:
        """Put the app away without ending it — only ever called with a tray up.

        The meeting, if there is one, keeps running: that is the difference
        between this and :meth:`quit_app`, and the reason the tray exists."""
        from stenograf.gui.tray import set_dock_icon

        if self.window is None:
            return
        self.window.hide()
        set_dock_icon(False)

    # -- shutdown ----------------------------------------------------------

    @Slot()
    def quit_app(self) -> None:
        """Leave for good — but let a running meeting finish first.

        In tray mode a meeting normally has no window in front of it, so quitting
        mid-meeting stops being the rare accident it was when the window *was*
        the app. Dropping capture on the floor here would cost the transcript, so
        the window goes away immediately (that is the feedback the click owes)
        and the app quits once :meth:`~stenograf.gui.meeting.MeetingScreen.shutdown`
        has stopped capture and the finalize has landed on disk. The menu bar
        says so meanwhile, and a second Quit gives up on it — the checkpoint
        survives, and a wait with no way out would be the worse bug."""
        self.hide_window()
        meeting = self._meeting()
        if self._quitting or not meeting.running:
            self._abandoned = self._quitting  # nothing left for join_meetings to do
            QCoreApplication.quit()
            return
        self._quitting = True
        self.quitting.emit()
        threading.Thread(target=self._finish_then_quit, name="gui-quit", daemon=True).start()

    @property
    def quitting_now(self) -> bool:
        """Whether a quit is already waiting on the meeting (the tray asks)."""
        return self._quitting

    def _finish_then_quit(self) -> None:
        meeting = self._meeting()
        try:
            meeting.shutdown()
        finally:
            meeting.post(QCoreApplication.quit)  # back onto the GUI thread to leave

    def join_meetings(self) -> None:
        """Finish meeting work an ended event loop would otherwise abandon.

        The fallback path, for every exit that is not :meth:`quit_app`: ⌘Q, and
        a window closed with no tray to close into. Meeting threads are daemons,
        so simply returning here kills whatever they are still doing; instead
        capture is stopped and the finalize (and the notes tail) awaited — see
        ``MeetingScreen.shutdown``. Say so on stderr first: if the app was
        started from a terminal, an unexplained pause reads as a hang."""
        meeting = self._meeting()
        if self._abandoned or not meeting.running:
            return
        print(
            "the app quit with a meeting still running — stopping capture and "
            "finishing it (finalize/notes); Ctrl-C abandons it, and "
            "`steno notes --last` regenerates missing notes afterwards",
            file=sys.stderr,
        )
        meeting.shutdown()

    def _meeting(self) -> MeetingScreen:
        from stenograf.gui.meeting import MeetingScreen

        meeting = self._screens["Meeting"]
        assert isinstance(meeting, MeetingScreen)
        return meeting


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
    gui.window = engine.rootObjects()[0]  # type: ignore[assignment]  # an ApplicationWindow is a QWindow
    return engine, gui


def run(*, tray: bool = False) -> int:
    """Open the app and run it until it quits; returns the exit code.

    With ``tray`` the app starts in the menu bar with no window at all — the
    login-item shape, and the one that idles at the wakeup floor. Where no tray
    host exists the flag degrades to a normal window, since the alternative is
    an app the user cannot reach."""
    from stenograf.gui.tray import install

    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setApplicationName("Stenograf")
    app.setApplicationDisplayName("Stenograf")
    app.setOrganizationName("stenograf")
    # The Linux half of the same identity: this is the app_id a Wayland window
    # carries, and it has to name the desktop entry `steno setup` wrote or the
    # window arrives iconless and separate from its own launcher. Ignored on
    # macOS and Windows. The constant lives with the entry that must match it.
    app.setDesktopFileName(DESKTOP_FILE_NAME)
    # Named here rather than left to the platform: launched from
    # `Stenograf.app` the Dock already shows the bundle's icon, but started
    # from a terminal — or on Linux and Windows — this is the only thing
    # standing between the app and a generic Python tile.
    app.setWindowIcon(QIcon(str(ASSETS / "icon.png")))

    _engine, gui = build(app)
    status_item = install(gui)
    if status_item is not None:
        # The window is now closeable without ending the app, so Qt's own rule
        # would quit us the moment it hides.
        app.setQuitOnLastWindowClosed(False)
    elif tray:
        print(
            "this desktop has no system tray, so --tray opened a window instead "
            "(stock GNOME needs the AppIndicator extension)",
            file=sys.stderr,
        )
    # Shown from here, not by the QML: a headless test can build the whole tree
    # (catching every QML error) without a window ever being realized.
    if tray and status_item is not None:
        # The window was never shown, so this only drops the Dock tile — an app
        # asked for the menu bar must not start life as a tile with no window.
        gui.hide_window()
    else:
        gui.show_window()
    code = app.exec()
    gui.join_meetings()
    return code


__all__ = ["MENU", "QML_DIR", "Screen", "StenografGui", "build", "run"]
