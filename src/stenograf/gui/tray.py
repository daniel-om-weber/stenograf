"""Menu-bar / system-tray mode.

A meeting is the one thing this tool does that outlives the window you started
it from: the video call is in front, stenograf is behind it, and the two
questions that matter for the next half hour — *is it still recording?* and
*stop it* — should not cost an application switch. So the app grows a status
item: the mark from its icon in the menu bar, tinted while a meeting is live,
with a menu that can start, stop and quit without a window existing at all.

That also makes the window optional, which is where the power argument lands:
a *visible* Qt Quick window is woken at the display's refresh rate no matter
how disciplined the QML is, and a hidden one wakes ~100× less (measured; the
full record is in PLAN.md), so closing to the tray is worth roughly two orders
of magnitude for the majority of a meeting's runtime. Hence the two behaviour
changes here, both conditional on a tray host actually existing:

- **closing the window hides it instead of quitting**, and a meeting in progress
  keeps running — the tray is what tells you it is still there;
- **the Dock tile goes with it on macOS** (:func:`set_dock_icon`), so the app
  really is menu-bar-only rather than a windowless Dock icon.

Where no tray host exists — stock GNOME without the AppIndicator extension is
the case that matters — :func:`install` returns ``None`` and none of that
happens: the window is the app, and closing it quits, exactly as before.

On Windows the icon itself comes from :mod:`stenograf.gui.wintray` rather than
from ``QSystemTrayIcon`` (:func:`_status_icon` picks) — why Qt's implementation
loses the user's show/hide choice is that module's docstring. Everything below
is written against the small surface both share, so this module has one
status-item story, not two.

Two landmines paid for here:

- **QSystemTrayIcon needs a QApplication, not a QGuiApplication.** It is a
  QtWidgets class and its menu is a real QWidget; under a bare QGuiApplication
  the process dies on ``qFatal`` ("Cannot create a QWidget without
  QApplication") before any Python exception can be raised. That is why
  :func:`stenograf.gui.app.run` constructs a ``QApplication``.
- **The menu is rebuilt on ``aboutToShow``, never on a timer.** Meeting state
  changes several times a second while captions arrive; re-labelling a menu
  nobody is looking at is exactly the idle work the redraw budget forbids. Only
  the icon and the tooltip follow the meeting live, and only when the *state*
  (not the elapsed clock) changes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, Qt, Slot
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from stenograf import ASSETS
from stenograf.log import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from stenograf.gui.app import StenografGui
    from stenograf.gui.meeting import MeetingScreen

    if sys.platform == "win32":
        from stenograf.gui.wintray import WindowsStatusIcon

        StatusIcon = QSystemTrayIcon | WindowsStatusIcon
    else:
        # wintray refuses to import off Windows and narrows itself away with it,
        # so naming its class here would be an unknown symbol on any other
        # platform's type check — off Windows there is only ever Qt's icon.
        StatusIcon = QSystemTrayIcon

MARK = ASSETS / "tray.svg"
"""The icon's two commas without their tile — a status item is artwork on nothing."""

_SIZES = (16, 18, 22, 32, 36, 44, 64)
"""Rasterized once each so the platform picks a crisp one at any scale factor."""

_TINT = {
    "rec": "#ff5f56",  # Theme.rec
    "busy": "#e8b339",  # Theme.busy
}
"""While a meeting is live the whole mark takes the phase colour the app's own
header uses. Idle keeps the brand inks (or a template image on macOS, below)."""


def _icon(tint: str | None, *, mask: bool = False) -> QIcon:
    """The mark, optionally re-inked in one flat colour.

    ``tint`` replaces every pixel's colour and keeps its alpha, which is what
    makes one asset serve three states; ``mask`` marks the result as a macOS
    template image, so the system inverts it for a light or dark menu bar."""
    renderer = QSvgRenderer(str(MARK))
    icon = QIcon()
    for size in _SIZES:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        if tint is not None:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(pixmap.rect(), QColor(tint))
        painter.end()
        icon.addPixmap(pixmap)
    icon.setIsMask(mask)
    return icon


def _status_icon(parent: QObject) -> StatusIcon:
    """The notification-area icon for this platform.

    Gated on the *platform plugin*, not on ``sys.platform`` alone, for the same
    reason :func:`set_dock_icon` is: under the offscreen plugin (the tests, CI)
    there is no shell to register with, and the native path would create a real
    window and a real icon for the test runner. A failure to build it is not
    fatal either — Qt's own icon loses the stable identity but still puts
    something in the menu bar, which is the part the user needs.
    """
    if sys.platform == "win32" and QGuiApplication.platformName() == "windows":
        try:
            from stenograf.gui.wintray import WindowsStatusIcon

            return WindowsStatusIcon(parent)
        except (OSError, ImportError) as exc:
            logger.warning(
                "could not create the Stenograf status icon (%s) — falling back to "
                "Qt's, which Windows may keep hidden in the overflow",
                exc,
            )
    return QSystemTrayIcon(parent)


def set_dock_icon(visible: bool) -> bool:
    """Show or hide the macOS Dock tile at runtime; ``False`` if that is not possible.

    ``NSApplicationActivationPolicyAccessory`` is what makes an app menu-bar-only,
    and it is settable while the app runs — measured, and the reason
    menu-bar mode needs no ``LSUIElement`` key in ``Stenograf.app``'s frozen
    Info.plist. The key was tried and rejected besides: a UIElement app cannot be
    brought forward by ``open``, so double-clicking the bundle opened the window
    *behind* whatever was in front.

    Done through ``objc_msgSend`` rather than pyobjc because AppKit is already
    loaded (Qt's cocoa plugin owns ``NSApp``) and one selector does not justify a
    dependency. Every ``objc_msgSend`` call needs its own signature — the real
    symbol is variadic, and ctypes must be told the exact argument types per
    call or the arguments go to the wrong registers."""
    # Not `sys.platform` alone: under the offscreen plugin (the tests, CI) there
    # is no Dock, and asking for `sharedApplication` would conjure an NSApp — and
    # a Dock tile for the test runner — where Qt deliberately made none.
    if sys.platform != "darwin" or QGuiApplication.platformName() != "cocoa":
        return False
    library = ctypes.util.find_library("objc")
    if library is None:  # pragma: no cover — macOS always has libobjc
        return False
    objc = ctypes.CDLL(library)
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    shared = ctypes.cast(
        objc.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    )
    application = shared(
        objc.objc_getClass(b"NSApplication"), objc.sel_registerName(b"sharedApplication")
    )
    if not application:  # pragma: no cover — no NSApp means no window either
        return False
    policy = ctypes.cast(
        objc.objc_msgSend,
        ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long),
    )
    regular, accessory = 0, 1
    return bool(
        policy(
            application,
            objc.sel_registerName(b"setActivationPolicy:"),
            regular if visible else accessory,
        )
    )


class Tray(QObject):
    """The status item: an icon that follows the meeting and a five-item menu.

    Every entry is a shortcut to something the window can already do, because
    the menu bar is a second front door to the same app, not a second app.
    """

    def __init__(self, shell: StenografGui) -> None:
        # Parented to the shell (itself owned by the application) for the same
        # reason the screens are: the tray must outlive the QML that it shows
        # and hides.
        super().__init__(shell)
        self.shell = shell
        self._state = ""  # forces the first _refresh through
        self._launched = True  # the startup activation is not a user asking for a window
        self._icons = {
            # macOS wants a template image, which it inverts per menu-bar
            # appearance; elsewhere a status icon is normally in colour, and
            # the asset's own two inks are already the app's.
            "idle": _icon("#000000", mask=True) if sys.platform == "darwin" else _icon(None),
            "rec": _icon(_TINT["rec"]),
            "busy": _icon(_TINT["busy"]),
        }

        self.menu = QMenu()
        self.status = self.menu.addAction("")
        self.status.setEnabled(False)
        self.menu.addSeparator()
        # Never disabled. It was, while the window was on screen — until Plasma's
        # own rendered menu showed it greyed out with the window merely *buried*
        # behind the video call (measured 2026-07-25): `isVisible()` means mapped,
        # not looked at, and the occluded window is the normal in-meeting shape
        # this whole mode is designed around. `show_window` raises and focuses
        # either way, so there is no state in which the entry has nothing to do.
        self.open_window = self._item("Open Stenograf", self._open)
        self.start = self._item("Start meeting…", self._start)
        self.stop = self._item("Stop && finalize", self._stop)  # && escapes the mnemonic
        self.folder = self._item("Open meetings folder", self._open_folder)
        self.menu.addSeparator()
        self.quit = self._item("Quit Stenograf", shell.quit_app)

        # Labelling happens when the menu opens, never while it is closed: the
        # meeting state changes several times a second under live captions.
        self.menu.aboutToShow.connect(self._relabel)

        meeting = self._meeting()
        meeting.changed.connect(self._refresh)
        shell.quitting.connect(self._refresh)
        self._show_icon(_status_icon(self))

        # From here on the window is closeable without ending the app — which
        # is the whole point, since a meeting outlives it.
        if shell.window is not None:
            shell.window.installEventFilter(self)
        if sys.platform == "darwin":
            # Re-opening the bundle is a macOS gesture with no counterpart
            # elsewhere: on Windows and Linux the status icon is the only way
            # back, and there an activation just means a window took focus.
            application = QApplication.instance()
            if application is not None:
                application.installEventFilter(self)

    def _show_icon(self, icon: StatusIcon) -> None:
        """Wire ``icon`` to the menu and the meeting, and put it on screen.

        A native icon the shell refuses is retried as Qt's, once. That refusal is
        real and reachable: ``NIM_ADD`` fails while Explorer is still starting,
        which is exactly when a login item runs. Ignoring it would be the worst
        outcome available, because everything downstream still believes there is
        a status item — :func:`install` returns a ``Tray``, ``run`` stops quitting
        on the last window, and :meth:`eventFilter` turns a close into a hide. In
        ``--tray`` mode, where no window is shown at all, the user would be left
        with an app that has neither a window nor an icon.
        """
        self.icon = icon
        icon.setContextMenu(self.menu)
        icon.activated.connect(self._clicked)
        self._state = ""  # a fresh icon carries no artwork yet
        self._refresh()
        # Only the native icon reports; QSystemTrayIcon.show() returns None, so
        # the fallback cannot recurse into itself.
        if icon.show() is False:
            logger.warning(
                "the Windows shell refused the Stenograf status icon — falling back to "
                "Qt's, which it may keep hidden in the overflow"
            )
            icon.deleteLater()
            self._show_icon(QSystemTrayIcon(self))

    def _item(self, text: str, handler: Callable[[], object]) -> QAction:
        action = self.menu.addAction(text)
        action.triggered.connect(handler)
        return action

    def _meeting(self) -> MeetingScreen:
        from stenograf.gui.meeting import MeetingScreen

        meeting = self.shell.screen("Meeting")
        assert isinstance(meeting, MeetingScreen)
        return meeting

    # -- what the meeting is doing -----------------------------------------

    def state(self) -> str:
        """``rec`` while capturing, ``busy`` while finishing, ``idle`` otherwise.

        ``phase`` alone will not do: it reads ``rec`` before any meeting has been
        started, and it reads ``done`` while the notes run is still going."""
        from stenograf.gui.meeting import Phase

        meeting = self._meeting()
        if not meeting.get("active"):
            return "idle"
        return "rec" if meeting.get("phase") == Phase.REC else "busy"

    def summary(self) -> str:
        """The disabled first menu entry: one line on what the app is doing."""
        if self.shell.quitting_now:
            return "Finishing the meeting before quitting…"
        state = self.state()
        if state == "rec":
            return f"Recording · {self._meeting().get('elapsed', '0:00')}"
        if state == "busy":
            return "Finishing the meeting…"
        return "No meeting running"

    # -- reacting ----------------------------------------------------------

    @Slot()
    def _refresh(self) -> None:
        """Follow the meeting's *state* — not its clock — into the icon."""
        state = self.state()
        if state == self._state:
            return
        finished = self._state in ("rec", "busy") and state == "idle"
        self._state = state
        self.icon.setIcon(self._icons[state])
        self.icon.setToolTip(f"Stenograf — {self.summary()}")
        if finished:
            self._announce()

    def _announce(self) -> None:
        """Say a meeting is over unless the user is already watching it end.

        "Already watching" is the *active* window, not a visible one. Qt's
        `isVisible()` is true for a window buried behind the video call — which
        is the shape the whole menu-bar mode assumes a meeting runs in — so
        keying off it would have suppressed this notification in precisely the
        common case and shown it only in the rare one. Measured through Plasma's
        rendered menu, which greyed *Open Stenograf* for an occluded window
        (2026-07-25). A focused window is already showing the finished
        transcript, and there the notification would be pure duplication."""
        window = self.shell.window
        if self.shell.quitting_now or (window is not None and window.isActive()):
            return
        if not QSystemTrayIcon.supportsMessages():
            return
        folder = str(self._meeting().get("folder", "")) or "your meetings folder"
        self.icon.showMessage("Meeting finished", folder, self._icons["idle"])

    @Slot()
    def _relabel(self) -> None:
        """Bring the menu up to date, called just before it is shown."""
        live = self.state() != "idle"
        self.status.setText(self.summary())
        self.start.setEnabled(not live and not self.shell.quitting_now)
        self.stop.setEnabled(self.state() == "rec")
        self.quit.setText(
            "Quit now (abandons the finalize)" if self.shell.quitting_now else "Quit Stenograf"
        )

    @Slot(QSystemTrayIcon.ActivationReason)
    def _clicked(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """A click on the icon opens the window — except on macOS.

        There the menu is what a click on a status item opens, and Qt emits
        ``Trigger`` alongside it; acting on that would raise the window every
        time someone reads the menu."""
        if sys.platform == "darwin":
            return
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._open()

    # -- the menu's five verbs ---------------------------------------------

    @Slot()
    def _open(self) -> None:
        self.shell.show_window()

    @Slot()
    def _start(self) -> None:
        """Straight to the setup form, wherever the window had been left."""
        self.shell.show_window()
        self.shell.reset_to("Setup")

    @Slot()
    def _stop(self) -> None:
        """Stop and finalize without opening anything — the reason this mode exists."""
        self._meeting().stop()

    @Slot()
    def _open_folder(self) -> None:
        """Hand the meetings folder to the file manager.

        The filesystem is this tool's index (the product scope forbids any
        browser of our own), so pointing at it is the whole feature."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from stenograf.flow import notes_home

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(notes_home())))

    @Slot()
    def _reopened(self) -> None:
        """macOS delivers a re-``open`` of a windowless app as an activation.

        Measured through the real bundle: double-clicking ``Stenograf.app`` while
        it sits in the menu bar does *not* start a second copy (LaunchServices
        still tracks the accessory app), but AppKit's default reopen handling has
        no window to order front, so without this the gesture does nothing at
        all. The tray menu would still work; a Mac user's first instinct is the
        icon they launched it from.

        The launch activation is skipped, and it is the reason this cannot just
        be "activate ⇒ show": ``--tray`` is activated once on startup like any
        other app, and obeying that would put a window on screen precisely where
        the user asked for none."""
        if self._launched:
            self._launched = False
            return
        window = self.shell.window
        if window is not None and not window.isVisible() and not self.shell.quitting_now:
            self.shell.show_window()

    # -- closing the window is not quitting --------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Turn the window's close button into "hide"; the meeting carries on."""
        if event.type() == QEvent.Type.Close and watched is self.shell.window:
            event.ignore()
            self.shell.hide_window()
            return True
        if event.type() == QEvent.Type.ApplicationActivate:
            self._reopened()
        return super().eventFilter(watched, event)


def install(shell: StenografGui) -> Tray | None:
    """Put the app in the menu bar, or report that this desktop has no room for it.

    ``None`` is a supported outcome, not a failure: stock GNOME ships no
    StatusNotifierItem host without the AppIndicator extension, and there the
    window stays the whole app."""
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    return Tray(shell)


__all__ = ["MARK", "Tray", "install", "set_dock_icon"]
