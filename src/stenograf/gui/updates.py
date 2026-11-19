"""The app's side of updating: one line on the home screen, one menu-bar entry.

Everything it does is a call into :mod:`stenograf.update`; what lives here is
only *when* — and the rule for that is that an update never gets in the way of
a meeting:

- nothing is shown, started or continued while a meeting runs, and a prepare
  in progress is cancelled when one starts (it costs CPU and network);
- the restart only happens with the app idle, on the click that asked for it —
  never as a side effect of quitting, since a logout would kill the swap
  halfway;
- there is no dialog and no notification, only the line.

The hourly timer is the app's second periodic timer besides the meeting clock.
It repaints nothing: the check behind it runs once a day, on a worker thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, QUrl, Slot
from PySide6.QtGui import QDesktopServices

from stenograf import __version__, update
from stenograf.gui.app import Screen

if TYPE_CHECKING:
    from pathlib import Path

    from stenograf.gui.app import StenografGui

_LOOK_EVERY_MS = 60 * 60 * 1000


class UpdateScreen(Screen):
    """Not a page: the object behind the update line, the tray entry and the
    Settings row. ``phase`` says which of them has something to show:

    ``none``, ``available`` (a click installs it), ``notice`` (this install
    cannot take it by itself; ``detail`` says why or how), ``preparing``,
    ``ready`` (prepared, waiting for an idle app and a click), ``restarting``,
    and ``updated`` / ``failed`` (the outcome of the last update, shown once)."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self._offer: update.Offer | None = None
        self._dismissed = ""
        self._prepared: Path | None = None
        self._cancel = threading.Event()
        self._tray = False
        self._timer: QTimer | None = None
        self.set(
            phase="none",
            version="",
            text="",
            detail="",
            quiet=False,
            summary=f"stenograf {__version__}",
        )

    def begin(self, *, tray: bool) -> None:
        """Called once the window is built: report, then start looking."""
        self._tray = tray
        update.mark_started()
        report = update.settle()
        if report is not None:
            self.set(
                phase="updated" if report.updated else "failed",
                version=report.version,
                text=report.message,
                detail="",
            )
        self.app._meeting().changed.connect(self._meeting_changed)
        self._look(force=False)
        self._timer = QTimer(self)
        self._timer.setInterval(_LOOK_EVERY_MS)
        self._timer.timeout.connect(lambda: self._look(force=False))
        self._timer.start()

    # -- looking ------------------------------------------------------------

    def _look(self, *, force: bool) -> None:
        def job() -> tuple[bool, update.Offer | None]:
            if not update.enabled():
                return False, None
            update.check(force=force)
            return True, update.offer(include_failed=force)

        self.work(job, done=self._looked, name="gui-update-check")

    def _looked(self, outcome: tuple[bool, update.Offer | None]) -> None:
        checking, found = outcome
        self._offer = found
        if not checking:
            summary = f"stenograf {__version__} — the update check is switched off"
        elif found is None:
            summary = f"stenograf {__version__} — no newer release known"
        else:
            summary = f"stenograf {__version__} — {found.version} is available"
        self.set(summary=summary)
        if self.get("phase") in ("none", "available", "notice"):
            self._show_offer()

    def _show_offer(self) -> None:
        found = self._offer
        if found is None or found.version == self._dismissed:
            self.set(phase="none", version="", text="", detail="")
            return
        self.set(
            phase="available" if found.installable else "notice",
            version=found.version,
            text=f"Stenograf {found.version} is available",
            detail="" if found.installable else (found.command or found.reason),
        )

    # -- intents ------------------------------------------------------------

    @Slot()
    def check_now(self) -> None:
        """The Settings row's button: ask again, and offer a failed version again."""
        self._dismissed = ""
        self.set(summary=f"stenograf {__version__} — checking…")
        self._look(force=True)

    @Slot()
    def install(self) -> None:
        """Prepare the offered version, then restart into it if the app is idle."""
        found = self._offer
        if found is None or not found.installable or self.get("phase") != "available":
            return
        if not self.idle():
            self.set(detail="Finish what is running first.")
            return
        self._cancel.clear()
        self._prepared = None  # an earlier attempt's folder is not this one's
        self.set(phase="preparing", text=f"Preparing the update to {found.version}…", detail="")
        version = found.version
        self.work(
            lambda: update.prepare(version, cancelled=self._cancel.is_set),
            done=self._prepared_ok,
            failed=self._prepare_failed,
            name="gui-update-prepare",
        )

    def _prepared_ok(self, folder: Path) -> None:
        if self._cancel.is_set():
            # Cancelled in the moment the trial run finished: still cancelled.
            self._show_offer()
            return
        self._prepared = folder
        if self.idle():
            self.restart()
        else:
            self.set(phase="ready", text=f"Stenograf {self.get('version')} is ready", detail="")

    def _prepare_failed(self, message: str) -> None:
        if self._cancel.is_set():
            self._show_offer()
            return
        self.set(phase="failed", text="The update could not be prepared.", detail=message)

    @Slot()
    def cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def restart(self) -> None:
        """Hand over to the updater and quit. Only on a click, only when idle."""
        if self._prepared is None:
            return
        if not self.idle():
            self.set(phase="ready", detail="Finish what is running first.")
            return
        try:
            update.hand_over(
                str(self.get("version")),
                self._prepared,
                restart=update.restart_command(tray=self._tray),
                wait_pids=update.launch_chain(),
                launcher_files=update.launcher_files(),
            )
        except update.UpdateError as exc:
            self.set(phase="failed", text="The update could not be started.", detail=str(exc))
            return
        self.set(phase="restarting", text="Restarting to update…", detail="")
        self.app.quit_app()

    @Slot()
    def dismiss(self) -> None:
        """Put the line away: a report for good, an offer until the next release."""
        if self.get("phase") in ("available", "notice"):
            self._dismissed = str(self.get("version"))
        self.set(phase="none")
        self._show_offer()

    @Slot()
    def open_notes(self) -> None:
        version = str(self.get("version"))
        if version:
            QDesktopServices.openUrl(QUrl(update.RELEASE_URL.format(version=version)))

    # -- staying out of a meeting's way ---------------------------------------

    def idle(self) -> bool:
        """No meeting, no transcription, no notes run."""
        if self.app._meeting().get("active"):
            return False
        return not any(
            screen is not None and screen.get("busy")
            for screen in (self.app.screen("Transcribe"), self.app.screen("Notes"))
        )

    def _meeting_changed(self) -> None:
        # The meeting's state changes several times a second under live
        # captions; only the edge is worth a repaint of the home screen.
        quiet = bool(self.app._meeting().get("active"))
        if quiet == self.get("quiet"):
            return
        self.set(quiet=quiet)
        if quiet and self.get("phase") == "preparing" and self._prepared is None:
            self.cancel()
