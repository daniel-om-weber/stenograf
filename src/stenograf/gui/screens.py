"""The five screens that are one library call each: setup, transcribe, notes, settings, doctor.

Each is a thirty-line controller over :mod:`stenograf.flow`, so they share
one file — splitting them would cost more import plumbing than it buys in
navigation. The meeting screen, which is not thirty lines, keeps its own
module.

The pattern every one of them follows: ``opened()`` refreshes whatever the
screen shows (these objects outlive their pages), a ``@Slot`` starts the work
through :meth:`~stenograf.gui.app.Screen.work`, and the outcome lands on
``state.status`` — the plain-text mirror the tests assert on.

**The notes screen stays a dumb folder picker** — never a meeting list with
titles, dates or summaries. That list would be the meeting browser the product
philosophy forbids; keeping the picker dumb enforces the lock structurally.
Folder, deliberately not file: the retired TUI's tree also accepted a bare
transcript JSON, and :func:`stenograf.flow.generate_notes_for` still handles
one, but that exotic shape is CLI territory (``steno notes PATH``) — a second
file dialog here would grow the picker for a case the folder covers in
practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Property, Slot

from stenograf.gui.app import Screen

if TYPE_CHECKING:
    from stenograf.gui.app import StenografGui

_AUTO = -1
"""Combo-box sentinel for "estimate the speaker count" (see the setup form)."""


def _url(path: Path) -> str:
    """A ``file:`` URL for a QML file dialog's starting folder (empty if absent)."""
    from PySide6.QtCore import QUrl

    return QUrl.fromLocalFile(str(path)).toString() if path.is_dir() else ""


def _local(url: str) -> Path:
    """The path a QML file dialog handed back (it deals in ``file:`` URLs)."""
    from PySide6.QtCore import QUrl

    return Path(QUrl(url).toLocalFile())


class SetupScreen(Screen):
    """The few choices that matter before capture starts.

    One concept per control: which sources to capture, whether to tell speakers
    apart (the per-channel counts only mean something while that is on),
    language, an optional title, the audio-recording opt-in and the notes
    toggle. Everything else — formats, vocabulary, re-ID, AEC, checkpoint
    cadence — comes from settings.toml exactly as it does for a flagless
    ``steno start``. The switches' standing defaults are refreshed on every
    visit, because settings.toml may have been edited between two of them."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self.opened()

    @Slot()
    def opened(self) -> None:
        from stenograf.flow import standing_settings

        standing = standing_settings()
        self.set(
            diarize=standing.speakers.diarization is True,
            recordAudio=standing.output.record_audio is True,
            notes=standing.notes.auto is True,
            error="",
        )

    @Property("QVariantList", constant=True)  # type: ignore[operator]  # PySide decorator
    def languages(self) -> list[dict[str, str]]:
        from stenograf.config import Language

        return [{"label": "Auto-detect", "value": "auto"}] + [
            {"label": language.name.title(), "value": language.value} for language in Language
        ]

    @Property("QVariantList", constant=True)  # type: ignore[operator]  # PySide decorator
    def counts(self) -> list[dict[str, object]]:
        # MeetingProfile caps counts at 8.
        return [{"label": "Auto-detect", "value": _AUTO}] + [
            {"label": str(n), "value": n} for n in range(1, 9)
        ]

    @Slot("QVariantMap")
    def start(self, form: dict) -> None:
        """Resolve the form into a request and hand it to the meeting screen.

        A form that cannot describe a meeting (both sources off, an unreadable
        settings file) stays open with the reason shown, through the same
        library call every front-end resolves with."""
        from stenograf.config import Language
        from stenograf.flow import MeetingRequestError, resolve_meeting_request

        def count(key: str) -> int | None:
            value = form.get(key)
            return None if value == _AUTO else int(value)  # type: ignore[arg-type]

        language = str(form.get("language", "auto"))
        try:
            request = resolve_meeting_request(
                mic=bool(form.get("mic")),
                system=bool(form.get("system")),
                diarize=bool(form.get("diarize")),
                local_speakers=count("local"),
                remote_speakers=count("remote"),
                language=None if language == "auto" else Language(language),
                title=str(form.get("title", "")),
                notes=bool(form.get("notes")),
                record_audio=bool(form.get("recordAudio")),
            )
        except MeetingRequestError as exc:
            self.set(error=str(exc))
            return
        self.set(error="")
        meeting = self.app.screen("Meeting")
        assert meeting is not None
        meeting.begin(request)  # type: ignore[attr-defined]  # the Meeting screen
        # Replace, not push: Back from the meeting belongs on Home, not on the
        # form that started it.
        self.app.replace("Meeting")


class TranscribeScreen(Screen):
    """Pick an audio (or video) file, run the finalize pipeline, watch it work.

    Leaving mid-run is refused: a worker thread cannot be interrupted safely,
    and letting it finish behind a closed page would surprise the user more."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self.set(
            file="",
            status="",
            busy=False,
            progress=0.0,
            home=_url(Path.home()),
        )

    @Slot(str)
    def choose(self, url: str) -> None:
        self.set(file=str(_local(url)), status="")

    @Slot()
    def start(self) -> None:
        from stenograf.flow import transcribe_recording

        path = Path(str(self._state.get("file", "")))
        if not path.is_file() or self._state.get("busy"):
            return
        self.set(busy=True, progress=0.0, status=f"transcribing {path.name}…")
        self.work(
            lambda: transcribe_recording(
                path,
                on_status=lambda message: self.post(self.set, status=message),
                on_windows=lambda done, total: self.post(self._windows, done, total),
            ),
            done=lambda result: self.set(busy=False, progress=1.0, status=result.summary()),
            failed=lambda message: self.set(busy=False, status=f"failed: {message}"),
            name="gui-transcribe",
        )

    def _windows(self, done: int, total: int) -> None:
        if done == 0:
            self.set(status=f"transcribing {total} windows…")
        self.set(progress=done / total if total else 0.0)


class NotesScreen(Screen):
    """Summarize a finished meeting: the newest one by default, or one you pick."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self.set(meeting="", status="", busy=False, home="")

    @Slot()
    def opened(self) -> None:
        """Pre-select the newest finished meeting (the ``notes --last`` default).

        Most notes runs happen right after the meeting they summarize, so the
        common case is one click; re-resolved per visit because the meeting that
        is newest changes while the app is open."""
        from stenograf.flow import notes_home
        from stenograf.output import latest_meeting_dir

        home = notes_home()
        latest = latest_meeting_dir(home)
        self.set(
            home=_url(home),
            meeting=str(latest) if latest is not None else "",
            status="" if latest is not None else f"No finished meeting in {home} yet.",
        )

    @Slot(str)
    def choose(self, url: str) -> None:
        self.set(meeting=str(_local(url)), status="")

    @Slot()
    def start(self) -> None:
        from stenograf.flow import generate_notes_for

        target = Path(str(self._state.get("meeting", "")))
        if not str(target) or self._state.get("busy"):
            return
        self.set(busy=True, status="generating notes…")
        self.work(
            lambda: generate_notes_for(
                target, on_progress=lambda message: self.post(self.set, status=f"notes: {message}")
            ),
            done=self._wrote,
            failed=lambda message: self.set(busy=False, status=f"failed: {message}"),
            name="gui-notes",
        )

    def _wrote(self, result: tuple) -> None:
        written, warnings = result
        status = f"wrote {', '.join(str(p) for p in written)}"
        if warnings:  # also recorded in the note's own footer
            status += f" — warning: {'; '.join(warnings)}"
        self.set(busy=False, status=status)


class SettingsScreen(Screen):
    """The effective configuration, read-only, plus a button that opens the file.

    A settings *form* is where UI effort balloons — explicitly out of scope. The
    file is the editor: *Open* hands settings.toml to whatever the desktop uses
    for it (creating the commented template first, exactly like ``steno settings
    edit``), and *Reload* re-reads it, so an edit is one alt-tab away."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self.lines: list[str] = []  # plain-text mirror (tests, debugging)
        self.set(text="", ok=True, path="")

    @Slot()
    def opened(self) -> None:
        from stenograf.flow import settings_report

        self.lines, ok = settings_report()
        self.set(text="\n".join(self.lines), ok=ok)

    @Slot()
    def edit(self) -> None:
        """Open settings.toml in the desktop's editor for ``.toml`` files."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from stenograf.settings import ensure_settings_file

        path, _created = ensure_settings_file()
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.set(text=f"Nothing on this machine opens .toml files — edit {path} directly.")


class DoctorScreen(Screen):
    """Machine-readiness report: permissions, OS version, helpers, models.

    The checks probe helpers, model caches and (on some platforms) spawn
    processes, so they run on a worker thread; ``run_checks`` returns a finished
    list, so the report renders in one go rather than streaming."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self.set(checks=[], status="", busy=False)

    @Slot()
    def opened(self) -> None:
        from stenograf import doctor

        if self._state.get("busy"):  # reopened before the last run reported back
            return
        self.set(checks=[], busy=True, status="running checks…")
        self.work(
            doctor.run_checks,
            done=self._report,
            failed=lambda message: self.set(busy=False, status=f"doctor failed: {message}"),
            name="gui-doctor",
        )

    def _report(self, checks: list) -> None:
        # ok → good, failed-but-optional → optional (an opt-in feature the
        # machine can healthily lack), failed → bad. QML colours from that.
        self.set(
            busy=False,
            checks=[
                {
                    "name": check.name,
                    "detail": check.detail,
                    "state": "good" if check.ok else ("optional" if check.optional else "bad"),
                }
                for check in checks
            ],
            status=(
                "Everything looks good."
                if (problems := sum(1 for c in checks if not (c.ok or c.optional))) == 0
                else f"{problems} problem(s) found — fix them and reopen this screen."
            ),
        )


__all__ = [
    "DoctorScreen",
    "NotesScreen",
    "SettingsScreen",
    "SetupScreen",
    "TranscribeScreen",
]
