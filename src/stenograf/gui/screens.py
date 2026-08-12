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
Folder, deliberately not file: :func:`stenograf.flow.generate_notes_for` also
handles a bare transcript JSON, but that exotic shape is CLI territory
(``steno notes PATH``) — a second file dialog here would grow the picker for a
case the folder covers in practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Property, Slot

from stenograf.gui.app import Screen

if TYPE_CHECKING:
    from stenograf.gui.app import StenografGui
    from stenograf.settings import Settings

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


def _device_entries(devices: list, pinned: str | None) -> list[dict[str, str]]:
    """The microphone picker's model, **with the entry to select first**.

    Nothing in the app writes a combo's ``currentIndex``: every picker relies on
    index 0 being the right starting point, and a model that arrives after the
    first paint resets the index to 0 anyway. Ordering the list here is
    therefore the whole selection mechanism — and it keeps the choice in one
    place instead of splitting it between Python and a QML binding.

    A configured device the run could not use still comes first, spelled out as
    such: the form must never quietly show "System default" for a run that would
    refuse to start.

    The default row's value is the literal that *clears* a standing selection
    (:data:`stenograf.flow.DEFAULT_MIC_DEVICE`), not the empty string — empty
    means "this form chose nothing", which is what a machine with no picker
    sends, and that must keep the standing selection rather than override it.
    """
    from stenograf.capture.helper import match_input_device
    from stenograf.flow import DEFAULT_MIC_DEVICE

    default = next((device.name for device in devices if device.is_default), "")
    rows = [
        {
            "label": f"System default ({default or 'none configured'})",
            "value": DEFAULT_MIC_DEVICE,
        }
    ]
    rows += [
        {
            "label": f"{device.name} (default)" if device.is_default else device.name,
            "value": device.id,
        }
        for device in devices
    ]
    if not pinned:
        return rows
    chosen, problem = match_input_device(devices, pinned)
    if chosen is None:
        return [{"label": f"{pinned} — {problem}", "value": pinned}, *rows]
    rows.sort(key=lambda row: row["value"] != chosen.id)  # stable: the pin moves to the front
    return rows


class SetupScreen(Screen):
    """The few choices that matter before capture starts.

    One concept per control: the meeting type, which sources to capture,
    whether to tell speakers apart (the per-channel counts only mean something
    while that is on), language, an optional title, the audio-recording opt-in
    and the notes toggle. Everything else — formats, vocabulary, re-ID, AEC,
    checkpoint cadence — comes from settings.toml exactly as it does for a
    flagless ``steno start``. The switches' standing defaults are refreshed on
    every visit, because settings.toml may have been edited between two of
    them — and so is the preset list, for the same reason."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self._generation = 0
        self.set(micDevices=[], micDevicesReady=False, micError="")
        self.opened()

    @Slot()
    def opened(self) -> None:
        from stenograf.flow import resolve_mic_device, standing_settings

        standing = standing_settings()
        # Through the same resolution the run uses, so the picker cannot
        # describe a selection differently from the meeting it starts.
        pinned = resolve_mic_device(None, standing)
        self.set(
            diarize=standing.speakers.diarization is True,
            recordAudio=standing.output.record_audio is True,
            notes=standing.notes.auto is True,
            presets=self._presets(standing),
            error="",
        )
        self._list_microphones(pinned)

    def _list_microphones(self, pinned: str | None) -> None:
        """Fill the microphone picker, off the GUI thread.

        Listing spawns the capture helper, and ``opened()`` runs on the event
        loop — a subprocess there would freeze the window on any machine whose
        sound server is slow to answer.

        The screen outlives its page, and ``opened()`` fires twice on the first
        visit (construction, then the page's own call) and again on every later
        one — deliberately, since a device may have been plugged in since. Only
        the newest answer is kept: an earlier listing that lands late (a slow
        sound server, or settings edited between two visits) is dropped rather
        than allowed to overwrite it."""
        from stenograf.capture.helper import list_input_devices

        self._generation += 1
        token = self._generation
        # Hidden until the new list lands: swapping a combo's model resets its
        # selection, and doing that under a visible control could discard a
        # choice the user had just made.
        self.set(micDevicesReady=False)
        self.work(
            list_input_devices,
            done=lambda devices: self._microphones(token, devices, pinned),
            failed=lambda message: self._no_microphones(token, message),
            name="gui-devices",
        )

    def _microphones(self, token: int, devices: list, pinned: str | None) -> None:
        if token != self._generation:
            return
        self.set(
            micDevices=_device_entries(devices, pinned),
            micDevicesReady=True,
            # A machine with nothing to record from says so: the picker has
            # nothing to show, and the failure would otherwise arrive as a
            # capture error seconds after Start.
            micError="" if devices else "No microphone is connected.",
        )

    def _no_microphones(self, token: int, message: str) -> None:
        """A listing that failed leaves the picker hidden and says why.

        Start still works: with no entry chosen the run follows the same device
        it would have without this control, and whatever is wrong with the
        capture stack will fail the meeting loudly a moment later."""
        if token != self._generation:
            return
        self.set(
            micDevices=[], micDevicesReady=False, micError=f"Microphones unavailable: {message}"
        )

    @staticmethod
    def _presets(standing: Settings) -> list[dict[str, str]]:
        """The meeting types on offer: ``[meetings.<name>]``, "None" first.

        ``""`` is the no-preset value, so a form that never touches the picker
        sends exactly what it sent before presets existed. Each entry carries
        the preset's own one-line summary (:meth:`MeetingPreset.summary`, the
        line ``steno presets`` prints) — the picker shows it under the
        drop-down, so choosing a type says what the type does. A machine with
        no presets, or an unreadable settings.toml, yields the "None" entry
        alone and the QML hides the control."""
        return [{"label": "None", "value": "", "hint": ""}] + [
            {"label": name, "value": name, "hint": standing.meetings[name].summary()}
            for name in sorted(standing.meetings)
        ]

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
        settings file, a preset settings.toml no longer defines) stays open with
        the reason shown, through the same library call every front-end resolves
        with."""
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
                # "" is the picker's no-preset entry; the library takes None.
                preset=str(form.get("preset", "")) or None,
                # "" is the picker's "system default" entry, and also what a
                # form sends when the listing failed and no picker was shown.
                mic_device=str(form.get("micDevice", "")) or None,
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
    edit``), and *Reload* re-reads it, so an edit is one alt-tab away.

    The meeting-type picker is the one control here, and it is still not a form:
    it re-renders the same read-only report with a ``[meetings.<name>]`` overlay
    applied, which is the only way to answer "what does this type actually
    change?" — a preset's overlay is sparse, so reading its section tells you
    what it sets but not what wins."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self.lines: list[str] = []  # plain-text mirror (tests, debugging)
        self.set(text="", ok=True, path="", preset="", presets=[])

    @Slot()
    def opened(self) -> None:
        from stenograf.flow import standing_settings

        # The list is refreshed first and the selection dropped if the file no
        # longer defines it, so an edit between two visits cannot leave the
        # picker showing a name that would now render as an error.
        presets = SetupScreen._presets(standing_settings())
        chosen = str(self.get("preset", ""))
        if chosen and all(entry["value"] != chosen for entry in presets):
            chosen = ""
        self.set(presets=presets)
        self.show(chosen)

    @Slot(str)
    def show(self, preset: str) -> None:
        """Render the report, optionally under a meeting preset (``""`` = none)."""
        from stenograf.flow import settings_report

        self.lines, ok = settings_report(preset or None)
        self.set(preset=preset, text="\n".join(self.lines), ok=ok)

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
