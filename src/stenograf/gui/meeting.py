"""The live meeting screen: a :class:`~stenograf.view.LiveView` that is also a QObject.

The only screen with real machinery. The meeting itself is
:class:`stenograf.flow.MeetingRun` on a worker thread; this class is the sink
for the events it emits, marshalling every one of them onto the GUI thread
before it touches a property (Qt objects belong to one thread, and the live
worker calls a view from its own).

Being both is deliberate: the events map one-to-one onto display state, so an
adapter object between the two would be pure indirection.

What the screen shows, and where it comes from:

- **committed captions** — appended through the ``committed`` signal into a QML
  ``ListModel``, never through the state map: a growing transcript must not
  re-evaluate every binding on the screen for each new line;
- **the interim area** — the open (bright) line plus the grey provisional tail,
  per channel, exactly as :class:`~stenograf.captions.CaptionStream` decides;
- **the header** — phase, elapsed, language, profile; the 1 Hz clock that
  advances it is the app's only periodic timer, and it also flushes an idle
  caption line — together the screen's whole redraw budget;
- **the footer** — the transient status line (model loading, notes progress)
  and, once allocated, the meeting folder, which a click opens in the file
  manager. They are separate fields on purpose: the notes step overwrites the
  status, and the folder must survive that. Once the transcript is in, a
  button copies it to the clipboard;
- **"who spoke?"** — after the finalize, one row per nameable speaker
  (:func:`stenograf.flow.meeting_speakers`). The run's notes wait until the
  names are saved or skipped, and leaving the screen counts as skipping:
  notes stranded behind a screen nobody returns to would be lost work. With
  ``[speakers] names_panel_timeout_s`` the panel skips itself after that long
  untouched — a second 1 Hz timer, alive only while that countdown runs.
  Each row plays the speaker's voice samples (:mod:`stenograf.playback`),
  which live in the run's memory and are dropped when the screen is left.

Stopping crosses back to capture through the callback the run hands over
(:meth:`set_stop`) — and it *blocks*, up to ~5 s waiting on the capture
subprocess to flush and exit, so it runs on its own thread. Doing it inline
would freeze the window for those seconds.
"""

from __future__ import annotations

import contextlib
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, Signal, Slot

from stenograf.captions import LIVE_LABEL, CaptionStream
from stenograf.gui.app import Screen, copy_transcript, open_folder
from stenograf.transcript import Transcript, format_timestamp
from stenograf.view import LiveView, profile_label

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from stenograf.asr.base import Word
    from stenograf.capture.base import Channel
    from stenograf.config import Language
    from stenograf.flow import MeetingRequest, MeetingRun, NamingOutcome, SpeakerSummary
    from stenograf.gui.app import StenografGui
    from stenograf.playback import Playback

class Phase(StrEnum):
    """Screen lifecycle: capture running → on-stop pass running → transcript shown
    (or the run never got off the ground). QML colours the dot from the value."""

    REC = "rec"
    FINALIZING = "finalizing"
    DONE = "done"
    FAILED = "failed"

    @property
    def label(self) -> str:
        """The header badge text (the value, shouted)."""
        return self.name


class MeetingScreen(Screen, LiveView):
    """One meeting, from the Start button to the finalized transcript on screen."""

    committed = Signal(str, str, str)
    """``(channel, speaker, text)`` — one finished caption line, appended to the
    log. ``channel`` is the :class:`~stenograf.capture.base.Channel` value
    (``"mic"``/``"system"``, empty for diarized entries) so QML colours rows by
    identity, not by comparing against the display label."""

    restored = Signal(list)
    """The finalize swap: the whole diarized transcript replaces the live captions."""

    cleared = Signal()
    """Drop everything on screen (a new meeting reuses this long-lived object)."""

    def __init__(self, app: StenografGui) -> None:
        super().__init__(app)
        self._captions = CaptionStream(self._emit_line)
        self._clock = QTimer(self)  # the ONLY periodic repaint in the app (1 Hz)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        self._names_clock = QTimer(self)  # the panel's countdown; idle otherwise
        self._names_clock.setInterval(1000)
        self._names_clock.timeout.connect(self._names_tick)

        self._request: MeetingRequest | None = None
        self._persist: Callable[[Transcript], object] | None = None
        self._stop_capture: Callable[[], None] | None = None
        self._thread: threading.Thread | None = None
        self._run: MeetingRun | None = None
        self._abort = threading.Event()
        """Cancels a run whose capture is still being built (`MeetingRun.abort`).

        Owned here rather than by the run so it exists before the worker
        thread does — Stop can land in the first milliseconds."""
        self._started = 0.0
        self._playback: Playback | None = None
        self._reset()

    def _set_phase(self, phase: Phase, **state: object) -> None:
        """One write for the pair QML reads: the phase id and its badge text."""
        self.set(phase=phase, phaseLabel=phase.label, **state)

    def _reset(self) -> None:
        self._names_clock.stop()
        self._stop_playback()
        self._set_phase(
            Phase.REC,
            active=False,
            elapsed="0:00",
            language="—",
            profile="",
            status="",
            folder="",
            transcriptReady=False,
            tails=[],
            canStop=False,
            speakers=[],
            knownNames="",
            namesOpen=False,
            namesBusy=False,
            namesCountdown=0,
            playing="",
            clipNext={},
        )

    # -- lifecycle ---------------------------------------------------------

    def begin(self, request: MeetingRequest) -> None:
        """Arm the next meeting; :meth:`opened` starts it once the page is up.

        Called by the setup form before it navigates here, so the header already
        describes the meeting while capture is still starting."""
        self._request = request
        self._reset()
        self.set(
            profile=profile_label(request.profile),
            language=request.profile.language.value if request.profile.language else "—",
        )

    @Slot()
    def opened(self) -> None:
        """The page is on screen: start the armed meeting (once).

        Starting here rather than at :meth:`begin` keeps the user watching
        the loading status in the header instead of a frozen window while the
        models come up."""
        request, self._request = self._request, None
        if request is None:  # re-entered the page without a new Start
            return
        self.cleared.emit()
        self._captions.clear()
        self._abort.clear()
        self._started = time.monotonic()
        self._clock.start()
        # Announced rather than derived: `phase` reads "rec" before anything has
        # started and "done" while the notes tail is still running, so the menu
        # bar (gui/tray.py) cannot tell from it whether a meeting is under way.
        self.set(active=True)
        self._thread = self.work(
            lambda: self._meeting(request),
            done=self._finished,
            failed=self._failed,
            name="gui-meeting",
        )

    def _meeting(self, request: MeetingRequest) -> Transcript | None:
        """The meeting, on the worker thread."""
        from stenograf.flow import MeetingRun

        run = MeetingRun(request, abort=self._abort, ask_names=True)
        # The transcript reaches disk at the ``finalized`` event, before the user
        # can close the window on the done screen.
        self._persist = run.persist
        self._run = run  # shutdown() needs its abandon-notes handle
        self.post(self.set, folder=str(run.out_dir))
        return run.run(self)

    def _finished(self, transcript: Transcript | None) -> None:
        """The meeting thread returned (capture stopped, finalize and notes done)."""
        self._clock.stop()
        self.set(active=False)
        if self._abort.is_set() and transcript is None:
            # The user cancelled before capture existed: nothing happened, so
            # there is nothing to show — back to the menu, like the Back button.
            self._reset()
            self.app.back()
            return
        if isinstance(transcript, Transcript):
            if self._state.get("phase") != Phase.DONE:  # a run that never emitted the event
                self._show(transcript)
            return
        self._set_phase(
            Phase.DONE,
            canStop=False,
            tails=[],
            status="the meeting ended before a transcript was produced; "
            "any .partial checkpoint is kept",
        )

    def _failed(self, message: str) -> None:
        self._clock.stop()
        self._set_phase(
            Phase.FAILED,
            active=False,
            canStop=False,
            status=message,
        )

    @property
    def running(self) -> bool:
        """Whether the meeting thread is still working (capture, finalize, notes)."""
        return self._thread is not None and self._thread.is_alive()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def shutdown(self) -> None:
        """The window closed with the meeting still going: end it, don't abandon it.

        Closing the window is the GUI's force-quit, and unlike the TUI's it can
        happen mid-capture — where a plain join would wait forever on a meeting
        nothing will ever stop. So stop capture first (the same callback the Stop
        button uses), then wait for what losing would actually cost: capture
        teardown and the finalize, which put the transcript on disk.

        The notes tail is the exception. It is regenerable (``steno notes
        --last``), the worker is a daemon, and an agentic ``[notes] command``
        backend legitimately runs for many minutes — joining it would block the
        quit for up to ``[notes] timeout_s`` with the window already gone. So a
        notes step that has not started is skipped (``abandon_notes``), and one
        already in flight is left to die with the process.

        A run whose capture never came up is cancelled outright (``abort``):
        no audio exists to protect, so the quit waits only a bounded moment
        for the worker to notice instead of joining a construction that may be
        wedged inside CoreAudio."""
        self._stop_playback()
        run = self._run
        if run is not None:
            run.abandon_notes.set()  # a notes step not yet started is skipped
        if self._stop_capture is None and self._state.get("phase") == Phase.REC:
            # Capture never came up: no audio exists, so there is nothing the
            # wait below could save. Cancel the construction and give the
            # worker a moment — a provider build wedged inside CoreAudio may
            # never return, and joining it would hang the quit for nothing.
            self._abort.set()
            thread = self._thread
            if thread is not None:
                thread.join(5.0)
            return
        if self._stop_capture is not None and self._state.get("phase") == Phase.REC:
            with contextlib.suppress(Exception):  # a stop error must not block the exit
                self._stop_capture()
        # abandon_notes is set, so notes_running can only be True here if notes
        # were already in flight before this shutdown began — never rechecked
        # into True during the wait.
        thread = self._thread
        while (
            thread is not None and thread.is_alive() and not (run is not None and run.notes_running)
        ):
            thread.join(0.2)

    # -- the 1 Hz tick -----------------------------------------------------

    def _tick(self) -> None:
        self.set(elapsed=format_timestamp(time.monotonic() - self._started))
        if self._captions.flush_if_idle():  # a stretch of speech ended
            self._render_tails()

    # -- intents -----------------------------------------------------------

    @Slot()
    def stop(self) -> None:
        """Stop & finalize while capturing; cancel while starting; leave when done."""
        phase = self._state.get("phase")
        if phase == Phase.REC and self._stop_capture is not None:
            self._set_phase(Phase.FINALIZING, canStop=False)
            threading.Thread(target=self._invoke_stop, name="gui-stop", daemon=True).start()
        elif phase == Phase.REC and self.running:
            # Capture is not up yet (the provider is still being built — exactly
            # where a wedged CoreAudio hangs, and when a user reaches for
            # Escape). Nothing is installed to stop, so cancel the run instead:
            # the worker checks the flag around construction and goes home, and
            # nothing is lost because no audio has been captured. A stop
            # callback landing concurrently is benign — the flag is ignored
            # once capture started, and the next press takes the branch above.
            self._abort.set()
            self.set(status="cancelling…")
        elif phase in (Phase.DONE, Phase.FAILED):
            self._stop_playback()
            if self._run is not None:
                self._run.settle_names()  # leaving unanswered = skipping
                self._run.discard_voice_clips()
            self.set(speakers=[], namesOpen=False)
            self.app.back()

    @Slot(str)
    def play(self, label: str) -> None:
        """Play ``label``'s next voice sample, or stop the one playing."""
        from stenograf import playback

        self.hold_names()
        playing = self._state.get("playing")
        self._stop_playback()
        if playing == label or self._run is None:
            return
        clips = self._run.voice_clips.get(label, [])
        if not clips:
            return
        saved = self._state.get("clipNext")
        cursor = dict(saved) if isinstance(saved, dict) else {}
        index = int(cursor.get(label, 0)) % len(clips)
        cursor[label] = index + 1
        self._playback = current = playback.Playback(
            clips[index], lambda error: self.post(self._played, current, error)
        )
        self.set(playing=label, clipNext=cursor)
        current.start()

    def _played(self, finished: Playback, error: str | None) -> None:
        if finished is not self._playback:  # a newer sample took over
            return
        self._playback = None
        self.set(playing="")
        if error:
            self.set(status=f"voice sample not played: {error}")

    def _stop_playback(self) -> None:
        current, self._playback = self._playback, None
        if current is not None:
            current.stop()
        if self._state.get("playing"):
            self.set(playing="")

    @Slot("QVariantMap", "QVariantList")
    def save_names(self, names: dict, forced: list) -> None:
        """Apply the typed names (``label → name``); ``forced`` labels join a
        profile whose saved voice they do not match. The panel closes and the
        notes start once every answer landed; a refused voice keeps it open."""
        run = self._run
        if run is None or self._state.get("namesBusy"):
            return
        self.hold_names()
        self.set(namesBusy=True)

        def job() -> tuple[list[NamingOutcome], list[SpeakerSummary], Transcript | None]:
            from stenograf import output
            from stenograf.flow import meeting_speakers

            outcomes = run.name_speakers(
                {str(k): str(v) for k, v in names.items()}, [str(f) for f in forced]
            )
            try:
                transcript = output.load_transcript(run.out_dir)[0]
            except ValueError:
                transcript = None
            return outcomes, meeting_speakers(run.out_dir, store_path=run.profile_store), transcript

        self.work(job, done=self._names_saved, failed=self._names_failed, name="gui-names")

    @Slot()
    def skip_names(self) -> None:
        """Close the panel unanswered; waiting notes start with the labels as they are."""
        self.hold_names()
        self.set(namesOpen=False)
        if self._run is not None:
            self._run.settle_names()

    @Slot()
    def hold_names(self) -> None:
        """The user is answering: stop the auto-close countdown for good."""
        self._names_clock.stop()
        self.set(namesCountdown=0)

    def _names_tick(self) -> None:
        current = self._state.get("namesCountdown", 0)
        left = (current if isinstance(current, int) else 0) - 1
        if left > 0:
            self.set(namesCountdown=left)
        else:
            self.skip_names()

    @Slot()
    def reveal(self) -> None:
        """Show the meeting's folder in the desktop's file manager."""
        folder = str(self._state.get("folder", ""))
        if folder and (problem := open_folder(Path(folder))):
            self.set(status=problem)

    @Slot(result=bool)
    def copy(self) -> bool:
        """Copy the finished transcript, names included; False (and why, on the
        status line) when there is none to copy."""
        folder = str(self._state.get("folder", ""))
        problem = copy_transcript(Path(folder)) if folder else "no meeting folder"
        if problem:
            self.set(status=f"transcript not copied: {problem}")
        return problem is None

    @Slot()
    def open_names(self) -> None:
        """Bring the panel back after it closed (names can be fixed any time)."""
        self.set(namesOpen=True)

    def _names_saved(
        self,
        result: tuple[list[NamingOutcome], list[SpeakerSummary], Transcript | None],
    ) -> None:
        outcomes, speakers, transcript = result
        if transcript is not None:
            self._restore(transcript)
        refused = {o.label: o for o in outcomes if o.refused_similarity is not None}
        rows = _speaker_rows(speakers, self._clip_counts())
        # A refused voice keeps its row (the label is unchanged) and the answer
        # the user typed, so "Add anyway" or a corrected name is one click away.
        for row in rows:
            outcome = refused.get(str(row["label"]))
            if outcome is not None:
                row["name"] = outcome.name
                row["refused"] = (
                    f"This voice doesn't sound like {outcome.name}'s saved voice "
                    f"(similarity {outcome.refused_similarity:.2f}). Add it to "
                    f"{outcome.name} anyway, or type another name."
                )
        errors = [f"{o.label}: {o.error}" for o in outcomes if o.error is not None]
        self.set(speakers=rows, namesBusy=False)
        if errors:
            self.set(status="; ".join(errors))
        if not refused and not errors:
            named = [o for o in outcomes if o.applied]
            if named:
                self.set(status=f"named {len(named)} speaker{'' if len(named) == 1 else 's'}")
            self.skip_names()

    def _names_failed(self, message: str) -> None:
        self.set(namesBusy=False, status=f"names not saved: {message}")

    def _invoke_stop(self) -> None:
        """Run the blocking capture teardown off the GUI thread (see the module docstring)."""
        try:
            self._stop_capture()  # type: ignore[misc]  # guarded by stop()
        except Exception as exc:  # noqa: BLE001 — a stop error must not wedge the UI
            self.post(self.set, status=f"stop failed: {exc}")

    # -- LiveView events (worker thread) → GUI thread ----------------------

    def set_stop(self, stop: Callable[[], None]) -> None:
        self._stop_capture = stop
        self.post(self.set, canStop=True)

    def commit(self, channel: Channel, words: Sequence[Word]) -> None:
        self.post(self._commit, channel, tuple(words))

    def interim(self, channel: Channel, text: str) -> None:
        self.post(self._interim, channel, text)

    def status(self, message: str) -> None:
        self.post(self.set, status=message)

    def language(self, language: Language) -> None:
        self.post(self.set, language=language.value)

    def finalizing(self) -> None:
        self.post(self._set_phase, Phase.FINALIZING)

    def finalized(self, transcript: Transcript) -> None:
        if self._persist is not None:
            try:
                self._persist(transcript)
            except Exception as exc:  # noqa: BLE001 — persistence must not sink the result
                self.error(f"could not write the transcript yet ({exc}); retrying on exit")
        self.post(self._show, transcript)

    def error(self, message: str) -> None:
        self.post(self.set, status=message)

    def speakers(self, speakers: list[SpeakerSummary], *, ask: bool) -> None:
        from stenograf.flow import known_speaker_names

        run = self._run
        known = (
            known_speaker_names(
                store_path=run.profile_store,
                attendee_names=run.request.profile.attendee_names,
            )
            if run is not None
            else []
        )
        self.post(
            self._offer_names, _speaker_rows(speakers, self._clip_counts()), ", ".join(known), ask
        )

    def _clip_counts(self) -> dict[str, int]:
        run = self._run
        return {label: len(c) for label, c in run.voice_clips.items()} if run else {}

    def _offer_names(self, rows: list[dict[str, object]], known: str, ask: bool) -> None:
        self.set(speakers=rows, knownNames=known, namesOpen=ask, clipNext={})
        timeout = self._run.request.settings.speakers.names_panel_timeout_s if self._run else None
        if ask and timeout:
            self.set(namesCountdown=max(1, round(timeout)))
            self._names_clock.start()

    # -- GUI-thread renderers ----------------------------------------------

    def _commit(self, channel: Channel, words: Sequence[Word]) -> None:
        self._captions.commit(channel, words)
        self._render_tails()

    def _interim(self, channel: Channel, text: str) -> None:
        self._captions.interim(channel, text)
        self._render_tails()

    def _emit_line(self, channel: Channel, text: str) -> None:
        """A line the caption stream finished — append it to the log."""
        self.committed.emit(channel.value, LIVE_LABEL[channel], text)

    def _render_tails(self) -> None:
        self.set(
            tails=[
                {
                    "channel": channel.value,
                    "speaker": LIVE_LABEL[channel],
                    "open": open_text,
                    "tail": tail,
                }
                for channel, open_text, tail in self._captions.tails()
            ]
        )

    def _show(self, transcript: Transcript) -> None:
        """Swap the live captions for the authoritative, diarized transcript."""
        self._captions.clear()
        self._clock.stop()
        self._restore(transcript)
        count = len(transcript.entries)
        self._set_phase(
            Phase.DONE,
            canStop=False,
            tails=[],
            transcriptReady=True,
            status=f"{count} {'entry' if count == 1 else 'entries'}",
            language=(
                transcript.language.value
                if transcript.language is not None
                else self._state.get("language", "—")
            ),
        )

    def _restore(self, transcript: Transcript) -> None:
        """Put ``transcript``'s entries on screen in place of whatever is there."""
        self.restored.emit(
            [
                {
                    "speaker": entry.speaker,
                    "time": format_timestamp(entry.start),
                    "text": entry.text,
                    "provisional": entry.provisional,
                }
                for entry in transcript.entries
            ]
        )


def _speaker_rows(
    speakers: Sequence[SpeakerSummary], clips: Mapping[str, int]
) -> list[dict[str, object]]:
    """The panel's rows, as QML reads them."""
    return [
        {
            "label": s.label,
            "recognised": s.recognised,
            "spoke": _spoke(s.seconds),
            "quotes": [
                {"text": f"“{quote}”", "line": line}
                for quote, line in zip(s.quotes, s.quote_lines, strict=True)
            ],
            "clips": clips.get(s.label, 0),
            "name": "",
            "refused": "",
        }
        for s in speakers
    ]


def _spoke(seconds: float) -> str:
    return f"{round(seconds)} s" if seconds < 60 else f"{round(seconds / 60)} min"


__all__ = ["MeetingScreen", "Phase"]
