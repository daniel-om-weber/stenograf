"""The live meeting screen: a :class:`~stenograf.view.LiveView` that is also a QObject.

The Qt counterpart of :mod:`stenograf.ui.meeting`, and the only screen with real
machinery. The meeting itself is :class:`stenograf.flow.MeetingRun` on a worker
thread; this class is the sink for the events it emits, marshalling every one of
them onto the GUI thread before it touches a property (Qt objects belong to one
thread, and the live worker calls a view from its own).

Being both is deliberate: the events map one-to-one onto display state, so an
adapter object between the two would be pure indirection.

What the screen shows, and where it comes from:

- **committed captions** — appended through the ``committed`` signal into a QML
  ``ListModel``, never through the state map: a growing transcript must not
  re-evaluate every binding on the screen for each new line;
- **the interim area** — the open (bright) line plus the grey provisional tail,
  per channel, exactly as :class:`~stenograf.captions.CaptionStream` decides;
  the TUI renders the same rows with markup instead of QML;
- **the header** — phase, elapsed, language, profile; the 1 Hz clock that
  advances it is the app's only periodic timer, and it also flushes an idle
  caption line (the whole redraw budget, from ``ui.meeting``);
- **the footer** — the transient status line (model loading, notes progress)
  and, once allocated, the meeting folder. They are separate fields on purpose:
  the notes step overwrites the status, and the folder must survive that.

Stopping crosses back to capture through the callback the run hands over
(:meth:`set_stop`) — and it *blocks*, up to ~5 s waiting on the capture
subprocess to flush and exit, so it runs on its own thread. Doing it inline
would freeze the window for those seconds.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, Signal, Slot

from stenograf.captions import LIVE_LABEL, CaptionStream
from stenograf.gui.app import Screen
from stenograf.transcript import Transcript
from stenograf.view import LiveView, clock, profile_label

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from stenograf.asr.base import Word
    from stenograf.capture.base import Channel
    from stenograf.config import Language
    from stenograf.flow import MeetingRequest
    from stenograf.gui.app import StenografGui

_PHASE_LABEL = {
    "rec": "REC",
    "finalizing": "FINALIZING",
    "done": "DONE",
    "failed": "FAILED",
}
"""Screen lifecycle: capture running → on-stop pass running → transcript shown
(or the run never got off the ground). QML colours the dot from the same key."""


class MeetingScreen(Screen, LiveView):
    """One meeting, from the Start button to the finalized transcript on screen."""

    committed = Signal(str, str)
    """``(speaker, text)`` — one finished caption line, appended to the log."""

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

        self._request: MeetingRequest | None = None
        self._persist: Callable[[Transcript], object] | None = None
        self._stop_capture: Callable[[], None] | None = None
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._reset()

    def _reset(self) -> None:
        self.set(
            phase="rec",
            phaseLabel=_PHASE_LABEL["rec"],
            elapsed="0:00",
            language="—",
            profile="",
            status="",
            folder="",
            tails=[],
            canStop=False,
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

        Starting here rather than at :meth:`begin` is what the Textual screen's
        ``on_ready`` hook does — the user watches the loading status in the
        header instead of a frozen window while the models come up."""
        request, self._request = self._request, None
        if request is None:  # re-entered the page without a new Start
            return
        self.cleared.emit()
        self._captions.clear()
        self._started = time.monotonic()
        self._clock.start()
        self._thread = self.work(
            lambda: self._meeting(request),
            done=self._finished,
            failed=self._failed,
            name="gui-meeting",
        )

    def _meeting(self, request: MeetingRequest) -> Transcript | None:
        """The meeting, on the worker thread."""
        from stenograf.flow import MeetingRun

        run = MeetingRun(request)
        # The transcript reaches disk at the ``finalized`` event, before the user
        # can close the window on the done screen.
        self._persist = run.persist
        self.post(self.set, folder=str(run.out_dir))
        return run.run(self)

    def _finished(self, transcript: Transcript | None) -> None:
        """The meeting thread returned (capture stopped, finalize and notes done)."""
        self._clock.stop()
        if isinstance(transcript, Transcript):
            if self._state.get("phase") != "done":  # a run that never emitted the event
                self._show(transcript)
            return
        self.set(
            phase="done",
            phaseLabel=_PHASE_LABEL["done"],
            canStop=False,
            tails=[],
            status="the meeting ended before a transcript was produced; "
            "any .partial checkpoint is kept",
        )

    def _failed(self, message: str) -> None:
        self._clock.stop()
        self.set(phase="failed", phaseLabel=_PHASE_LABEL["failed"], canStop=False, status=message)

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
        button uses), then wait: the finalize and the notes tail run to
        completion and the transcript lands on disk, exactly as if Stop had been
        pressed."""
        if self._stop_capture is not None and self._state.get("phase") == "rec":
            with contextlib.suppress(Exception):  # a stop error must not block the exit
                self._stop_capture()
        self.join()

    # -- the 1 Hz tick -----------------------------------------------------

    def _tick(self) -> None:
        self.set(elapsed=clock(time.monotonic() - self._started))
        if self._captions.flush_if_idle():  # a stretch of speech ended
            self._render_tails()

    # -- intents -----------------------------------------------------------

    @Slot()
    def stop(self) -> None:
        """Stop & finalize while capturing; leave once there is nothing to stop."""
        phase = self._state.get("phase")
        if phase == "rec" and self._stop_capture is not None:
            self.set(phase="finalizing", phaseLabel=_PHASE_LABEL["finalizing"], canStop=False)
            threading.Thread(target=self._invoke_stop, name="gui-stop", daemon=True).start()
        elif phase in ("done", "failed"):
            self.app.back()

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
        self.post(self.set, phase="finalizing", phaseLabel=_PHASE_LABEL["finalizing"])

    def finalized(self, transcript: Transcript) -> None:
        if self._persist is not None:
            try:
                self._persist(transcript)
            except Exception as exc:  # noqa: BLE001 — persistence must not sink the result
                self.error(f"could not write the transcript yet ({exc}); retrying on exit")
        self.post(self._show, transcript)

    def error(self, message: str) -> None:
        self.post(self.set, status=message)

    # -- GUI-thread renderers ----------------------------------------------

    def _commit(self, channel: Channel, words: Sequence[Word]) -> None:
        self._captions.commit(channel, words)
        self._render_tails()

    def _interim(self, channel: Channel, text: str) -> None:
        self._captions.interim(channel, text)
        self._render_tails()

    def _emit_line(self, channel: Channel, text: str) -> None:
        """A line the caption stream finished — append it to the log."""
        self.committed.emit(LIVE_LABEL[channel], text)

    def _render_tails(self) -> None:
        self.set(
            tails=[
                {"speaker": LIVE_LABEL[channel], "open": open_text, "tail": tail}
                for channel, open_text, tail in self._captions.tails()
            ]
        )

    def _show(self, transcript: Transcript) -> None:
        """Swap the live captions for the authoritative, diarized transcript."""
        self._captions.clear()
        self._clock.stop()
        self.restored.emit(
            [
                {
                    "speaker": entry.speaker,
                    "time": clock(entry.start),
                    "text": entry.text,
                    "provisional": entry.provisional,
                }
                for entry in transcript.entries
            ]
        )
        count = len(transcript.entries)
        self.set(
            phase="done",
            phaseLabel=_PHASE_LABEL["done"],
            canStop=False,
            tails=[],
            status=f"{count} {'entry' if count == 1 else 'entries'}",
            language=(
                transcript.language.value
                if transcript.language is not None
                else self._state.get("language", "—")
            ),
        )


__all__ = ["MeetingScreen"]
