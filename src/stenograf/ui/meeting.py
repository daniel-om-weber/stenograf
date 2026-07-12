"""The live-caption meeting screen — the second :class:`~stenograf.view.LiveView`.

Built as a standalone Textual app in Phase 2, Task 6; converted to a
:class:`~textual.screen.Screen` in Phase 7, Task 2 (PLAN.md §5) so one
codepath serves two entries:

- **``steno start``** runs :class:`~stenograf.ui.app.StenografApp` with this
  screen as its *default* (root) screen via :meth:`TextualLiveView.serve`; the
  screen exits the whole app when dismissed.
- **The launcher** pushes it onto the Home stack; dismissing returns to Home
  with the finalized transcript as the screen result.

The screen itself: a pinned header (REC / elapsed / language / profile), an
append-only ``RichLog`` of committed captions, a dim per-channel interim tail
(channel-coarse ``You``/``Remote`` — the real ``Local-N``/``Remote-M`` labels
appear only after the on-stop finalize swap), and a footer.

Design constraints from the plan:

- **Minimal redraw.** One 1 Hz clock is the only periodic repaint (it advances
  the header's elapsed time and flushes an idle open caption line); everything
  else updates on an event. The frame cap and animation kill-switch are owned
  by the app shell (:mod:`stenograf.ui._fps` + ``StenografApp.on_mount``), so
  they cover this screen wherever it runs.
- **Worker → UI crossing.** The live worker calls the view from its own thread;
  every UI mutation is marshalled onto the Textual event loop via
  :meth:`App.call_from_thread` (Textual's supported ``loop.call_soon_threadsafe``
  wrapper). Updates arriving before the screen is mounted or after it stops are
  dropped — the UI is best-effort, the transcript is authoritative.
- **Ctrl-C is a captured key event**, not a ``KeyboardInterrupt``: under Textual
  the terminal is in raw mode, so the quit binding must *deliberately* cross back
  to the capture side via a ``stop`` callback (``provider.stop``) to end capture
  gracefully; the meeting then finalizes and the screen dismisses on its own.

The heavy import (textual) lives here, not in :mod:`stenograf.view`, so the plain
stdout view stays dependency-light; ``steno start`` imports this module only when
the TUI is actually used.
"""

from __future__ import annotations

import stenograf.ui._fps  # noqa: F401  — must precede the textual imports (frame cap)

# isort: split

import contextlib
import threading
import time
from collections.abc import Callable, Sequence
from enum import Enum

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from stenograf.asr.base import Word
from stenograf.capture.base import Channel
from stenograf.config import Language, MeetingProfile
from stenograf.transcript import Transcript
from stenograf.ui.app import StenografApp
from stenograf.view import LiveView

_LIVE_LABEL = {Channel.MIC: "You", Channel.SYSTEM: "Remote"}
_LABEL_STYLE = {Channel.MIC: "bold cyan", Channel.SYSTEM: "bold magenta"}
_LINE_GAP = 1.5  # committed run continues while the gap to the next words is under this

_LINE_FLUSH_CHARS = 250
"""Once the open committed line grows past this, it moves into the scrolling log
immediately. The open-line merge exists for the speculative pass's few-word
commits; the window pass commits a whole ~30 s window per batch, and during a
long remote stretch budget-closed windows join with sub-second gaps, so without
this bound the line grows for minutes inside the height-capped interim area —
invisible below its fourth row (the "UI frozen while remote talks" bug)."""

_IDLE_FLUSH_S = 5.0
"""Wall-clock seconds without a new commit before the open line flushes to the
log anyway. The last window of a stretch of speech otherwise sits in the interim
area until some future commit displaces it — minutes, in a quiet meeting."""

_INTERIM_TAIL_CHARS = 200
"""At most this much of the open line (its tail) renders in the interim area.
The area clips at the bottom, so only the freshest words may occupy it."""


class Phase(Enum):
    """Screen lifecycle: drives the header rendering and what a quit keypress does.

    ``CAPTURING`` → live pass running; ``FINALIZING`` → capture stopped, on-stop
    pass running; ``DONE`` → transcript shown, quit just leaves. Each member
    carries its header lead text and color.
    """

    CAPTURING = ("● REC", "red")
    FINALIZING = ("◼ finalizing", "yellow")
    DONE = ("✓ done", "green")

    def __init__(self, lead: str, color: str) -> None:
        self.lead = lead
        self.color = color


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _profile_label(profile: MeetingProfile) -> str:
    def part(count: int | None) -> str:
        return "auto" if count is None else str(count)

    if profile.local_speakers == 0:
        return f"remote {part(profile.remote_speakers)}"
    if profile.remote_speakers == 0:
        return f"local {part(profile.local_speakers)}"
    return f"local {part(profile.local_speakers)} · remote {part(profile.remote_speakers)}"


class MeetingScreen(Screen[Transcript | None]):
    """Header, captions log, interim tail, footer — the live meeting view.

    UI mutations happen through the ``push_*`` methods, which must run on the app
    thread (the :class:`TextualLiveView` adapter marshals them there). The screen
    keeps plain-text mirrors of what it renders (``committed_lines``, the live
    tail, the header) so behaviour is assertable without scraping widget internals.
    """

    DEFAULT_CSS = """
    #header { dock: top; height: 1; background: $panel; color: $text; padding: 0 1; }
    #captions { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    /* The live "bottom line": the open committed run renders normally (bright);
       only its grey tail is dimmed, via [dim] markup — so this widget must NOT be
       globally muted, or committed and provisional text would look identical. */
    #interim { height: auto; max-height: 4; padding: 0 1; }
    """

    BINDINGS = [Binding("ctrl+c,q", "stop", "Stop & finalize", priority=True, show=True)]

    def __init__(
        self,
        *,
        profile: MeetingProfile,
        language: Language | None = None,
        stop: Callable[[], None] | None = None,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._language = language
        self.stop_callback = stop
        self._on_ready = on_ready

        self._phase = Phase.CAPTURING
        self._status = ""
        self._start = 0.0
        self._transcript: Transcript | None = None  # the screen result (finalize swap)

        # One interleaved committed stream (the RichLog), tracked like the plain
        # view: a run continues on the open line until the channel changes or a
        # pause opens. The open line lives in the interim area (bright) and moves
        # into the log on break — or once it outgrows _LINE_FLUSH_CHARS or sits
        # idle past _IDLE_FLUSH_S, so continuous speech cannot hold text out of
        # the log (and out of sight) indefinitely.
        self._open_channel: Channel | None = None
        self._open_words: list[str] = []
        self._last_end = 0.0
        self._last_commit_at = 0.0  # wall clock of the newest commit (idle flush)
        self._interim: dict[Channel, str] = {}

        # Plain-text mirrors for tests / debugging.
        self.committed_lines: list[str] = []
        self.ready = threading.Event()  # set once mounted; the view gates on it

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        log = RichLog(id="captions", markup=True, wrap=True, highlight=False, auto_scroll=True)
        yield log
        yield Static(id="interim")
        yield Footer()

    def on_mount(self) -> None:
        self._start = time.monotonic()
        self._render_header()
        self.set_interval(1.0, self._tick)  # the ONLY periodic repaint (1 Hz)
        self.ready.set()
        if self._on_ready is not None:
            self._on_ready()

    def on_unmount(self) -> None:
        self.ready.clear()

    # -- periodic (1 Hz) ---------------------------------------------------

    def _tick(self) -> None:
        self._render_header()
        # Idle flush: commits stopped arriving (the stretch of speech ended), so
        # the open line cannot continue soon — move it into the log rather than
        # leaving it stranded in the interim area until a future commit displaces
        # it. Costs at most one extra log line if speech resumes within the gap.
        if self._open_words and time.monotonic() - self._last_commit_at > _IDLE_FLUSH_S:
            self._flush_open_line()
            self._render_interim()

    # -- captions ----------------------------------------------------------

    def push_committed(self, channel: Channel, words: Sequence[Word]) -> None:
        if not words:
            return
        text = [w.text for w in words]
        continues = (
            self._open_channel == channel
            and self._open_words
            and words[0].start - self._last_end <= _LINE_GAP
        )
        if continues:
            self._open_words.extend(text)
        else:
            self._flush_open_line()
            self._open_channel = channel
            self._open_words = list(text)
        self._last_end = words[-1].end
        self._last_commit_at = time.monotonic()
        # Size bound: past the cap the open line reads as a paragraph already, so
        # move it into the log *now*. A window-mode batch (~30 s of speech) lands
        # in the log the moment it commits instead of accumulating — clipped and
        # invisible — in the height-capped interim area.
        if len(" ".join(self._open_words)) >= _LINE_FLUSH_CHARS:
            self._flush_open_line()
        self._render_interim()

    def push_interim(self, channel: Channel, text: str) -> None:
        if text:
            self._interim[channel] = text
        else:
            self._interim.pop(channel, None)
        self._render_interim()

    def _flush_open_line(self) -> None:
        """Move the growing committed line into the append-only log."""
        if self._open_channel is None or not self._open_words:
            return
        label, style = _LIVE_LABEL[self._open_channel], _LABEL_STYLE[self._open_channel]
        text = " ".join(self._open_words)
        self.query_one("#captions", RichLog).write(f"[{style}]{label}[/]  {text}")
        self.committed_lines.append(f"{label}  {text}")
        self._open_channel = None
        self._open_words = []

    def _render_interim(self) -> None:
        """The dim per-channel tail: the open committed line (bright) + grey tail."""
        lines: list[str] = []
        for channel in (Channel.MIC, Channel.SYSTEM):
            parts: list[str] = []
            if channel == self._open_channel and self._open_words:
                open_text = " ".join(self._open_words)
                if len(open_text) > _INTERIM_TAIL_CHARS:  # the area clips at the bottom
                    open_text = "…" + open_text[-_INTERIM_TAIL_CHARS:]
                parts.append(open_text)
            tail = self._interim.get(channel, "")
            if tail:
                parts.append(f"[dim]{tail}[/]")
            if parts:
                style = _LABEL_STYLE[channel]
                lines.append(f"[{style}]{_LIVE_LABEL[channel]}[/]  " + " ".join(parts))
        self.query_one("#interim", Static).update("\n".join(lines))

    # -- out-of-band notices ----------------------------------------------

    def push_status(self, message: str) -> None:
        self._status = message
        self._render_header()

    def push_language(self, language: Language) -> None:
        self._language = language
        self._render_header()

    def push_finalizing(self) -> None:
        self._phase = Phase.FINALIZING
        self._render_header()

    def push_finalized(self, transcript: Transcript) -> None:
        """Swap the live captions for the authoritative, diarized transcript."""
        self._flush_open_line()
        self._interim.clear()
        self._open_channel = None
        self._open_words = []
        self.query_one("#interim", Static).update("")
        log = self.query_one("#captions", RichLog)
        log.clear()
        self.committed_lines = []
        for entry in transcript.entries:
            marker = " [dim](overlap)[/]" if entry.provisional else ""
            stamp = f"[dim]{_clock(entry.start)}[/]"
            log.write(f"[bold]{entry.speaker}[/] {stamp}  {entry.text}{marker}")
            self.committed_lines.append(f"{entry.speaker}  {entry.text}")
        if transcript.language is not None:
            self._language = transcript.language
        self._phase = Phase.DONE
        self._transcript = transcript
        n = len(transcript.entries)
        self._status = f"{n} {'entry' if n == 1 else 'entries'} · q to exit"
        self._render_header()

    def push_error(self, message: str) -> None:
        self._status = message
        self._render_header()

    # -- header ------------------------------------------------------------

    def header_text(self) -> str:
        elapsed = _clock(time.monotonic() - self._start) if self._start else "0:00"
        lang = self._language.value if self._language else "—"
        bits = [self._phase.lead, elapsed, lang, _profile_label(self._profile)]
        if self._status:
            bits.append(self._status)
        return "  ·  ".join(bits)

    def _render_header(self) -> None:
        header = self.query_one("#header", Static)
        header.update(f"[{self._phase.color}]{self.header_text()}[/]")

    # -- quit --------------------------------------------------------------

    def action_stop(self) -> None:
        """Ctrl-C / q: end capture (crossing to the worker), then let it finalize.

        The first press while capturing hands off to ``stop_callback`` — capture
        ends, the meeting thread runs the finalize pass, and it dismisses the
        screen when done. A second press (impatient, still finalizing) forces the
        exit. Once the transcript is shown, quit just leaves.

        ``stop_callback`` (``provider.stop``) *blocks* — up to ~5 s waiting on the
        capture subprocess to flush and exit — so it runs on a background thread, not
        the event loop: doing it inline would freeze the whole TUI for those seconds
        and, worse, deaden this very binding so the impatient second Ctrl-C could not
        force an exit. The teardown finishes on its own; the meeting thread then
        finalizes and dismisses the screen.
        """
        if self._phase is Phase.CAPTURING and self.stop_callback is not None:
            self._phase = Phase.FINALIZING
            self._render_header()
            threading.Thread(target=self._invoke_stop, name="tui-stop", daemon=True).start()
        else:
            self._leave()

    def _leave(self) -> None:
        """Return control to whoever showed the screen (the two-entry rule).

        Pushed by the launcher (something is under it on the stack): dismiss
        back with the finalized transcript — or ``None`` if the meeting never
        produced one — as the screen result. Run as the app's default screen
        (``steno start``): there is nothing under it to return to, so leaving
        exits the whole app.
        """
        if len(self.app.screen_stack) > 1:
            self.dismiss(self._transcript)
        else:
            self.app.exit()

    def _invoke_stop(self) -> None:
        """Run the blocking capture teardown off the event loop (see action_stop)."""
        try:
            self.stop_callback()  # type: ignore[misc]  # guarded by action_stop
        except Exception as exc:  # never let a stop error wedge the UI
            with contextlib.suppress(Exception):  # the app may already be gone
                self.app.call_from_thread(self.push_error, f"stop failed: {exc}")


class TextualLiveView(LiveView):
    """LiveView backed by :class:`MeetingScreen`, marshalling events onto its loop.

    Construct with the meeting profile, then :meth:`serve` a ``meeting`` callable
    (typically ``recorder.run(..., live=True, view=self, ...)``): a
    :class:`StenografApp` runs on the calling (main) thread with the meeting
    screen as its root while the meeting runs on a background thread, and the
    orchestrator's structured events cross back to the UI thread. ``serve``
    returns the meeting's result (the authoritative transcript) once the app
    exits.
    """

    def __init__(
        self,
        profile: MeetingProfile,
        *,
        language: Language | None = None,
        stop: Callable[[], None] | None = None,
        persist: Callable[[Transcript], object] | None = None,
        app: StenografApp | None = None,
    ) -> None:
        # ``persist`` runs on the meeting thread at the ``finalized`` event,
        # before the UI swap: the CLI wires its write-transcript-files closure
        # here so the meeting is on disk while the app still shows the "done"
        # screen (crash/force-quit there must not lose it). Exceptions are
        # surfaced via :meth:`error`, never raised — the caller retries after
        # :meth:`serve` returns.
        #
        # ``app`` selects the entry mode: None (the CLI) builds a private shell
        # with the meeting screen as its root and :meth:`serve` runs it; the
        # launcher passes its already-running app instead, pushes
        # :attr:`screen` itself, and must NOT call :meth:`serve`.
        self._screen = MeetingScreen(profile=profile, language=language, stop=stop)
        self._app = app if app is not None else StenografApp(initial=self._screen)
        self._persist = persist
        self._meeting_thread: threading.Thread | None = None

    @property
    def app(self) -> StenografApp:
        return self._app

    @property
    def screen(self) -> MeetingScreen:
        return self._screen

    def set_stop(self, stop: Callable[[], None]) -> None:
        self._screen.stop_callback = stop

    # -- LiveView events → UI thread --------------------------------------

    def commit(self, channel: Channel, words: Sequence[Word]) -> None:
        self._marshal(self._screen.push_committed, channel, tuple(words))

    def interim(self, channel: Channel, text: str) -> None:
        self._marshal(self._screen.push_interim, channel, text)

    def status(self, message: str) -> None:
        self._marshal(self._screen.push_status, message)

    def language(self, language: Language) -> None:
        self._marshal(self._screen.push_language, language)

    def finalizing(self) -> None:
        self._marshal(self._screen.push_finalizing)

    def finalized(self, transcript: Transcript) -> None:
        if self._persist is not None:
            try:
                self._persist(transcript)
            except Exception as exc:  # noqa: BLE001 — persistence must not sink the result
                self.error(f"could not write the transcript yet ({exc}); retrying on exit")
        self._marshal(self._screen.push_finalized, transcript)

    def error(self, message: str) -> None:
        self._marshal(self._screen.push_error, message)

    def _marshal(self, fn: Callable[..., object], *args: object) -> None:
        """Run a UI mutation on the app's event loop; drop it if the screen isn't live.

        ``call_from_thread`` is Textual's thread-safe hop onto its event loop
        (``loop.call_soon_threadsafe`` under the hood). It refuses to run on the
        app's own thread and raises once the loop is gone; both are fine to ignore —
        the UI is provisional and the finalize pass is the real transcript.
        """
        if not self._screen.ready.is_set():
            return
        # best-effort UI: call_from_thread raises once the loop is gone or if run
        # off a worker thread — either way the finalize pass is the real transcript.
        with contextlib.suppress(Exception):
            self._app.call_from_thread(fn, *args)

    # -- lifecycle ---------------------------------------------------------

    def serve(self, meeting: Callable[[], Transcript]) -> Transcript:
        """Run the TUI (this thread) while ``meeting`` runs on a background thread.

        Returns the meeting's transcript once the app exits; re-raises whatever the
        meeting raised. The background thread renders the finalize swap and leaves
        the screen when the meeting returns (capture stopped and finalized).
        """
        result: dict[str, object] = {}
        self.arm_meeting(meeting, result)
        self._app.run()

        # The UI has exited, but the meeting thread may still be running the on-stop
        # finalize (e.g. the user force-quit the TUI while it was finalizing). Join it
        # so the authoritative transcript is always collected — the finalize, once
        # started, is never dropped just because the UI closed early.
        if self._meeting_thread is not None:
            self._meeting_thread.join()

        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        return result.get("transcript")  # type: ignore[return-value]

    def arm_meeting(self, meeting: Callable[[], Transcript], result: dict[str, object]) -> None:
        """Wire ``on_ready`` to run ``meeting`` on a background thread once mounted.

        :meth:`serve` (the CLI) calls this itself; the launcher flow calls it
        directly before pushing :attr:`screen`, then reads ``result`` in the
        screen's dismiss callback. Also the seam that makes the
        meeting → finalize → exit flow exercisable under Textual's ``run_test``
        harness (which drives the loop itself instead of :meth:`serve`'s
        blocking ``run``). The thread is held on ``self._meeting_thread`` so
        :meth:`serve` can join it before reading the result.
        """

        def run_meeting() -> None:
            try:
                result["transcript"] = meeting()
            except BaseException as exc:  # noqa: BLE001 — surfaced to the caller in serve
                result["error"] = exc
            finally:
                # Show the result only while the app is still up; if it already
                # exited (force-quit), skip the UI hop and the wait for it.
                if self._app.is_running:
                    self._screen.ready.wait(timeout=5)  # don't finish before the mount
                    with contextlib.suppress(Exception):  # app already gone
                        self._app.call_from_thread(self._finish, result)

        def start_meeting() -> None:
            self._meeting_thread = threading.Thread(
                target=run_meeting, name="tui-meeting", daemon=True
            )
            self._meeting_thread.start()

        self._screen._on_ready = start_meeting

    def _finish(self, result: dict[str, object]) -> None:
        """On the UI thread: ensure the finalize result is shown, then wait to quit.

        The orchestrator emits ``finalized`` itself now (``run`` calls
        ``view.finalized`` before returning), so on the normal path the swap has
        already happened and the screen is in the ``done`` phase — nothing to do but
        keep it up for the user to read and dismiss. The push here is a
        fallback for a meeting that returns a transcript without emitting the
        event (e.g. a test stand-in). A meeting that raised has nothing to show, so
        the screen leaves (which exits the app when it is the root).
        """
        transcript = result.get("transcript")
        if isinstance(transcript, Transcript):
            if self._screen._phase is not Phase.DONE:
                self._screen.push_finalized(transcript)
        else:  # the meeting raised — nothing to show, so just leave
            self._screen._leave()


__all__ = ["MeetingScreen", "Phase", "TextualLiveView"]
