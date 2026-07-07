"""Textual live-caption TUI — the second :class:`~stenograf.view.LiveView`.

Phase 2, Task 6 (PLAN.md §5). A pinned header (REC / elapsed / language /
profile), an append-only ``RichLog`` of committed captions, a dim per-channel
interim tail (channel-coarse ``You``/``Remote`` — the real ``Local-N``/
``Remote-M`` labels appear only after the on-stop finalize swap), and a footer.

Design constraints from the plan:

- **Minimal redraw.** One 1 Hz clock is the only periodic repaint (it advances
  the header's elapsed time); everything else updates on an event. Animations are
  disabled (``animation_level = "none"``) and the frame cap is pinned low
  (``TEXTUAL_FPS=15``, applied before textual is imported — the value is baked
  into ``Screen.UPDATE_PERIOD`` at import time).
- **Worker → UI crossing.** The live worker calls the view from its own thread;
  every UI mutation is marshalled onto the Textual event loop via
  :meth:`App.call_from_thread` (Textual's supported ``loop.call_soon_threadsafe``
  wrapper). Updates arriving before the app is mounted or after it stops are
  dropped — the UI is best-effort, the transcript is authoritative.
- **Ctrl-C is a captured key event**, not a ``KeyboardInterrupt``: under Textual
  the terminal is in raw mode, so the quit binding must *deliberately* cross back
  to the capture side via a ``stop`` callback (``provider.stop``) to end capture
  gracefully; the meeting then finalizes and the app exits on its own.

The heavy import (textual) lives here, not in :mod:`stenograf.view`, so the plain
stdout view stays dependency-light; ``steno start`` imports this module only when
the TUI is actually used.
"""

from __future__ import annotations

import os

# Must precede the textual import: textual reads TEXTUAL_FPS once, at import, into
# constants.MAX_FPS and Screen.UPDATE_PERIOD. Honour a value the user already set
# (setdefault) but otherwise pin the low frame cap the minimal-redraw budget wants
# (PLAN.md §5). Re-pinned defensively just below in case textual was imported first.
os.environ.setdefault("TEXTUAL_FPS", "15")

import contextlib  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402

import textual.constants  # noqa: E402
import textual.screen  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.widgets import Footer, RichLog, Static  # noqa: E402

from stenograf.asr.base import Word  # noqa: E402
from stenograf.capture.base import Channel  # noqa: E402
from stenograf.config import Language, MeetingProfile  # noqa: E402
from stenograf.transcript import Transcript  # noqa: E402
from stenograf.view import LiveView  # noqa: E402

# Re-pin the frame cap regardless of import order: MAX_FPS/UPDATE_PERIOD are baked
# from the env var at textual import time, so an earlier import would leave them at
# the 60 fps default. UPDATE_PERIOD is the screen-refresh interval, read only when
# the app first builds its update timer (not yet — no app is running at import), so
# assigning it here reliably bounds the redraw rate.
_FPS = int(os.environ["TEXTUAL_FPS"])
textual.constants.MAX_FPS = _FPS
textual.screen.UPDATE_PERIOD = 1 / _FPS

_LIVE_LABEL = {Channel.MIC: "You", Channel.SYSTEM: "Remote"}
_LABEL_STYLE = {Channel.MIC: "bold cyan", Channel.SYSTEM: "bold magenta"}
_LINE_GAP = 1.5  # committed run continues while the gap to the next words is under this


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


class LiveApp(App[None]):
    """The Textual application: header, captions log, interim tail, footer.

    UI mutations happen through the ``push_*`` methods, which must run on the app
    thread (the :class:`TextualLiveView` adapter marshals them there). The app
    keeps plain-text mirrors of what it renders (``committed_lines``, the live
    tail, the header) so behaviour is assertable without scraping widget internals.
    """

    CSS = """
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

        # Phase drives the header and what a quit keypress does.
        # capturing → live pass running; finalizing → capture stopped, on-stop
        # pass running; done → transcript shown, quit just exits.
        self._phase = "capturing"
        self._status = ""
        self._start = 0.0

        # One interleaved committed stream (the RichLog), tracked like the plain
        # view: a run continues on the open line until the channel changes or a
        # pause opens. The open line lives in the interim area (bright) and moves
        # into the log on break, so committed text is visible immediately.
        self._open_channel: Channel | None = None
        self._open_words: list[str] = []
        self._last_end = 0.0
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
        self.animation_level = "none"  # minimal redraw: no CSS/scroll animations
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
                parts.append(" ".join(self._open_words))
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
        self._phase = "finalizing"
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
        self._phase = "done"
        n = len(transcript.entries)
        self._status = f"{n} {'entry' if n == 1 else 'entries'} · q to exit"
        self._render_header()

    def push_error(self, message: str) -> None:
        self._status = message
        self._render_header()

    # -- header ------------------------------------------------------------

    def header_text(self) -> str:
        elapsed = _clock(time.monotonic() - self._start) if self._start else "0:00"
        lead = {"capturing": "● REC", "finalizing": "◼ finalizing", "done": "✓ done"}[self._phase]
        lang = self._language.value if self._language else "—"
        bits = [lead, elapsed, lang, _profile_label(self._profile)]
        if self._status:
            bits.append(self._status)
        return "  ·  ".join(bits)

    def _render_header(self) -> None:
        header = self.query_one("#header", Static)
        color = {"capturing": "red", "finalizing": "yellow", "done": "green"}[self._phase]
        header.update(f"[{color}]{self.header_text()}[/]")

    # -- quit --------------------------------------------------------------

    def action_stop(self) -> None:
        """Ctrl-C / q: end capture (crossing to the worker), then let it finalize.

        The first press while capturing hands off to ``stop_callback`` — capture
        ends, the meeting thread runs the finalize pass, and it exits the app when
        done. A second press (impatient, still finalizing) forces an exit. Once the
        transcript is shown, quit just exits.
        """
        if self._phase == "capturing" and self.stop_callback is not None:
            self._phase = "finalizing"
            self._render_header()
            try:
                self.stop_callback()
            except Exception as exc:  # never let a stop error wedge the UI
                self.push_error(f"stop failed: {exc}")
        else:
            self.exit()


class TextualLiveView(LiveView):
    """LiveView backed by :class:`LiveApp`, marshalling events onto its event loop.

    Construct with the meeting profile, then :meth:`serve` a ``meeting`` callable
    (typically ``recorder.run(..., live=True, view=self, ...)``): the app runs on
    the calling (main) thread while the meeting runs on a background thread, and
    the orchestrator's structured events cross back to the UI thread. ``serve``
    returns the meeting's result (the authoritative transcript) once the app exits.
    """

    def __init__(
        self,
        profile: MeetingProfile,
        *,
        language: Language | None = None,
        stop: Callable[[], None] | None = None,
    ) -> None:
        self._app = LiveApp(profile=profile, language=language, stop=stop)
        self._meeting_thread: threading.Thread | None = None

    @property
    def app(self) -> LiveApp:
        return self._app

    def set_stop(self, stop: Callable[[], None]) -> None:
        self._app.stop_callback = stop

    # -- LiveView events → UI thread --------------------------------------

    def commit(self, channel: Channel, words: Sequence[Word]) -> None:
        self._marshal(self._app.push_committed, channel, tuple(words))

    def interim(self, channel: Channel, text: str) -> None:
        self._marshal(self._app.push_interim, channel, text)

    def status(self, message: str) -> None:
        self._marshal(self._app.push_status, message)

    def language(self, language: Language) -> None:
        self._marshal(self._app.push_language, language)

    def finalizing(self) -> None:
        self._marshal(self._app.push_finalizing)

    def finalized(self, transcript: Transcript) -> None:
        self._marshal(self._app.push_finalized, transcript)

    def error(self, message: str) -> None:
        self._marshal(self._app.push_error, message)

    def _marshal(self, fn: Callable[..., object], *args: object) -> None:
        """Run a UI mutation on the app's event loop; drop it if the app isn't live.

        ``call_from_thread`` is Textual's thread-safe hop onto its event loop
        (``loop.call_soon_threadsafe`` under the hood). It refuses to run on the
        app's own thread and raises once the loop is gone; both are fine to ignore —
        the UI is provisional and the finalize pass is the real transcript.
        """
        if not self._app.ready.is_set():
            return
        # best-effort UI: call_from_thread raises once the loop is gone or if run
        # off a worker thread — either way the finalize pass is the real transcript.
        with contextlib.suppress(Exception):
            self._app.call_from_thread(fn, *args)

    # -- lifecycle ---------------------------------------------------------

    def serve(self, meeting: Callable[[], Transcript]) -> Transcript:
        """Run the TUI (this thread) while ``meeting`` runs on a background thread.

        Returns the meeting's transcript once the app exits; re-raises whatever the
        meeting raised. The background thread renders the finalize swap and exits
        the app when the meeting returns (capture stopped and finalized).
        """
        result: dict[str, object] = {}
        self._arm_meeting(meeting, result)
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

    def _arm_meeting(self, meeting: Callable[[], Transcript], result: dict[str, object]) -> None:
        """Wire ``on_ready`` to run ``meeting`` on a background thread once mounted.

        Split out of :meth:`serve` so the meeting → finalize → exit flow is
        exercisable under Textual's ``run_test`` harness (which drives the loop
        itself instead of calling :meth:`serve`'s blocking ``run``). The thread is
        held on ``self._meeting_thread`` so :meth:`serve` can join it before reading
        the result.
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
                    self._app.ready.wait(timeout=5)  # don't finish before the app mounts
                    with contextlib.suppress(Exception):  # app already gone
                        self._app.call_from_thread(self._finish, result)

        def start_meeting() -> None:
            self._meeting_thread = threading.Thread(
                target=run_meeting, name="tui-meeting", daemon=True
            )
            self._meeting_thread.start()

        self._app._on_ready = start_meeting

    def _finish(self, result: dict[str, object]) -> None:
        """On the UI thread: ensure the finalize result is shown, then wait to quit.

        The orchestrator emits ``finalized`` itself now (``run`` calls
        ``view.finalized`` before returning), so on the normal path the swap has
        already happened and the app is in the ``done`` phase — nothing to do but
        keep the app up for the user to read and dismiss. The push here is a
        fallback for a meeting that returns a transcript without emitting the
        event (e.g. a test stand-in). A meeting that raised has nothing to show, so
        the app exits.
        """
        transcript = result.get("transcript")
        if isinstance(transcript, Transcript):
            if self._app._phase != "done":
                self._app.push_finalized(transcript)
        else:  # the meeting raised — nothing to show, so just exit
            self._app.exit()


__all__ = ["LiveApp", "TextualLiveView"]
