"""Live-caption views: the event interface + a plain-stdout implementation.

Phase 2, Task 5 (PLAN.md §5). The live pass (``LiveWorker`` → ``LiveDecoder``)
emits a stream of events — committed words, a provisional grey tail, and the
out-of-band notices (status, language lock, the finalize hand-off). A
:class:`LiveView` is the sink for those events; a concrete view renders them
however it likes. This module ships the first, dependency-free renderer,
:class:`PlainLiveView`, which streams committed captions to stdout with
``click.echo`` — usable over a pipe, into a log file, or any non-TTY. The
Textual TUI (Task 6) is a second :class:`LiveView` behind the same interface.

Live captions are **channel-coarse**: the live pass does not diarize, so it can
only say which channel spoke (``You`` = mic/local, ``Remote`` = system audio).
The on-stop finalize replaces the whole live transcript with diarized
``Local-N``/``Remote-M`` speakers (PLAN.md §2), surfaced via :meth:`finalized`.
In a non-TTY stream the captions already printed cannot be rewritten, so the
plain view drops the interim tail (there is no cursor to erase it) and prints
only committed text; the live grey tail is the Textual view's concern.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import click

from stenograf.asr.base import Word
from stenograf.capture.base import Channel
from stenograf.config import Language
from stenograf.live import StreamingUpdate
from stenograf.transcript import Transcript

_LIVE_LABEL = {Channel.MIC: "You", Channel.SYSTEM: "Remote"}
"""Channel-coarse caption labels for the live pass (PLAN.md Task 6). Distinct
from the checkpoint's ``Local``/``Remote`` (session ``_CHANNEL_COARSE``): the
live *display* addresses the user as ``You``. The finalize swap replaces both
with diarized ``Local-N``/``Remote-M`` labels."""

_LINE_GAP = 1.5
"""A committed run continues the same on-screen line while it stays on one
channel and the gap to the next words is under this many seconds; a larger gap
starts a new line, so the plain log reads in utterance-sized paragraphs."""


class LiveView:
    """Sink for live-pass events, rendered by a concrete view.

    Every event is a no-op by default, so a view overrides only what it renders
    (and the bare base doubles as a null view). The orchestrator drives a view
    through :meth:`update` — committed + interim words for a channel, straight
    from the worker's ``on_update`` — plus the out-of-band notices
    :meth:`status`, :meth:`language`, :meth:`finalizing`, :meth:`finalized`, and
    :meth:`error`. A view may hold display resources (the Textual TUI does), so
    it is a context manager whose :meth:`close` tears them down.
    """

    def __enter__(self) -> LiveView:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release any display resources (no-op for a plain stream)."""

    # -- streamed captions -------------------------------------------------

    def update(self, channel: Channel, update: StreamingUpdate) -> None:
        """Dispatch one worker ``StreamingUpdate`` to :meth:`commit`/:meth:`interim`.

        Matches the ``OnUpdate`` signature, so ``on_update=view.update`` wires the
        worker straight to a view. Newly committed words are handed over first
        (append-only, stable), then the current provisional tail — which may be
        empty, clearing a tail that just committed in full.
        """
        if update.committed:
            self.commit(channel, update.committed)
        self.interim(channel, update.interim)

    def commit(self, channel: Channel, words: Sequence[Word]) -> None:
        """Words a channel just finalized (shown black, never rewritten)."""

    def interim(self, channel: Channel, text: str) -> None:
        """A channel's current provisional tail (shown grey, replaced each feed)."""

    # -- out-of-band notices ----------------------------------------------

    def status(self, message: str) -> None:
        """A progress line (model load, capture start, interrupt, …)."""

    def language(self, language: Language) -> None:
        """The meeting language, once detected and locked."""

    def finalizing(self) -> None:
        """The live pass has stopped; the heavy on-stop finalize is running."""

    def finalized(self, transcript: Transcript) -> None:
        """The authoritative transcript that supersedes the live captions."""

    def error(self, message: str) -> None:
        """A recoverable error (e.g. the live pass stopped early)."""


class PlainLiveView(LiveView):
    """Streams committed captions to a (typically non-TTY) stream via ``click.echo``.

    The first shippable live view (PLAN.md §5): no Textual dependency, works over
    a pipe or into a file. Committed words stream onto a per-channel line — the
    line continues while one channel keeps talking and breaks when the channel
    changes or a pause opens, so the log reads as utterance-sized paragraphs. The
    provisional grey tail is dropped: a non-TTY stream has no cursor to erase it,
    and committed text is the durable contract.

    All output passes through one lock: commits arrive on the worker thread while
    the status/language/finalize notices arrive on the main thread, and without
    the lock a caption line and a notice could interleave mid-write.
    """

    def __init__(self, echo: Callable[..., None] = click.echo) -> None:
        self._echo = echo
        self._lock = threading.Lock()
        self._open = False  # a caption line is mid-write (no trailing newline yet)
        self._line_channel: Channel | None = None
        self._last_end = 0.0  # end time of the last word on the open line

    def commit(self, channel: Channel, words: Sequence[Word]) -> None:
        if not words:
            return
        with self._lock:
            text = " ".join(w.text for w in words)
            continues = (
                self._open
                and channel == self._line_channel
                and words[0].start - self._last_end <= _LINE_GAP
            )
            if continues:
                self._echo(f" {text}", nl=False)
            else:
                self._break_line()
                self._echo(f"[{_clock(words[0].start)}] {_LIVE_LABEL[channel]}: {text}", nl=False)
                self._open = True
                self._line_channel = channel
            self._last_end = words[-1].end

    def status(self, message: str) -> None:
        self._notice(message)

    def language(self, language: Language) -> None:
        self._notice(f"language: {language.value}")

    def finalizing(self) -> None:
        self._notice("finalizing — the on-stop pass replaces the live captions")

    def finalized(self, transcript: Transcript) -> None:
        speakers = len({e.speaker for e in transcript.entries})
        self._notice(f"finalized: {len(transcript.entries)} entries, {speakers} speakers")

    def error(self, message: str) -> None:
        with self._lock:
            self._break_line()
            self._echo(click.style(f"error: {message}", fg="red"), err=True)

    def _notice(self, message: str) -> None:
        """Print an out-of-band line, first closing any open caption line."""
        with self._lock:
            self._break_line()
            self._echo(message)

    def _break_line(self) -> None:
        """Terminate the open caption line so the next output starts fresh.

        Caller must hold ``self._lock``. The caption line was written with
        ``nl=False``, so an empty echo supplies its missing newline.
        """
        if self._open:
            self._echo("")
            self._open = False
            self._line_channel = None


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
