"""Live-caption views: the event interface + a plain-stdout implementation.

The live pass (``LiveWorker`` → ``LiveDecoder``)
emits a stream of events — committed words, a provisional grey tail, and the
out-of-band notices (status, language lock, the finalize hand-off). A
:class:`LiveView` is the sink for those events; a concrete view renders them
however it likes. This module ships the first, dependency-free renderer,
:class:`PlainLiveView`, which streams committed captions to stdout with
``click.echo`` — the terminal live mode, and equally usable over a pipe or
into a log file. The Qt meeting screen is a second view behind the same
interface (:mod:`stenograf.gui.meeting`).

Live captions are **channel-coarse**: the live pass does not diarize, so it can
only say which channel spoke (``You`` = mic/local, ``Remote`` = system audio).
The on-stop finalize replaces the whole live transcript with diarized
``Local-N``/``Remote-M`` speakers, surfaced via :meth:`finalized`.
Captions already printed cannot be rewritten, so the plain view drops the
interim tail (there is no cursor to erase it) and prints only committed text;
the live grey tail is the Qt view's concern.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import click

from stenograf.asr.base import Word
from stenograf.captions import LIVE_LABEL, CaptionStream
from stenograf.capture.base import Channel
from stenograf.config import Language, MeetingProfile
from stenograf.live import StreamingUpdate
from stenograf.transcript import Transcript


class LiveView:
    """Sink for live-pass events, rendered by a concrete view.

    Every event is a no-op by default, so a view overrides only what it renders
    (and the bare base doubles as a null view). The orchestrator drives a view
    through :meth:`update` — committed + interim words for a channel, straight
    from the worker's ``on_update`` — plus the out-of-band notices
    :meth:`status`, :meth:`language`, :meth:`finalizing`, :meth:`finalized`, and
    :meth:`error`. A view may hold display resources, so it is a context
    manager whose :meth:`close` tears them down.
    """

    def __enter__(self) -> LiveView:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release any display resources (no-op for a plain stream)."""

    def set_stop(self, stop: Callable[[], None]) -> None:
        """Offer the view a callback that ends capture (a Stop button or key).

        Called by the run as soon as there *is* something to stop — before the
        models load, so the control is live during a slow cold start. A view
        with no way to interrupt (the plain stream: Ctrl-C reaches the process
        directly) ignores it."""

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
    """Streams committed captions line-by-line via ``click.echo``.

    The terminal live view (and the first shippable one): works over a pipe or
    into a file just as well as on a TTY. The line rules — continue on one
    channel, break on a channel change, a pause, or past the length cap — are
    :class:`~stenograf.captions.CaptionStream`'s; this view only renders its
    decisions, echoing each commit the moment it arrives (never buffering
    until a line completes) by printing the not-yet-echoed remainder of the
    stream's open line. The provisional grey tail is dropped: a non-TTY
    stream has no cursor to erase it, and committed text is the durable
    contract.

    All output passes through one lock: commits arrive on the worker thread while
    the status/language/finalize notices arrive on the main thread, and without
    the lock a caption line and a notice could interleave mid-write.
    """

    def __init__(self, echo: Callable[..., None] = click.echo) -> None:
        self._echo = echo
        self._lock = threading.Lock()
        self._stream = CaptionStream(self._line_done)
        self._printed = 0  # chars of the stream's open line already echoed
        self._start = 0.0  # start time of the newest commit, for a fresh header

    def commit(self, channel: Channel, words: Sequence[Word]) -> None:
        if not words:
            return
        with self._lock:
            self._start = words[0].start
            self._stream.commit(channel, words)  # may fire _line_done
            open_text = " ".join(self._stream.open_words)
            remainder = open_text[self._printed :]
            if remainder:
                if self._printed == 0:
                    remainder = f"[{clock(self._start)}] {LIVE_LABEL[channel]}: {remainder}"
                self._echo(remainder, nl=False)
                self._printed = len(open_text)

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
            self._stream.flush()  # close the open caption line first
            self._echo(click.style(f"error: {message}", fg="red"), err=True)

    def _notice(self, message: str) -> None:
        """Print an out-of-band line, first closing any open caption line."""
        with self._lock:
            self._stream.flush()
            self._echo(message)

    def _line_done(self, channel: Channel, text: str) -> None:
        """The stream completed a line: echo whatever of it is still unprinted
        (the length-cap flush fires mid-commit, before this view has echoed the
        words that crossed the cap) and terminate it.

        Runs inside :class:`CaptionStream` calls made under ``self._lock`` —
        it must not take the lock itself. The caption line was written with
        ``nl=False``, so the empty echo supplies its missing newline."""
        remainder = text[self._printed :]
        if remainder and self._printed == 0:
            # The whole line arrived and flushed within one commit.
            remainder = f"[{clock(self._start)}] {LIVE_LABEL[channel]}: {remainder}"
        if remainder:
            self._echo(remainder, nl=False)
        self._echo("")
        self._printed = 0


def clock(seconds: float) -> str:
    """``m:ss`` (``h:mm:ss`` past an hour) — timestamps and elapsed time alike."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def profile_label(profile: MeetingProfile) -> str:
    """What a meeting screen's header says about the sources being captured.

    ``local 2 · remote auto``, or just the live half when one source is off."""

    def part(count: int | None) -> str:
        return "auto" if count is None else str(count)

    if profile.local_speakers == 0:
        return f"remote {part(profile.remote_speakers)}"
    if profile.remote_speakers == 0:
        return f"local {part(profile.local_speakers)}"
    return f"local {part(profile.local_speakers)} · remote {part(profile.remote_speakers)}"
