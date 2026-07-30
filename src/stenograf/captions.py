"""Live captions → readable lines: the segmentation both live views share.

A live view receives *words*, not lines: the worker commits a few words at a
time (the speculative pass) or a whole ~30 s window at once (the window pass),
plus a provisional tail that is replaced on every feed. Turning that into
something a person can read — one growing line per run of speech, broken when
the channel changes or a pause opens, flushed into the scrollback before it
outgrows the screen — is a pile of small timing rules that must not exist twice.
So it lives here, and a meeting screen (today: the Qt one) only *renders*
what this decides.

The rules, and why each exists:

- a committed run **continues** the open line while it stays on one channel and
  the gap to the next words is under :data:`LINE_GAP` seconds — so the log reads
  in utterance-sized paragraphs rather than in decoder batches;
- the open line **flushes** into the scrollback once it passes
  :data:`LINE_FLUSH_CHARS`, because a window-pass batch commits ~30 s of speech
  at once and during a long remote stretch budget-closed windows join with
  sub-second gaps: without the bound the line grows for minutes inside a
  height-capped interim area, invisible below its fourth row (the "UI frozen
  while remote talks" bug);
- it also flushes after :data:`IDLE_FLUSH_S` seconds without a commit, or the
  last window of a stretch of speech sits in the interim area until some future
  commit displaces it — minutes, in a quiet meeting;
- only the last :data:`INTERIM_TAIL_CHARS` of the open line are offered for the
  interim area, which clips at the bottom: only the freshest words may occupy it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from stenograf.capture.base import Channel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from stenograf.asr.base import Word

LIVE_LABEL = {Channel.MIC: "You", Channel.SYSTEM: "Remote"}
"""Channel-coarse caption labels. The live pass does not diarize, so it can only
say *which channel* spoke; the on-stop finalize swaps in ``Local-N``/``Remote-M``."""

LINE_GAP = 1.5
"""Seconds of silence that end the open line (see the module docstring)."""

LINE_FLUSH_CHARS = 250
"""Character length past which the open line moves to the scrollback at once."""

IDLE_FLUSH_S = 5.0
"""Wall-clock seconds without a commit before the open line flushes anyway."""

INTERIM_TAIL_CHARS = 200
"""How much of the open line's tail may render in the (clipping) interim area."""


class CaptionStream:
    """One interleaved committed stream, plus the per-channel provisional tails.

    ``on_line(channel, text)`` fires whenever a line is complete and belongs in
    the view's scrollback — the only output; everything still in flight is read
    back through :meth:`tails` after each event."""

    def __init__(self, on_line: Callable[[Channel, str], None]) -> None:
        self._on_line = on_line
        self._open_channel: Channel | None = None
        self._open_words: list[str] = []
        self._last_end = 0.0  # end time of the newest word on the open line
        self._last_commit_at = 0.0  # wall clock of the newest commit (idle flush)
        self._interim: dict[Channel, str] = {}

    @property
    def open_words(self) -> list[str]:
        """The words on the line still being built (empty between runs)."""
        return list(self._open_words)

    def commit(self, channel: Channel, words: Sequence[Word]) -> None:
        """Take newly committed words, continuing or breaking the open line."""
        if not words:
            return
        text = [w.text for w in words]
        continues = (
            self._open_channel == channel
            and self._open_words
            and words[0].start - self._last_end <= LINE_GAP
        )
        if continues:
            self._open_words.extend(text)
        else:
            self.flush()
            self._open_channel = channel
            self._open_words = list(text)
        self._last_end = words[-1].end
        self._last_commit_at = time.monotonic()
        # Past the cap the open line reads as a paragraph already, so move it to
        # the scrollback *now* rather than letting it accumulate — clipped and
        # invisible — in the interim area.
        if len(" ".join(self._open_words)) >= LINE_FLUSH_CHARS:
            self.flush()

    def interim(self, channel: Channel, text: str) -> None:
        """Replace a channel's provisional tail (empty text clears it)."""
        if text:
            self._interim[channel] = text
        else:
            self._interim.pop(channel, None)

    def flush(self) -> None:
        """Emit the open line, if any, and start a new one."""
        if self._open_channel is None or not self._open_words:
            return
        self._on_line(self._open_channel, " ".join(self._open_words))
        self._open_channel = None
        self._open_words = []

    def flush_if_idle(self) -> bool:
        """Flush a line whose commits stopped arriving; whether anything moved.

        Called from the view's 1 Hz tick. Costs at most one extra line if speech
        resumes within the gap."""
        if self._open_words and time.monotonic() - self._last_commit_at > IDLE_FLUSH_S:
            self.flush()
            return True
        return False

    def tails(self) -> list[tuple[Channel, str, str]]:
        """What the interim area should show: ``(channel, open text, dim tail)``.

        In fixed channel order, skipping channels with nothing in flight. The
        open text is the bright, already-committed part of the line (truncated
        to its freshest :data:`INTERIM_TAIL_CHARS`); the tail is the grey
        provisional text. Either may be empty, never both."""
        rows: list[tuple[Channel, str, str]] = []
        for channel in (Channel.MIC, Channel.SYSTEM):
            open_text = ""
            if channel == self._open_channel and self._open_words:
                open_text = " ".join(self._open_words)
                if len(open_text) > INTERIM_TAIL_CHARS:  # the area clips at the bottom
                    open_text = "…" + open_text[-INTERIM_TAIL_CHARS:]
            tail = self._interim.get(channel, "")
            if open_text or tail:
                rows.append((channel, open_text, tail))
        return rows

    def clear(self) -> None:
        """Drop everything in flight — the finalize swap replaces it all."""
        self._open_channel = None
        self._open_words = []
        self._interim.clear()


__all__ = [
    "IDLE_FLUSH_S",
    "INTERIM_TAIL_CHARS",
    "LINE_FLUSH_CHARS",
    "LINE_GAP",
    "LIVE_LABEL",
    "CaptionStream",
]
