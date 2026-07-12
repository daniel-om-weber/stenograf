"""Shared machinery for the queue-streaming capture providers (Linux, Windows).

Both providers have the same shape — one pump thread per captured channel
feeding a single queue that ``frames()`` drains — and the same lifecycle:
``start()`` anchors a shared session clock, each channel pins itself to it at
its first delivered frame, and a pump ending unexpectedly tears down its
siblings so the meeting ends visibly and finalizes rather than silently
continuing half-captured. Subclasses contribute only their transport: opening
a channel's stream, the blocking read loop, and how a stop reaches the
streams. macOS stays separate — its helper is a single subprocess owning both
channels, read synchronously by ``frames()`` itself.

The load-bearing timestamp invariant lives in :class:`SessionClock`: a
frame's timestamp derives from the channel's cumulative delivered sample
count, never from arrival jitter, so gaps in *delivery* never shift audio in
session time. The one sanctioned exception is the forward re-anchor for
transports whose sample stream can under-run session time (WASAPI loopback
wall-clock-estimates silence gaps); ``reanchor_tolerance_s`` is ``inf``
everywhere else.
"""

from __future__ import annotations

import math
import threading
import time
from abc import abstractmethod
from collections.abc import Callable, Iterator
from queue import SimpleQueue

import numpy as np

from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel


class SessionClock:
    """One shared t=0 for all channels; sample-derived per-channel stamps.

    ``start()`` anchors the session; ``stamp(channel, nsamples)`` returns the
    session-time timestamp for a frame of ``nsamples`` that just finished
    arriving on ``channel``. A channel anchors itself at its first frame —
    arrival time minus the frame's duration, since those samples were captured
    over the preceding frame-length — and every later stamp is
    ``anchor + delivered / SAMPLE_RATE``.

    A channel's sample-derived stamp may fall behind its arrival-derived one
    when the transport under-fills a silence gap (WASAPI loopback). When that
    lag exceeds ``reanchor_tolerance_s`` the channel re-anchors forward —
    forward only: per-channel timestamps must stay monotonic, and
    ``SessionStore`` pads the skipped span with silence.

    Each channel is stamped from its own pump thread; per-channel state is
    disjoint, so no lock is needed.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        reanchor_tolerance_s: float = math.inf,
    ) -> None:
        self._clock = clock
        self._tolerance = reanchor_tolerance_s
        self._t0: float | None = None
        self._anchors: dict[Channel, float] = {}
        self._delivered: dict[Channel, int] = {}

    def start(self) -> None:
        self._t0 = self._clock()
        self._anchors.clear()
        self._delivered.clear()

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def stamp(self, channel: Channel, nsamples: int) -> float:
        assert self._t0 is not None, "stamp() before start()"
        elapsed = self._clock() - self._t0
        started_at = max(0.0, elapsed - nsamples / SAMPLE_RATE)
        anchor = self._anchors.get(channel, started_at)
        timestamp = anchor + self._delivered.get(channel, 0) / SAMPLE_RATE
        if started_at - timestamp > self._tolerance:
            anchor += started_at - timestamp
            timestamp = started_at
        self._anchors[channel] = anchor
        self._delivered[channel] = self._delivered.get(channel, 0) + nsamples
        return timestamp


class QueueStreamingProvider[TransportT](CaptureProvider):
    """Base for providers that pump one thread per channel into a shared queue.

    Subclasses set ``_thread_prefix`` and implement the transport:

    - ``_open_channel(channel)`` runs on the ``start()`` caller's thread and
      returns the per-channel transport handle (e.g. a subprocess) that
      ``_pump`` receives; return ``None``-like state and open inside ``_pump``
      when the transport is thread-bound (COM).
    - ``_pump(channel, transport)`` runs on the channel's daemon thread: a
      blocking read loop that calls ``_emit(channel, samples)`` per frame and
      returns at end of stream. It may poll ``_stop_event``.
    - ``_stop_transport()`` makes every pump's read loop end. It must be
      idempotent and thread-safe: ``stop()`` is called from several threads
      (the capture loop on max_seconds, the meeting thread on close, the TUI's
      quit binding) and from a pump thread itself on an unexpected stream
      death.

    The base owns the queue, the sentinel protocol (a pump enqueues its
    ``Channel`` when it ends; ``frames()`` finishes once every started channel
    has), and sibling teardown (a pump ending while the provider is not
    stopping calls ``stop()``).
    """

    _thread_prefix = "capture"

    def __init__(
        self,
        *,
        frame_ms: int,
        clock: Callable[[], float] = time.monotonic,
        reanchor_tolerance_s: float = math.inf,
    ) -> None:
        self._frame_samples = max(1, SAMPLE_RATE * frame_ms // 1000)
        self._clock = SessionClock(clock=clock, reanchor_tolerance_s=reanchor_tolerance_s)
        self._queue: SimpleQueue[AudioFrame | Channel] = SimpleQueue()
        self._started: set[Channel] = set()
        self._threads: dict[Channel, threading.Thread] = {}
        self._stop_event = threading.Event()

    def start(self, channels: set[Channel]) -> None:
        self._clock.start()
        self._started = set(channels)
        self._stop_event.clear()
        for channel in sorted(channels):
            transport = self._open_channel(channel)
            thread = threading.Thread(
                target=self._run_pump,
                args=(channel, transport),
                name=f"{self._thread_prefix}-{channel.value}",
                daemon=True,
            )
            self._threads[channel] = thread
            thread.start()

    def frames(self) -> Iterator[AudioFrame]:
        if not self._clock.started:
            raise RuntimeError("frames() called before start()")
        open_channels = set(self._started)
        while open_channels:
            item = self._queue.get()
            if isinstance(item, AudioFrame):
                yield item
            else:  # a channel sentinel: that pump ended
                open_channels.discard(item)

    def stop(self) -> None:
        self._stop_event.set()
        self._stop_transport()

    def _run_pump(self, channel: Channel, transport: TransportT) -> None:
        try:
            self._pump(channel, transport)
        finally:
            if not self._stop_event.is_set():
                self.stop()
            self._queue.put(channel)

    def _emit(self, channel: Channel, samples: np.ndarray) -> None:
        """Stamp a frame onto the session clock and hand it to ``frames()``."""
        timestamp = self._clock.stamp(channel, len(samples))
        self._queue.put(AudioFrame(channel=channel, timestamp=timestamp, samples=samples))

    @abstractmethod
    def _open_channel(self, channel: Channel) -> TransportT:
        """Open one channel's stream (on the ``start()`` thread)."""

    @abstractmethod
    def _pump(self, channel: Channel, transport: TransportT) -> None:
        """Blocking read loop: ``_emit`` frames until end of stream."""

    @abstractmethod
    def _stop_transport(self) -> None:
        """End every pump's read loop (idempotent, thread-safe)."""
