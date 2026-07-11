"""Windows capture provider — WASAPI shared-mode streams via ``soundcard``.

Windows exposes system audio as WASAPI *loopback* capture on a render device,
so no native helper is needed: the ``soundcard`` package opens shared-mode
capture streams for both channels —

- mic    → the default input device
- system → loopback capture on the default output device

**Decision D (PLAN.md §5 Phase 6) resolved to ``soundcard``** over
``pyaudiowpatch``, spiked on real hardware (Windows 11 notebook, 2026-07-11):
one API covers mic and loopback, and it initializes WASAPI with
``AUTOCONVERTPCM | SRC_DEFAULT_QUALITY``, so Windows resamples server-side to
our wire rate like parec does on Linux — no Python resampler dependency.
Measured: recorder open 15–80 ms, first frame < 300 ms, exact frame-sized
delivery cadence, a 440 Hz test tone recovered bit-clean through loopback
(the ~1 s startup gap SoundCard showed under PipeWire is a Pulse-backend
artifact; it does not occur on the native Windows backend).

Two WASAPI behaviours the design leans on:

- **Loopback delivers no packets while nothing renders.** ``soundcard``
  papers over this by synthesizing zeros from the measured idle time, so the
  stream stays continuous — but the fill is wall-clock *estimated*, so
  sample-count-derived timestamps can drift from session time across long
  silences. The pump re-anchors whenever its derived clock falls behind the
  arrival-derived one by more than ``_REANCHOR_TOLERANCE_S`` (forward only:
  per-channel timestamps must stay monotonic, and ``SessionStore`` pads the
  skipped span with silence).
- **COM apartments are per-thread**, so each channel's device is resolved and
  its recorder opened *inside* its own pump thread (spike-verified working).

Both channels are stamped against one clock: ``start()`` anchors a shared
t=0 and each channel pins itself to it at its first delivered frame, like
the other providers. Both devices pin to the defaults at meeting start; a
mid-meeting default switch is not followed (unlike ``@DEFAULT_MONITOR@`` on
Linux — WASAPI has no equivalent alias). No code path writes audio to disk.
"""

from __future__ import annotations

import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from queue import SimpleQueue

import numpy as np

from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel

DEFAULT_FRAME_MS = 200
"""Frame size delivered to the core (~200 ms, matching the other providers)."""

_REANCHOR_TOLERANCE_S = 0.5
"""How far a channel's sample-derived clock may fall behind its
arrival-derived clock before the pump re-anchors. Generous enough that
delivery jitter (one frame + WASAPI buffering, ~0.25 s worst measured) never
trips it; tight enough that a mis-estimated silence gap cannot skew the AEC's
far-end alignment or the transcript for the rest of the meeting."""


class CaptureUnavailableError(RuntimeError):
    """Live capture cannot run here (no soundcard package, no default device)."""


def _import_soundcard():
    try:
        import soundcard
    except Exception as exc:  # ImportError, or COM/cffi init failures
        raise CaptureUnavailableError(
            f"the soundcard package is unavailable ({exc}) — reinstall stenograf, "
            "or `pip install soundcard`, to capture on Windows"
        ) from exc
    # Loopback silence gaps set WASAPI's discontinuity flag when audio
    # resumes; that is expected and handled (zero-fill + re-anchor), so the
    # per-gap warning would only spam the terminal/TUI.
    if (category := getattr(soundcard, "SoundcardRuntimeWarning", None)) is not None:
        warnings.filterwarnings("ignore", category=category)
    return soundcard


def default_devices(channels: set[Channel]) -> dict[Channel, str]:
    """What each channel would record from right now.

    Resolves the default devices the same way the pumps will at start, so a
    missing package or an absent default device fails *before* capture (and
    models) start, and so the CLI can name what the meeting will record —
    the loopback-of-default-output choice is invisible otherwise.
    """
    soundcard = _import_soundcard()
    devices = {}
    for channel in sorted(channels):
        device = _default_device(soundcard, channel)
        suffix = " (loopback)" if channel is Channel.SYSTEM else ""
        devices[channel] = f"{device.name}{suffix}"
    return devices


def _default_device(soundcard, channel: Channel):
    """The soundcard device a channel records from (mic, or output loopback)."""
    if channel is Channel.MIC:
        try:
            return soundcard.default_microphone()
        except Exception as exc:
            raise CaptureUnavailableError(
                f"no default microphone ({exc}) — check Windows sound settings"
            ) from exc
    try:
        speaker = soundcard.default_speaker()
        return soundcard.get_microphone(speaker.id, include_loopback=True)
    except Exception as exc:
        raise CaptureUnavailableError(
            f"no default output device to loopback-capture ({exc}) — check Windows sound settings"
        ) from exc


class WindowsCaptureProvider(CaptureProvider):
    """Streams frames from one WASAPI capture stream per captured channel.

    ``backend`` overrides the soundcard module (a fake in tests); production
    imports the real one and fails at construction when it is missing,
    mirroring ``find_helper`` on macOS and the parec check on Linux. ``clock``
    overrides the session clock (tests drive the re-anchor logic with it).

    Each channel gets one pump thread that owns its device end to end (COM
    objects are apartment-bound): it opens the recorder, downmixes the
    device's float32 channels to mono int16, and stamps ~200 ms frames onto
    the shared session clock; ``frames()`` drains their queue.
    """

    def __init__(
        self,
        *,
        backend=None,
        frame_ms: int = DEFAULT_FRAME_MS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._soundcard = backend if backend is not None else _import_soundcard()
        self._clock = clock
        self._frame_samples = max(1, SAMPLE_RATE * frame_ms // 1000)
        self._queue: SimpleQueue[AudioFrame | Channel] = SimpleQueue()
        self._t0: float | None = None
        self._started: set[Channel] = set()
        self._threads: dict[Channel, threading.Thread] = {}
        self._stop_event = threading.Event()

    def start(self, channels: set[Channel]) -> None:
        self._t0 = self._clock()
        self._started = set(channels)
        self._stop_event.clear()
        for channel in sorted(channels):
            thread = threading.Thread(
                target=self._pump, args=(channel,), name=f"wasapi-{channel.value}", daemon=True
            )
            self._threads[channel] = thread
            thread.start()

    def frames(self) -> Iterator[AudioFrame]:
        if self._t0 is None:
            raise RuntimeError("frames() called before start()")
        open_channels = set(self._started)
        while open_channels:
            item = self._queue.get()
            if isinstance(item, AudioFrame):
                yield item
            else:  # a channel sentinel: that pump ended
                open_channels.discard(item)

    def stop(self) -> None:
        # Idempotent + thread-safe (an Event); pumps notice within one frame
        # read (~frame_ms + WASAPI's silence threshold) and release their
        # devices on the way out. Also called *from* a pump thread on an
        # unexpected stream death, hence the current-thread guard.
        self._stop_event.set()
        current = threading.current_thread()
        for thread in self._threads.values():
            if thread is not current:
                thread.join(timeout=5)

    def _pump(self, channel: Channel) -> None:
        """Own one channel end to end: device, recorder, framing, timestamps.

        The channel anchors itself to the shared clock at its first frame —
        arrival time minus the frame's duration — and derives every later
        timestamp from the sample count, so delivery jitter never shifts
        audio in session time. The one exception is the forward re-anchor
        after an under-filled loopback silence (module docstring). On end it
        enqueues its channel as a sentinel; an unexpected death also tears
        down the other channel so the meeting ends visibly and finalizes,
        rather than silently continuing half-captured.
        """
        anchor: float | None = None
        delivered = 0
        try:
            device = _default_device(self._soundcard, channel)
            with device.recorder(samplerate=SAMPLE_RATE) as recorder:
                while not self._stop_event.is_set():
                    block = recorder.record(self._frame_samples)
                    samples = _to_mono_int16(block)
                    if not len(samples):
                        continue
                    elapsed = self._clock() - (self._t0 or 0.0)
                    started_at = max(0.0, elapsed - len(samples) / SAMPLE_RATE)
                    if anchor is None:
                        anchor = started_at
                    timestamp = anchor + delivered / SAMPLE_RATE
                    if started_at - timestamp > _REANCHOR_TOLERANCE_S:
                        anchor += started_at - timestamp
                        timestamp = started_at
                    delivered += len(samples)
                    self._queue.put(
                        AudioFrame(channel=channel, timestamp=timestamp, samples=samples)
                    )
        except Exception as exc:
            # The other providers inherit their subprocess's stderr; this is
            # the in-process equivalent so the user sees why a stream died.
            print(f"stenograf: {channel.value} capture stream died: {exc}", file=sys.stderr)
        finally:
            if not self._stop_event.is_set():
                self.stop()
            self._queue.put(channel)


def _to_mono_int16(block: np.ndarray) -> np.ndarray:
    """Downmix a float32 frames×channels block to the wire format."""
    mono = block.mean(axis=1) if block.ndim == 2 else block
    return np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
