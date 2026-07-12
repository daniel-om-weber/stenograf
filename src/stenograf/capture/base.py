"""Capture provider interface.

A capture provider delivers two independent mono PCM streams — microphone
(local speakers) and system audio (remote speakers) — as in-memory frames.
Providers are platform-specific:

- macOS: a signed Swift helper subprocess (Core Audio process tap + mic),
  speaking a framed protocol over a Unix socket / stdio.
- Linux: one ``parec`` subprocess per channel (PipeWire/PulseAudio sources).
- Windows: in-process via WASAPI (mic + loopback, the soundcard package).

The core never learns where the audio came from; it only consumes
``AudioFrame`` objects. No provider may ever write audio to disk.

Also home to :class:`GapPaddedBuffer`, the one implementation of the
timestamp-anchored buffering arithmetic that consumers of frames
(echo canceller, recording tee) rely on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

SAMPLE_RATE = 16_000
"""All frames carry mono int16 PCM at this rate; providers resample."""

DEFAULT_FRAME_MS = 200
"""Frame size providers deliver to the core (~200 ms; PLAN.md §2)."""


def frame_samples(frame_ms: int) -> int:
    """Samples per delivered frame — the one frame-size computation."""
    return max(1, SAMPLE_RATE * frame_ms // 1000)

ORDER_TOLERANCE_SAMPLES = SAMPLE_RATE // 100  # 10 ms
"""Backward timestamp jitter tolerated before a frame is treated as a
stream-ordering error. Providers deliver frames monotonically per channel; a
larger backward jump means the stream desynced, and appending the frame anyway
would silently misalign everything after it (see SessionStore / WavTee)."""


class CaptureUnavailableError(RuntimeError):
    """Live capture cannot run here — missing capture stack (parec/pactl, the
    soundcard package), no default device, or OS privacy settings deny access.

    One class for every platform backend, so callers (CLI channel preview,
    doctor) can catch it without knowing which provider raised it."""


class Channel(StrEnum):
    MIC = "mic"  # local speaker(s)
    SYSTEM = "system"  # remote speakers (meeting-app output)


@dataclass(frozen=True)
class AudioFrame:
    channel: Channel
    timestamp: float
    """Seconds since session start, for the first sample of the frame.

    Both channels are stamped against one clock, so a mic frame and a system
    frame bearing the same timestamp were captured at the same instant. The
    echo canceller relies on this to align the far-end reference."""
    samples: np.ndarray
    """Mono int16 PCM at SAMPLE_RATE."""

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


class GapPaddedBuffer(ABC):
    """Timestamp-anchored int16 sample stream — subclasses provide the storage.

    The one home of the forward-gap / backward-jump arithmetic that keeps a
    channel's sample index and its timeline in agreement. Both consumers —
    the echo canceller's tracks and the recording tee's pending channels —
    build on it; a divergence between the two would silently misalign the
    recording or the AEC's far-end reference.

    Frames land at ``round(timestamp * SAMPLE_RATE)``. The stream anchors at
    the first frame by default; constructing with ``anchor=0`` anchors at
    session t=0 instead, so a late first frame gets its head padded (the
    recording tee aligns every file to the capture clock's t=0 this way). A
    forward gap wider than ``pad_gaps_over`` samples is filled with silence
    to keep the timeline honest; gaps up to it (delivery jitter) are absorbed
    by appending flush. A backward jump past ``ORDER_TOLERANCE_SAMPLES``
    raises — appending anyway would silently misalign everything after it.
    """

    def __init__(self, *, label: str, pad_gaps_over: int = 0, anchor: int | None = None) -> None:
        self._label = label
        self._pad_gaps_over = pad_gaps_over
        self._end = anchor
        """Absolute end of the stored stream, in samples; None until anchored."""

    def add(self, timestamp: float, samples: np.ndarray) -> None:
        """Append a frame's samples, silence-padding any gap before them."""
        offset = round(timestamp * SAMPLE_RATE)
        if self._end is None:
            self._end = offset
        gap = offset - self._end
        if gap < -ORDER_TOLERANCE_SAMPLES:
            raise ValueError(
                f"{self._label} frame went backwards {-gap / SAMPLE_RATE:.3f}s "
                f"(timestamp {timestamp:.3f}s): the capture stream desynced; "
                "frames arrive in order per channel"
            )
        if gap > self._pad_gaps_over:
            self._place(np.zeros(gap, dtype=np.int16))
            self._end += gap
        samples = np.asarray(samples, dtype=np.int16)
        self._place(samples)
        self._end += len(samples)

    @abstractmethod
    def _place(self, samples: np.ndarray) -> None:
        """Store ``samples`` at the end of the stream."""


class CaptureProvider(ABC):
    """Delivers live audio frames for the requested channels.

    Platform modules where the OS offers a device *choice* (linux, windows)
    additionally expose a module-level ``default_devices(channels)`` preflight
    that names what a meeting would record. Deliberately not part of this ABC:
    it must run before construction (a missing capture stack fails there, and
    the CLI reports it before models load), and macOS/file have no equivalent
    (the signed helper owns device selection; file replay has no devices).
    ``stenograf.loaders`` dispatches it per platform.
    """

    @abstractmethod
    def start(self, channels: set[Channel]) -> None:
        """Begin capture; may trigger OS permission prompts on first use."""

    @abstractmethod
    def frames(self) -> Iterator[AudioFrame]:
        """Yield frames across all started channels until ``stop`` is called."""

    @abstractmethod
    def stop(self) -> None:
        """End capture and release devices; ``frames`` iterators finish."""
