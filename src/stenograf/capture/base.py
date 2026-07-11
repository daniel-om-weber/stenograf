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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

SAMPLE_RATE = 16_000
"""All frames carry mono int16 PCM at this rate; providers resample."""

ORDER_TOLERANCE_SAMPLES = SAMPLE_RATE // 100  # 10 ms
"""Backward timestamp jitter tolerated before a frame is treated as a
stream-ordering error. Providers deliver frames monotonically per channel; a
larger backward jump means the stream desynced, and appending the frame anyway
would silently misalign everything after it (see SessionStore / WavTee)."""


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


class CaptureProvider(ABC):
    """Delivers live audio frames for the requested channels."""

    @abstractmethod
    def start(self, channels: set[Channel]) -> None:
        """Begin capture; may trigger OS permission prompts on first use."""

    @abstractmethod
    def frames(self) -> Iterator[AudioFrame]:
        """Yield frames across all started channels until ``stop`` is called."""

    @abstractmethod
    def stop(self) -> None:
        """End capture and release devices; ``frames`` iterators finish."""
