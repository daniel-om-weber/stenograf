"""File-replay capture provider (dev/test).

Replays one or two audio files as a live capture session — mic and/or system
channel — so the meeting orchestrator runs end-to-end before the native
capture helper exists. Frames from both channels are yielded in timestamp
order, matching what a real provider delivers.

By default it emits as fast as the consumer reads (batch finalize dev/test).
With ``paced=True`` it releases each frame at its wall-clock timestamp, so
``--replay`` exercises the live pass at genuine meeting cadence (PLAN.md §5
Task 3) — the property that stresses the LiveWorker's real-time behaviour.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from stenograf.audio import load_audio, to_int16
from stenograf.capture.base import (
    DEFAULT_FRAME_MS,
    SAMPLE_RATE,
    AudioFrame,
    CaptureProvider,
    Channel,
    frame_samples,
)


class FileCaptureProvider(CaptureProvider):
    """Emits the given files as framed PCM on their channels."""

    def __init__(
        self,
        sources: dict[Channel, Path | str],
        *,
        frame_ms: int = DEFAULT_FRAME_MS,
        paced: bool = False,
    ):
        self._sources = {ch: Path(p) for ch, p in sources.items()}
        self._frame = frame_samples(frame_ms)
        self._paced = paced
        self._loaded: dict[Channel, np.ndarray] = {}
        self._stopped = False

    def start(self, channels: set[Channel]) -> None:
        self._stopped = False
        self._loaded = {
            ch: to_int16(load_audio(path)) for ch, path in self._sources.items() if ch in channels
        }

    def frames(self) -> Iterator[AudioFrame]:
        pending = []  # (timestamp, channel, samples), merged across channels
        for channel, pcm in self._loaded.items():
            for start in range(0, len(pcm), self._frame):
                chunk = pcm[start : start + self._frame]
                pending.append((start / SAMPLE_RATE, channel, chunk))
        pending.sort(key=lambda f: (f[0], f[1]))
        origin = time.monotonic()
        for timestamp, channel, chunk in pending:
            if self._stopped:
                return
            if self._paced:
                # Release the frame at its wall-clock arrival time; a stop is seen
                # within at most one frame (the sleep is bounded by frame_ms).
                delay = origin + timestamp - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield AudioFrame(channel=channel, timestamp=timestamp, samples=chunk)

    def stop(self) -> None:
        self._stopped = True
