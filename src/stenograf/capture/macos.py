"""macOS capture provider — spawns the signed Swift helper and reads its frames.

The helper (``stenocap``) captures system audio via a Core Audio process tap
and the mic via AVAudioEngine, resamples both to mono 16 kHz int16, and streams
them as framed PCM over its stdout. This provider spawns it, parses the frames
into :class:`AudioFrame` objects, and terminates it on stop. No Python package
exposes the process-tap API, which is why the native helper exists (PLAN.md §2).

Wire protocol (helper → us), little-endian, stdout carries frames only
(status/errors go to the helper's stderr):

    each frame = channel:u8  timestamp:f64  count:u32  samples:count×i16

``channel`` is 0 for mic, 1 for system; ``timestamp`` is seconds since capture
start on a **clock shared by both channels** (the helper anchors each channel to
the Mach host time of its first buffer), so equal timestamps mean simultaneous
capture; ``samples`` is mono 16 kHz int16 PCM. There is no control channel back
to the helper — selecting channels is done with argv flags, and stopping is a
SIGINT/SIGTERM.
"""

from __future__ import annotations

import os
import signal
import struct
import subprocess
import threading
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

import numpy as np

from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel

_HEADER = struct.Struct("<BdI")  # channel u8, timestamp f64, count u32
_CHANNEL_CODE = {0: Channel.MIC, 1: Channel.SYSTEM}
_CHANNEL_FLAG = {Channel.MIC: "--mic", Channel.SYSTEM: "--system"}
_MAX_FRAME_SAMPLES = SAMPLE_RATE * 10  # sanity bound to catch stream desync (10 s)

HELPER_NAME = "stenocap"
_ENV_OVERRIDE = "STENOGRAF_CAPTURE_HELPER"


class HelperNotFoundError(RuntimeError):
    """The native capture helper binary could not be located."""


def find_helper() -> Path:
    """Locate the ``stenocap`` binary: env override, packaged bin, then dev build."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)

    packaged = resources.files("stenograf") / "bin" / HELPER_NAME
    if packaged.is_file():
        path = Path(str(packaged))
        # Some install paths drop the executable bit from wheel contents;
        # restore it on our own binary rather than failing with EACCES.
        if not os.access(path, os.X_OK):
            path.chmod(path.stat().st_mode | 0o755)
        return path

    # Dev fallback: native/helper/stenocap in the source tree.
    dev = Path(__file__).resolve().parents[3] / "native" / "helper" / HELPER_NAME
    if dev.is_file():
        return dev

    raise HelperNotFoundError(
        f"capture helper '{HELPER_NAME}' not found. Build it with "
        f"native/helper/build.sh, or set {_ENV_OVERRIDE} to its path."
    )


class MacOSCaptureProvider(CaptureProvider):
    """Streams frames from the ``stenocap`` subprocess.

    ``command`` overrides the launch command (a path or an argv prefix) — used
    to point at a fake helper in tests; production locates the signed binary via
    :func:`find_helper`.

    Echo cancellation is *not* done here. The helper used to expose a ``--aec``
    flag backed by Voice Processing IO; measured on macOS 26 it emitted no mic
    frames at all and attenuated the system channel by ~36 dB, so it was removed
    (see native/README.md). Echo is cancelled downstream, with the system channel
    as the far-end reference.
    """

    def __init__(self, *, command: str | Path | list[str] | None = None) -> None:
        if command is None:
            self._prefix = [str(find_helper())]
        elif isinstance(command, list):
            self._prefix = command
        else:
            self._prefix = [str(command)]
        self._proc: subprocess.Popen[bytes] | None = None
        # stop() is called from several threads (the capture loop on max_seconds,
        # the meeting thread on close, and the TUI's quit binding), so serialize it.
        self._stop_lock = threading.Lock()

    def start(self, channels: set[Channel]) -> None:
        argv = list(self._prefix)
        argv += [_CHANNEL_FLAG[ch] for ch in (Channel.MIC, Channel.SYSTEM) if ch in channels]
        # stdout is the binary frame stream; stderr (status/prompts) is inherited
        # so the user sees TCC prompts and any capture errors on their terminal.
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE)

    def frames(self) -> Iterator[AudioFrame]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("frames() called before start()")
        stream = self._proc.stdout
        # This iterator owns the pipe: stop() runs on other threads and must not
        # close it under an in-flight read, so the stream is closed here — at
        # end of stream or when the consumer abandons the iterator.
        try:
            while True:
                frame = read_frame(stream)
                if frame is None:
                    return  # helper closed its stdout (stopped or exited)
                yield frame
        finally:
            stream.close()

    def stop(self) -> None:
        # Idempotent + thread-safe: claim the process under the lock and null it so
        # a concurrent or repeat stop() (max_seconds, meeting-thread close, TUI quit)
        # is a no-op and only one caller ever signals/reaps it. The blocking wait
        # runs outside the lock so a second caller returns immediately.
        with self._stop_lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # helper flushes and exits on SIGINT
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # stdout is deliberately not closed here: frames() may be blocked in a
        # read on another thread, and the helper's exit already ends the stream.


def _read_exact(stream, n: int) -> bytes | None:
    """Read exactly ``n`` bytes, or return ``None`` at a clean end of stream."""
    chunks = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None  # EOF; a partial read here means the helper died mid-frame
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream) -> AudioFrame | None:
    """Parse one frame from the helper's stdout, or ``None`` at end of stream."""
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    code, timestamp, count = _HEADER.unpack(header)
    if code not in _CHANNEL_CODE or count > _MAX_FRAME_SAMPLES:
        raise ValueError(f"malformed capture frame: channel={code} count={count}")
    payload = _read_exact(stream, count * 2)
    if payload is None:
        return None
    samples = np.frombuffer(payload, dtype="<i2").astype(np.int16)
    return AudioFrame(channel=_CHANNEL_CODE[code], timestamp=timestamp, samples=samples)
