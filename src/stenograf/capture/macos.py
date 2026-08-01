"""macOS capture provider — spawns the signed Swift helper and reads its frames.

The helper (``stenocap``) captures system audio via a Core Audio process tap
and the mic via AVAudioEngine, resamples both to mono 16 kHz int16, and streams
them as framed PCM over its stdout. This provider spawns it, parses the frames
into :class:`AudioFrame` objects, and terminates it on stop. No Python package
exposes the process-tap API, which is why the native helper exists.

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
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from stenograf.capture.base import (
    SAMPLE_RATE,
    AudioFrame,
    CaptureHelperError,
    CaptureProvider,
    Channel,
)
from stenograf.capture.streaming import read_exact, relay_lines

if TYPE_CHECKING:
    from collections.abc import Callable

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

    # Dev fallback: native/stenocap-macos/stenocap in the source tree.
    dev = Path(__file__).resolve().parents[3] / "native" / "stenocap-macos" / HELPER_NAME
    if dev.is_file():
        return dev

    raise HelperNotFoundError(
        f"capture helper '{HELPER_NAME}' not found. Build it with "
        f"native/stenocap-macos/build.sh, or set {_ENV_OVERRIDE} to its path."
    )


class MacOSCaptureProvider(CaptureProvider):
    """Streams frames from the ``stenocap`` subprocess.

    ``command`` overrides the launch command (a path or an argv prefix) — used
    to point at a fake helper in tests; production locates the signed binary via
    :func:`find_helper`.

    ``on_log`` receives the helper's stderr, one decoded line at a time.
    ``None`` (the CLI) inherits stderr so capture errors land on the terminal
    as ever; the Qt meeting screen installs a sink instead (loaders.CaptureLog)
    — a GUI process has no terminal for a raw stderr write to reach.

    Echo cancellation is *not* done here. The helper used to expose a ``--aec``
    flag backed by Voice Processing IO; measured on macOS 26 it emitted no mic
    frames at all and attenuated the system channel by ~36 dB, so it was removed
    (see native/README.md). Echo is cancelled downstream, with the system channel
    as the far-end reference.
    """

    def __init__(
        self,
        *,
        command: str | Path | list[str] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        if command is None:
            self._prefix = [str(find_helper())]
        elif isinstance(command, list):
            self._prefix = command
        else:
            self._prefix = [str(command)]
        self._on_log = on_log
        self._proc: subprocess.Popen[bytes] | None = None
        self._queue: queue.SimpleQueue[AudioFrame | Exception | None] | None = None
        self._drainer: threading.Thread | None = None
        self._log_relay: threading.Thread | None = None
        self._channels: set[Channel] = set()
        # The last helper stderr lines (on_log mode only): a crash's FATAL line
        # goes into the CaptureHelperError message so the failure toast says
        # *why*, not just that the helper died.
        self._stderr_tail: deque[str] = deque(maxlen=8)
        self._stop_requested = False
        self._respawned = False
        # stop() is called from several threads (the capture loop on max_seconds,
        # the meeting thread on close, and the GUI's stop button), so serialize it.
        self._stop_lock = threading.Lock()

    def start(self, channels: set[Channel]) -> None:
        self._channels = set(channels)
        self._stop_requested = False
        self._respawned = False
        # The queue outlives a helper respawn (frames() re-reads the attribute
        # per item, but one queue for both spawns is simpler and the first
        # drainer's end-of-stream marker is consumed before the second starts).
        self._queue = queue.SimpleQueue()
        with self._stop_lock:
            self._spawn()

    def _spawn(self) -> None:
        """Launch the helper and its drain/relay threads. Caller holds _stop_lock."""
        argv = list(self._prefix)
        argv += [_CHANNEL_FLAG[ch] for ch in (Channel.MIC, Channel.SYSTEM) if ch in self._channels]
        # stdout is the binary frame stream. stderr (status/errors) is inherited
        # by default so capture errors land on the plain CLI's terminal — but
        # with an on_log sink it is piped and relayed line-by-line instead,
        # keeping the helper's chatter off the TUI's screen (class docstring).
        stderr = None if self._on_log is None else subprocess.PIPE
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr)
        assert self._proc.stdout is not None
        if self._on_log is not None:
            assert self._proc.stderr is not None
            self._log_relay = threading.Thread(
                target=relay_lines,
                args=(self._proc.stderr, self._tail_and_forward),
                name="stenocap-log",
                daemon=True,
            )
            self._log_relay.start()
        # The pipe is drained on a dedicated thread, whatever the consumer does.
        # If the consumer stalls with the pipe full (it holds 64 KB ≈ 1 s of
        # frames), the helper blocks in write(), Core Audio decides its IO
        # callback is unresponsive and kills the tap — permanently, mid-meeting;
        # ebf660a and 7dd1510 both reached production through this. The queue is
        # unbounded on purpose: the meeting already lives in RAM (SessionStore),
        # so buffering here (~64 KB/s) costs nothing new, while dropping frames
        # would lose meeting audio to a stall the consumer recovers from.
        self._drainer = threading.Thread(
            target=_drain_pipe,
            args=(self._proc.stdout, self._queue),
            name="stenocap-drain",
            daemon=True,
        )
        self._drainer.start()

    def _tail_and_forward(self, line: str) -> None:
        # Tail first: relay_lines suppresses sink exceptions per line, and the
        # crash diagnostics must survive even a broken on_log.
        self._stderr_tail.append(line)
        self._on_log(line)  # type: ignore[misc]  # only installed when on_log is set

    def frames(self) -> Iterator[AudioFrame]:
        if self._queue is None:
            raise RuntimeError("frames() called before start()")
        delivered = False
        while True:
            item = self._queue.get()
            if item is None:
                # Helper closed its stdout — a stop, a natural end, or a crash.
                if not self._ended_cleanly(delivered):
                    continue  # respawned after a startup crash — keep reading
                return
            if isinstance(item, Exception):
                raise item  # stream desync, noticed by the drain thread
            delivered = True
            yield item

    def _ended_cleanly(self, delivered: bool) -> bool:
        """At end-of-stream: was this a clean stop (True) or a helper crash?

        A crash before any frame is retried once — measured 2026-07-20, a
        concurrent capture app (OBS, a second stenograf) makes coreaudiod
        flakily fail or hang the helper's device setup, and an immediate
        second attempt often succeeds. A second startup crash, or any crash
        after audio flowed, raises :class:`CaptureHelperError`; returning a
        clean end here is what used to finalize an empty transcript that
        looked like a successful meeting.
        """
        proc = self._proc
        if proc is None or self._stop_requested:
            return True  # stop() claimed it — a requested stop
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # stdout closed but process wedged
            proc.kill()
            code = proc.wait()
        if code == 0:
            return True  # helper's own clean exit (external SIGINT/SIGTERM)
        # Join the stderr relay so the helper's last lines (the FATAL) are in
        # the tail before we compose the error.
        if self._log_relay is not None:
            self._log_relay.join(timeout=2)
        detail = next((ln for ln in reversed(self._stderr_tail) if "FATAL" in ln), None)
        if detail is None:
            detail = f"capture helper exited with status {code}"
            if self._stderr_tail:
                detail += f" (last: {self._stderr_tail[-1]})"
        if not delivered and not self._respawned:
            self._respawned = True
            self._announce(f"capture failed at startup ({detail}); retrying once")
            time.sleep(1.0)  # give coreaudiod a beat to settle
            with self._stop_lock:
                if self._stop_requested:
                    return True
                self._spawn()
            return False
        raise CaptureHelperError(detail)

    def _announce(self, message: str) -> None:
        line = f"capture: {message}"
        if self._on_log is not None:
            self._on_log(line)
        else:  # plain CLI — stderr is where the helper's own chatter lands
            print(line, file=sys.stderr)

    def stop(self) -> None:
        # Idempotent + thread-safe: claim the process under the lock and null it so
        # a concurrent or repeat stop() (max_seconds, meeting-thread close, TUI quit)
        # is a no-op and only one caller ever signals/reaps it. The blocking wait
        # runs outside the lock so a second caller returns immediately.
        with self._stop_lock:
            self._stop_requested = True
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
        # The helper's exit ends the stream: the drain thread sees EOF, queues
        # the end-of-stream marker for frames(), closes the pipe, and exits.
        # The stderr relay (if piping) sees EOF the same way; join it too so
        # the helper's last lines ("stopped") reach the sink before stop()
        # returns — the CLI replays buffered problems right after.
        for thread in (self._drainer, self._log_relay):
            if thread is not None:
                thread.join(timeout=5)


def _drain_pipe(stream, out: queue.SimpleQueue) -> None:
    """Pump helper frames into the queue at capture rate, whatever downstream does.

    Runs on its own thread and owns the pipe: it closes it at end of stream,
    after queueing the ``None`` end-of-stream marker. A malformed frame (stream
    desync) is queued as the exception for ``frames()`` to re-raise.
    """
    try:
        while True:
            frame = read_frame(stream)
            if frame is None:
                out.put(None)
                return
            out.put(frame)
    except Exception as exc:
        out.put(exc)
    finally:
        stream.close()


def read_frame(stream) -> AudioFrame | None:
    """Parse one frame from the helper's stdout, or ``None`` at end of stream."""
    header = read_exact(stream, _HEADER.size)
    if header is None:
        return None
    code, timestamp, count = _HEADER.unpack(header)
    if code not in _CHANNEL_CODE or count > _MAX_FRAME_SAMPLES:
        raise ValueError(f"malformed capture frame: channel={code} count={count}")
    payload = read_exact(stream, count * 2)
    if payload is None:
        return None
    samples = np.frombuffer(payload, dtype="<i2").astype(np.int16)
    return AudioFrame(channel=_CHANNEL_CODE[code], timestamp=timestamp, samples=samples)
