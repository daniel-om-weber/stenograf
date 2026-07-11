"""Linux capture provider — one ``parec`` subprocess per channel.

PipeWire (through its PulseAudio compatibility layer) and plain PulseAudio
expose system audio as every sink's *monitor* source, so no native helper is
needed: ``parec`` — the PulseAudio client shipped alongside ``pactl`` —
streams any source as raw PCM on stdout, resampled server-side to our wire
format (mono 16 kHz s16le). One subprocess per channel:

- mic    → ``--device=@DEFAULT_SOURCE@``  (the default input device)
- system → ``--device=@DEFAULT_MONITOR@`` (the default output's monitor)

Decision B (PLAN.md §5 Phase 5 sub-plan) was settled on the CachyOS notebook
(PipeWire 1.6.7) against the SoundCard ``include_loopback`` candidate: parec
adds no Python dependency, mirrors the macOS helper's subprocess-streaming
architecture, and measured a clean ~86 ms delivery cadence where SoundCard
showed a ~1 s startup gap. Measured behaviours the device names rely on:

- ``@DEFAULT_MONITOR@`` **follows** a default-sink change mid-capture: the
  user plugs in a headset, WirePlumber moves the meeting app's playback
  stream to it, and the capture moves with it. ``@DEFAULT_SOURCE@`` pins to
  the mic that was default at start (an acceptable asymmetry — the remote
  channel is the one that must survive an output-device switch).
- A deviceless ``parec`` does NOT track the pulse default under
  pipewire-pulse (WirePlumber auto-routes it instead, which never picks a
  monitor), so the special names are load-bearing, not convenience.
- A muted sink's monitor delivers silence (sink volume does not attenuate
  it). A meeting the user can hear is by definition not muted, so this only
  bites test rigs — a fresh ``module-null-sink`` loads muted here.

Both channels are stamped against one clock: ``start()`` anchors a shared
t=0 and each channel pins itself to it at its first delivered frame, like
the macOS helper anchoring each channel to the host time of its first
buffer. No code path writes audio to disk.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from queue import SimpleQueue

import numpy as np

from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel

DEFAULT_FRAME_MS = 200
"""Frame size delivered to the core (~200 ms, matching the macOS helper)."""

_PAREC_LATENCY_MS = 100
"""Target stream latency asked of the server — keeps delivery well under the
frame size so a frame's arrival time tracks its capture time."""

_CHANNEL_DEVICE = {Channel.MIC: "@DEFAULT_SOURCE@", Channel.SYSTEM: "@DEFAULT_MONITOR@"}


class CaptureUnavailableError(RuntimeError):
    """Live capture cannot run here (no parec/pactl, no server, no device)."""


def default_devices(channels: set[Channel]) -> dict[Channel, str]:
    """What each channel's ``@DEFAULT_*@`` alias resolves to right now.

    Asks ``pactl`` so a missing tool, an unreachable sound server, or an
    absent default device fails *before* capture starts (the CLI turns this
    into a clean error at provider construction), and so the CLI can tell the
    user which devices the meeting will actually record.
    """
    devices = {}
    for channel in sorted(channels):
        if channel is Channel.MIC:
            devices[channel] = _pactl_default("get-default-source", "microphone")
        else:
            devices[channel] = _pactl_default("get-default-sink", "output device") + ".monitor"
    return devices


def _pactl_default(command: str, what: str) -> str:
    try:
        proc = subprocess.run(["pactl", command], capture_output=True, text=True, timeout=10)
    except FileNotFoundError as exc:
        raise CaptureUnavailableError(
            "pactl not found — live capture needs the PulseAudio client tools "
            "(package pipewire-pulse or pulseaudio-utils, depending on distro)"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CaptureUnavailableError(f"pactl {command} timed out — sound server hung?") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"exit {proc.returncode}"
        raise CaptureUnavailableError(
            f"pactl {command} failed ({detail}) — is PipeWire or PulseAudio running?"
        )
    name = proc.stdout.strip()
    if not name:
        raise CaptureUnavailableError(f"no default {what} is configured (pactl {command})")
    return name


class LinuxCaptureProvider(CaptureProvider):
    """Streams frames from one ``parec`` subprocess per captured channel.

    ``command`` overrides the parec launch command (a path or an argv prefix)
    — used to point at a fake in tests; production requires ``parec`` on PATH
    and fails at construction otherwise, mirroring ``find_helper`` on macOS.

    Each channel gets its own reader thread that slices the byte stream into
    ~200 ms frames and stamps them onto the shared session clock; ``frames()``
    drains their queue. The threads also decouple parec's pipe from a stalled
    consumer — the queue grows (bounded by RAM, ~64 KB/s/channel) instead of
    the pipe filling and the server overrunning the stream.
    """

    def __init__(self, *, command: str | list[str] | None = None, frame_ms: int = DEFAULT_FRAME_MS):
        if command is None:
            if shutil.which("parec") is None:
                raise CaptureUnavailableError(
                    "parec not found — live capture needs the PulseAudio client tools "
                    "(package pipewire-pulse or pulseaudio-utils, depending on distro)"
                )
            self._prefix = ["parec"]
        elif isinstance(command, list):
            self._prefix = command
        else:
            self._prefix = [str(command)]
        self._frame_samples = max(1, SAMPLE_RATE * frame_ms // 1000)
        self._procs: dict[Channel, subprocess.Popen[bytes]] = {}
        self._queue: SimpleQueue[AudioFrame | Channel] = SimpleQueue()
        self._t0: float | None = None
        self._started: set[Channel] = set()
        self._stopping = False
        # stop() is called from several threads (the capture loop on max_seconds,
        # the meeting thread on close, the TUI's quit binding, and a reader on an
        # unexpected stream death), so serialize claiming the processes.
        self._stop_lock = threading.Lock()

    def start(self, channels: set[Channel]) -> None:
        self._t0 = time.monotonic()
        self._started = set(channels)
        self._stopping = False
        for channel in sorted(channels):
            argv = [
                *self._prefix,
                f"--device={_CHANNEL_DEVICE[channel]}",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                "--format=s16le",
                "--raw",
                f"--latency-msec={_PAREC_LATENCY_MS}",
                "--client-name=stenograf",
                f"--stream-name={channel.value}",
            ]
            # stdout is the raw PCM stream; stderr (server errors) is inherited
            # so the user sees why a stream died on their terminal.
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE)
            self._procs[channel] = proc
            threading.Thread(
                target=self._pump, args=(channel, proc), name=f"parec-{channel.value}", daemon=True
            ).start()

    def frames(self) -> Iterator[AudioFrame]:
        if self._t0 is None:
            raise RuntimeError("frames() called before start()")
        open_channels = set(self._started)
        while open_channels:
            item = self._queue.get()
            if isinstance(item, AudioFrame):
                yield item
            else:  # a channel sentinel: that reader hit end of stream
                open_channels.discard(item)

    def stop(self) -> None:
        # Idempotent + thread-safe: claim the processes under the lock so only
        # one caller signals/reaps them; the blocking waits run outside it.
        with self._stop_lock:
            procs, self._procs = self._procs, {}
            self._stopping = True
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    def _pump(self, channel: Channel, proc: subprocess.Popen[bytes]) -> None:
        """Read one channel's byte stream into timestamped frames.

        The channel anchors itself to the shared clock at its first frame —
        arrival time minus the frame's duration, since those samples were
        captured over the preceding frame-length — and derives every later
        timestamp from the sample count, so gaps in *delivery* never shift
        audio in session time. On end of stream (a stop, or the stream dying)
        it enqueues its channel as a sentinel; an unexpected death also tears
        down the other channel so the meeting ends visibly and finalizes,
        rather than silently continuing half-captured.
        """
        stream = proc.stdout
        assert stream is not None  # Popen(stdout=PIPE) above
        frame_bytes = self._frame_samples * 2
        anchor: float | None = None
        delivered = 0
        try:
            while True:
                data = _read_up_to(stream, frame_bytes)
                if len(data) < 2:
                    return
                samples = np.frombuffer(data[: len(data) & ~1], dtype="<i2").astype(np.int16)
                if anchor is None:
                    elapsed = time.monotonic() - (self._t0 or 0.0)
                    anchor = max(0.0, elapsed - len(samples) / SAMPLE_RATE)
                timestamp = anchor + delivered / SAMPLE_RATE
                delivered += len(samples)
                self._queue.put(AudioFrame(channel=channel, timestamp=timestamp, samples=samples))
        finally:
            stream.close()
            if not self._stopping:
                self.stop()
            self._queue.put(channel)


def _read_up_to(stream, n: int) -> bytes:
    """Read ``n`` bytes, or whatever remains before end of stream."""
    chunks = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
