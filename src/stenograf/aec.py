"""Acoustic echo cancellation: subtract the speakers from the microphone.

The default way to sit in an online meeting is speakers + built-in mic. The
remote participants come out of the speakers, back into the mic, and land on the
mic channel — where the pipeline attributes them to *you*. Measured on this
machine, that echo sits ~24 dB above the mic's noise floor: loud enough to
transcribe, so it produces duplicate lines under a ``Local-N`` label.

We already capture the far end: the system channel *is* the audio the speakers
played. So this is a textbook echo-cancellation setup — near end (mic), far end
(system reference) — and we hand both to WebRTC's AEC3 via livekit's
``AudioProcessingModule``. The mic channel is replaced by its cleaned version;
the system channel passes through untouched (nothing echoes into the tap).

Why AEC3 and not Apple's Voice Processing IO, which is one flag away in the
capture helper: VPIO ducks other applications' audio, which would attenuate the
remote speech we are trying to transcribe (measured: −36 dB on the system
channel), and its AVAudioEngine binding delivered no mic frames at all on macOS
26. See native/README.md. Chrome ships both and defaults to this one.

Two properties of AEC3 shape the design:

- It consumes **exactly 10 ms frames** (160 samples at 16 kHz), far end first,
  then near end, for the same instant. Hence the tick pump below.
- Its internal delay estimator does the real work; ``set_stream_delay_ms`` is a
  hint. Feeding a deliberately wrong 500 ms hint measured the same 26 dB ERLE as
  the correct 25 ms, so the constant below is a nicety, not a load-bearing value.

Measured across the PLAN-AEC.md scenario matrix (quiet/loud, batch/live,
built-in/Bluetooth, double-talk), a canceller with a live reference leaks
nothing the ASR decodes. What residual echo survives comes from *losing* the
reference — a stalled or mis-clocked tap — which ``far_end_missing_ticks``
counts; a cross-channel text backstop at merge time arms itself on that signal
(``session.MeetingRecorder``), and the CLI warns when it had to act.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stenograf.capture.base import (
    ORDER_TOLERANCE_SAMPLES,
    SAMPLE_RATE,
    AudioFrame,
    CaptureProvider,
    Channel,
    GapPaddedBuffer,
)
from stenograf.recording import WavTee

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

TICK_SAMPLES = SAMPLE_RATE // 100
"""AEC3 processes exactly 10 ms at a time; neither channel may be fed anything else."""

DEFAULT_DELAY_MS = 25
"""Measured speaker → air → mic → tap round trip on a MacBook Pro (24.6 ms)."""

_FAR_HISTORY_S = 0.5
"""Reference kept behind the current tick, so a late mic frame can still pair up."""

_MAX_HOLD_S = 0.5
"""Mic backlog tolerated while waiting for the reference. Past this we cancel
against silence rather than stall the live captions: a stalled tap (device
change, sleep/wake) must degrade to "no cancellation", never to "no captions"."""


class _Track(GapPaddedBuffer):
    """A contiguous run of one channel's samples, and the timestamp of sample 0.

    Providers deliver frames monotonically per channel, but a device that drops
    buffers leaves a hole. The shared buffer pads it with silence, keeping the
    sample index and the timeline in agreement — the only reason the far-end
    reference can be looked up by timestamp at all. Jitter-sized gaps are
    absorbed (``pad_gaps_over``) rather than zero-stuffed mid-speech.
    """

    def __init__(self, channel: Channel) -> None:
        super().__init__(label=channel.value, pad_gaps_over=ORDER_TOLERANCE_SAMPLES)
        self._buf = np.zeros(0, dtype=np.int16)

    def _place(self, samples: np.ndarray) -> None:
        if self._buf.size:
            self._buf = np.concatenate([self._buf, samples])
        else:
            self._buf = samples.astype(np.int16, copy=True)

    @property
    def available(self) -> int:
        return self._buf.size

    @property
    def start_ts(self) -> float | None:
        """Timestamp of sample 0 — derived, so ``take`` advances it implicitly."""
        if self._end is None:
            return None
        return (self._end - self._buf.size) / SAMPLE_RATE

    def take(self, count: int) -> np.ndarray:
        """Remove and return the first ``count`` samples, advancing the timeline."""
        assert self._end is not None
        head, self._buf = self._buf[:count], self._buf[count:]
        return head

    def window(self, ts: float, count: int) -> np.ndarray | None:
        """``count`` samples starting at ``ts``, or None if they have not arrived.

        A window starting before this track does is padded with silence: the mic
        opens after the tap, so the reverse case (mic first) only happens when
        the tap is late, and silence is the honest reference for audio nobody
        has heard yet.
        """
        if self.start_ts is None:
            return None
        index = int(round((ts - self.start_ts) * SAMPLE_RATE))
        if index >= 0:
            if index + count > self._buf.size:
                return None
            return self._buf[index : index + count]
        pad = min(-index, count)
        if pad == count:
            return np.zeros(count, dtype=np.int16)
        tail = self._buf[: count - pad]
        if tail.size < count - pad:
            return None
        return np.concatenate([np.zeros(pad, dtype=np.int16), tail])

    def trim_before(self, ts: float) -> None:
        start = self.start_ts
        if start is None or ts <= start:
            return
        drop = min(int((ts - start) * SAMPLE_RATE), self._buf.size)
        if drop > 0:
            self._buf = self._buf[drop:]  # start_ts is derived; it advances with the slice


class EchoCanceller:
    """Replaces mic frames with echo-cancelled ones, using system as reference.

    Feed it every frame in arrival order; it returns the frames to forward. The
    system channel is returned immediately and unchanged, so remote transcription
    is never delayed or altered. Mic frames are buffered until the reference
    covers the same instant, then emitted 10 ms at a time (re-aggregated into one
    frame per call, preserving the input's timestamps).

    With no system channel there is nothing to cancel against, and every frame
    passes straight through.
    """

    def __init__(
        self,
        channels: set[Channel],
        *,
        delay_ms: int = DEFAULT_DELAY_MS,
        noise_suppression: bool = False,
        cancel: bool = True,
    ) -> None:
        self.enabled = cancel and Channel.MIC in channels and Channel.SYSTEM in channels
        self.far_end_missing_ticks = 0
        self._delay_ms = delay_ms
        self._near = _Track(Channel.MIC)
        self._far = _Track(Channel.SYSTEM)
        self._apm = None
        if self.enabled:
            # Imported here: the native lib is ~9 MB and no other path needs it.
            from livekit import rtc

            self._rtc = rtc
            self._apm = rtc.AudioProcessingModule(
                echo_cancellation=True,
                # AGC pumps the gain and fights the ASR front end; NS is off by
                # default because this is an accuracy-first transcriber and the
                # suppressor colours speech. Neither is needed to cancel echo.
                auto_gain_control=False,
                noise_suppression=noise_suppression,
                high_pass_filter=True,
            )

    def process(self, frame: AudioFrame) -> list[AudioFrame]:
        if not self.enabled:
            return [frame]
        if frame.channel is Channel.SYSTEM:
            self._far.add(frame.timestamp, frame.samples)
            return [frame]
        self._near.add(frame.timestamp, frame.samples)
        return self._drain()

    def drain(self) -> list[AudioFrame]:
        """Flush the tail at end of stream, padding the last partial tick."""
        if not self.enabled or self._near.available == 0:
            return []
        return self._drain(flush=True)

    def _drain(self, *, flush: bool = False) -> list[AudioFrame]:
        hold = int(_MAX_HOLD_S * SAMPLE_RATE)
        ticks: list[np.ndarray] = []
        first_ts: float | None = None

        while self._near.available > 0:
            partial = self._near.available < TICK_SAMPLES
            if partial and not flush:
                break
            ts = self._near.start_ts
            assert ts is not None

            far = self._far.window(ts, TICK_SAMPLES)
            if far is None:
                # The reference has not caught up. Wait — unless the mic backlog
                # says the tap has stopped, in which case forward uncancelled
                # audio rather than freezing the captions.
                if not flush and self._near.available < hold:
                    break
                far = np.zeros(TICK_SAMPLES, dtype=np.int16)
                if not flush:
                    # Only mid-capture counts as reference loss. At end-of-stream
                    # flush the far end has legitimately ended a hair before the
                    # mic tail; counting those ticks made every clean meeting
                    # report a phantom "ran without its reference for 0.0s" and
                    # spuriously armed the echo-text backstop.
                    self.far_end_missing_ticks += 1

            count = min(TICK_SAMPLES, self._near.available)
            near = self._near.take(count)
            if count < TICK_SAMPLES:
                near = np.concatenate([near, np.zeros(TICK_SAMPLES - count, dtype=np.int16)])

            cleaned = self._tick(far, near)[:count]
            if first_ts is None:
                first_ts = ts
            ticks.append(cleaned)
            self._far.trim_before(ts - _FAR_HISTORY_S)

        if first_ts is None:
            return []
        return [AudioFrame(channel=Channel.MIC, timestamp=first_ts, samples=np.concatenate(ticks))]

    def _tick(self, far: np.ndarray, near: np.ndarray) -> np.ndarray:
        """One 10 ms step: reference first, then the mic, per AEC3's contract."""
        assert self._apm is not None
        reverse = self._rtc.AudioFrame(far.tobytes(), SAMPLE_RATE, 1, TICK_SAMPLES)
        self._apm.process_reverse_stream(reverse)
        self._apm.set_stream_delay_ms(self._delay_ms)
        capture = self._rtc.AudioFrame(near.tobytes(), SAMPLE_RATE, 1, TICK_SAMPLES)
        self._apm.process_stream(capture)
        return np.frombuffer(bytes(capture.data), dtype=np.int16)


class AecDump:
    """The mic/lpb/enh WAV triple that ``eval/aec_score.py`` scores.

    AECMOS naming: ``mic.wav`` is the near end as the device heard it,
    ``lpb.wav`` (loopback) is the far-end reference the canceller saw, and
    ``enh.wav`` is the mic as the ASR receives it. All three are mono 16 kHz
    and share the capture clock's t=0 (``WavTee`` pads each file's head up to
    its first frame's timestamp), so they are sample-aligned for scoring.

    Opt-in via ``--aec-dump``: like ``--record-audio``, this writes meeting
    audio to disk.
    """

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._mic = WavTee(directory / "mic.wav", {Channel.MIC})
        self._lpb = WavTee(directory / "lpb.wav", {Channel.SYSTEM})
        self._enh = WavTee(directory / "enh.wav", {Channel.MIC})

    def add_input(self, frame: AudioFrame) -> None:
        (self._mic if frame.channel is Channel.MIC else self._lpb).add(frame)

    def add_output(self, frame: AudioFrame) -> None:
        if frame.channel is Channel.MIC:
            self._enh.add(frame)

    def close(self) -> None:
        self._mic.close()
        self._lpb.close()
        self._enh.close()


class EchoCancellingProvider(CaptureProvider):
    """Wraps a provider, cancelling speaker bleed out of its mic channel.

    ``cancel=False`` keeps the wrapper as a pure pass-through — used with
    ``dump_dir`` to record the uncancelled baseline the eval rig compares
    against (``--no-aec --aec-dump``).
    """

    def __init__(
        self,
        inner: CaptureProvider,
        *,
        delay_ms: int = DEFAULT_DELAY_MS,
        noise_suppression: bool = False,
        cancel: bool = True,
        dump_dir: Path | None = None,
    ) -> None:
        self._inner = inner
        self._delay_ms = delay_ms
        self._noise_suppression = noise_suppression
        self._cancel = cancel
        self._dump_dir = dump_dir
        self._dump: AecDump | None = None
        self._canceller: EchoCanceller | None = None
        self._iterating = False

    @property
    def canceller(self) -> EchoCanceller | None:
        return self._canceller

    def start(self, channels: set[Channel]) -> None:
        self._canceller = EchoCanceller(
            channels,
            delay_ms=self._delay_ms,
            noise_suppression=self._noise_suppression,
            cancel=self._cancel,
        )
        if self._dump_dir is not None:
            self._dump = AecDump(self._dump_dir)
        self._inner.start(channels)

    def frames(self) -> Iterator[AudioFrame]:
        assert self._canceller is not None, "frames() called before start()"
        # This iterator owns the dump while it runs: a stop() mid-meeting must
        # not close the WAVs under it — the canceller tail drained after the
        # inner stream ends still lands in enh.wav.
        self._iterating = True
        try:
            for frame in self._inner.frames():
                if self._dump is not None:
                    self._dump.add_input(frame)
                for produced in self._canceller.process(frame):
                    if self._dump is not None:
                        self._dump.add_output(produced)
                    yield produced
            for produced in self._canceller.drain():
                if self._dump is not None:
                    self._dump.add_output(produced)
                yield produced
        finally:
            self._iterating = False
            self._close_dump()

    def stop(self) -> None:
        self._inner.stop()
        # start() → stop() without ever iterating frames(): nobody else will
        # close the dump's three WAV handles.
        if not self._iterating:
            self._close_dump()

    def _close_dump(self) -> None:
        dump, self._dump = self._dump, None
        if dump is not None:
            dump.close()
