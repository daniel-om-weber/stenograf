"""Opt-in audio recording — audio touches disk only on explicit request.

Off by default: stenograf's guarantee is that audio stays in RAM. When the user
explicitly passes ``--record-audio``, this tee additionally appends the incoming
PCM to a file as it arrives — mic on the left channel, system audio on the
right (mono when only one channel is captured), preserving the channel
separation in a file any player opens.

The target's suffix picks the format. The default ``audio.opus`` pipes the PCM
through the bundled ffmpeg's libopus at 32 kbps per channel — transparent to
both transcription and speaker embeddings at ~a tenth of WAV's size (evidence:
eval/README.md "Stored-audio codec") — and ``steno transcribe`` reads it back
through the same ffmpeg. A ``.wav`` target (an explicit ``--record-audio
foo.wav``, and every AEC dump track) writes the capture wire format instead:
mono 16 kHz int16 per channel, bit-exact.

Both sinks are crash-safe like the incremental text checkpoints. The WAV
header's size fields are rewritten after every drain, so a process killed
mid-meeting leaves a playable file missing only the last, not-yet-aligned
tail; Ogg pages are self-delimiting, so a killed Opus encoder leaves a
readable file missing only its buffered tail (measured 2026-08-06: ~5 s at
SIGKILL). And because a recording the user asked for must survive encoder
trouble, a spawn failure or a mid-meeting encoder death degrades the tee to a
``.wav`` beside the target from that point on, reported via
:attr:`AudioTee.fallback_path`.
"""

from __future__ import annotations

import contextlib
import struct
import subprocess
import wave
from collections import deque
from pathlib import Path

import numpy as np

from stenograf.audio import ffmpeg_exe
from stenograf.capture.base import (
    SAMPLE_RATE,
    AudioFrame,
    Channel,
    GapPaddedBuffer,
)

_BYTES_PER_SAMPLE = 2  # int16
_BITS_PER_SAMPLE = 16
_PCM = 1  # WAV format tag

# Per channel because the tee's channels are independent voice feeds (mic,
# system), not a stereo image sharing content. The value itself is a shipped
# default whose evidence lives in eval/README.md "Stored-audio codec".
_OPUS_BITRATE_PER_CHANNEL = 32_000

# Stereo layout is fixed so recordings are always mic-left / system-right.
_CHANNEL_ORDER = (Channel.MIC, Channel.SYSTEM)


def read_channels(path: Path | str, channels: list[Channel]) -> dict[Channel, np.ndarray]:
    """Read a ``.wav`` tee recording back into its per-channel int16 streams.

    The exact inverse of the tee's fixed layout (mic left, system right; mono when
    a single channel was recorded). ``channels`` is the ordered channel list
    the recording holds — the
    meeting's captured channels (``mic`` before ``system``, matching the tee) — and
    disambiguates a mono file, which the WAV header alone cannot (a mono recording
    is mic-only *or* system-only depending on the meeting mode). Its length must
    equal the file's channel count.

    Raises ``ValueError`` if the file is not the 16 kHz 16-bit PCM WAV the tee
    writes, or if its channel count does not match ``channels`` — so it targets a
    recorded meeting, not an arbitrary imported source file.
    """
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != _BYTES_PER_SAMPLE or w.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"{Path(path).name} is not a 16 kHz 16-bit PCM WAV; "
                "re-finalize needs a stenograf --record-audio recording"
            )
        nchannels = w.getnchannels()
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if nchannels != len(channels):
        raise ValueError(
            f"recording has {nchannels} channel(s) but the meeting expects "
            f"{len(channels)} ({', '.join(c.value for c in channels)})"
        )
    if nchannels == 1:
        return {channels[0]: frames}
    columns = frames.reshape(-1, nchannels)
    return {ch: np.ascontiguousarray(columns[:, i]) for i, ch in enumerate(channels)}


class _WavSink:
    """Interleaved int16 PCM into a RIFF file, header re-patched every write."""

    def __init__(self, path: Path, nchannels: int) -> None:
        self._nchannels = nchannels
        self._frames = 0  # per-channel PCM frames flushed to disk
        self._file = path.open("wb")
        self._file.write(self._header_bytes())

    def write(self, block: bytes) -> None:
        self._file.write(block)
        self._frames += len(block) // (self._nchannels * _BYTES_PER_SAMPLE)
        self._file.flush()
        self._file.seek(0)
        self._file.write(self._header_bytes())
        self._file.seek(0, 2)  # resume appending at end of file
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def _header_bytes(self) -> bytes:
        data_bytes = self._frames * self._nchannels * _BYTES_PER_SAMPLE
        byte_rate = SAMPLE_RATE * self._nchannels * _BYTES_PER_SAMPLE
        block_align = self._nchannels * _BYTES_PER_SAMPLE
        return b"".join(
            [
                b"RIFF",
                struct.pack("<I", 36 + data_bytes),
                b"WAVE",
                b"fmt ",
                struct.pack(
                    "<IHHIIHH",
                    16,  # fmt chunk size
                    _PCM,
                    self._nchannels,
                    SAMPLE_RATE,
                    byte_rate,
                    block_align,
                    _BITS_PER_SAMPLE,
                ),
                b"data",
                struct.pack("<I", data_bytes),
            ]
        )


class _OpusSink:
    """Interleaved int16 PCM into the bundled ffmpeg, encoding Ogg Opus.

    Invariant: exactly one of ``_proc`` (live encoder) and ``_fallback`` (the
    encoder failed; a :class:`_WavSink` beside the target carries on) is set.
    """

    def __init__(self, path: Path, nchannels: int) -> None:
        self._path = path
        self._nchannels = nchannels
        self._proc: subprocess.Popen[bytes] | None = None
        self._fallback: _WavSink | None = None
        self.fallback_path: Path | None = None
        try:
            self._proc = subprocess.Popen(
                [
                    ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    str(nchannels),
                    "-i",
                    "-",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    str(nchannels * _OPUS_BITRATE_PER_CHANNEL),
                    "-f",
                    "ogg",  # muxer pinned, not inferred from the target's suffix
                    str(path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            self._engage_fallback()

    def write(self, block: bytes) -> None:
        sink = self._fallback
        if sink is None:
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write(block)
                return
            except (OSError, ValueError):  # broken pipe / closed stdin: encoder died
                sink = self._engage_fallback()
        sink.write(block)

    def close(self) -> None:
        self._reap()
        if self._fallback is not None:
            self._fallback.close()

    def _engage_fallback(self) -> _WavSink:
        self._reap()
        self.fallback_path = self._path.with_suffix(".wav")
        self._fallback = _WavSink(self.fallback_path, self._nchannels)
        return self._fallback

    def _reap(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            with contextlib.suppress(OSError):
                self._proc.stdin.close()
        try:
            # Closing stdin lets ffmpeg finish the file; it encodes ≫ realtime,
            # so a full minute means it is hung, and Ogg stays readable through
            # a kill.
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None


class AudioTee:
    """Streams captured frames to disk, aligned by timestamp.

    Stereo (mic left, system right) when both channels are recorded, mono
    otherwise. The path's suffix picks the format: ``.wav`` is raw PCM,
    anything else Ogg Opus. Feed frames with :meth:`add`; call :meth:`close`
    to flush the tail and finalize the file.
    """

    def __init__(self, path: Path | str, channels: set[Channel]) -> None:
        self._channels = [ch for ch in _CHANNEL_ORDER if ch in channels]
        if not self._channels:
            raise ValueError("AudioTee needs at least one channel to record")
        self._nchannels = len(self._channels)
        self._pending = {ch: _PendingChannel() for ch in self._channels}
        self._path = Path(path)
        self._sink: _WavSink | _OpusSink
        if self._path.suffix.lower() == ".wav":
            self._sink = _WavSink(self._path, self._nchannels)
        else:
            self._sink = _OpusSink(self._path, self._nchannels)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def fallback_path(self) -> Path | None:
        """Where recording continued after an Opus encoder failure, if it did."""
        return self._sink.fallback_path if isinstance(self._sink, _OpusSink) else None

    def add(self, frame: AudioFrame) -> None:
        """Buffer a frame and flush whatever is now aligned across channels."""
        pending = self._pending.get(frame.channel)
        if pending is None:
            return  # a channel we're not recording
        pending.add(frame.timestamp, frame.samples)
        self._drain(min(p.available for p in self._pending.values()))

    def close(self) -> None:
        """Flush the remaining tail, padding shorter channels with silence."""
        longest = max((p.available for p in self._pending.values()), default=0)
        self._drain(longest, pad=True)
        self._sink.close()

    def _drain(self, count: int, *, pad: bool = False) -> None:
        if count <= 0:
            return
        columns = [self._pending[ch].take(count, pad=pad) for ch in self._channels]
        if self._nchannels == 1:
            block = columns[0]
        else:
            block = np.empty(count * self._nchannels, dtype=np.int16)
            for i, column in enumerate(columns):
                block[i :: self._nchannels] = column
        self._sink.write(block.tobytes())


class _PendingChannel(GapPaddedBuffer):
    """FIFO of a channel's not-yet-written int16 samples, gap-padded by time.

    Anchored at session t=0 with every gap padded exactly (``pad_gaps_over=0``),
    so the recorded file's clock stays honest and all of a meeting's recordings
    share the capture clock's t=0.
    """

    def __init__(self) -> None:
        super().__init__(label="recorded", anchor=0)
        self._chunks: deque[np.ndarray] = deque()
        self._available = 0

    def _place(self, samples: np.ndarray) -> None:
        self._chunks.append(samples)
        self._available += len(samples)

    @property
    def available(self) -> int:
        return self._available

    def take(self, count: int, *, pad: bool = False) -> np.ndarray:
        """Pop ``count`` samples from the front; pad with silence if short."""
        out = np.empty(count, dtype=np.int16)
        filled = 0
        while filled < count and self._chunks:
            chunk = self._chunks[0]
            take = min(len(chunk), count - filled)
            out[filled : filled + take] = chunk[:take]
            if take == len(chunk):
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[take:]
            filled += take
            self._available -= take
        if filled < count:
            if not pad:
                raise ValueError("take() past available samples without pad=True")
            out[filled:] = 0
        return out
