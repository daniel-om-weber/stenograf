"""Opt-in audio recording — audio touches disk only on explicit request.

Off by default: stenograf's guarantee is that audio stays in RAM. When the user
explicitly passes ``--record-audio``, this tee additionally appends the incoming
PCM to a WAV file as it arrives — mic on the left channel, system audio on the
right (mono when only one channel is captured), preserving the channel
separation in a file any player opens. The only other disk path is the equally
opt-in AEC debug dump (``--aec-dump``, :class:`stenograf.aec.AecDump`), which
reuses this tee.

Written at the capture wire format (mono 16 kHz int16 per channel): enough for
speech playback and exactly what re-transcription via ``steno transcribe``
needs. Crash-safe like the incremental text checkpoints — the header's size
fields are rewritten after every drain, so a process killed mid-meeting still
leaves a playable file missing only the last, not-yet-aligned tail.
"""

from __future__ import annotations

import struct
import wave
from collections import deque
from pathlib import Path

import numpy as np

from stenograf.capture.base import (
    ORDER_TOLERANCE_SAMPLES,
    SAMPLE_RATE,
    AudioFrame,
    Channel,
)

_BYTES_PER_SAMPLE = 2  # int16
_BITS_PER_SAMPLE = 16
_PCM = 1  # WAV format tag

# Stereo layout is fixed so recordings are always mic-left / system-right.
_CHANNEL_ORDER = (Channel.MIC, Channel.SYSTEM)


def read_channels(path: Path | str, channels: list[Channel]) -> dict[Channel, np.ndarray]:
    """Read a :class:`WavTee` recording back into its per-channel int16 streams.

    The exact inverse of the tee's fixed layout (mic left, system right; mono when
    a single channel was recorded), used by the AEC eval rig to score dump triples.
    ``channels`` is the ordered channel list the recording holds — the
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


class WavTee:
    """Streams captured frames to a 16 kHz int16 WAV, aligned by timestamp.

    Stereo (mic left, system right) when both channels are recorded, mono
    otherwise. Feed frames with :meth:`add`; call :meth:`close` to flush the
    tail and finalize the header.
    """

    def __init__(self, path: Path | str, channels: set[Channel]) -> None:
        self._channels = [ch for ch in _CHANNEL_ORDER if ch in channels]
        if not self._channels:
            raise ValueError("WavTee needs at least one channel to record")
        self._nchannels = len(self._channels)
        self._pending = {ch: _PendingChannel() for ch in self._channels}
        self._path = Path(path)
        self._file = self._path.open("wb")
        self._frames_written = 0  # per-channel PCM frames flushed to disk
        self._file.write(self._header_bytes())

    @property
    def path(self) -> Path:
        return self._path

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
        self._file.close()

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
        self._file.write(block.tobytes())
        self._frames_written += count
        self._patch_header()

    def _patch_header(self) -> None:
        self._file.flush()
        self._file.seek(0)
        self._file.write(self._header_bytes())
        self._file.seek(0, 2)  # resume appending at end of file
        self._file.flush()

    def _header_bytes(self) -> bytes:
        data_bytes = self._frames_written * self._nchannels * _BYTES_PER_SAMPLE
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


class _PendingChannel:
    """FIFO of a channel's not-yet-written int16 samples, gap-padded by time."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._available = 0
        self._received = 0  # total samples placed, including silence padding

    def add(self, timestamp: float, samples: np.ndarray) -> None:
        offset = round(timestamp * SAMPLE_RATE)
        if offset < self._received - ORDER_TOLERANCE_SAMPLES:
            # Backward past jitter tolerance: appending here would misalign the
            # recorded file from this frame on. Frames arrive in order per channel.
            raise ValueError(
                f"frame went backwards {(self._received - offset) / SAMPLE_RATE:.3f}s "
                f"(timestamp {timestamp:.3f}s); frames must arrive in order"
            )
        if offset > self._received:  # gap → pad silence to keep the clock honest
            gap = offset - self._received
            self._chunks.append(np.zeros(gap, dtype=np.int16))
            self._available += gap
            self._received += gap
        samples = np.asarray(samples, dtype=np.int16)
        self._chunks.append(samples)
        self._available += len(samples)
        self._received += len(samples)

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
