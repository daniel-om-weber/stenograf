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
tail. The encoder runs with per-packet flushing (``-flush_packets 1``) so
Ogg's self-delimiting pages give the same contract — a killed encoder leaves
a readable file missing a few seconds of tail (measured 2026-08-06: 2–5 s at
SIGKILL; without the flag ffmpeg holds a 256 KB block and a sub-minute
recording dies at zero bytes).

A recording the user asked for must also survive encoder trouble without
touching the capture loop, whose contract is that nothing on it may stall
(:class:`stenograf.session.CaptureLoop`). PCM therefore reaches ffmpeg through
a writer thread; if the encoder cannot be spawned, dies, or stops draining
(full disk, dead network mount), the tee kills it and degrades to a ``.wav``
next to the target from that point on, reported via
:attr:`AudioTee.fallback_path`.

Another program may also take the name away while the file is written. On
2026-11-27 Google Drive, mirroring a meeting folder, downloaded its copy of
the still-empty ``audio.opus`` over the one being written, and both of that
day's meetings went into files that no longer had a name. So the tee opens
every target itself and hands ffmpeg that open file; on close, a target whose
name no longer leads to the tee's file is written back from it
(:attr:`AudioTee.restored`). Windows needs none of this: no other program can
rename over or delete a file the tee holds open there. What the write-back
leaves uncovered is in DECISIONS.md.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from stenograf.audio import ffmpeg_exe
from stenograf.capture.base import (
    SAMPLE_RATE,
    AudioFrame,
    Channel,
    GapPaddedBuffer,
)
from stenograf.proc import no_window

_BYTES_PER_SAMPLE = 2  # int16
_BITS_PER_SAMPLE = 16
_PCM = 1  # WAV format tag

# Per channel because the tee's channels are independent voice feeds (mic,
# system), not a stereo image sharing content. The value itself is a shipped
# default whose evidence lives in eval/README.md "Stored-audio codec".
_OPUS_BITRATE_PER_CHANNEL = 32_000

# An encoder whose pipe has accepted nothing for this long, with data queued,
# is wedged (full disk, dead mount) and killed. Progress-based, not
# queue-depth-based: a producer burst must never kill a healthy encoder.
_WEDGE_SECONDS = 10

# If the queue backs up this far the writer itself is stuck (a hung mount in
# the fallback open); new blocks are dropped so capture stays live and memory
# stays bounded.
_DROP_SECONDS = 60

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


@dataclass(frozen=True)
class _Kept:
    """Where a closed sink's file ended up."""

    path: Path | None
    """The file holding the recording; ``None`` once it is lost or deleted, or
    moved to a name the tee does not know."""
    restored: bool = False
    error: str | None = None


def _keep_at(path: Path, file: BinaryIO) -> _Kept:
    """Make sure ``path`` still leads to the recording in ``file``, writing it
    back if another program took the name away.

    Call it only once nothing writes to ``file`` any more — it reads from the
    start. By then the open ``file`` may be the recording's only copy, so the
    write-back is synced to disk and lands under ``path`` atomically: a reader
    sees the other program's file or the whole recording, never half of one.
    Never raises; a failure is the result's ``error``.
    """
    if sys.platform == "win32":
        # Nothing can rename over or delete a file held open here — and the
        # write-back could not replace a name this handle still holds.
        return _Kept(path)
    dest = Path(os.path.realpath(path))  # a symlinked target keeps its link
    try:
        file.flush()
        held = os.fstat(file.fileno())
        with contextlib.suppress(FileNotFoundError):
            if os.path.samestat(os.stat(dest), held):
                return _Kept(path)
    except OSError as exc:
        return _Kept(path, error=f"{path.name} could not be checked at the end ({exc})")
    if held.st_nlink > 0:  # still named somewhere: moved, not replaced
        what, otherwise = f"{path.name} was moved during the meeting", "it kept its new name"
    else:
        what = f"another program replaced or removed {path.name} during the meeting"
        otherwise = "the recording is lost"
    try:
        fd, name = tempfile.mkstemp(prefix=f"{dest.name}.", suffix=".restore", dir=dest.parent)
    except OSError as exc:
        return _Kept(None, error=f"{what} and could not be written back ({exc}); {otherwise}")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            file.seek(0)
            shutil.copyfileobj(file, out)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, stat.S_IMODE(held.st_mode))  # not mkstemp's owner-only mode
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink()
        return _Kept(None, error=f"{what} and could not be written back ({exc}); {otherwise}")
    try:
        os.replace(tmp, dest)
    except OSError as exc:
        return _Kept(
            tmp,
            error=f"{what} and could not be written back under its name ({exc}); "
            f"the recording is in {tmp.name}",
        )
    return _Kept(path, restored=True)


class _WavSink:
    """Interleaved int16 PCM into a RIFF file, header re-patched every write."""

    def __init__(self, path: Path, nchannels: int) -> None:
        self._path = path
        self._nchannels = nchannels
        self._frames = 0  # per-channel PCM frames flushed to disk
        self._kept: _Kept | None = None
        self._file = path.open("w+b")  # readable, for :func:`_keep_at`
        self._file.write(self._header_bytes())

    @property
    def restored(self) -> bool:
        return self._kept is not None and self._kept.restored

    @property
    def error(self) -> str | None:
        return None if self._kept is None else self._kept.error

    @property
    def saved(self) -> list[Path]:
        """Where the recording is; meaningful once closed."""
        if self._kept is None:
            return [self._path]
        return [] if self._kept.path is None else [self._kept.path]

    def write(self, block: bytes) -> None:
        self._file.write(block)
        self._frames += len(block) // (self._nchannels * _BYTES_PER_SAMPLE)
        self._file.flush()
        self._file.seek(0)
        self._file.write(self._header_bytes())
        self._file.seek(0, 2)  # resume appending at end of file
        self._file.flush()

    def close(self) -> None:
        if self._file.closed:
            return
        try:
            self._kept = _keep_at(self._path, self._file)
        finally:
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

    The capture thread only enqueues (and, once degraded, appends to the
    already-open WAV fallback); the writer thread owns the encoder pipe and is
    the only place a fallback is *opened*, so a hung filesystem can wedge only
    the writer, never capture. On any encoder failure — spawn error, death, or
    no pipe progress for the wedge window — the sink degrades to a
    :class:`_WavSink` at ``<target>.wav`` (the name is appended, never
    substituted, so an unrelated sibling ``.wav`` is safe) and every later
    block goes there. Blocks already queued for the dead encoder are dropped
    rather than replayed: the fallback's timeline starts at the switch, and
    out-of-order PCM would corrupt it.

    ``_proc`` stays set after degradation so :meth:`close` can reap it —
    "encoder is live" is ``_fallback is None and error is None``, never
    ``_proc is not None``.
    """

    def __init__(self, path: Path, nchannels: int) -> None:
        self._path = path
        self._nchannels = nchannels
        self._proc: subprocess.Popen[bytes] | None = None
        self._fallback: _WavSink | None = None
        self.fallback_path: Path | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._queued = 0  # bytes handed to the writer thread, not yet written
        self._last_progress = time.monotonic()
        self._closing = False
        self._drop_bytes = _DROP_SECONDS * SAMPLE_RATE * nchannels * _BYTES_PER_SAMPLE
        self._writer: threading.Thread | None = None
        self.restored = False
        """A file was taken away meanwhile and written back on close."""
        self._file: BinaryIO | None = None
        self._kept: _Kept | None = None
        try:
            # get_ffmpeg_exe() raises RuntimeError when no binary exists —
            # doctor.py catches the same; here it must degrade, not abort the
            # meeting. Asked first, so a missing binary creates no target.
            exe = ffmpeg_exe()
            # ffmpeg writes into this file, not to the path, so the tee still
            # holds the recording if another program takes the name away.
            self._file = path.open("w+b")
            self._proc = subprocess.Popen(
                [
                    exe,
                    "-hide_banner",
                    "-loglevel",
                    "error",
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
                    # Without per-packet flushing ffmpeg holds a 256 KB output
                    # block: a killed sub-minute recording is zero bytes. With
                    # it, a kill costs 2–5 s (measured 2026-08-06).
                    "-flush_packets",
                    "1",
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                stdin=subprocess.PIPE,
                stdout=self._file,
                stderr=subprocess.DEVNULL,
                creationflags=no_window(),
            )
        except (OSError, RuntimeError) as exc:
            self._engage_fallback(exc)  # pre-capture, so blocking open is fine
            return
        self._writer = threading.Thread(target=self._pump, name="opus-tee", daemon=True)
        self._writer.start()

    def write(self, block: bytes) -> None:
        with self._lock:
            fallback = self._fallback
            if fallback is None:
                if self.error is not None:
                    return  # encoder AND fallback failed; drop, reported at close
                stalled = (
                    self._queued > 0
                    and time.monotonic() - self._last_progress > _WEDGE_SECONDS
                )
                if stalled and self._proc is not None:
                    # No pipe progress for the whole window: the encoder is
                    # wedged. Kill it — the writer's blocked write fails and
                    # engages the fallback. Repeat kills of a dead process are
                    # no-ops.
                    self._proc.kill()
                if self._queued > self._drop_bytes:
                    return  # the fallback open itself is stuck: drop, stay live
                self._queued += len(block)
        if fallback is None:
            self._queue.put(block)
        else:
            fallback.write(block)  # an OSError here is AudioTee's to guard

    def close(self) -> None:
        self._closing = True
        if self._writer is not None:
            self._queue.put(None)
            # A healthy encoder drains the queue ≫ realtime, so a long join
            # means it is wedged: kill to unblock the writer's pipe write.
            self._writer.join(timeout=10)
            if self._writer.is_alive() and self._proc is not None:
                self._proc.kill()
                self._writer.join(timeout=10)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        with self._lock:
            fallback = self._fallback
        if fallback is not None:
            try:
                fallback.close()
            except OSError as exc:
                self._note_error(f"the wav continuation could not be finished ({exc})")
            if fallback.error is not None:
                self._note_error(fallback.error)
            self.restored = fallback.restored
        self._finish_target(abandoned=fallback is not None or self.error is not None)

    @property
    def saved(self) -> list[Path]:
        """Where the recording is — target first, then any ``.wav``
        continuation; meaningful once closed."""
        if self._kept is None:
            target = [self._path]
        else:
            target = [] if self._kept.path is None else [self._kept.path]
        return target + ([] if self._fallback is None else self._fallback.saved)

    def _finish_target(self, *, abandoned: bool) -> None:
        """Close the target once ffmpeg is gone: drop it if it holds no audio,
        else make sure its name still leads to it."""
        file, self._file = self._file, None
        if file is None:
            self._kept = _Kept(None)
            return
        drop = False
        try:
            held = os.fstat(file.fileno())
            if abandoned and held.st_size < 1024:
                # Ogg headers at most, no audio: delete it rather than
                # advertise an empty recording — unless the name already
                # belongs to another program's file.
                with contextlib.suppress(OSError):
                    drop = os.path.samestat(os.stat(self._path), held)
                self._kept = _Kept(None)
                return
            self._kept = _keep_at(self._path, file)
            if self._kept.restored:
                self.restored = True
            if self._kept.error is not None:
                self._note_error(self._kept.error)
        finally:
            file.close()
            if drop:  # after the close: Windows refuses to delete an open file
                with contextlib.suppress(OSError):
                    self._path.unlink()

    def _note_error(self, message: str) -> None:
        with self._lock:
            self.error = message if self.error is None else f"{self.error}; {message}"

    def _pump(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        dead = False
        while (block := self._queue.get()) is not None:
            wrote = False
            if not dead:
                try:
                    proc.stdin.write(block)
                    wrote = True
                except (OSError, ValueError) as exc:  # broken pipe: encoder died
                    dead = True
                    # This block's slot in the timeline is already lost to the
                    # pipe, so it is dropped, not replayed into the fallback.
                    self._engage_fallback(exc)
            with self._lock:
                self._queued -= len(block)
                if wrote:
                    self._last_progress = time.monotonic()
        with contextlib.suppress(OSError, ValueError):
            proc.stdin.close()  # EOF lets a live encoder finalize the file

    def _engage_fallback(self, cause: BaseException) -> None:
        """Open the ``.wav`` continuation; on a second failure record it instead.

        Runs on the writer thread (or in ``__init__`` before capture is wired)
        and opens the file OUTSIDE the lock: a hung mount here must wedge only
        this thread — capture keeps enqueuing until the drop bound and the tee
        stays responsive.
        """
        if self._closing:
            with self._lock:
                self.error = f"opus encoder died while finishing ({cause}); tail lost"
            return
        target = self._path.with_name(self._path.name + ".wav")
        try:
            fallback = _WavSink(target, self._nchannels)
        except OSError as exc:
            with self._lock:
                self.error = f"opus encoder failed ({cause}); wav fallback failed ({exc})"
            return
        with self._lock:
            self._fallback = fallback
            self.fallback_path = target


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
        self._error: str | None = None
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

    @property
    def error(self) -> str | None:
        """Why the recording failed, mid-meeting or on close, if it did."""
        if self._error is not None:
            return self._error
        return self._sink.error

    @property
    def restored(self) -> bool:
        """Another program took a file away while it was written, and
        :meth:`close` wrote it back."""
        return self._sink.restored

    @property
    def saved(self) -> list[Path]:
        """The existing files that hold this recording, after :meth:`close` —
        possibly none, possibly a name other than :attr:`path`."""
        return [p for p in self._sink.saved if p.exists()]

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
        if self._error is None:  # a failed sink was already closed in _drain
            try:
                self._sink.close()
            except OSError as exc:
                self._error = str(exc)

    def _drain(self, count: int, *, pad: bool = False) -> None:
        if count <= 0:
            return
        # Always consume the pending buffers, even once the sink is dead —
        # they would otherwise grow for the rest of the meeting.
        columns = [self._pending[ch].take(count, pad=pad) for ch in self._channels]
        if self._error is not None:
            return
        if self._nchannels == 1:
            block = columns[0]
        else:
            block = np.empty(count * self._nchannels, dtype=np.int16)
            for i, column in enumerate(columns):
                block[i :: self._nchannels] = column
        try:
            self._sink.write(block.tobytes())
        except OSError as exc:
            # A recording must never end the meeting: a full disk under either
            # sink stops the recording, not the capture (reported at close).
            self._error = str(exc)
            with contextlib.suppress(OSError):
                self._sink.close()


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
