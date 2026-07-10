"""Audio loading and sample-format helpers.

Everything downstream of capture works on one format: mono 16 kHz float32
in [-1, 1]. Plain 16-bit WAV files are read natively; everything else
(.mov, .m4a, other rates/channel counts) goes through ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def to_float32(samples: np.ndarray) -> np.ndarray:
    """int16 PCM → float32 in [-1, 1]; float input passes through."""
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    return np.asarray(samples, dtype=np.float32)


def to_int16(samples: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] → int16 PCM (the capture wire format); int16 passes through.

    The exact inverse of :func:`to_float32` for in-range values (×32768 with
    rounding), so a float32-loaded file fed back into the int16 domain — file
    replay, the split-channel transcribe — reproduces the original samples."""
    if samples.dtype == np.int16:
        return samples
    return np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int16)


def sample_index(seconds: float) -> int:
    """Seconds → sample index, the ONE conversion for ASR window boundaries.

    The live window pass and the batch finalize pass slice the same windows
    from the same audio; using any other conversion (``round``, a different
    epsilon) in one of them shifts a slice by one sample, which is enough to
    flip a marginal word in the decode. Keep them byte-identical: every
    boundary goes through this truncation.
    """
    return int(seconds * SAMPLE_RATE)


def load_audio(path: Path | str) -> np.ndarray:
    """Load any audio/video file as mono 16 kHz float32."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return _load_wav(path)
        except _NeedsFfmpeg:
            pass
    return _load_via_ffmpeg(path)


class _NeedsFfmpeg(Exception):
    """WAV variant the stdlib reader can't handle (rate/channels/encoding)."""


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Raw interleaved int16 frames plus the channel count of a plain 16 kHz WAV."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise _NeedsFfmpeg
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getnchannels()


def _load_wav(path: Path) -> np.ndarray:
    frames, channels = _read_wav(path)
    if channels > 1:
        frames = frames.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return to_float32(frames)


def load_audio_channels(path: Path | str) -> list[np.ndarray]:
    """Load a file as per-channel 16 kHz float32 — one mono array per channel.

    The channel-preserving counterpart of :func:`load_audio`, for recordings
    whose channels are separate voice feeds rather than one stereo image: a
    ``--record-audio`` tee (mic left, system right), a dual-channel call
    recording. A mono file loads as a single-element list.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            frames, channels = _read_wav(path)
        except _NeedsFfmpeg:
            pass
        else:
            columns = frames.reshape(-1, channels)
            return [to_float32(np.ascontiguousarray(columns[:, i])) for i in range(channels)]
    channels = audio_channel_count(path)
    if channels <= 1:
        return [_load_via_ffmpeg(path)]
    raw = _load_via_ffmpeg(path, channels=channels).reshape(-1, channels)
    return [np.ascontiguousarray(raw[:, i]) for i in range(channels)]


def audio_channel_count(path: Path | str) -> int:
    """Channel count from the container header — no PCM decode.

    Falls back to 1 when it cannot tell (no ffprobe, unreadable header): those
    files then take the mono decode path exactly as before.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnchannels()
        except (wave.Error, EOFError):
            pass  # not a readable RIFF — fall through to ffprobe
    if shutil.which("ffprobe") is None:
        return 1
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "csv=p=0",
        str(path),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(proc.stdout.strip()) if proc.returncode == 0 else 1
    except ValueError:
        return 1


_ENVELOPE_BLOCK_S = 0.1
"""Loudness-envelope resolution for :func:`channels_look_independent` — coarse
enough to ignore phase/pan differences, fine enough to track speech turns."""

INDEPENDENT_MAX_CORRELATION = 0.5
"""Envelope correlation below which two channels read as separate voice feeds.
Measured far from both populations: a real stenograf tee scores ≈ -0.2 (mic and
system activity are disjoint, and the mic is echo-cancelled), while a stereo
image of one room scores ≈ 1.0 (every voice is in both channels)."""


def channels_look_independent(left: np.ndarray, right: np.ndarray) -> tuple[bool, float | None]:
    """Do two channels carry separate voice feeds rather than one stereo image?

    Splitting a stereo *image* per channel would double-transcribe every
    speaker, so the split must only happen when the channels' activity is
    genuinely disjoint. The discriminator is the Pearson correlation of the
    channels' loudness envelopes: level- and pan-invariant, and the two cases
    sit far apart (see :data:`INDEPENDENT_MAX_CORRELATION`).

    Returns ``(independent, correlation)``; correlation is ``None`` — never
    independent — when either channel is constant (a dead channel means there
    is nothing to split) or the clip is shorter than one envelope block.
    """
    block = int(_ENVELOPE_BLOCK_S * SAMPLE_RATE)
    length = min(len(left), len(right)) // block * block
    if length == 0:
        return False, None

    def envelope(samples: np.ndarray) -> np.ndarray:
        blocks = to_float32(samples)[:length].astype(np.float64).reshape(-1, block)
        return np.sqrt((blocks**2).mean(axis=1))

    left_env, right_env = envelope(left), envelope(right)
    if left_env.std() == 0.0 or right_env.std() == 0.0:
        return False, None
    correlation = float(np.corrcoef(left_env, right_env)[0, 1])
    return correlation < INDEPENDENT_MAX_CORRELATION, correlation


def _load_via_ffmpeg(path: Path, *, channels: int = 1) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"reading {path.suffix or 'this'} files requires ffmpeg on PATH "
            "(brew install ffmpeg); only mono 16 kHz 16-bit WAV works without it"
        )
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(channels), "-ar", str(SAMPLE_RATE),
        "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode().strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
