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


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise _NeedsFfmpeg
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        channels = w.getnchannels()
    if channels > 1:
        frames = frames.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return to_float32(frames)


def _load_via_ffmpeg(path: Path) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"reading {path.suffix or 'this'} files requires ffmpeg on PATH "
            "(brew install ffmpeg); only mono 16 kHz 16-bit WAV works without it"
        )
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode().strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
