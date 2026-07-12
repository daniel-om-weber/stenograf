"""Shared paths, manifest access, and audio I/O for the eval harness.

The harness (``uv run --group eval eval/<script>.py``) has two tiers: the
Phase 0 candidate-comparison scripts (score, extract, adjudicate, …) are
standalone and deliberately do not import the stenograf package, so they can
evaluate candidates the package never shipped; the verification scripts
(parity, live, diarize) exist precisely to exercise the *real* package
backends and import it on purpose. This module serves both, so it stays
package-free.
"""

from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

EVAL_DIR = Path(__file__).parent
EXAMPLES_DIR = EVAL_DIR.parent / "examples"
AUDIO_DIR = EVAL_DIR / "audio"
REFS_DIR = EVAL_DIR / "refs"
OUT_DIR = EVAL_DIR / "out"
MANIFEST = EVAL_DIR / "manifest.json"


@dataclass(frozen=True)
class EvalSegment:
    id: str
    source: str
    """Filename inside examples/."""
    start: float
    end: float
    language: str | None
    """"de" / "en"; None until determined (LID scan or listening)."""
    notes: str = ""

    @property
    def source_path(self) -> Path:
        return EXAMPLES_DIR / self.source

    @property
    def wav_path(self) -> Path:
        return AUDIO_DIR / f"{self.id}.wav"

    @property
    def ref_path(self) -> Path:
        return REFS_DIR / f"{self.id}.txt"

    def hyp_path(self, backend: str) -> Path:
        return OUT_DIR / backend / f"{self.id}.json"


def load_manifest() -> list[EvalSegment]:
    entries = json.loads(MANIFEST.read_text())
    segments = [EvalSegment(**entry) for entry in entries]
    ids = [segment.id for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate segment ids in manifest.json")
    return segments


def to_wav16k(
    src: Path,
    dst: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    duration: float | None = None,
) -> None:
    """ffmpeg any input → mono 16 kHz s16 WAV, optionally cutting a window.

    The one encoding every eval artifact uses — the same wire format the
    package captures. ``start``/``end`` bound the cut; ``duration`` is the
    ``-t`` alternative to ``end`` for fixed-length probes."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def read_pcm16(path: Path) -> np.ndarray:
    """A mono 16 kHz s16 WAV as an int16 array; raises ValueError otherwise."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != 16_000 or w.getsampwidth() != 2:
            raise ValueError(f"{path} is not a mono 16 kHz int16 WAV")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def split_at_silences(
    path: Path, target_s: float = 30.0, search_s: float = 5.0
) -> list[tuple[float, float]]:
    """Split a mono 16 kHz s16 WAV into ~target_s windows, cutting at the
    quietest 300 ms near each target boundary (poor man's VAD — good enough
    to avoid slicing through words)."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    total_s = len(samples) / rate
    if total_s <= target_s * 1.5:
        return [(0.0, total_s)]

    window = int(rate * 0.3)
    bounds = [0.0]
    while total_s - bounds[-1] > target_s * 1.5:
        center = bounds[-1] + target_s
        lo = int((center - search_s) * rate)
        hi = int((center + search_s) * rate) - window
        offsets = range(lo, hi, window // 3)
        quietest = min(
            offsets, key=lambda o: float(np.abs(samples[o : o + window].astype(np.int32)).mean())
        )
        bounds.append((quietest + window // 2) / rate)
    bounds.append(total_s)
    return list(zip(bounds, bounds[1:], strict=False))
