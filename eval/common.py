"""Shared paths and manifest access for the Phase 0 eval harness.

The harness is standalone tooling (``uv run --group eval eval/<script>.py``);
it deliberately does not import the stenograf package.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

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
