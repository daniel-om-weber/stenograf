"""Read and write NIST RTTM — the reference/hypothesis format for diarization.

An RTTM ``SPEAKER`` line is ten whitespace-separated fields::

    SPEAKER <file-id> <chan> <onset> <duration> <NA> <NA> <speaker> <NA> <NA>

We only use file-id, onset, duration, and speaker; the rest are fixed. This
module is deliberately pure (stdlib only) so the DER scorer and its tests never
need audio, models, or the stenograf package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    """One speaker's contiguous span, in seconds on the file's clock."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_rttm(path: Path) -> list[Turn]:
    """Read every ``SPEAKER`` turn from an RTTM file, sorted by start time.

    Blank lines, comments (``;;``/``#``), and non-``SPEAKER`` records are
    ignored, and zero/negative-duration turns are dropped."""
    turns: list[Turn] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith((";;", "#")):
            continue
        fields = line.split()
        if fields[0] != "SPEAKER":
            continue
        if len(fields) < 8:
            raise ValueError(f"{path}:{lineno}: malformed SPEAKER line: {raw!r}")
        onset, duration = float(fields[3]), float(fields[4])
        if duration <= 0:
            continue
        turns.append(Turn(speaker=fields[7], start=onset, end=onset + duration))
    return sorted(turns, key=lambda t: (t.start, t.end))


def format_rttm(turns: list[Turn], file_id: str) -> str:
    """Render turns as RTTM text (channel 1, decimals to the millisecond)."""
    lines = [
        f"SPEAKER {file_id} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
        for t in sorted(turns, key=lambda t: (t.start, t.end))
    ]
    return "\n".join(lines) + "\n" if lines else ""


def write_rttm(path: Path, turns: list[Turn], file_id: str) -> None:
    Path(path).write_text(format_rttm(turns, file_id))


def speakers(turns: list[Turn]) -> list[str]:
    """Distinct speaker labels, in first-appearance order."""
    seen: list[str] = []
    for t in turns:
        if t.speaker not in seen:
            seen.append(t.speaker)
    return seen
