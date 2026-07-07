"""Transcript data model — the only artifact stenograf ever writes to disk."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from stenograf.asr.base import Word
from stenograf.config import Language, MeetingProfile


@dataclass(frozen=True)
class TranscriptEntry:
    speaker: str
    """Display label: ``Local-1`` / ``Remote-2``, or a profile name."""
    text: str
    start: float
    end: float
    provisional: bool = False
    """True for overlapping-speech regions where attribution is unreliable."""
    words: tuple[Word, ...] = ()
    """The entry's word-level timestamps, on the session clock, in order.

    Retained so the JSON output honours §Outputs' word-level-timestamp promise
    and so subtitle export (SRT/VTT) can re-flow long speaker turns into short,
    time-accurate cues. Empty only when the ASR backend emits no word timestamps
    (e.g. a Whisper/Voxtral path) — ``text`` is always the source of truth."""


@dataclass
class Transcript:
    language: Language | None
    """``None`` when neither given nor (yet) auto-detected."""
    profile: MeetingProfile
    entries: list[TranscriptEntry] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "language": self.language.value if self.language else None,
                "profile": asdict(self.profile),
                "entries": [asdict(e) for e in self.entries],
            },
            ensure_ascii=False,
            indent=2,
        )

    def to_markdown(self) -> str:
        lines = []
        for entry in self.entries:
            marker = " *(overlap)*" if entry.provisional else ""
            lines.append(f"**{entry.speaker}** [{_fmt(entry.start)}]{marker}: {entry.text}")
        return "\n\n".join(lines) + "\n"


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
