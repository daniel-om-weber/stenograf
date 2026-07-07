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
            default=str,  # the profile's speaker_profile_store may be a Path
        )

    def to_markdown(self) -> str:
        lines = []
        for entry in self.entries:
            marker = " *(overlap)*" if entry.provisional else ""
            lines.append(f"**{entry.speaker}** [{_fmt(entry.start)}]{marker}: {entry.text}")
        return "\n\n".join(lines) + "\n"

    def to_srt(self) -> str:
        """Render as SubRip (SRT) subtitles — one numbered cue per re-flowed chunk."""
        blocks = []
        for i, cue in enumerate(_build_cues(self.entries), start=1):
            text = _wrap(cue.text)
            if cue.speaker:
                text = f"{cue.speaker}: {text}"
            blocks.append(f"{i}\n{_ts(cue.start, ',')} --> {_ts(cue.end, ',')}\n{text}\n")
        return "\n".join(blocks)

    def to_vtt(self) -> str:
        """Render as WebVTT subtitles, speaker attribution via ``<v>`` voice spans."""
        blocks = ["WEBVTT\n"]
        for cue in _build_cues(self.entries):
            payload = _wrap(_escape_vtt(cue.text))
            if cue.speaker:
                payload = f"<v {_escape_vtt(cue.speaker)}>{payload}</v>"
            blocks.append(f"{_ts(cue.start, '.')} --> {_ts(cue.end, '.')}\n{payload}\n")
        return "\n".join(blocks)


@dataclass(frozen=True)
class _Cue:
    start: float
    end: float
    speaker: str
    text: str


# Subtitle re-flow budgets. A TranscriptEntry is a whole gap-split speaker turn —
# far too long to show as one subtitle — so SRT/VTT export re-flows each entry's
# retained word timestamps (Phase 3 Task 0e) into short cues.
_MAX_CUE_CHARS = 84  # ~two 42-char subtitle lines of spoken text
_MAX_CUE_SECONDS = 6.0  # a single cue never lingers longer than this
_MAX_CUE_GAP = 1.0  # a pause this long inside a turn starts a new cue
_WRAP_WIDTH = 42  # physical subtitle line width


def _build_cues(entries: list[TranscriptEntry]) -> list[_Cue]:
    """Re-flow all entries into short subtitle cues, ordered by start time.

    Overlapping Local/Remote cues are kept as-is (both formats permit them); the
    speaker label on each cue disambiguates who is speaking.
    """
    cues = [cue for entry in entries for cue in _entry_cues(entry)]
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def _entry_cues(entry: TranscriptEntry) -> list[_Cue]:
    if not entry.words:
        # A wordless backend (Whisper/Voxtral) leaves nothing to re-flow: emit the
        # whole turn as one cue on its own time span. `text` is always authoritative.
        return [_Cue(entry.start, entry.end, entry.speaker, entry.text)]

    cues: list[_Cue] = []
    run: list[Word] = []

    def flush() -> None:
        nonlocal run
        if run:
            cues.append(
                _Cue(run[0].start, run[-1].end, entry.speaker, " ".join(w.text for w in run))
            )
        run = []

    for word in entry.words:
        if run:
            chars = len(" ".join(w.text for w in run)) + 1 + len(word.text)
            if (
                chars > _MAX_CUE_CHARS
                or word.end - run[0].start > _MAX_CUE_SECONDS
                or word.start - run[-1].end > _MAX_CUE_GAP
            ):
                flush()
        run.append(word)
    flush()
    return cues


def _wrap(text: str, width: int = _WRAP_WIDTH) -> str:
    """Greedily wrap cue text onto lines of at most ``width`` characters."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _escape_vtt(text: str) -> str:
    """Escape the WebVTT-significant characters so cue payloads stay well-formed."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ts(seconds: float, sep: str) -> str:
    """Format ``HH:MM:SS<sep>mmm`` — ``sep`` is ``,`` for SRT, ``.`` for VTT."""
    ms = round(max(seconds, 0.0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
