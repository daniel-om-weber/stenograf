"""The finalize pass: VAD windows → batch ASR → diarization → merged entries.

This is the accuracy core (PLAN.md §2). It operates on one channel of mono
16 kHz PCM; the meeting orchestrator runs it per channel (mic / system) and
interleaves the results. ``steno transcribe`` runs it on a file.
"""

from __future__ import annotations

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE
from stenograf.config import Language
from stenograf.diarization.base import Diarizer, SpeakerTurn
from stenograf.transcript import TranscriptEntry
from stenograf.vad import SileroVAD, pack_windows

MAX_ENTRY_GAP = 1.5
"""Silence (s) between words of one speaker that still reads as one entry."""


def finalize_channel(
    samples,
    *,
    asr: ASRBackend,
    language: Language | None,
    vad: SileroVAD | None = None,
    diarizer: Diarizer | None = None,
    num_speakers: int | None = None,
    on_progress=None,
) -> list[TranscriptEntry]:
    """Transcribe one channel; returns entries with raw ``S<n>`` speaker labels.

    ``diarizer=None`` or ``num_speakers=1`` attributes everything to ``S0``.
    ``on_progress`` is called as ``on_progress(stage: str, done: int, total: int)``.
    """
    duration = len(samples) / SAMPLE_RATE
    if vad is not None:
        windows = pack_windows(vad.speech_segments(samples), duration)
    else:
        windows = [(0.0, duration)] if duration > 0 else []

    segments: list[Segment] = []
    for i, (start, end) in enumerate(windows):
        if on_progress is not None:
            on_progress("asr", i, len(windows))
        window = samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        segments.extend(_shift(seg, start) for seg in asr.transcribe(window, language))
    segments.sort(key=lambda seg: seg.start)

    if diarizer is None or num_speakers == 1:
        return [
            TranscriptEntry(speaker="S0", text=seg.text, start=seg.start, end=seg.end)
            for seg in segments
        ]

    if on_progress is not None:
        on_progress("diarization", 0, 1)
    turns = diarizer.diarize(samples, num_speakers)
    words = [word for seg in segments for word in seg.words]
    if not words:
        # A backend that emits text but no word timestamps (a contract
        # violation for diarized use — see ASRBackend) would otherwise drop the
        # whole transcript here. Fall back to attributing each segment as a unit
        # by its time span rather than losing the text.
        words = [Word(text=seg.text, start=seg.start, end=seg.end) for seg in segments]
    return merge_words_turns(words, turns)


def _shift(seg: Segment, offset: float) -> Segment:
    return Segment(
        text=seg.text,
        start=seg.start + offset,
        end=seg.end + offset,
        words=tuple(
            Word(text=w.text, start=w.start + offset, end=w.end + offset, confidence=w.confidence)
            for w in seg.words
        ),
    )


def merge_words_turns(
    words: list[Word],
    turns: list[SpeakerTurn],
    *,
    max_gap: float = MAX_ENTRY_GAP,
) -> list[TranscriptEntry]:
    """Assign each word a speaker and group runs into transcript entries.

    A word takes the speaker of the turn covering its midpoint. Inside
    overlapping turns the largest-overlap turn wins and the entry is flagged
    provisional; words outside every turn take the nearest turn's speaker.
    """
    entries: list[TranscriptEntry] = []
    run: list[Word] = []
    run_speaker = ""
    run_provisional = False

    def close_run() -> None:
        nonlocal run, run_provisional
        if run:
            entries.append(
                TranscriptEntry(
                    speaker=run_speaker,
                    text=" ".join(w.text for w in run),
                    start=run[0].start,
                    end=run[-1].end,
                    provisional=run_provisional,
                )
            )
        run = []
        run_provisional = False

    for word in sorted(words, key=lambda w: w.start):
        speaker, provisional = _assign(word, turns)
        if run and (speaker != run_speaker or word.start - run[-1].end > max_gap):
            close_run()
        run.append(word)
        run_speaker = speaker
        run_provisional = run_provisional or provisional
    close_run()
    return entries


def group_words(
    words: list[Word], speaker: str, *, max_gap: float = MAX_ENTRY_GAP
) -> list[TranscriptEntry]:
    """Group one un-diarized speaker's words into entries, split on gaps > max_gap.

    The live checkpoint (Option B, PLAN.md §3) turns a channel's committed live
    words into readable entries the same way :func:`merge_words_turns` groups a
    diarization turn — one entry per continuous run of speech — but with no
    speaker assignment: every word is attributed to ``speaker`` (a channel-coarse
    ``Local``/``Remote`` label, since the live pass does not diarize). Words must
    already be in time order.
    """
    entries: list[TranscriptEntry] = []
    run: list[Word] = []

    def close_run() -> None:
        nonlocal run
        if run:
            entries.append(
                TranscriptEntry(
                    speaker=speaker,
                    text=" ".join(w.text for w in run),
                    start=run[0].start,
                    end=run[-1].end,
                )
            )
        run = []

    for word in words:
        if run and word.start - run[-1].end > max_gap:
            close_run()
        run.append(word)
    close_run()
    return entries


def _assign(word: Word, turns: list[SpeakerTurn]) -> tuple[str, bool]:
    if not turns:
        return "S0", False
    midpoint = (word.start + word.end) / 2
    covering = [t for t in turns if t.start <= midpoint < t.end]
    if len(covering) == 1:
        return covering[0].speaker, False
    if covering:  # overlapping speech
        best = max(covering, key=lambda t: min(t.end, word.end) - max(t.start, word.start))
        return best.speaker, True
    nearest = min(turns, key=lambda t: max(t.start - midpoint, midpoint - t.end))
    return nearest.speaker, False


def relabel_speakers(
    entries: list[TranscriptEntry], template: str = "Speaker {n}"
) -> list[TranscriptEntry]:
    """Map raw ``S<n>`` labels to display names, numbered by first appearance."""
    mapping: dict[str, str] = {}
    result = []
    for entry in entries:
        if entry.speaker not in mapping:
            mapping[entry.speaker] = template.format(n=len(mapping) + 1)
        result.append(
            TranscriptEntry(
                speaker=mapping[entry.speaker],
                text=entry.text,
                start=entry.start,
                end=entry.end,
                provisional=entry.provisional,
            )
        )
    return result
